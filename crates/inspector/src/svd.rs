/// Jacobi symmetric eigendecomposition and rank-health metrics.
///
/// All public functions take plain `Vec<f64>` / `&[f64]` so they have
/// no dependency on candle and are easy to unit-test.
use serde::{Deserialize, Serialize};

/// Health classification of a LoRA layer's rank usage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RankHealth {
    /// Effective rank ≈ 1 or top-1 energy ≈ 1: layer is dominated by a single direction.
    Collapsed,
    /// Balance >= 0.75: rank is well-utilised.
    Good,
    /// Balance >= 0.50: moderate rank utilisation.
    Ok,
    /// Balance >= 0.25: poor but non-trivial rank utilisation.
    Weak,
    /// Balance < 0.25: very poor rank utilisation.
    Poor,
}

/// Per-layer rank utilisation metrics derived from the singular values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankMetrics {
    /// Declared rank of the LoRA layer (size of the low-rank bottleneck).
    pub nominal_rank: usize,
    /// Shannon entropy-based effective rank: exp(H) where H = -Σ p·ln(p), p = s²/Σs².
    pub effective_rank: f64,
    /// Fraction of singular-value energy carried by the top singular value.
    pub top1_energy: f64,
    /// effective_rank / nominal_rank — 1.0 means perfectly balanced.
    pub balance: f64,
    /// Ratio s[0]/s[1] (None if rank < 2 or s[1] ≈ 0).
    pub dominance: Option<f64>,
    /// Qualitative health rating derived from balance and top1_energy.
    pub health: RankHealth,
}

/// Compute `RankMetrics` from a slice of singular values and the nominal rank.
///
/// `svs` need not be sorted and may contain zeros.
/// `nominal_rank` is the declared rank of the LoRA layer.
pub fn rank_metrics_from_svs(svs: &[f64], nominal_rank: usize) -> RankMetrics {
    const EPSILON: f64 = 1e-10;

    // Compute energy (squared singular values) and total energy
    let energies: Vec<f64> = svs.iter().map(|&s| s * s).collect();
    let total_energy: f64 = energies.iter().sum();

    // effective_rank via Shannon entropy over the energy distribution
    let effective_rank = if total_energy < EPSILON {
        1.0
    } else {
        let entropy: f64 = energies
            .iter()
            .filter(|&&e| e > EPSILON)
            .map(|&e| {
                let p = e / total_energy;
                -p * p.ln()
            })
            .sum();
        entropy.exp()
    };

    // top-1 energy fraction
    let top1_energy = if total_energy < EPSILON {
        1.0
    } else {
        energies.iter().cloned().fold(0.0_f64, f64::max) / total_energy
    };

    // balance = effective_rank / nominal_rank, clamped to [0,1]
    let balance = if nominal_rank == 0 {
        0.0
    } else {
        (effective_rank / nominal_rank as f64).min(1.0)
    };

    // dominance = s[0] / s[1] when meaningful
    let mut sorted_svs = svs.to_vec();
    sorted_svs.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let dominance = if sorted_svs.len() >= 2 && sorted_svs[1] > EPSILON {
        Some(sorted_svs[0] / sorted_svs[1])
    } else {
        None
    };

    // Health classification — check COLLAPSED first
    let health = if nominal_rank == 1 && (effective_rank - 1.0).abs() < 1e-6 {
        // rank-1 layer fully utilizing its single dimension — not collapsed
        RankHealth::Good
    } else if (effective_rank - 1.0).abs() < 1e-6 || top1_energy > 1.0 - 1e-6 {
        RankHealth::Collapsed
    } else if balance >= 0.75 {
        RankHealth::Good
    } else if balance >= 0.50 {
        RankHealth::Ok
    } else if balance >= 0.25 {
        RankHealth::Weak
    } else {
        RankHealth::Poor
    };

    RankMetrics {
        nominal_rank,
        effective_rank,
        top1_energy,
        balance,
        dominance,
        health,
    }
}

/// Compute `RankMetrics` from candle `up` and `down` tensors.
///
/// Calls `singular_values` then `rank_metrics_from_svs`.
pub fn rank_metrics(
    up: &candle_core::Tensor,
    down: &candle_core::Tensor,
) -> crate::Result<RankMetrics> {
    let svs = singular_values(up, down)?;
    let rank = svs.len();
    Ok(rank_metrics_from_svs(&svs, rank))
}

// ── Row-major helpers ────────────────────────────────────────────────────────

/// A^T A for row-major A of shape (rows × cols). Returns (cols × cols) row-major.
/// A[k,i] = a[k*cols + i]
fn matmul_rm_ata(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0;
            for k in 0..rows {
                s += a[k * cols + i] * a[k * cols + j];
            }
            out[i * cols + j] = s;
        }
    }
    out
}

/// A @ A^T for row-major A of shape (rows_a × cols). Returns (rows_a × rows_a) row-major.
/// B[i,j] = sum_k a[i*cols + k] * a[j*cols + k]
fn matmul_rm_abt(a: &[f64], rows_a: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows_a * rows_a];
    for i in 0..rows_a {
        for j in 0..rows_a {
            let mut s = 0.0;
            for k in 0..cols {
                s += a[i * cols + k] * a[j * cols + k];
            }
            out[i * rows_a + j] = s;
        }
    }
    out
}

/// A^T @ B for row-major A (rows × cols_a) and B (rows × cols_b). Returns (cols_a × cols_b) row-major.
/// Out[i,j] = sum_k a[k*cols_a + i] * b[k*cols_b + j]
fn matmul_rm_atb(a: &[f64], b: &[f64], rows: usize, cols_a: usize, cols_b: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; cols_a * cols_b];
    for i in 0..cols_a {
        for j in 0..cols_b {
            let mut s = 0.0;
            for k in 0..rows {
                s += a[k * cols_a + i] * b[k * cols_b + j];
            }
            out[i * cols_b + j] = s;
        }
    }
    out
}

/// Jacobi eigendecomp wrapper that accepts a row-major symmetric matrix
/// and delegates to the column-major `jacobi_sym`.
///
/// Symmetric matrix: A[i,j]==A[j,i] so row-major and col-major byte layout is identical.
/// We can pass m_rm directly to the col-major jacobi_sym.
/// However jacobi_sym returns col-major eigenvectors; callers of jacobi_sym_rm
/// expect row-major, so we must transpose.
fn jacobi_sym_rm(m_rm: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let (vals, vecs_cm) = jacobi::jacobi_sym(m_rm, n);
    // vecs_cm col-major: vecs_cm[row + col*n]
    // convert to row-major: vecs_rm[row*n + col]
    let vecs_rm: Vec<f64> = (0..n)
        .flat_map(|row| (0..n).map(move |col| (row, col)))
        .map(|(row, col)| vecs_cm[row + col * n])
        .collect();
    (vals, vecs_rm)
}

/// Compute singular values of `up @ down` from row-major buffers.
///
/// - `up_rm`:   row-major [out_features × rank]
/// - `down_rm`: row-major [rank × in_features]
/// - Returns singular values in descending order (length = rank).
pub fn singular_values_rm(
    up_rm: &[f64],
    down_rm: &[f64],
    out: usize,
    rank: usize,
    in_features: usize,
) -> Vec<f64> {
    debug_assert_eq!(up_rm.len(), out * rank, "up_rm length mismatch");
    debug_assert_eq!(down_rm.len(), rank * in_features, "down_rm length mismatch");

    // A = up^T @ up  (rank × rank, row-major)
    let a = matmul_rm_ata(up_rm, out, rank);
    // B = down @ down^T  (rank × rank, row-major)
    let b = matmul_rm_abt(down_rm, rank, in_features);

    // Eigendecomp: A = Q_a D_a Q_a^T,  eigenvalues = S_up^2
    let (lam_a, q_a) = jacobi_sym_rm(&a, rank);
    // Eigendecomp: B = Q_b D_b Q_b^T,  eigenvalues = S_dn^2
    let (lam_b, q_b) = jacobi_sym_rm(&b, rank);

    // S_up = sqrt(|lam_a|), S_dn = sqrt(|lam_b|)
    let s_up: Vec<f64> = lam_a.iter().map(|v| v.abs().sqrt()).collect();
    let s_dn: Vec<f64> = lam_b.iter().map(|v| v.abs().sqrt()).collect();

    // C = diag(s_up) @ Q_a^T @ Q_b @ diag(s_dn)   (rank × rank)
    // Step 1: T1 = Q_a^T @ Q_b  (row-major, cols_a=rank, cols_b=rank, rows=rank)
    let t1 = matmul_rm_atb(&q_a, &q_b, rank, rank, rank);
    // Step 2: scale rows by s_up and cols by s_dn
    let mut c = vec![0.0f64; rank * rank];
    for row in 0..rank {
        for col in 0..rank {
            c[row * rank + col] = s_up[row] * t1[row * rank + col] * s_dn[col];
        }
    }

    // Singular values of C = singular values of up @ down
    // Compute C^T C (row-major) and eigendecomp
    let ctc = matmul_rm_ata(&c, rank, rank);
    let (lam_c, _) = jacobi_sym_rm(&ctc, rank);

    // Singular values = sqrt(eigenvalues of C^T C), descending
    lam_c.iter().map(|v| v.abs().sqrt()).collect()
}

// ── Candle tensor interface ──────────────────────────────────────────────────

/// Reshape a conv weight tensor to 2-D.
/// [d0, d1, 1, 1] → [d0, d1], [d0, d1, d2, d3] → [d0, d1*d2*d3], 2-D unchanged.
pub fn flatten_to_2d(t: &candle_core::Tensor) -> crate::Result<candle_core::Tensor> {
    match t.dims() {
        [d0, d1, 1, 1] => Ok(t.reshape(&[*d0, *d1])?),
        [d0, d1, d2, d3] => Ok(t.reshape(&[*d0, *d1 * *d2 * *d3])?),
        [_, _] => Ok(t.clone()),
        dims => Err(crate::InspectorError::Msg(format!(
            "flatten_to_2d: unsupported tensor dims {:?}",
            dims
        ))),
    }
}

/// Compute singular values of `up @ down` from candle tensors.
///
/// Handles conv shapes [out, rank, 1, 1] transparently.
/// Returns singular values in descending order (length = rank).
pub fn singular_values(
    up: &candle_core::Tensor,
    down: &candle_core::Tensor,
) -> crate::Result<Vec<f64>> {
    let up2 = flatten_to_2d(up)?;
    let down2 = flatten_to_2d(down)?;

    let out = up2.dim(0)?;
    let rank = up2.dim(1)?;
    let in_features = down2.dim(1)?;

    // to_vec2 produces row-major (standard Rust nested Vec layout)
    // BF16 is not directly castable to F64; promote to F32 first.
    let up_f32 = match up2.dtype() {
        candle_core::DType::BF16 => up2.to_dtype(candle_core::DType::F32)?,
        _ => up2.clone(),
    };
    let up_rm: Vec<f64> = up_f32
        .to_dtype(candle_core::DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;
    let down_f32 = match down2.dtype() {
        candle_core::DType::BF16 => down2.to_dtype(candle_core::DType::F32)?,
        _ => down2.clone(),
    };
    let down_rm: Vec<f64> = down_f32
        .to_dtype(candle_core::DType::F64)?
        .flatten_all()?
        .to_vec1::<f64>()?;

    Ok(singular_values_rm(&up_rm, &down_rm, out, rank, in_features))
}

/// Compute singular values of `up @ down` using the rank×rank core trick.
///
/// - `up_cm`:   column-major [out_features × rank]
/// - `down_cm`: column-major [rank × in_features]
/// - Returns singular values in descending order (length = rank).
#[cfg(test)]
pub(crate) fn singular_values_from_vecs(
    up_cm: &[f64],
    down_cm: &[f64],
    out: usize,
    rank: usize,
    in_features: usize,
) -> Vec<f64> {
    use jacobi::jacobi_sym;

    debug_assert_eq!(up_cm.len(), out * rank, "up_cm length mismatch");
    debug_assert_eq!(down_cm.len(), rank * in_features, "down_cm length mismatch");

    // A = up^T @ up  (rank × rank, column-major)
    let a = matmul_cm_ata(up_cm, out, rank);
    // B = down @ down^T  (rank × rank, column-major)
    let b = matmul_cm_abt(down_cm, rank, in_features);

    // Eigendecomp: A = Q_a D_a Q_a^T,  eigenvalues = S_up^2
    let (lam_a, q_a) = jacobi_sym(&a, rank);
    // Eigendecomp: B = Q_b D_b Q_b^T,  eigenvalues = S_dn^2
    let (lam_b, q_b) = jacobi_sym(&b, rank);

    // S_up = sqrt(|lam_a|), S_dn = sqrt(|lam_b|)
    let s_up: Vec<f64> = lam_a.iter().map(|v| v.abs().sqrt()).collect();
    let s_dn: Vec<f64> = lam_b.iter().map(|v| v.abs().sqrt()).collect();

    // C = diag(s_up) @ Q_a^T @ Q_b @ diag(s_dn)   (rank × rank)
    // Step 1: T1 = Q_a^T @ Q_b
    let t1 = matmul_cm_atb(&q_a, &q_b, rank, rank, rank);
    // Step 2: scale rows by s_up and cols by s_dn
    let mut c = vec![0.0f64; rank * rank];
    for col in 0..rank {
        for row in 0..rank {
            c[row + col * rank] = s_up[row] * t1[row + col * rank] * s_dn[col];
        }
    }

    // Singular values of C = singular values of up @ down
    // Compute C^T C and eigendecomp
    let ctc = matmul_cm_ata(&c, rank, rank);
    let (lam_c, _) = jacobi_sym(&ctc, rank);

    // Singular values = sqrt(eigenvalues of C^T C), descending
    lam_c.iter().map(|v| v.abs().sqrt()).collect()
}

// ── helpers ──────────────────────────────────────────────────────────────────

/// A^T A for column-major A of shape (rows × cols). Returns (cols × cols) col-major.
#[cfg(test)]
fn matmul_cm_ata(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; cols * cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0;
            for k in 0..rows {
                s += a[k + i * rows] * a[k + j * rows];
            }
            out[i + j * cols] = s;
        }
    }
    out
}

/// A @ A^T for column-major A of shape (rows_a × cols_a). Returns (rows_a × rows_a) col-major.
///
/// B[i,j] = sum_k a[i + k*rows_a] * a[j + k*rows_a]
#[cfg(test)]
fn matmul_cm_abt(a: &[f64], rows_a: usize, cols_a: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows_a * rows_a];
    for i in 0..rows_a {
        for j in 0..rows_a {
            let mut s = 0.0;
            for k in 0..cols_a {
                s += a[i + k * rows_a] * a[j + k * rows_a];
            }
            out[i + j * rows_a] = s;
        }
    }
    out
}

/// A^T @ B for col-major A (rows × cols_a) and B (rows × cols_b). Returns (cols_a × cols_b) col-major.
#[cfg(test)]
fn matmul_cm_atb(a: &[f64], b: &[f64], rows: usize, cols_a: usize, cols_b: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; cols_a * cols_b];
    for j in 0..cols_b {
        for i in 0..cols_a {
            let mut s = 0.0;
            for k in 0..rows {
                s += a[k + i * rows] * b[k + j * rows];
            }
            out[i + j * cols_a] = s;
        }
    }
    out
}

pub mod jacobi {
    /// Jacobi eigendecomposition of a symmetric n×n matrix stored
    /// column-major in `m` (length n*n).
    ///
    /// Returns `(eigenvalues, eigenvectors_col_major)` sorted descending
    /// by eigenvalue.
    pub fn jacobi_sym(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        debug_assert_eq!(
            m.len(),
            n * n,
            "jacobi_sym: input length {} != n*n={}",
            m.len(),
            n * n
        );
        let mut a = m.to_vec(); // working copy, column-major
                                // eigenvector matrix starts as identity
        let mut v = vec![0.0f64; n * n];
        for i in 0..n {
            v[i * n + i] = 1.0;
        }

        let max_sweeps = 100 * n * n;
        let eps = f64::EPSILON * 4.0;

        for _ in 0..max_sweeps {
            // Find largest off-diagonal |a[p,q]| (column-major: a[p + q*n])
            let mut max_val = 0.0f64;
            let mut p = 0usize;
            let mut q = 1usize;
            for col in 0..n {
                for row in 0..col {
                    let val = a[row + col * n].abs();
                    if val > max_val {
                        max_val = val;
                        p = row;
                        q = col;
                    }
                }
            }
            if max_val < eps {
                break;
            }

            // Compute rotation angle
            let app = a[p + p * n];
            let aqq = a[q + q * n];
            let apq = a[p + q * n];
            let tau = (aqq - app) / (2.0 * apq);
            let t = if tau >= 0.0 {
                1.0 / (tau + (1.0 + tau * tau).sqrt())
            } else {
                -1.0 / (-tau + (1.0 + tau * tau).sqrt())
            };
            let c = 1.0 / (1.0 + t * t).sqrt();
            let s = t * c;

            // Update diagonal
            a[p + p * n] = app - t * apq;
            a[q + q * n] = aqq + t * apq;
            a[p + q * n] = 0.0;
            a[q + p * n] = 0.0;

            // Update remaining rows/cols
            for r in 0..n {
                if r == p || r == q {
                    continue;
                }
                let arp = a[r + p * n];
                let arq = a[r + q * n];
                a[r + p * n] = c * arp - s * arq;
                a[p + r * n] = a[r + p * n];
                a[r + q * n] = s * arp + c * arq;
                a[q + r * n] = a[r + q * n];
            }

            // Accumulate eigenvectors
            for r in 0..n {
                let vrp = v[r + p * n];
                let vrq = v[r + q * n];
                v[r + p * n] = c * vrp - s * vrq;
                v[r + q * n] = s * vrp + c * vrq;
            }
        }

        // Eigenvalues are now on the diagonal
        let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i + i * n], i)).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<f64> = pairs.iter().map(|(val, _)| *val).collect();

        // Reorder eigenvectors (columns) to match sorted eigenvalues
        let mut eigenvectors = vec![0.0f64; n * n];
        for (new_col, (_, old_col)) in pairs.iter().enumerate() {
            for row in 0..n {
                eigenvectors[row + new_col * n] = v[row + old_col * n];
            }
        }

        (eigenvalues, eigenvectors)
    }
}

#[cfg(test)]
mod tests {
    use super::jacobi::jacobi_sym;

    fn nearly_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    /// 2×2 diagonal — eigenvalues must come back sorted descending.
    #[test]
    fn diagonal_2x2() {
        // [[4, 0], [0, 1]] — eigenvalues 4, 1
        let m = vec![4.0, 0.0, 0.0, 1.0]; // column-major
        let (vals, vecs) = jacobi_sym(&m, 2);
        assert!(nearly_eq(vals[0], 4.0), "got {}", vals[0]);
        assert!(nearly_eq(vals[1], 1.0), "got {}", vals[1]);
        // eigenvectors should be identity (columns = [1,0] and [0,1])
        assert!(nearly_eq(vecs[0].abs(), 1.0)); // col 0, row 0
        assert!(nearly_eq(vecs[1].abs(), 0.0)); // col 0, row 1
    }

    /// 2×2 symmetric off-diagonal.
    #[test]
    fn symmetric_2x2() {
        // [[2, 1], [1, 2]] — eigenvalues 3, 1
        let m = vec![2.0, 1.0, 1.0, 2.0]; // column-major: col0=[2,1], col1=[1,2]
        let (vals, _vecs) = jacobi_sym(&m, 2);
        assert!(nearly_eq(vals[0], 3.0), "got {}", vals[0]);
        assert!(nearly_eq(vals[1], 1.0), "got {}", vals[1]);
    }

    /// 3×3 known eigenvalues.
    #[test]
    fn symmetric_3x3() {
        // [[4,1,0],[1,3,0],[0,0,2]]
        // 2×2 submatrix [[4,1],[1,3]]: char poly λ²-7λ+11=0 → λ=(7±√5)/2
        // Sorted descending: (7+√5)/2 ≈ 4.618, (7-√5)/2 ≈ 2.382, 2.0
        let m = vec![4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        let (vals, _vecs) = jacobi_sym(&m, 3);
        let lam0 = (7.0 + 5.0_f64.sqrt()) / 2.0;
        let lam1 = (7.0 - 5.0_f64.sqrt()) / 2.0;
        assert!(nearly_eq(vals[0], lam0), "got {}", vals[0]);
        assert!(nearly_eq(vals[1], lam1), "got {}", vals[1]);
        assert!(nearly_eq(vals[2], 2.0), "got {}", vals[2]);
    }

    #[test]
    fn eigenvectors_orthonormal() {
        let m = vec![4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        let (_vals, vecs) = jacobi_sym(&m, 3);
        // V^T V should be identity: sum of col_i * col_j = delta_ij
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3).map(|k| vecs[k + i * 3] * vecs[k + j * 3]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-9,
                    "V^T V [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn scalar_n1() {
        let m = vec![7.0];
        let (vals, vecs) = jacobi_sym(&m, 1);
        assert!((vals[0] - 7.0).abs() < 1e-12);
        assert!((vecs[0] - 1.0).abs() < 1e-12);
    }

    use super::singular_values_from_vecs;

    /// Known 2×2 case: up=[1,0,0,1] (identity 2×2), down=[2,0,0,3] (diag).
    /// up @ down = [[2,0],[0,3]] — singular values 3, 2.
    #[test]
    fn singular_values_2x2_identity_up() {
        // up: 2×2 identity (out=2, rank=2), column-major
        let up = vec![1.0, 0.0, 0.0, 1.0]; // [[1,0],[0,1]]
                                           // down: 2×2 diagonal (rank=2, in=2), column-major
        let down = vec![2.0, 0.0, 0.0, 3.0]; // [[2,0],[0,3]]
        let svs = singular_values_from_vecs(&up, &down, 2, 2, 2);
        assert!((svs[0] - 3.0).abs() < 1e-6, "sv[0]={}", svs[0]);
        assert!((svs[1] - 2.0).abs() < 1e-6, "sv[1]={}", svs[1]);
    }

    #[test]
    fn singular_values_nontrivial() {
        // up: 3×2 col-major: [[1,2],[3,4],[5,6]]
        // col-major storage: col0=[1,3,5], col1=[2,4,6] → [1,3,5,2,4,6]
        // down: 2×3 col-major: [[7,8,9],[10,11,12]]
        // col-major storage: col0=[7,10], col1=[8,11], col2=[9,12] → [7,10,8,11,9,12]
        // up @ down = [[27,30,33],[61,68,75],[95,106,117]]
        // numpy: np.linalg.svd([[27,30,33],[61,68,75],[95,106,117]], compute_uv=False)
        // ≈ [225.029, 0.160, ~0.0]  (rank-2 product so third sv ≈ 0)
        let up_cm = vec![1.0f64, 3.0, 5.0, 2.0, 4.0, 6.0]; // col-major 3×2
        let down_cm = vec![7.0f64, 10.0, 8.0, 11.0, 9.0, 12.0]; // col-major 2×3
        let svs = singular_values_from_vecs(&up_cm, &down_cm, 3, 2, 3);
        // Top singular value should be ≈ 225.029 (verified with numpy)
        assert!((svs[0] - 225.029).abs() < 0.01, "sv[0]={:.4}", svs[0]);
        // Second sv ≈ 0.160 (rank-2 so this is small but nonzero)
        assert!(svs[1] < 1.0 && svs[1] > 0.01, "sv[1]={:.4}", svs[1]);
    }

    /// 2×4 up × 4×3 down — singular values of the product.
    #[test]
    fn singular_values_rectangular() {
        // up: out=2, rank=4  (column-major, 2×4)
        // down: rank=4, in=3 (column-major, 4×3)
        // We pick up = [[1,0,0,0],[0,1,0,0]] (first 2 rows of 4×4 identity)
        // and down = [[5,0,0],[0,4,0],[0,0,3],[0,0,0]]
        // up@down = [[5,0,0],[0,4,0]] — singular values 5,4
        let up = vec![1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; // col-major 2×4
        let down = vec![5.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]; // col-major 4×3
        let svs = singular_values_from_vecs(&up, &down, 2, 4, 3);
        // rank=4 so 4 singular values; top 2 should be 5,4
        assert!((svs[0] - 5.0).abs() < 1e-5, "sv[0]={}", svs[0]);
        assert!((svs[1] - 4.0).abs() < 1e-5, "sv[1]={}", svs[1]);
    }

    // ── Candle tensor tests ─────────────────────────────────────────────────

    use super::singular_values;

    /// Non-symmetric candle test: up 3×2, down 2×3.
    ///
    /// up = [[1,2],[3,4],[5,6]] (row-major: [1,2,3,4,5,6])
    /// down = [[7,8,9],[10,11,12]] (row-major: [7,8,9,10,11,12])
    /// up @ down = [[27,30,33],[61,68,75],[95,106,117]]
    /// top singular value ≈ 225.029 (verified with numpy)
    #[test]
    fn singular_values_candle_nonsymmetric() {
        let dev = &candle_core::Device::Cpu;
        let up = candle_core::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2), dev)
            .unwrap();
        let down =
            candle_core::Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (2, 3), dev)
                .unwrap();
        let svs = singular_values(&up, &down).unwrap();
        assert!((svs[0] - 225.029).abs() < 0.01, "sv[0]={:.4}", svs[0]);
        // Second sv should be small but nonzero (rank-2 product)
        assert!(svs[1] < 1.0 && svs[1] > 0.01, "sv[1]={:.4}", svs[1]);
    }

    /// Identity × diagonal: up 2×2 identity, down 2×2 diagonal [[3,0],[0,2]].
    /// up @ down = [[3,0],[0,2]] — singular values 3, 2.
    #[test]
    fn singular_values_candle_identity_diag() {
        let dev = &candle_core::Device::Cpu;
        let up = candle_core::Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), dev).unwrap();
        let down = candle_core::Tensor::from_vec(vec![3.0f32, 0.0, 0.0, 2.0], (2, 2), dev).unwrap();
        let svs = singular_values(&up, &down).unwrap();
        assert!((svs[0] - 3.0).abs() < 1e-5, "sv[0]={:.6}", svs[0]);
        assert!((svs[1] - 2.0).abs() < 1e-5, "sv[1]={:.6}", svs[1]);
    }

    /// Rank-1 case: up [3,1] = [[2],[0],[0]], down [1,3] = [[1,0,0]].
    /// product = [[2,0,0],[0,0,0],[0,0,0]] — single sv = 2.0
    #[test]
    fn singular_values_rank1_candle() {
        use super::singular_values;
        use candle_core::{Device, Tensor};
        let dev = &Device::Cpu;
        let up = Tensor::from_vec(vec![2.0f32, 0.0, 0.0], (3, 1), dev).unwrap();
        let down = Tensor::from_vec(vec![1.0f32, 0.0, 0.0], (1, 3), dev).unwrap();
        let svs = singular_values(&up, &down).unwrap();
        assert!((svs[0] - 2.0).abs() < 1e-5, "sv[0]={}", svs[0]);
    }

    // ── RankMetrics tests ───────────────────────────────────────────────────

    use super::{rank_metrics_from_svs, RankHealth};

    #[test]
    fn rank_metrics_balanced() {
        // equal svs → effective_rank == nominal_rank, balance == 1.0
        let svs = vec![1.0f64, 1.0, 1.0];
        let m = rank_metrics_from_svs(&svs, 3);
        assert!(
            (m.effective_rank - 3.0).abs() < 1e-6,
            "effective_rank={}",
            m.effective_rank
        );
        assert!((m.balance - 1.0).abs() < 1e-6, "balance={}", m.balance);
        assert!(
            (m.top1_energy - 1.0 / 3.0).abs() < 1e-6,
            "top1_energy={}",
            m.top1_energy
        );
        assert!(
            matches!(m.health, RankHealth::Good),
            "health={:?}",
            m.health
        );
    }

    #[test]
    fn rank_metrics_collapsed() {
        let svs = vec![5.0f64, 0.0, 0.0, 0.0];
        let m = rank_metrics_from_svs(&svs, 4);
        assert!(
            (m.top1_energy - 1.0).abs() < 1e-10,
            "top1_energy={}",
            m.top1_energy
        );
        assert!(
            matches!(m.health, RankHealth::Collapsed),
            "health={:?}",
            m.health
        );
    }

    /// Conv weight shape [out, rank, 1, 1] should be handled by flatten_to_2d.
    #[test]
    fn singular_values_candle_conv_shape() {
        use super::super::svd::flatten_to_2d;
        let dev = &candle_core::Device::Cpu;
        // up: [3, 2, 1, 1] — same data as 3×2 non-symmetric test
        let up4 =
            candle_core::Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2, 1, 1), dev)
                .unwrap();
        let up2 = flatten_to_2d(&up4).unwrap();
        assert_eq!(up2.dims(), &[3, 2], "flatten_to_2d shape mismatch");

        // Verify singular_values works end-to-end with the flattened tensor
        let down =
            candle_core::Tensor::from_vec(vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0], (2, 3), dev)
                .unwrap();
        let svs = singular_values(&up4, &down).unwrap();
        assert!((svs[0] - 225.029).abs() < 0.01, "sv[0]={:.4}", svs[0]);
    }

    #[test]
    fn rank_metrics_rank1_is_good() {
        // A single nonzero sv with nominal_rank=1 should be Good, not Collapsed
        let svs = vec![3.0f64];
        let m = rank_metrics_from_svs(&svs, 1);
        assert!(
            matches!(m.health, RankHealth::Good),
            "expected Good, got {:?}",
            m.health
        );
        assert!((m.balance - 1.0).abs() < 1e-6);
        assert_eq!(m.dominance, None);
    }

    #[test]
    fn rank_metrics_dominated_not_collapsed() {
        // [10, 1, 0, 0, 0, 0, 0, 0] — heavy but has 2 nonzero svs, not collapsed
        let svs = vec![10.0f64, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let m = rank_metrics_from_svs(&svs, 8);
        // effective_rank ≈ 1.06 — dominated but not collapsed
        assert!(
            !matches!(m.health, RankHealth::Collapsed),
            "should not be Collapsed, got {:?}",
            m.health
        );
        assert!(m.top1_energy < 1.0 - 1e-6);
    }
}
