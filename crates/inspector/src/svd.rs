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

    singular_values_from_grams_rm(&a, &b, rank)
}

/// Plain row-major `A @ B` for square `n × n` matrices. `rank` is small
/// (typically well under a few hundred), so an unaccelerated triple loop
/// here is negligible compared to the eigendecompositions around it.
fn matmul_rm_square(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                s += a[i * n + k] * b[k * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

/// Continuation of `singular_values_rm` starting from the already-computed
/// `rank × rank` Gram matrices `A = up^T @ up` and `B = down @ down^T`
/// (both row-major). Factored out so callers with a `rank × out`/`rank × in`
/// contraction (which dominates cost for wide LoRA layers) can compute the
/// Gram matrices with a faster backend than the plain Rust loops here.
///
/// Only needs **one** eigendecomposition-with-eigenvectors plus one
/// eigenvalues-only decomposition, rather than three, via the identity:
/// `eig((up@down)(up@down)^T) == eig(up @ B @ up^T) == eig(B^(1/2) @ A @ B^(1/2))`
/// (nonzero eigenvalues of `X@Y` equal those of `Y@X` for `X = up @ B^(1/2)`,
/// `Y = B^(1/2) @ up^T`). So the squared singular values of `up @ down` are
/// exactly the eigenvalues of `M = B^(1/2) @ A @ B^(1/2)`.
fn singular_values_from_grams_rm(a: &[f64], b: &[f64], rank: usize) -> Vec<f64> {
    // Eigendecomp: B = Q_b D_b Q_b^T, used to build B^(1/2) = Q_b diag(sqrt(|D_b|)) Q_b^T.
    let (lam_b, q_b) = jacobi_sym_rm(b, rank);
    let s_b: Vec<f64> = lam_b.iter().map(|v| v.abs().sqrt()).collect();

    let mut b_half = vec![0.0f64; rank * rank];
    for i in 0..rank {
        for j in 0..rank {
            let mut s = 0.0;
            for k in 0..rank {
                s += q_b[i * rank + k] * s_b[k] * q_b[j * rank + k];
            }
            b_half[i * rank + j] = s;
        }
    }

    // M = B^(1/2) @ A @ B^(1/2)  (rank × rank, symmetric)
    let m = matmul_rm_square(&matmul_rm_square(&b_half, a, rank), &b_half, rank);

    // Eigenvalues of M are the squared singular values of up @ down; eigenvectors
    // unused, so skip accumulating them (symmetric matrix: row-major == column-major).
    let lam_m = jacobi::jacobi_eigenvalues_only(&m, rank);

    lam_m.iter().map(|v| v.abs().sqrt()).collect()
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
///
/// The `rank × out`/`rank × in_features` Gram matrices (`up^T @ up`,
/// `down @ down^T`) are computed via candle's matmul (backed by a real GEMM
/// kernel) rather than the plain nested-loop `f64` fallback used by
/// [`singular_values_rm`], since for wide LoRA layers (large `out`/`in_features`)
/// that contraction dominates cost. Only the resulting `rank × rank` matrices
/// are pulled out to `f64` for the Jacobi eigendecomposition, which is cheap
/// regardless of implementation at typical LoRA ranks.
pub fn singular_values(
    up: &candle_core::Tensor,
    down: &candle_core::Tensor,
) -> crate::Result<Vec<f64>> {
    let up2 = flatten_to_2d(up)?.to_dtype(candle_core::DType::F32)?;
    let down2 = flatten_to_2d(down)?.to_dtype(candle_core::DType::F32)?;

    let rank = up2.dim(1)?;

    // A = up^T @ up  (rank × rank), B = down @ down^T (rank × rank)
    let a = up2.t()?.matmul(&up2)?.to_dtype(candle_core::DType::F64)?;
    let b = down2
        .matmul(&down2.t()?)?
        .to_dtype(candle_core::DType::F64)?;

    let a_rm: Vec<f64> = a.flatten_all()?.to_vec1::<f64>()?;
    let b_rm: Vec<f64> = b.flatten_all()?.to_vec1::<f64>()?;

    Ok(singular_values_from_grams_rm(&a_rm, &b_rm, rank))
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
    /// Off-diagonal Frobenius norm of a column-major symmetric `n×n` matrix.
    fn off_diag_norm(a: &[f64], n: usize) -> f64 {
        let mut s = 0.0;
        for col in 0..n {
            for row in 0..col {
                let v = a[row + col * n];
                s += 2.0 * v * v;
            }
        }
        s.sqrt()
    }

    /// Cyclic Jacobi eigendecomposition of a symmetric n×n matrix stored
    /// column-major in `m` (length n*n).
    ///
    /// Sweeps through all `(p, q)` pairs in a fixed order each pass rather
    /// than searching for the globally-largest off-diagonal entry every
    /// rotation (classical Jacobi): a full sweep is O(n^2) rotations with no
    /// per-rotation search, converging in a small constant number of sweeps,
    /// vs. classical Jacobi's O(n^2) search repeated for as many rotations as
    /// it takes to converge (effectively O(n^4) for realistic matrices).
    ///
    /// Returns `(eigenvalues, eigenvectors_col_major)` sorted descending by
    /// eigenvalue.
    pub fn jacobi_sym(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        jacobi_sym_impl(m, n, true)
    }

    /// Same as [`jacobi_sym`] but skips accumulating eigenvectors, which is
    /// roughly half the per-rotation cost, for callers that only need
    /// eigenvalues.
    pub fn jacobi_eigenvalues_only(m: &[f64], n: usize) -> Vec<f64> {
        jacobi_sym_impl(m, n, false).0
    }

    fn jacobi_sym_impl(m: &[f64], n: usize, want_vectors: bool) -> (Vec<f64>, Vec<f64>) {
        debug_assert_eq!(
            m.len(),
            n * n,
            "jacobi_sym: input length {} != n*n={}",
            m.len(),
            n * n
        );
        let mut a = m.to_vec(); // working copy, column-major
        let mut v = if want_vectors {
            let mut v = vec![0.0f64; n * n];
            for i in 0..n {
                v[i * n + i] = 1.0;
            }
            v
        } else {
            Vec::new()
        };

        // Convergence is measured relative to the matrix's own scale so this
        // works for Gram matrices of any magnitude, not just those near 1.0.
        let scale = a
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let tol = f64::EPSILON * 4.0 * scale;
        // Cyclic Jacobi converges quadratically once off-diagonal entries are
        // small; a small constant number of sweeps is enough regardless of n.
        let max_sweeps = 60;

        for _ in 0..max_sweeps {
            if off_diag_norm(&a, n) < tol {
                break;
            }

            for q in 1..n {
                for p in 0..q {
                    let apq = a[p + q * n];
                    // Skip negligible/exactly-zero pivots: rotating on them is a
                    // no-op at best, and an exact zero would divide 0.0/0.0 (NaN)
                    // below. Classical Jacobi never reached this line for a zero
                    // pivot since it always picked the largest remaining entry;
                    // a fixed cyclic sweep visits every pair, so this guard is
                    // required for correctness, not just performance.
                    if apq == 0.0 {
                        continue;
                    }

                    let app = a[p + p * n];
                    let aqq = a[q + q * n];
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
                    if want_vectors {
                        for r in 0..n {
                            let vrp = v[r + p * n];
                            let vrq = v[r + q * n];
                            v[r + p * n] = c * vrp - s * vrq;
                            v[r + q * n] = s * vrp + c * vrq;
                        }
                    }
                }
            }
        }

        // Eigenvalues are now on the diagonal
        let mut pairs: Vec<(f64, usize)> = (0..n).map(|i| (a[i + i * n], i)).collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let eigenvalues: Vec<f64> = pairs.iter().map(|(val, _)| *val).collect();

        let eigenvectors = if want_vectors {
            // Reorder eigenvectors (columns) to match sorted eigenvalues
            let mut eigenvectors = vec![0.0f64; n * n];
            for (new_col, (_, old_col)) in pairs.iter().enumerate() {
                for row in 0..n {
                    eigenvectors[row + new_col * n] = v[row + old_col * n];
                }
            }
            eigenvectors
        } else {
            Vec::new()
        };

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

    /// Deterministic xorshift PRNG so regression fixtures are reproducible
    /// without pulling in a `rand` dependency.
    fn xorshift_vec(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn xorshift_sym(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 2.0 - 1.0
        };
        let mut m = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let v = next();
                m[i + j * n] = v;
                m[j + i * n] = v;
            }
        }
        m
    }

    /// Regression guard for the upcoming eigendecomposition-count reduction
    /// and Jacobi rewrite: pins today's `singular_values` output for a
    /// fixed-seed, moderately-sized (rank 32) random up/down pair.
    #[test]
    fn singular_values_matches_golden_reference() {
        use candle_core::{Device, Tensor};
        let dev = &Device::Cpu;
        let (out, rank, in_features) = (256, 32, 256);
        let up = Tensor::from_vec(xorshift_vec(out * rank, 1), (out, rank), dev).unwrap();
        let down = Tensor::from_vec(
            xorshift_vec(rank * in_features, 2),
            (rank, in_features),
            dev,
        )
        .unwrap();

        let svs = singular_values(&up, &down).unwrap();
        assert_eq!(svs.len(), rank);

        // Singular values must be sorted descending and non-negative.
        assert!(svs.windows(2).all(|w| w[0] >= w[1] - 1e-9));
        assert!(svs.iter().all(|&s| s >= 0.0));

        // Golden reference captured from this same implementation prior to
        // the eigendecomposition-count reduction / cyclic Jacobi rewrite.
        // Tolerance is relative (1e-3) since the Gram matrices are formed in
        // f32 (see `singular_values`), which already caps achievable precision
        // well below f64, independent of eigensolver choice.
        let golden_top5 = [
            123.21000480950784,
            120.68052912415921,
            116.7523811023853,
            114.77819517802014,
            111.50699348929611,
        ];
        for (i, &g) in golden_top5.iter().enumerate() {
            let rel = (svs[i] - g).abs() / g.abs().max(1e-9);
            assert!(rel < 1e-3, "sv[{i}]={} golden={g} rel={rel}", svs[i]);
        }
    }

    /// Randomized correctness sanity check for the Jacobi eigensolver:
    /// A @ V ≈ V @ diag(eigenvalues) and V is orthonormal.
    #[test]
    fn jacobi_sym_reconstructs_random_symmetric_matrix() {
        use super::jacobi::jacobi_sym;
        let n = 48;
        let m = xorshift_sym(n, 7);
        let (vals, vecs) = jacobi_sym(&m, n);

        // V^T V ≈ I
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|k| vecs[k + i * n] * vecs[k + j * n]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-8,
                    "V^T V [{i},{j}] = {dot}, expected {expected}"
                );
            }
        }

        // A @ V ≈ V @ diag(vals), checked column by column.
        for col in 0..n {
            for row in 0..n {
                let av: f64 = (0..n).map(|k| m[row + k * n] * vecs[k + col * n]).sum();
                let vlambda = vecs[row + col * n] * vals[col];
                assert!(
                    (av - vlambda).abs() < 1e-6,
                    "A@V[{row},{col}]={av} V@Lambda={vlambda}"
                );
            }
        }
    }

    /// Cyclic Jacobi visits every `(p, q)` pair each sweep, including exact
    /// zeros — unlike classical (max-pivot-search) Jacobi, which never landed
    /// on a zero entry. A block-diagonal matrix whose off-block entries are
    /// exactly zero AND whose diagonal happens to repeat across two different
    /// blocks is the true failure case for pair `(p, q)` straddling those
    /// blocks: `apq == 0.0` AND `aqq - app == 0.0`, so `tau = 0.0 / 0.0` is
    /// NaN without the zero-pivot guard, poisoning the whole matrix.
    ///
    /// (A matrix that's *already* fully diagonal doesn't exercise this: the
    /// off-diagonal-norm convergence check exits before any rotation runs.
    /// This one has genuine off-diagonal energy in one block, so it must
    /// iterate, and lands on the zero/equal-diagonal pair in the process.)
    #[test]
    fn jacobi_sym_handles_zero_pivot_with_equal_diagonal_without_nan() {
        use super::jacobi::jacobi_sym;
        let n = 4;
        // Column-major 4x4, block-diagonal: rows/cols {0,1} form a 2x2 block
        // with real off-diagonal energy; rows/cols {2,3} are zero off-diagonal
        // with equal diagonal value 2.0 (matching pair (2,3) exactly).
        let m = vec![
            1.0, 0.5, 0.0, 0.0, // col 0
            0.5, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 2.0, 0.0, // col 2
            0.0, 0.0, 0.0, 2.0, // col 3
        ];
        let (vals, vecs) = jacobi_sym(&m, n);

        assert!(vals.iter().all(|v| v.is_finite()), "vals={:?}", vals);
        assert!(vecs.iter().all(|v| v.is_finite()), "vecs contain NaN/inf");

        let mut expected = [1.5, 0.5, 2.0, 2.0];
        expected.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for (got, want) in vals.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-9, "got={got} want={want}");
        }
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
