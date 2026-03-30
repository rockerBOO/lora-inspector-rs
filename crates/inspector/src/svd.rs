/// Jacobi symmetric eigendecomposition and rank-health metrics.
///
/// All public functions take plain `Vec<f64>` / `&[f64]` so they have
/// no dependency on candle and are easy to unit-test.

pub mod jacobi {
    /// Jacobi eigendecomposition of a symmetric n×n matrix stored
    /// column-major in `m` (length n*n).
    ///
    /// Returns `(eigenvalues, eigenvectors_col_major)` sorted descending
    /// by eigenvalue.
    pub fn jacobi_sym(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
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
}
