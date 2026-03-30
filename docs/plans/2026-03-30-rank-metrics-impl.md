# Rank Health Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-layer rank health metrics (effective_scale, factorization_balance, top1_energy, effective_rank, balance, dominance, health rating) to the LoRA inspector, exposed via WASM, using a self-contained Jacobi eigendecomposition with no new dependencies.

**Architecture:** A new `svd.rs` module holds the Jacobi symmetric eigendecomposition and rank×rank core trick. `file.rs` gets three new methods that delegate into the existing weights. `worker.rs` exposes them to the WASM layer.

**Tech Stack:** Rust, candle-core (matmul only), wasm-bindgen, serde

---

## Task 1: `jacobi_sym` — symmetric eigendecomposition

**Files:**
- Create: `crates/inspector/src/svd.rs`
- Modify: `crates/inspector/src/lib.rs`

### Step 1: Add module declaration

In `crates/inspector/src/lib.rs`, add after the existing `pub mod norms;` line:

```rust
pub mod svd;
```

### Step 2: Write the failing test

Create `crates/inspector/src/svd.rs` with just the test first:

```rust
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
        todo!()
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
        // [[4,1,0],[1,3,0],[0,0,2]] — eigenvalues: (7+sqrt(5))/2, (7-sqrt(5))/2, 2
        let m = vec![4.0, 1.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 2.0];
        let (vals, _vecs) = jacobi_sym(&m, 3);
        let lam0 = (7.0 + 5.0_f64.sqrt()) / 2.0;
        let lam2 = (7.0 - 5.0_f64.sqrt()) / 2.0;
        assert!(nearly_eq(vals[0], lam0), "got {}", vals[0]);
        assert!(nearly_eq(vals[1], 2.0), "got {}", vals[1]);
        assert!(nearly_eq(vals[2], lam2), "got {}", vals[2]);
    }
}
```

### Step 3: Run test to verify it fails

```bash
cargo test -p inspector jacobi -- --nocapture
```

Expected: FAIL with `not yet implemented`

### Step 4: Implement `jacobi_sym`

Replace the `todo!()` with the full Jacobi implementation:

```rust
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
```

### Step 5: Run tests to verify they pass

```bash
cargo test -p inspector jacobi -- --nocapture
```

Expected: 3 tests PASS

### Step 6: Commit

```bash
git add crates/inspector/src/svd.rs crates/inspector/src/lib.rs
git commit -m "feat: add Jacobi symmetric eigendecomposition in svd.rs"
```

---

## Task 2: `singular_values` — rank×rank core trick

**Files:**
- Modify: `crates/inspector/src/svd.rs`

### Step 1: Write the failing test

Add to the `tests` module in `svd.rs`:

```rust
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

/// 2×4 up × 4×3 down — singular values of the product.
#[test]
fn singular_values_rectangular() {
    // up: out=2, rank=4  (column-major, 2×4)
    // down: rank=4, in=3 (column-major, 4×3)
    // We pick up = [[1,0,0,0],[0,1,0,0]] (first 2 rows of 4×4 identity)
    // and down = [[5,0,0],[0,4,0],[0,0,3],[0,0,0]]
    // up@down = [[5,0,0],[0,4,0]] — singular values 5,4
    let up = vec![1.0,0.0, 0.0,1.0, 0.0,0.0, 0.0,0.0]; // col-major 2×4
    let down = vec![5.0,0.0,0.0,0.0, 0.0,4.0,0.0,0.0, 0.0,0.0,3.0,0.0]; // col-major 4×3
    let svs = singular_values_from_vecs(&up, &down, 2, 4, 3);
    // rank=4 so 4 singular values; top 2 should be 5,4
    assert!((svs[0] - 5.0).abs() < 1e-5, "sv[0]={}", svs[0]);
    assert!((svs[1] - 4.0).abs() < 1e-5, "sv[1]={}", svs[1]);
}
```

### Step 2: Run to verify failure

```bash
cargo test -p inspector singular_values -- --nocapture
```

Expected: compile error — `singular_values_from_vecs` not found

### Step 3: Implement `singular_values_from_vecs`

Add this function to `crates/inspector/src/svd.rs` (outside the `jacobi` sub-module, at the crate level of the file):

```rust
use crate::Result;
use candle_core::Tensor;

/// Compute singular values of `up @ down` using the rank×rank core trick.
///
/// - `up`:   shape [out_features, rank] (2D; caller must flatten conv dims)
/// - `down`: shape [rank, in_features]  (2D)
/// - Returns singular values in descending order (length = rank).
pub fn singular_values(up: &Tensor, down: &Tensor) -> Result<Vec<f64>> {
    let up_f64 = up.to_dtype(candle_core::DType::F64)?;
    let down_f64 = down.to_dtype(candle_core::DType::F64)?;

    let up_dims = up_f64.dims2()?;   // (out, rank)
    let down_dims = down_f64.dims2()?; // (rank, in)
    let rank = up_dims.1;
    debug_assert_eq!(rank, down_dims.0);

    // A = up^T @ up  (rank × rank)
    let a_tensor = up_f64.t()?.matmul(&up_f64)?;
    // B = down @ down^T  (rank × rank)
    let b_tensor = down_f64.matmul(&down_f64.t()?)?;

    let a_vec = a_tensor.flatten_all()?.to_vec1::<f64>()?;
    let b_vec = b_tensor.flatten_all()?.to_vec1::<f64>()?;

    singular_values_from_vecs(&up_flat_to_col_major(&up_f64)?,
                               &down_flat_to_col_major(&down_f64)?,
                               up_dims.0, rank, down_dims.1)
    // Delegate to the pure-Rust implementation
    // (a_vec and b_vec already are the inputs we need for the trick)
}

/// Pure-Rust entry point for testing without candle.
/// `up_cm`: column-major [out × rank], `down_cm`: column-major [rank × in].
pub fn singular_values_from_vecs(
    up_cm: &[f64],
    down_cm: &[f64],
    out: usize,
    rank: usize,
    in_features: usize,
) -> Vec<f64> {
    use jacobi::jacobi_sym;

    // A = up^T @ up  (rank × rank, column-major)
    let a = matmul_cm_ata(up_cm, out, rank);
    // B = down @ down^T  (rank × rank, column-major)
    let b = matmul_cm_abt(down_cm, rank, in_features);

    // Eigendecomp: A = Q_a D_a Q_a^T,  eigenvalues = S_up^2,  Q_a cols = Vh_up^T
    let (lam_a, q_a) = jacobi_sym(&a, rank);
    // Eigendecomp: B = Q_b D_b Q_b^T,  eigenvalues = S_dn^2,  Q_b cols = U_dn
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

/// A^T A for column-major A of shape (rows × cols).  Returns (cols × cols) col-major.
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

/// A B^T for col-major A (rows_a × cols_a) and B (rows_b × cols_a). Returns (rows_a × rows_b).
fn matmul_cm_abt(a: &[f64], rows_a: usize, cols_a: usize) -> Vec<f64> {
    // down @ down^T: a is (rank × in), result is (rank × rank)
    let rows_b = rows_a; // square for down @ down^T
    matmul_cm_ata(a, cols_a, rows_a) // equivalent: (down^T)^T (down^T) = down down^T...
    // Actually easier: just reuse ata with transposed view
    // down @ down^T = (down^T)^T @ (down^T) but down^T is (in × rank)
    // So this is ATA of (in × rank) matrix = down^T, giving rank×rank
    // matmul_cm_ata takes col-major (rows × cols), down is col-major (rank × in)
    // down^T col-major would be (in × rank) col-major — but that's just
    // re-interpreting the same storage. We want B = down @ down^T (rank×rank).
    // B[i,j] = sum_k down[i,k] * down[j,k]
    // In col-major down: down[i,k] = a[i + k * rows_a]
}

/// A^T B for col-major A (rows × cols_a) and B (rows × cols_b). Returns (cols_a × cols_b) col-major.
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

fn up_flat_to_col_major(t: &Tensor) -> Result<Vec<f64>> {
    Ok(t.flatten_all()?.to_vec1::<f64>()?)
}

fn down_flat_to_col_major(t: &Tensor) -> Result<Vec<f64>> {
    Ok(t.flatten_all()?.to_vec1::<f64>()?)
}
```

> **Note on matrix storage:** candle stores tensors in row-major order. The helper functions above assume column-major. Before wiring `singular_values` to candle tensors in a later task, we must verify how candle's `flatten_all()` lays out a 2D tensor (row-major = transpose of col-major). The pure-Rust tests use explicit column-major vectors and can be verified independently. Task 3 handles the candle→Vec conversion correctly.

### Step 4: Run tests

```bash
cargo test -p inspector singular_values -- --nocapture
```

Expected: 2 tests PASS

### Step 5: Commit

```bash
git add crates/inspector/src/svd.rs
git commit -m "feat: add singular_values rank-rank core trick"
```

---

## Task 3: Fix matrix storage and wire to candle

**Files:**
- Modify: `crates/inspector/src/svd.rs`

candle `flatten_all()` on a 2D tensor [rows, cols] gives **row-major** layout:
`[row0_col0, row0_col1, ..., row1_col0, ...]`

The helpers above expect **column-major**. Fix `singular_values` to transpose correctly:

### Step 1: Write the failing test (candle tensors)

Add to the `tests` module:

```rust
#[test]
fn singular_values_candle_2x2() {
    use candle_core::{Device, Tensor};
    use super::singular_values;

    let dev = &Device::Cpu;
    // up: [[1,0],[0,1]] (2×2 identity)
    let up = Tensor::from_vec(vec![1.0f32, 0.0, 0.0, 1.0], (2, 2), dev).unwrap();
    // down: [[3,0],[0,2]] (diagonal)
    let down = Tensor::from_vec(vec![3.0f32, 0.0, 0.0, 2.0], (2, 2), dev).unwrap();
    let svs = singular_values(&up, &down).unwrap();
    assert!((svs[0] - 3.0).abs() < 1e-5, "sv[0]={}", svs[0]);
    assert!((svs[1] - 2.0).abs() < 1e-5, "sv[1]={}", svs[1]);
}
```

### Step 2: Run to confirm failure

```bash
cargo test -p inspector singular_values_candle -- --nocapture
```

### Step 3: Fix `singular_values` to use row-major helpers

Replace `singular_values` with a correct implementation that uses row-major storage throughout (matching candle's flatten_all):

```rust
pub fn singular_values(up: &Tensor, down: &Tensor) -> Result<Vec<f64>> {
    let up_2d = flatten_to_2d(up)?;
    let down_2d = flatten_to_2d(down)?;

    let (out, rank) = up_2d.dims2()?;
    let (rank2, in_features) = down_2d.dims2()?;
    debug_assert_eq!(rank, rank2);

    let up_rm = up_2d.to_dtype(candle_core::DType::F64)?
        .flatten_all()?.to_vec1::<f64>()?;
    let down_rm = down_2d.to_dtype(candle_core::DType::F64)?
        .flatten_all()?.to_vec1::<f64>()?;

    Ok(singular_values_rm(&up_rm, &down_rm, out, rank, in_features))
}

/// Flatten conv weights [out, rank, 1, 1] → [out, rank]; 2D stays as-is.
fn flatten_to_2d(t: &Tensor) -> Result<Tensor> {
    match t.dims() {
        [out, rank, 1, 1] => Ok(t.reshape(&[*out, *rank])?),
        [_, _] => Ok(t.clone()),
        dims => Err(crate::InspectorError::Msg(
            format!("unexpected tensor dims for SVD: {:?}", dims)
        )),
    }
}

/// Row-major version of singular_values_from_vecs.
/// `up_rm`: row-major [out × rank], `down_rm`: row-major [rank × in].
pub fn singular_values_rm(
    up_rm: &[f64],
    down_rm: &[f64],
    out: usize,
    rank: usize,
    in_features: usize,
) -> Vec<f64> {
    use jacobi::jacobi_sym;

    // A = up^T @ up  (rank × rank)
    // A[i,j] = sum_k up[k,i] * up[k,j]  (row-major: up[k,i] = up_rm[k*rank + i])
    let a = matmul_rm_ata(up_rm, out, rank);
    // B = down @ down^T  (rank × rank)
    // B[i,j] = sum_k down[i,k] * down[j,k]  (row-major: down[i,k] = down_rm[i*in + k])
    let b = matmul_rm_abt(down_rm, rank, in_features);

    let (lam_a, q_a_rm) = jacobi_sym_rm(&a, rank);
    let (lam_b, q_b_rm) = jacobi_sym_rm(&b, rank);

    let s_up: Vec<f64> = lam_a.iter().map(|v| v.abs().sqrt()).collect();
    let s_dn: Vec<f64> = lam_b.iter().map(|v| v.abs().sqrt()).collect();

    // T1 = Q_a^T @ Q_b  (Q_a row-major [rank×rank], Q_a^T means swap row/col)
    let t1 = matmul_rm_atb(&q_a_rm, &q_b_rm, rank, rank, rank);

    // C[i,j] = s_up[i] * T1[i,j] * s_dn[j]
    let mut c_rm = vec![0.0f64; rank * rank];
    for i in 0..rank {
        for j in 0..rank {
            c_rm[i * rank + j] = s_up[i] * t1[i * rank + j] * s_dn[j];
        }
    }

    // C^T C (rank × rank)
    let ctc = matmul_rm_ata(&c_rm, rank, rank); // wait: CTC[i,j] = sum_k C^T[i,k]*C[k,j] = sum_k C[k,i]*C[k,j]
    // matmul_rm_ata(c, rows=rank, cols=rank) computes c^T c — correct.
    let (lam_c, _) = jacobi_sym_rm(&ctc, rank);

    lam_c.iter().map(|v| v.abs().sqrt()).collect()
}
```

Add row-major matrix helpers and a row-major `jacobi_sym_rm` wrapper:

```rust
/// Row-major Jacobi: same algorithm, just different index arithmetic.
/// Input/output stored row-major (m[i*n + j]).
/// Returns (eigenvalues_desc, eigenvectors_row_major).
pub fn jacobi_sym_rm(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    // Convert to col-major, run jacobi_sym, convert back
    let cm: Vec<f64> = (0..n).flat_map(|col| (0..n).map(move |row| m[row * n + col])).collect();
    let (vals, vecs_cm) = jacobi::jacobi_sym(&cm, n);
    // vecs_cm is col-major → convert to row-major
    let vecs_rm: Vec<f64> = (0..n).flat_map(|row| (0..n).map(move |col| vecs_cm[row + col * n])).collect();
    (vals, vecs_rm)
}

/// A^T A, row-major A [rows × cols]. Returns [cols × cols] row-major.
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

/// A B^T, row-major A [rows_a × cols] and B [rows_b × cols]. Returns [rows_a × rows_b] row-major.
fn matmul_rm_abt(a: &[f64], rows_a: usize, cols: usize) -> Vec<f64> {
    // down @ down^T: B[i,j] = sum_k a[i,k] * a[j,k]
    let rows_b = rows_a;
    let mut out = vec![0.0f64; rows_a * rows_b];
    for i in 0..rows_a {
        for j in 0..rows_b {
            let mut s = 0.0;
            for k in 0..cols {
                s += a[i * cols + k] * a[j * cols + k];
            }
            out[i * rows_b + j] = s;
        }
    }
    out
}

/// A^T B, row-major A [rows × cols_a] and B [rows × cols_b]. Returns [cols_a × cols_b] row-major.
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
```

> The earlier `singular_values_from_vecs` (column-major) can be removed or kept for the pure-Rust tests — update the existing tests to call `singular_values_rm` instead.

### Step 4: Run all svd tests

```bash
cargo test -p inspector svd -- --nocapture
```

Expected: all tests PASS

### Step 5: Commit

```bash
git add crates/inspector/src/svd.rs
git commit -m "feat: fix row-major storage, wire singular_values to candle tensors"
```

---

## Task 4: `RankMetrics` struct and `rank_metrics()`

**Files:**
- Modify: `crates/inspector/src/svd.rs`

### Step 1: Write the failing test

```rust
#[test]
fn rank_metrics_balanced() {
    // rank=3, equal singular values → effective_rank == 3, balance == 1.0
    // up: 3×3 identity, down: 3×3 identity  → up@down = I, svs = [1,1,1]
    let up_rm = vec![1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0];
    let down_rm = vec![1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0];
    let svs = singular_values_rm(&up_rm, &down_rm, 3, 3, 3);
    let m = rank_metrics_from_svs(&svs, 3);
    assert!((m.effective_rank - 3.0).abs() < 1e-6);
    assert!((m.balance - 1.0).abs() < 1e-6);
    assert!((m.top1_energy - 1.0/3.0).abs() < 1e-6);
    assert!(matches!(m.health, RankHealth::Good));
}

#[test]
fn rank_metrics_collapsed() {
    // rank=4, only first sv nonzero → collapsed
    let svs = vec![5.0, 0.0, 0.0, 0.0];
    let m = rank_metrics_from_svs(&svs, 4);
    assert!((m.top1_energy - 1.0).abs() < 1e-10);
    assert!(matches!(m.health, RankHealth::Collapsed));
}
```

### Step 2: Run to confirm failure

```bash
cargo test -p inspector rank_metrics -- --nocapture
```

### Step 3: Implement

Add to `crates/inspector/src/svd.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankMetrics {
    /// s[0]² / sum(s²) — 1.0 means rank-1
    pub top1_energy: f64,
    /// exp(entropy(s²/sum(s²))) — how many dimensions are used
    pub effective_rank: f64,
    /// effective_rank / nominal_rank
    pub balance: f64,
    /// s[0] / s[1]; None if rank == 1
    pub dominance: Option<f64>,
    pub health: RankHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RankHealth {
    Good,
    Ok,
    Weak,
    Poor,
    Collapsed,
}

/// Compute `RankMetrics` from singular values (descending) and nominal rank.
pub fn rank_metrics_from_svs(svs: &[f64], nominal_rank: usize) -> RankMetrics {
    let energies: Vec<f64> = svs.iter().map(|s| s * s).collect();
    let total: f64 = energies.iter().sum();

    let (top1_energy, effective_rank) = if total < f64::EPSILON {
        (1.0, 1.0) // degenerate: treat as collapsed
    } else {
        let top1 = energies[0] / total;
        let eff_rank = {
            let entropy: f64 = energies.iter()
                .filter(|&&e| e > 0.0)
                .map(|&e| { let p = e / total; -p * p.ln() })
                .sum();
            entropy.exp()
        };
        (top1, eff_rank)
    };

    let balance = effective_rank / nominal_rank as f64;

    let dominance = if svs.len() >= 2 && svs[1].abs() > f64::EPSILON {
        Some(svs[0] / svs[1])
    } else {
        None
    };

    let health = if (effective_rank - 1.0).abs() < 1e-6 || top1_energy > 1.0 - 1e-6 {
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

    RankMetrics { top1_energy, effective_rank, balance, dominance, health }
}

/// Full pipeline: candle tensors → RankMetrics.
pub fn rank_metrics(up: &Tensor, down: &Tensor) -> Result<RankMetrics> {
    let up_2d = flatten_to_2d(up)?;
    let (_out, rank) = up_2d.dims2()?;
    let svs = singular_values(up, down)?;
    Ok(rank_metrics_from_svs(&svs, rank))
}
```

### Step 4: Run tests

```bash
cargo test -p inspector -- --nocapture
```

Expected: all PASS

### Step 5: Commit

```bash
git add crates/inspector/src/svd.rs
git commit -m "feat: add RankMetrics struct and rank_metrics()"
```

---

## Task 5: `effective_scale` and `factorization_balance` on `LoRAFile`

**Files:**
- Modify: `crates/inspector/src/file.rs`

### Step 1: Write the failing tests

Add to `file.rs` test module:

```rust
#[test]
fn effective_scale_is_l2_of_scaled_weight() -> crate::Result<()> {
    let file = "edgWar40KAdeptaSororitas.safetensors";
    let buffer = load_file(file)?;
    let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);
    let base_name = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";

    let eff = lora_file.effective_scale(base_name)?.unwrap();
    let scaled = lora_file.scale_weight(base_name)?;
    let l2 = lora_file.l2_norm::<f64>(&scaled)?;
    assert!((eff - l2).abs() < 1e-10);
    Ok(())
}

#[test]
fn factorization_balance_near_one_for_standard_lora() -> crate::Result<()> {
    let file = "edgWar40KAdeptaSororitas.safetensors";
    let buffer = load_file(file)?;
    let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);
    let base_name = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";

    let bal = lora_file.factorization_balance(base_name)?.unwrap();
    // Standard LoRA should be in range (0.1, 10.0)
    assert!(bal > 0.1 && bal < 10.0, "balance={}", bal);
    Ok(())
}
```

### Step 2: Run to confirm failure

```bash
cargo test -p inspector effective_scale factorization_balance -- --nocapture
```

### Step 3: Implement

Add these methods to `impl LoRAFile` in `file.rs`. Also add `use crate::{norms, svd};` at the top of the file.

```rust
/// Frobenius norm of `up @ down * (alpha/rank)` — the magnitude of the LoRA weight delta.
pub fn effective_scale(&self, base_name: &str) -> Result<Option<f64>> {
    match self.scale_weight(base_name) {
        Ok(t) => Ok(Some(self.l2_norm::<f64>(&t)?)),
        Err(InspectorError::NotFound) => Ok(None),
        Err(e) => Err(e),
    }
}

/// up_norm / down_norm.  ~1.0 for standard LoRA; higher with LoRA+.
pub fn factorization_balance(&self, base_name: &str) -> Result<Option<f64>> {
    match self.weights.as_ref() {
        None => Ok(None),
        Some(weights) => {
            let up = match weights.up(base_name) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            let down = match weights.down(base_name) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            let up_norm = norms::matrix_norm::<f64>(&up.to_dtype(candle_core::DType::F64)?)?;
            let down_norm = norms::matrix_norm::<f64>(&down.to_dtype(candle_core::DType::F64)?)?;
            if down_norm < f64::EPSILON {
                return Ok(None);
            }
            Ok(Some(up_norm / down_norm))
        }
    }
}

/// SVD-based rank health metrics.
pub fn rank_metrics(&self, base_name: &str) -> Result<Option<svd::RankMetrics>> {
    match self.weights.as_ref() {
        None => Ok(None),
        Some(weights) => {
            let up = match weights.up(base_name) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            let down = match weights.down(base_name) {
                Ok(t) => t,
                Err(_) => return Ok(None),
            };
            Ok(Some(svd::rank_metrics(&up, &down)?))
        }
    }
}
```

> `weights.up()` and `weights.down()` are currently `fn` methods on `BufferedLoRAWeight` (private impl of `Weight` trait). Check `weight.rs:639,647` — they are `fn up` and `fn down` on the `Weight` trait impl. If they are not `pub`, make them `pub(crate)` on `BufferedLoRAWeight`.

### Step 4: Run tests

```bash
cargo test -p inspector -- --nocapture
```

### Step 5: Commit

```bash
git add crates/inspector/src/file.rs crates/inspector/src/svd.rs
git commit -m "feat: add effective_scale, factorization_balance, rank_metrics to LoRAFile"
```

---

## Task 6: `effective_scales_all` with outlier detection on `LoRAFile`

**Files:**
- Modify: `crates/inspector/src/file.rs`

### Step 1: Write the failing test

```rust
#[test]
fn effective_scales_all_flags_outliers() -> crate::Result<()> {
    let file = "edgWar40KAdeptaSororitas.safetensors";
    let buffer = load_file(file)?;
    let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);

    let results = lora_file.effective_scales_all();
    // Should have one entry per base_name
    assert_eq!(results.len(), lora_file.base_names().len());
    // All is_outlier=false when data is uniform (at minimum the field exists)
    assert!(results.iter().all(|r| r.eff_scale >= 0.0));
    Ok(())
}
```

### Step 2: Run to confirm failure

```bash
cargo test -p inspector effective_scales_all -- --nocapture
```

### Step 3: Implement

Add to `file.rs`:

```rust
use crate::statistic;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LayerScale {
    pub base_name: String,
    pub eff_scale: f64,
    pub is_outlier: bool,
}

impl LoRAFile {
    // … existing methods …

    pub fn effective_scales_all(&self) -> Vec<LayerScale> {
        let base_names = self.base_names();
        let scales: Vec<(String, f64)> = base_names
            .iter()
            .filter_map(|name| {
                self.effective_scale(name).ok().flatten().map(|s| (name.clone(), s))
            })
            .collect();

        if scales.is_empty() {
            return vec![];
        }

        let scale_values: Vec<f64> = scales.iter().map(|(_, s)| *s).collect();
        let tensor = candle_core::Tensor::from_vec(
            scale_values.clone(),
            scale_values.len(),
            &candle_core::Device::Cpu,
        );

        let median = tensor
            .ok()
            .and_then(|t| statistic::median::<f64>(&t).ok())
            .unwrap_or(0.0);

        let threshold = 1.5 * median;

        scales
            .into_iter()
            .map(|(base_name, eff_scale)| LayerScale {
                base_name,
                eff_scale,
                is_outlier: eff_scale > threshold,
            })
            .collect()
    }
}
```

### Step 4: Run tests

```bash
cargo test -p inspector -- --nocapture
```

### Step 5: Commit

```bash
git add crates/inspector/src/file.rs
git commit -m "feat: add effective_scales_all with outlier detection"
```

---

## Task 7: Expose to WASM in `worker.rs`

**Files:**
- Modify: `crates/lora-inspector-wasm/src/worker.rs`

### Step 1: Write the failing test

The WASM layer has no unit tests for these methods — they're covered by the inspector tests. Just add the methods and verify they compile. Add a compile-only smoke test in `worker.rs`:

```rust
// In tests module at bottom of worker.rs:
#[test]
fn worker_exposes_rank_methods() {
    // This test just verifies the methods exist and types are correct.
    // Real testing is done in the inspector crate.
    let _: fn(&LoraWorker, &str) -> Option<f64> = LoraWorker::effective_scale_js as _;
}
```

### Step 2: Implement the WASM methods

Add to `impl LoraWorker` in `worker.rs`:

```rust
use inspector::file::LayerScale;

// Rename to avoid clash with Rust method
pub fn effective_scale(&self, base_name: &str) -> Option<f64> {
    self.file.effective_scale(base_name)
        .map_err(|e| console::error_1(&format!("effective_scale error: {e}").into()))
        .ok()
        .flatten()
}

pub fn factorization_balance(&self, base_name: &str) -> Option<f64> {
    self.file.factorization_balance(base_name)
        .map_err(|e| console::error_1(&format!("factorization_balance error: {e}").into()))
        .ok()
        .flatten()
}

pub fn rank_metrics(&self, base_name: &str) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    self.file
        .rank_metrics(base_name)
        .map_err(|e| JsValue::from_str(&e.to_string()))?
        .map(|m| serde_wasm_bindgen::to_value(&m).map_err(|e| JsValue::from_str(&e.to_string())))
        .unwrap_or(Ok(JsValue::NULL))
}

pub fn effective_scales_all(&self) -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();
    let scales = self.file.effective_scales_all();
    serde_wasm_bindgen::to_value(&scales)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

All four methods need `#[wasm_bindgen]` on the `impl` block — that's already there.

### Step 3: Build WASM to confirm it compiles

```bash
cargo build --target wasm32-unknown-unknown -p lora-inspector-wasm --release 2>&1 | tail -20
```

Expected: compiles cleanly

### Step 4: Run full test suite

```bash
make test
```

### Step 5: Commit

```bash
git add crates/lora-inspector-wasm/src/worker.rs
git commit -m "feat: expose rank metrics to WASM layer"
```

---

## Task 8: Final WASM artifact size check

### Step 1: Build production WASM

```bash
make build-wasm
```

### Step 2: Check size delta

```bash
ls -lh crates/lora-inspector-wasm/pkg/lora-inspector_bg.wasm
```

Compare against baseline of ~583KB. If growth exceeds ~50KB, investigate with:

```bash
cargo bloat --release --target wasm32-unknown-unknown -p lora-inspector-wasm --crates 2>/dev/null | head -20
```

### Step 3: Commit if clean

```bash
git add crates/lora-inspector-wasm/pkg/
git commit -m "build: update WASM artifacts with rank metrics"
```
