# Rank Health Metrics Design

## Goal

Add per-layer rank health analysis to the LoRA inspector without adding new Rust dependencies. All computation runs in the browser via WASM on CPU.

## Metrics to expose

| Metric                  | Formula                           | Notes                                                      |
| ----------------------- | --------------------------------- | ---------------------------------------------------------- |
| `eff_scale`             | `l2_norm(up @ down * alpha/rank)` | Already computed as `scale_weight`; expose as named metric |
| `up_norm`               | `frobenius_norm(up)`              | Raw up matrix magnitude                                    |
| `down_norm`             | `frobenius_norm(down)`            | Raw down matrix magnitude                                  |
| `factorization_balance` | `up_norm / down_norm`             | ~1.0 normal, higher with LoRA+                             |
| `top1_energy`           | `s[0]² / sum(s²)`                 | 1.0 = collapsed to rank-1                                  |
| `effective_rank`        | `exp(entropy(s²/sum(s²)))`        | How many dimensions actually used                          |
| `balance`               | `effective_rank / nominal_rank`   | Rank utilization                                           |
| `dominance`             | `s[0] / s[1]`                     | Ratio of top two singular values                           |
| `health`                | threshold on balance              | GOOD/OK/WEAK/POOR/COLLAPSED                                |

### Rank health thresholds

| Rating    | Condition           |
| --------- | ------------------- |
| COLLAPSED | effective_rank == 1 |
| GOOD      | balance ≥ 0.75      |
| OK        | balance ≥ 0.50      |
| WEAK      | balance ≥ 0.25      |
| POOR      | balance < 0.25      |

### Outlier detection

`eff_scale > 1.5 × median(all eff_scales)` flags a layer as outlier.

---

## Implementation approach

### No new dependencies

nalgebra and faer would increase the WASM artifact size. Instead, a self-contained Jacobi symmetric eigendecomposition (~100 lines) handles all SVD needs. This works because:

- All Jacobi calls operate only on **rank×rank matrices** (typically 16×16 to 64×64).
- Large `out_features`/`in_features` dimensions are handled by candle matmul, not Jacobi.
- Rank ≤ 64 converges fast. If rank > 128 becomes common, nalgebra can replace Jacobi later.

### Accuracy trade-off

We compute eigenvalues of `C^T C` rather than direct Jacobi SVD. This halves floating-point precision for very small singular values. For rank health ratings (GOOD/OK/WEAK/POOR) this is not a problem in practice.

---

## Computation chain

### Singular values of `up @ down` (rank×rank core trick)

```
Step 1: A = up^T @ up           (rank×rank, via candle matmul)
Step 2: jacobi_sym(A) → (λ_A, Q_A)   where λ_A = S_up², Q_A columns = Vh_up^T
Step 3: B = down @ down^T        (rank×rank, via candle matmul)
Step 4: jacobi_sym(B) → (λ_B, Q_B)   where λ_B = S_dn², Q_B columns = U_dn
Step 5: C = diag(√λ_A) @ Q_A^T @ Q_B @ diag(√λ_B)   (rank×rank, pure Rust)
Step 6: jacobi_sym(C^T @ C) → eigenvalues = squared singular values of up @ down
Step 7: sort descending, compute metrics
```

`jacobi_sym` returns `(eigenvalues: Vec<f64>, eigenvectors: Vec<f64>)` in column-major order.
Steps 1–4 need eigenvectors to form C. Step 6 needs only eigenvalues.

---

## Module structure

### New file: `crates/inspector/src/svd.rs`

- `jacobi_sym(m: &[f64], n: usize) -> (Vec<f64>, Vec<f64>)`
  Jacobi eigendecomposition of symmetric n×n matrix (column-major flat storage).
  Returns `(eigenvalues, eigenvectors_column_major)`, eigenvalues descending.

- `singular_values(up: &Tensor, down: &Tensor) -> Result<Vec<f64>>`
  Implements the full rank×rank core trick. Converts candle tensors to `Vec<f64>`, runs chain above.

- `rank_metrics(up: &Tensor, down: &Tensor, nominal_rank: usize) -> Result<RankMetrics>`
  Calls `singular_values`, computes all metrics.

### New types in `svd.rs`

```rust
#[derive(Debug, Serialize, Deserialize)]
pub struct RankMetrics {
    pub top1_energy: f64,
    pub effective_rank: f64,
    pub balance: f64,
    pub dominance: Option<f64>,  // None if rank == 1
    pub health: RankHealth,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RankHealth { Good, Ok, Weak, Poor, Collapsed }
```

### Changes to `crates/inspector/src/file.rs` (or `weight.rs`)

Add methods on `LoRAFile`:

- `effective_scale(base_name) -> Result<f64>`
- `factorization_balance(base_name) -> Result<f64>`
- `rank_metrics(base_name) -> Result<RankMetrics>`

### Changes to `crates/lora-inspector-wasm/src/worker.rs`

New WASM-exported methods on `LoraWorker`:

- `effective_scale(base_name: &str) -> Option<f64>`
- `factorization_balance(base_name: &str) -> Option<f64>`
- `rank_metrics(base_name: &str) -> Result<JsValue, JsValue>`
- `effective_scales_all() -> Result<JsValue, JsValue>`
  Returns `Vec<{base_name, eff_scale, is_outlier}>` using 1.5× median threshold.

---

## Upgrade path

If rank > 128 performance becomes a bottleneck:

- Replace `jacobi_sym` with `nalgebra::linalg::SymmetricEigen` using `default-features = false`.
- Public API of `singular_values` and `rank_metrics` stays identical.
- No WASM layer changes needed.
