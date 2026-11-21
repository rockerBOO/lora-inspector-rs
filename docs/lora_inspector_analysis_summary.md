# LoRA Inspector — Analysis Summary

This document consolidates the metrics, algorithms, and analysis patterns from the LoRA toolset for re-implementation in a LoRA inspector tool.

---

## Core Concepts

### Effective Scale

The single most important per-layer metric. Measures the actual weight delta the LoRA applies at strength 1.0.

```
effective_scale = norm(up @ down) * (alpha / rank)
```

- `up` = lora_up.weight (`out × rank`)
- `down` = lora_down.weight (`rank × in`)
- `alpha` = scalar per-layer multiplier
- `rank` = LoRA rank

Use this to compare layers, detect outliers, and assess overall LoRA strength.

### Rank×Rank Core Trick (fast SVD)

Avoids materializing the full `(out × in)` matrix. Instead:

1. SVD of `up` → `U_up, S_up, Vh_up` (size `out × rank`)
2. SVD of `down` → `U_dn, S_dn, Vh_dn` (size `rank × in`)
3. Form core: `C = diag(S_up) @ Vh_up @ U_dn @ diag(S_dn)` — size `rank × rank`
4. SVD of `C` → singular values equal those of `up @ down`

This is the preferred method for large Flux-style LoRAs.

### Full SVD (accurate but slower)

Materialize `M = up @ down` then SVD the full matrix. Needed for per-element statistics and exact energy distributions. Only practical for smaller ranks or when detail is required.

---

## Layer Naming

LoRA key pattern: `lora_unet_{block_type}_{N}_{sub_layer}.lora_{up|down}.weight`

### Block types

- `double_blocks_N` — double-stream transformer blocks
- `single_blocks_N` — single-stream transformer blocks

### Layer families (within a block)

| Family       | Key fragment                         |
| ------------ | ------------------------------------ |
| `txt_attn`   | txt_attn projections (double blocks) |
| `img_attn`   | img_attn projections (double blocks) |
| `txt_mlp`    | txt MLP layers (double blocks)       |
| `img_mlp`    | img MLP layers (double blocks)       |
| `linear`     | linear1/linear2 (single blocks)      |
| `modulation` | modulation layers                    |

### Base model key mapping

```
lora_unet_double_blocks_0_img_attn_proj → double_blocks.0.img_attn.proj.weight
lora_unet_single_blocks_3_linear1       → single_blocks.3.linear1.weight
```

---

## Metrics Reference

### Per-layer metrics

| Metric           | Formula                            | Meaning                                                 |
| ---------------- | ---------------------------------- | ------------------------------------------------------- |
| `eff_scale`      | `norm(up @ down) * (alpha / rank)` | Magnitude of LoRA weight delta                          |
| `up_norm`        | `frobenius_norm(up)`               | Up matrix magnitude                                     |
| `down_norm`      | `frobenius_norm(down)`             | Down matrix magnitude                                   |
| `up_down_ratio`  | `up_norm / down_norm`              | Factorization balance; ~1.0 standard, higher with LoRA+ |
| `top1_energy`    | `s[0]² / sum(s²)`                  | Fraction of energy in dominant direction; 1.0 = rank-1  |
| `effective_rank` | `exp(entropy(s²/sum(s²)))`         | Shannon entropy measure; how many dimensions are used   |
| `balance`        | `effective_rank / nominal_rank`    | Rank utilization fraction                               |
| `dominance`      | `s[0] / s[1]`                      | Ratio of top two singular values                        |
| `delta_norm`     | `norm(up @ down * alpha/rank)`     | Frobenius norm of full weight delta                     |

### Rank health thresholds (`effective_rank / nominal_rank`)

| Rating    | Threshold          |
| --------- | ------------------ |
| GOOD      | ≥ 0.75             |
| OK        | ≥ 0.50             |
| WEAK      | ≥ 0.25             |
| POOR      | < 0.25             |
| COLLAPSED | effective_rank = 1 |

### Base model comparison metrics (requires base model)

| Metric        | Formula                                             | Meaning                                         |
| ------------- | --------------------------------------------------- | ----------------------------------------------- |
| `base_norm`   | `frobenius_norm(base_weight)`                       | Scale of base weight                            |
| `delta_ratio` | `eff_scale / base_norm`                             | LoRA size relative to base (0.33 = 1/3 of base) |
| `spec_ratio`  | `spectral_norm(LoRA) / spectral_norm(base)`         | Dominant direction comparison                   |
| `cos_sim`     | cosine similarity of dominant left singular vectors | Direction alignment                             |

Base spectral norm is computed via power iteration (32 iterations) to avoid full SVD on large matrices.

### Outlier detection

Flag a layer as outlier when:

```
eff_scale > threshold × median(all_eff_scales)
```

Default threshold: 1.5×. Configurable.

### Direction alignment classification (with base model)

Uses `spec_ratio` and `cos_sim` (threshold 0.5):

| Category     | Condition                        | Meaning                                   |
| ------------ | -------------------------------- | ----------------------------------------- |
| `overrides`  | spec_ratio > 1 AND cos_sim > 0.5 | LoRA overpowers base in its own direction |
| `new-large`  | spec_ratio > 1 AND cos_sim ≤ 0.5 | Large LoRA pointing elsewhere             |
| `reinforces` | spec_ratio ≤ 1 AND cos_sim > 0.5 | Adds to existing base direction (normal)  |
| `novel`      | spec_ratio ≤ 1 AND cos_sim ≤ 0.5 | New learning at healthy scale             |

### LoRA-to-LoRA comparison metrics

| Metric        | Formula                                  | Meaning                                |
| ------------- | ---------------------------------------- | -------------------------------------- |
| `cos_sim`     | cosine of dominant left singular vectors | Direction similarity between two LoRAs |
| `scale_ratio` | `eff_scale(A) / eff_scale(B)`            | Relative magnitude                     |

Direction alignment categories (cos_sim only):

- `aligned` > 0.9
- `similar` > 0.5
- `diverged` > 0.2
- `orthogonal` ≤ 0.2

### DoRA column magnitudes

Compute per-output-dimension magnitude of `up @ down`. High coefficient of variation (CV > 0.5) means the LoRA effect is concentrated in a few output dimensions rather than spread uniformly.

### Tensor statistics (per up/down matrix)

- Mean, std, kurtosis, skew
- Frobenius norm, max absolute value
- High kurtosis = heavy-tailed / outlier weights

---

## Block-level Aggregation

After computing per-layer metrics, aggregate by block (`double_blocks_N` / `single_blocks_N`):

- Mean effective rank ratio
- Mean and median delta norm
- Outlier count

### Block recommendations

| Recommendation | Condition                                                |
| -------------- | -------------------------------------------------------- |
| `DROP`         | mean_er_ratio < 0.12 AND mean_delta_norm < 60% of median |
| `REDUCE`       | mean_er_ratio < 0.25                                     |
| `KEEP`         | mean_er_ratio ≥ 0.75                                     |
| `OK`           | otherwise                                                |

---

## Reduction Methods (for inspector actions)

These are the operations available for modifying a LoRA. An inspector tool can offer these as actions after analysis.

### Alpha scaling (modifies alpha only, weights unchanged)

- `global` — multiply all alphas by a factor (e.g. 0.75, 0.5)
- `selective` — multiply alpha only on outlier layers

Preserves weight direction and rank structure. Does NOT preserve factorization balance semantics in the same way as weight scaling.

### Weight scaling (modifies weights, preserves alpha)

Apply `sqrt(factor)` symmetrically to both `up` and `down`:

```
new_up  = up  * sqrt(factor)
new_down = down * sqrt(factor)
# Result: norm(new_up @ new_down) = factor * norm(up @ down)
```

Preserves LoRA+ factorization balance. Recommended over alpha scaling when maintaining balance matters.

Methods:

- `norm-global` — scale all layers to current median
- `norm-selective` — cap outliers to median, leave others untouched
- `max-norm` — cap any layer above an absolute value; best for pre-training prep

### Spectral methods (requires base model)

- `spectral` — normalize largest singular value to median spectral norm
- `spec-ratio` — reduce layers where `spec_ratio > target` (default 0.9)
- `orth` — orthogonalize dominant directions against base model's top singular vectors

### Rank reduction

- `svd` — truncate to keep top-k energy fraction (e.g. 0.75)
- `energy` — adaptive per-layer: drop components below energy threshold (e.g. 0.01)

### Pruning (removes keys entirely)

- `layer-prune` — remove all keys for layers above threshold × median
- `block-prune` — remove all keys for a block type: `double`, `single`, `double-txt`, `double-img`

### Use-case guidance

| Goal                          | Recommended method                         |
| ----------------------------- | ------------------------------------------ |
| Inference only (smaller file) | `layer-prune` or `block-prune double-txt`  |
| Continued training            | `max-norm` with absolute cap               |
| Targeted reduction only       | `norm-selective`                           |
| Remove base model dominance   | `spec-ratio` or `orth`                     |
| Compress rank                 | `energy` (adaptive) or `svd` (fixed ratio) |

---

## Visualization Patterns

Five plot types useful in an inspector:

| Plot                | Type              | X                           | Y / Color                       | Purpose                                              |
| ------------------- | ----------------- | --------------------------- | ------------------------------- | ---------------------------------------------------- |
| `distributions`     | Violin + jitter   | variant                     | effective scale                 | Shows full distribution shape per variant            |
| `heatmap`           | Heatmap           | layer (sorted by eff_scale) | variant                         | Which layers each method affects                     |
| `reduction_ratio`   | Diverging heatmap | layer                       | `variant/original` ratio        | How much each layer was reduced (green < 1, red > 1) |
| `family_summary`    | Grouped bar       | layer family                | median eff_scale                | How methods affect architectural components          |
| `sv_energy_scatter` | Scatter           | effective scale             | top1_energy (colored by family) | High-scale vs rank-1 relationship                    |

---

## Key/Tensor Structure

Each LoRA layer has three tensors:

- `{prefix}.lora_up.weight` — shape `(out_features, rank)`
- `{prefix}.lora_down.weight` — shape `(rank, in_features)`
- `{prefix}.alpha` — scalar (stored as 0-dim tensor)

Optional (DoRA):

- `{prefix}.dora_scale` — per-output-dimension scale vector

LoRA+ detection: read from metadata key `ss_network_args` → `loraplus_lr_ratio`.

---

## Implementation Priorities

For a minimal LoRA inspector, implement in this order:

1. **Load & parse** — read safetensors, group keys by layer prefix, extract up/down/alpha/rank
2. **Effective scale** — compute per layer, show overall statistics and outliers
3. **Block/family grouping** — aggregate by block and family
4. **Singular value analysis** — rank×rank core trick for top1_energy, effective_rank, balance
5. **Factorization balance** — up_norm/down_norm ratio, flag inconsistencies
6. **Outlier flagging** — threshold × median, sorted table
7. **Base model comparison** — delta_ratio, spec_ratio, direction alignment (optional, needs base)
8. **DoRA column CV** — detect output-dimension concentration (optional)
9. **LoRA-to-LoRA comparison** — cos_sim and scale_ratio between two LoRAs (optional)
