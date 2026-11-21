# Layout Reorganization Plan

Date: 2026-04-05

## Context

Following the initial design review and 6-section layout implementation, a second pass
identified fields placed in the wrong sections based on their semantic meaning. This plan
tracks the moves, fixes, and formatting improvements to make the layout more coherent.

---

## Changes

### 1. Precision → out of Optimizer, into Training

Move the precision row from the Optimizer section into the Training section.
Includes the worker-derived `Precision` component (actual weight dtype) alongside
the config precision fields — they're related and belong together.

Fields moved: Precision (worker), Mixed precision, Full fp16, Full bf16, fp8 base,
Gradient checkpointing CPU offload.

**Files:** `Main.jsx`

---

### 2. Seed → Training

Seed is a training reproducibility parameter, not optimizer-specific.

- Remove from `Optimizer.jsx`
- Add to `EpochStep.jsx` (same row as epoch/steps)

**Files:** `Optimizer.jsx`, `EpochStep.jsx`

---

### 3. Gradient Checkpointing → Training

A training memory/performance setting, not a network property.

- Remove from `Network.jsx`
- Add alongside Seed/Epoch in Training

**Files:** `Network.jsx`, `EpochStep.jsx`

---

### 4. CaptionDropout + Max token length → Dataset

These are dataset processing parameters (how captions are handled during training),
not training hyperparameters.

- Remove `<CaptionDropout>` from Training section in `Main.jsx`
- Add it inside the Dataset section via `Dataset.jsx`

**Files:** `Main.jsx`, `Dataset.jsx`

---

### 5. Noise → Experimental (two groups)

Noise augmentation is experimental/advanced diffusion training. Remove from Training
section and add two separate groups to `Experimental.jsx`. Each group only renders
if any key in the group is present in metadata.

- **"Noise offset"**: `ss_noise_offset`, `ss_noise_offset_random_strength`,
  `ss_adaptive_noise_scale`
- **"MultiRes noise"**: `ss_multires_noise_iterations`, `ss_multires_noise_discount`,
  `ss_ip_noise_gamma`, `ss_ip_noise_gamma_random_strength`

**Files:** `Main.jsx`, `Experimental.jsx`

---

### 6. SNR → Experimental

SNR-related loss weighting fields are noise-related training techniques, not core
loss configuration.

- Remove `ss_debiased_estimation`, `ss_min_snr_gamma`, `ss_zero_terminal_snr`
  from `Loss.jsx`
- Add a **"SNR"** group to `Experimental.jsx` (only renders when any key present)

**Files:** `Loss.jsx`, `Experimental.jsx`

---

### 7. Learning rate: OR → AND

`LRScheduler.jsx` currently shows the global `ss_learning_rate` when _either_
`ss_unet_lr` or `ss_text_encoder_lr` is absent/None (OR logic). When either specific
LR is set, the global LR is still partially shown.

Correct behavior: show global LR only when _both_ are absent/None. When specific LRs
are set they fully replace the global LR — showing it alongside them is misleading.

**Files:** `LRScheduler.jsx`

---

### 8. sd-scripts commit hash → only when set

`PretrainedModel.jsx` always renders `ss_sd_scripts_commit_hash`. Musubi-tuner runs
don't have this key. Guard with `metadata.has()`.

**Files:** `PretrainedModel.jsx`

---

### 9. Model hash — wider display

The 64-char SHA256 hash is truncated by the grid column. Give the hash container
`gridColumn: "1 / span 2"` (or similar) so it can wrap/display fully.

**Files:** `PretrainedModel.jsx`

---

### 10. Elapsed → h/m/s format

`ModelSpec.jsx` formats elapsed time as decimal minutes (`toPrecision(4) minutes`).
Replace with a formatter that outputs `Xh Ym Zs`, omitting zero-valued leading units
(e.g. `1h 50m 0s` not `0h 1h 50m 0s`).

**Files:** `ModelSpec.jsx`

---

## Section contents after changes

| Section      | Additions                                           | Removals                                                           |
| ------------ | --------------------------------------------------- | ------------------------------------------------------------------ |
| Training     | Precision row, Seed, Gradient Checkpointing         | Noise, CaptionDropout                                              |
| Optimizer    | —                                                   | Seed, Precision row                                                |
| Dataset      | CaptionDropout                                      | —                                                                  |
| Network      | —                                                   | Gradient Checkpointing                                             |
| Experimental | Noise offset group, MultiRes noise group, SNR group | —                                                                  |
| Loss         | —                                                   | SNR fields (debiased estimation, min SNR gamma, zero terminal SNR) |
| Metadata     | Hash wider, elapsed h/m/s, sd-scripts hash guarded  | —                                                                  |
