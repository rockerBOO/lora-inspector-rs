# Dataset Organization Cleanup

Date: 2026-04-13

## Context

Several issues were found while adding `ss_tag_frequency` support and reviewing the dataset
section against the kohya-ss and musubi-tuner dataset documentation:

1. **`Max token length` only appears in the old-format (KohyaDataset) path.** It lives at
   `ss_max_token_length` in both old and new formats, but the `ss_datasets` path never shows it.

2. **`CaptionDropout` is a misleading name.** The component only holds the three caption
   dropout rates, but it renders next to `Max token length` which is unrelated to dropout.
   A more accurate name reflects that these are caption training settings.

3. **`Subset.jsx` is a stub.** It only shows `image_dir`. The kohya-ss docs define subsets
   as having `keep_tokens`, `flip_aug`, `num_repeats`, `class_tokens`, `is_reg`, `caption_extension`,
   `color_aug`, `face_crop_aug_range`, `random_crop`, `shuffle_caption`, and others. None of
   these are currently shown.

---

## Changes

### 1. Add `Max token length` to the `ss_datasets` path

`ss_max_token_length` is a top-level metadata key present in both old and new formats.
Currently it is only read in `Dataset.jsx` for the `KohyaDataset` code path.

Add it above `CaptionDropout` in the `ss_datasets` code paths (both the shared-settings
branch and the per-dataset branch).

**Files:** `Dataset.jsx`

---

### 2. Rename `CaptionDropout` → `CaptionSettings`

`CaptionDropout.jsx` is not about dropout alone — at minimum it sits next to
`Max token length`, and the three dropout rate fields are caption training configuration,
not a standalone concept. Rename to `CaptionSettings` to reflect its actual scope.

The component body stays the same for now (three dropout rate fields). The name change
makes the intent clearer and avoids confusion when `Max token length` is added nearby.

**Files:**

- `dataset/CaptionDropout.jsx` → `dataset/CaptionSettings.jsx` (rename file)
- `Dataset.jsx` (update import and JSX usage)
- `components/index.js` (update re-export)
- `tests/dataset-analysis-components.test.jsx` (update import path)
- Any other files that import `CaptionDropout`

---

### 3. Implement `Subset.jsx`

The subset object from kohya-ss `ss_datasets` entries can contain:

| Field                   | Description                       |
| ----------------------- | --------------------------------- |
| `image_dir`             | Image directory path              |
| `class_tokens`          | Trigger words / class tokens      |
| `num_repeats`           | Repeat count for this subset      |
| `keep_tokens`           | Number of tokens to keep at front |
| `shuffle_caption`       | Boolean                           |
| `flip_aug`              | Boolean                           |
| `color_aug`             | Boolean                           |
| `face_crop_aug_range`   | Array e.g. `[1.0, 3.0]`           |
| `random_crop`           | Boolean                           |
| `caption_extension`     | e.g. `.txt`                       |
| `is_reg`                | Boolean — regularization subset   |
| `caption_prefix`        | String prepended to all captions  |
| `caption_suffix`        | String appended to all captions   |
| `keep_tokens_separator` | Separator string e.g. `"          |
| `secondary_separator`   | Separator string e.g. `";;;"`     |
| `enable_wildcard`       | Boolean                           |

Display strategy: show `image_dir` as a heading, then render all present fields in
compact rows. Guard every field individually — subsets are sparse and different trainers
emit different subsets of these fields.

Only render a field when it is present in the subset object. Do not show empty rows.

**Files:** `dataset/Subset.jsx`

---

## Structural result in `Dataset.jsx`

After these changes, both code paths render in the same order:

```
[Dataset content]     Buckets (ss_datasets) or KohyaDataset (ss_dataset_dirs)
[Max token length]    ss_max_token_length  ← now in both paths
[CaptionSettings]     ss_caption_dropout_*, etc.  ← renamed from CaptionDropout
[Tag frequencies]     TopLevelTagFrequency (ss_tag_frequency)
```

---

## Testing

Add or update tests in `dataset-analysis-components.test.jsx`:

- `Max token length` renders in the `ss_datasets` path when `ss_max_token_length` is present
- `Max token length` absent when key is not in metadata
- `CaptionSettings` import path works (after rename)
- `Subset` renders `image_dir` heading
- `Subset` renders `class_tokens` when present
- `Subset` renders `num_repeats`, `keep_tokens`, `is_reg` when present
- `Subset` does not render rows for absent fields

---

## Files changed

| File                                         | Change                                                                                     |
| -------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `dataset/Dataset.jsx`                        | Add `Max token length` to `ss_datasets` paths; update `CaptionDropout` → `CaptionSettings` |
| `dataset/CaptionDropout.jsx`                 | Rename to `CaptionSettings.jsx`                                                            |
| `dataset/Subset.jsx`                         | Implement full subset field display                                                        |
| `components/index.js`                        | Update re-export for renamed file                                                          |
| `tests/dataset-analysis-components.test.jsx` | New tests; update import paths                                                             |
