# Dataset Display Expansion Plan

Date: 2026-04-05

## Context

The current `Buckets.jsx` component handles `ss_datasets` entries but was written against
the retro-anime musubi-tuner format (image directory, basic bucket settings). The musubi-tuner
dataset config supports image datasets, video datasets, JSONL-sourced datasets, control image
datasets, and FramePack-specific settings. A video-trained LoRA currently shows almost no
dataset information.

---

## Changes

### 1. Dataset source identification

Each dataset entry has exactly one of these source fields. Currently only `image_directory`
is shown; the others are silently ignored.

Show whichever is present:

- `image_directory` — image folder with caption text files
- `video_directory` — video folder with caption text files
- `image_jsonl_file` — image dataset sourced from JSONL metadata
- `video_jsonl_file` — video dataset sourced from JSONL metadata

Label each clearly so the user knows which type of dataset was used. A video dataset
and an image dataset look identical right now.

**Files:** `Buckets.jsx`

---

### 2. Cache directory

`cache_directory` is present in musubi-tuner `ss_datasets` entries. Currently not shown.
Useful for debugging and understanding the training setup.

Show when present, guarded with `"cache_directory" in dataset`.

**Files:** `Buckets.jsx`

---

### 3. Video-specific fields

Only relevant when `video_directory` or `video_jsonl_file` is present. Show as a group,
all guarded individually.

Fields:

- `target_frames` — array e.g. `[1, 25, 45]`, join with `,` for display
- `frame_extraction` — `head` / `chunk` / `slide` / `uniform` / `full`
- `frame_stride` — integer, only meaningful for `slide` extraction
- `frame_sample` — integer, only meaningful for `uniform` extraction
- `max_frames` — integer cap, relevant for `full` extraction
- `source_fps` — decimal, present when fps conversion is active

Only render the video fields row when any video key is present, so image datasets
are not cluttered with empty video fields.

**Files:** `Buckets.jsx`

---

### 4. Control image fields

We currently show `has_control` as a boolean. Expand to show the actual control
configuration when present:

- `control_directory` — path to control images folder
- `control_resolution` — array e.g. `[1024, 1024]`, format as `WxH`
- `no_resize_control` — boolean, shown when present

Replace the bare `has_control` boolean with these richer fields. If only `has_control`
is present (no directory or resolution), fall back to showing it.

**Files:** `Buckets.jsx`

---

### 5. FramePack-specific fields

Only present for FramePack training. Show as a group when any `fp_*` key is present:

- `fp_latent_window_size` — integer
- `fp_1f_clean_indices` — array, format as `[0, 1]`
- `fp_1f_target_index` — integer
- `fp_1f_no_post` — boolean

**Files:** `Buckets.jsx`

---

### 6. Multiple target (Qwen-Image-Layered)

- `multiple_target` — boolean, shown when present

**Files:** `Buckets.jsx`

---

## Structural approach in Buckets.jsx

Reorganise the component into logical groups rather than a flat row of all fields.
Each group only renders when relevant keys are present:

```
[Source row]         image_directory | video_directory | image_jsonl_file | video_jsonl_file
[Bucket row]         enable_bucket, bucket_no_upscale, min/max reso (kohya), resolution
[Repeats row]        num_repeats, caption_extension, cache_directory
[Video row]          target_frames, frame_extraction, frame_stride, frame_sample, max_frames, source_fps
[Control row]        control_directory, control_resolution, no_resize_control
[FramePack row]      fp_latent_window_size, fp_1f_clean_indices, fp_1f_target_index, fp_1f_no_post
[Special row]        multiple_target
[Bucket info]        BucketInfo (existing)
[Subsets]            Subset list (existing, kohya ss_datasets format)
[Tag frequencies]    TagFrequency (existing, kohya ss_datasets format)
```

Each row is a `<div className="row space-apart">` that only renders when at least one
field in the group is present.

---

## Helper functions needed

- `formatArray(arr)` — joins array with `,` for display (used for `target_frames`,
  `fp_1f_clean_indices`, `control_resolution`)
- `formatResolution(resolution)` — already exists, keep as-is

---

## Testing

Add tests to `dataset-analysis-components.test.jsx` covering:

- Video dataset fields render when `video_directory` + `target_frames` present
- JSONL source shown for `image_jsonl_file` and `video_jsonl_file`
- Control fields render when `control_directory` present; `has_control` alone still works
- FramePack group renders when any `fp_*` key present
- Video fields absent for image-only dataset (no noise in image datasets)
- `target_frames` array formatted as comma-separated string

---

## Files changed

| File                                         | Change                                                                                       |
| -------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `dataset/Buckets.jsx`                        | All display changes — source type, video fields, control fields, FramePack fields, cache dir |
| `tests/dataset-analysis-components.test.jsx` | New tests for each new field group                                                           |
