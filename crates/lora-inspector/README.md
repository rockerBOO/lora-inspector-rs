# LoRA Inspector CLI

Command-line interface for analyzing LoRA (Low-Rank Adaptation) model files.

## Installation

```bash
cargo install --path .
```

## Usage

```bash
# Print the header and block-weight sections
lora-inspector inspect FILE

# Add the effective-rank section (parallelised across layers)
lora-inspector inspect FILE --rank

# Machine-readable output, no scraping required
lora-inspector inspect FILE --json | jq '.summary'
```

`inspect FILE` prints a width-adaptive terminal report: a header with the
file's network information, then the block-weight section. `--rank` adds a
per-layer effective-rank section. `--json` replaces the terminal report with a
single machine-readable object.

The report adapts to the terminal width — a flat table at 100+ columns, a
sparkline view at 60–99, one line per family below that. `--json` output never
depends on width or TTY.

Progress is written to stderr, never stdout, and only while `--rank` runs on a
TTY (suppress it with `--quiet`).

The other two shipped subcommands:

```bash
lora-inspector block-weights --file model.safetensors --output-format text
lora-inspector compare-metadata --file1 model1.safetensors --file2 model2.safetensors
```

`block-weights` is deprecated (see Compatibility); `compare-metadata` is
unchanged.

## Flags

| Flag             | Default       | Description                                                                                                                                                                                                               |
| ---------------- | ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--rank`         | off           | Compute and render the effective-rank section (off by default).                                                                                                                                                           |
| `--json`         | off           | Print a single machine-readable JSON object instead of the terminal report; no ANSI, no progress, no width logic.                                                                                                         |
| `--no-blocks`    | off           | Skip the block section. Without `--rank` this loads only the file header, never the tensor payload.                                                                                                                       |
| `--metadata`     | off           | Include the raw safetensors metadata map in `--json` output.                                                                                                                                                              |
| `--slow`         | off           | Allow the materialising fallback for layers without a plain 2-D up/down pair — missing pairs (LoHA/LoKr/OFT/BOFT) or factors that are not 2-D after singleton collapse (convolution kernels, LoCon, tucker); much slower. |
| `--width <N>`    | detected      | Override the detected terminal width in columns.                                                                                                                                                                          |
| `--top <N>`      | 8             | Rows in the "most loaded blocks" table in the wide (100+ column) tier; the compact tier shows a fixed top 3.                                                                                                              |
| `--jobs <N>`     | 0 (all cores) | Worker threads for the rank pass; `0` means available cores.                                                                                                                                                              |
| `--ascii`        | off           | Use ASCII glyphs only, no Unicode bars or sparklines.                                                                                                                                                                     |
| `--color <mode>` | auto          | When to use ANSI colour: `auto`, `always`, or `never`.                                                                                                                                                                    |
| `--quiet`        | off           | Suppress the rank-progress spinner on stderr.                                                                                                                                                                             |

## JSON

With `--json`, `inspect` prints exactly one object on stdout. A failure prints
exactly one object `{"error": {"kind": "io|format|unsupported", "message": "..."}}`
on stderr and exits `1`. clap's usage errors (exit `2`) bypass this contract
entirely and print plain text. Exit code `0` means success.

### Field reference

| field                                   | type         | meaning                                                                                                                                                                                      |
| --------------------------------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `schema_version`                        | int          | contract version; `1` for this shape                                                                                                                                                         |
| `load`                                  | string       | `full` or `header_only`; under `header_only`, payload-derived fields (`network.alpha`, `blocks`, `ungrouped`, `skipped`) are `[]` or `null` by construction                                  |
| `file.path`                             | string       | path as given on the command line                                                                                                                                                            |
| `file.size_bytes`                       | int          | file length                                                                                                                                                                                  |
| `file.precision`                        | string\|null | `fp16`, `bf16`, `fp32`, … of the weight tensors                                                                                                                                              |
| `network.module`                        | string       | resolved module, e.g. `kohya-ss/lora`, `peft`                                                                                                                                                |
| `network.type`                          | string\|null | `LoRA`, `LoCon`, `LoHA`, `LoKr`, `OFT`, …                                                                                                                                                    |
| `network.family`                        | string       | `UNet`, `TextEncoder`, or `Unknown`                                                                                                                                                          |
| `network.rank`                          | int[]        | declared ranks, sorted; `[]` when not declared                                                                                                                                               |
| `network.alpha`                         | float[]      | declared alphas, sorted; `[]` when not declared                                                                                                                                              |
| `network.layers.total`                  | int          | base names walked                                                                                                                                                                            |
| `network.layers.unet` / `.text_encoder` | int          | base names per group family, not substring matching                                                                                                                                          |
| `metadata`                              | object\|null | raw `__metadata__`; only with `--metadata`                                                                                                                                                   |
| `blocks[]`                              | row[]        | one row per (group, index), display order; `null` with `--no-blocks`                                                                                                                         |
| `blocks[].group`                        | string       | `Down`, `Input`, `Mid`, `Middle`, `Up`, `Output`, `Double`, `Single`, `SingleTransformer`, `Transformer`, `Refiner`, `TokenRefiner`, `TxtFusion`, `Blocks`, `Layers`, `TextEncoder`, `Other` |
| `blocks[].family`                       | string       | `UNet` \| `TextEncoder` \| `Other`                                                                                                                                                           |
| `blocks[].label`                        | string       | short display tag: `IN04`, `DB03`, `TE11`; bare (`MID`) when there is no index                                                                                                               |
| `blocks[].index`                        | int\|null    | block index taken from the key                                                                                                                                                               |
| `blocks[].layers`                       | int          | base names in this block                                                                                                                                                                     |
| `blocks[].mean` / `.median` / `.max`    | float        | of the per-layer block weight                                                                                                                                                                |
| `blocks[].outlier`                      | bool         | mean exceeds 1.5× the median **block** mean                                                                                                                                                  |
| `ungrouped`                             | object\|null | `{layers, prefixes}` for keys matching no family; `null` when not computed                                                                                                                   |
| `skipped`                               | object\|null | `{layers, reason}` for layers without a low-rank factor; `null` when not computed                                                                                                            |
| `rank[]`                                | row[]        | one row per layer, sorted by `base_name`; `null` unless `--rank`                                                                                                                             |
| `rank[].eff_scale`                      | float\|null  | the same metric as `blocks[].mean`, per layer                                                                                                                                                |
| `rank[].outlier`                        | bool         | layer scale exceeds 1.5× the median **layer** scale                                                                                                                                          |
| `rank[].health`                         | string       | `Good`, `Ok`, `Weak`, `Poor`, `Collapsed`                                                                                                                                                    |
| `summary.block_weight`                  | object       | `{median, peak: {label, value, ratio_to_median}, outliers}`                                                                                                                                  |
| `summary.rank`                          | object\|null | `{layers, by_health, median_balance, weak_or_worse, elapsed_ms}`; `by_health` keys are alphabetical                                                                                          |

### Rules for reading it

- **Flat arrays in display order.** `blocks` and `rank` are flat lists of rows;
  `blocks` rows repeat their own `group`/`family`/`label` and `rank` rows repeat
  their own `group`/`label`, so `jq '.blocks[] | select(.mean > 1)'` needs no
  join. `blocks` are in display order (group rank, then index, then label);
  `rank` rows are sorted by `base_name`.
- **`null` means not computed; `[]` means computed and nothing found.** With
  `--no-blocks`, `blocks`, `ungrouped` and `skipped` are `null`; without
  `--rank`, `rank` is `null`. A file that grouped nothing gives `"blocks": []`.
- **Floats are rounded to 6 significant digits.** Human output and JSON show
  the same value; the human view re-formats block-weight and rank values to 2
  decimals for alignment.
- **`load: header_only` means payload-derived fields are empty by
  construction.** The header-only path (`--no-blocks` without `--rank`) reads
  only `8 + header_len` bytes; alpha tensors live in the payload, so
  `network.alpha` is necessarily `[]` there, and any other empty field is
  covered by `load`, not by per-field sentinels.
- **Errors are one object on stderr with exit 1.** Under `--json`, a failure
  emits `{"error": {"kind": "…", "message": "…"}}` and no partial report;
  `kind` is one of `io`, `format`, `unsupported`. Usage errors exit `2` as
  plain text, not as this object.
- **`schema_version` bumps only on rename, retype or removal.** Additive fields
  do not bump it.

## What the numbers mean

- **Block weight** (`blocks[].mean`, and the terminal `block weight` section)
  is the Frobenius norm of the scaled delta at strength 1.0 —
  `||(alpha/rank)·(up @ down)||_F` — computed from the rank×rank Gram matrices
  (`up^T up` and `down down^T`) so the full product is never materialised.
  Per-block rows are averages over the layers in that block.
- **Rank health** (`rank[].health`) comes from the singular-value energy
  distribution of the layer. `effective_rank` is the Shannon-entropy effective
  rank (`exp(H)`), `balance` is `effective_rank / nominal_rank` (1.0 means
  fully balanced), `top1_energy` is the energy fraction in the top singular
  value, and `dominance` is `s[0]/s[1]`. The `Collapsed` test (near-total
  energy in one direction) runs before the balance bands, so a weak balance
  can read `Collapsed` rather than its balance band; otherwise the bands are `Good`
  (balance ≥ 0.75), `Ok` (≥ 0.50), `Weak` (≥ 0.25), `Poor` (< 0.25).
- **`--slow`** switches the block pass to the materialising fallback for layers
  without a plain 2-D up/down pair (see the flag table), and is far slower;
  without it those layers are skipped and counted as `skipped`.
- **`--rank`** is CPU-bound (the Jacobi SVD is the project's own code, not a GPU
  path) and parallelised across layers with `--jobs`.

## Performance

Measured end-to-end with `inspect` on this machine (Ryzen 7 5800X, 8 cores,
release build, `opt-level=3` + LTO, warm page cache), `time` wall-clock:

| file                                                     | size   | rank | layers | `inspect` | `inspect --rank` |
| -------------------------------------------------------- | ------ | ---- | ------ | --------- | ---------------- |
| `boo.safetensors` (SD1.5)                                | 4.8 MB | 4    | 264    | 0.17 s    | 0.20 s           |
| `h3/H3_Motion_BoosterV2.safetensors`                     | 155 MB | 16   | 208    | 0.44 s    | 0.55 s           |
| `wan2.2/14b/23High noise-Cumshot Aesthetics.safetensors` | 307 MB | 32   | 400    | 0.99 s    | 1.25 s           |
| `flux/urban_decay_v20_rank64_bf16-step02016.safetensors` | 269 MB | 64   | 114    | 1.09 s    | 1.44 s           |

Wall time is dominated by reading the payload into memory for the larger files;
the block and rank passes themselves are the difference between the two modes.

## Building for speed

Default release builds stay portable. For a local non-portable build on this
class of CPU:

```bash
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release -p lora-inspector
```

Release builds already use `opt-level=3` + LTO. The size profile
(`opt-level = "z"`) applies only to the wasm release bundle, pinned via
`RUSTFLAGS` in the Makefile, never inherited by this binary.

## Compatibility

- `block-weights` remains as a deprecated wrapper around the new pipeline: it
  renders the block section only and keeps `--output-format json|text`. In this
  release its `--output-format json` shape changed — it now emits the same
  `schema_version: 1` object as `inspect --json` instead of the old
  `norms`/`base_names`/`non_block_weights`/`blocks` map keyed by `unknown_N`.
  This is a breaking CLI output change; the release bump is
  `scripts/bump_release.py minor`, which defaults to `1.7.0-beta.1` (per
  `CLAUDE.md`).
- `compare-metadata` is unchanged.

## Optional Features

- CUDA support: Enable with `--features cuda`. The report always runs on the
  CPU; the feature only keeps the GPU dependency reachable.

## Contributing

- Suggest improvements or report issues on GitHub
- Pull requests welcome

## License

See the root project LICENSE file
