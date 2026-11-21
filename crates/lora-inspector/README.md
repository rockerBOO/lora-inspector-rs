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

## Flags

| Flag             | Default       | Description                                                                                                       |
| ---------------- | ------------- | ----------------------------------------------------------------------------------------------------------------- |
| `--rank`         | off           | Compute and render the effective-rank section (off by default).                                                   |
| `--json`         | off           | Print a single machine-readable JSON object instead of the terminal report; no ANSI, no progress, no width logic. |
| `--no-blocks`    | off           | Skip the block section. Without `--rank` this loads only the file header, never the tensor payload.               |
| `--metadata`     | off           | Include the raw safetensors metadata map in `--json` output.                                                      |
| `--slow`         | off           | Allow the materialising fallback for layers without an up/down pair (LoHA/LoKr/OFT/BOFT); much slower.            |
| `--width <N>`    | detected      | Override the detected terminal width in columns.                                                                  |
| `--top <N>`      | 8             | Rows in the "most loaded blocks" table.                                                                           |
| `--jobs <N>`     | 0 (all cores) | Worker threads for the rank pass; `0` means available cores.                                                      |
| `--ascii`        | off           | Use ASCII glyphs only, no Unicode bars or sparklines.                                                             |
| `--color <mode>` | auto          | When to use ANSI colour: `auto`, `always`, or `never`.                                                            |
| `--quiet`        | off           | Suppress the rank-progress spinner on stderr.                                                                     |

## JSON

With `--json`, `inspect` prints exactly one object on stdout. A failure prints
exactly one object `{"error": {"kind": "io|format|unsupported|usage", "message": "..."}}`
on stderr and exits `1`; usage errors from the argument parser exit `2`.
Exit code `0` means success.

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
  every row repeats its own `group`/`family`/`label`, so `jq '.blocks[] | select(.mean > 1)'` needs no join. `blocks` are in display order (group rank,
  then index, then label); `rank` rows are sorted by `base_name`.
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
  `kind` is one of `io`, `format`, `unsupported`, `usage`. Usage errors exit `2`.
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
  value, and `dominance` is `s[0]/s[1]`. The health word is `Good`
  (balance ≥ 0.75), `Ok` (≥ 0.50), `Weak` (≥ 0.25), `Poor` (< 0.25), or
  `Collapsed` (effectively a single direction).
- **`--slow`** switches the block pass to the materialising fallback for layers
  without a plain up/down pair, and is far slower; without it those layers are
  skipped and counted as `skipped`.
- **`--rank`** is CPU-bound (the Jacobi SVD is the project's own code, not a GPU
  path) and parallelised across layers with `--jobs`.

## Performance

Measured on this machine (Ryzen 7 5800X, 8 cores, release build with
`opt-level=3` + LTO):

| command                                  | file                              | wall time |
| ---------------------------------------- | --------------------------------- | --------- |
| `inspect boo.safetensors`                | SD1.5, rank 4, 264 layers         | 0.16 s    |
| `inspect boo.safetensors --rank`         | same                              | 0.20 s    |
| `inspect urban_decay_v20_rank64… --rank` | Flux, rank 64, 114 layers, 269 MB | 1.75 s    |

The block section (header + per-layer effective scale) runs in 0.14–0.41 s for
the whole file on this class of machine; the rank pass runs 2–9 ms per layer at
typical ranks and ~51 ms/layer at rank 128 (≈74 s single-threaded, ≈10 s at 8
threads on a 1440-layer LTX file). For large files the wall time is dominated
by reading the payload into memory — the 269 MB Flux file above spends about a
second of its 1.75 s just loading before the report runs.

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
  `norms`/`base_names`/`blocks` map keyed by `unknown_N`. This is a breaking
  CLI output change.
- `compare-metadata` is unchanged.

## Optional Features

- CUDA support: Enable with `--features cuda`. The report always runs on the
  CPU; the feature only keeps the GPU dependency reachable.

## Contributing

- Suggest improvements or report issues on GitHub
- Pull requests welcome

## License

See the root project LICENSE file
