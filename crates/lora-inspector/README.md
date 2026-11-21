# LoRA Inspector CLI

Command-line interface for analyzing LoRA (Low-Rank Adaptation) model files.

## Installation

```bash
cargo install --path .
```

## Usage

```bash
# Print the header and block sections
lora-inspector inspect model.safetensors

# Add the effective-rank section (parallelised across layers)
lora-inspector inspect model.safetensors --rank

# Machine-readable single-object output
lora-inspector inspect model.safetensors --json
```

### `inspect`

```
lora-inspector inspect <FILE> [OPTIONS]

  --rank                 compute and render the effective-rank section (off by default)
  --json                 machine-readable output; no ANSI, no progress, no width logic
  --no-blocks            skip the block section; header/metadata only (header-only load)
  --metadata             include the raw safetensors metadata map in --json output
  --slow                 allow the materialising fallback for networks without an
                         up/down pair; off by default
  --width <N>            override detected terminal width
  --top <N>              rows in the "most loaded blocks" table (default 8)
  --jobs <N>             worker threads for the rank pass (default: available cores)
  --ascii                ASCII glyphs only
  --color <auto|always|never>   default auto (TTY and not NO_COLOR)
  --quiet                suppress progress on stderr
```

The header and block sections are always printed; `--rank` adds the
effective-rank section. `--no-blocks` without `--rank` loads only the file
header, so inspecting a multi-gigabyte file costs only its header. Progress is
written to stderr, never stdout, and only while `--rank` runs on a TTY.

### `block-weights` (deprecated)

`block-weights` is deprecated in favour of `inspect`. It renders the block
section only and keeps `--output-format json|text`.

```bash
lora-inspector block-weights --file model.safetensors --output-format text
```

### `compare-metadata`

```bash
lora-inspector compare-metadata --file1 model1.safetensors --file2 model2.safetensors
```

## JSON output

With `--json`, `inspect` prints exactly one object on stdout. A failure prints
exactly one object `{"error": {"kind": "io|format|unsupported|usage", "message": "..."}}`
on stderr and exits `1`; usage errors from the argument parser exit `2`.
Exit code `0` means success.

The document is flat and self-describing: `blocks` and `rank` are lists of rows
whose group/family/label repeat on every row, `null` means "not computed" and
`[]` means "computed, nothing found", and every float is rounded to six
significant digits. `--width` never changes the serialized document.

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
| `network.alpha`                         | float[]      | declared alphas, sorted; `[]` when not declared or when the header-only load ran (alpha values live in the payload)                                                                          |
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

## Optional Features

- CUDA support: Enable with `--features cuda`. The report always runs on the
  CPU; the feature only keeps the GPU dependency reachable.

## Contributing

- Suggest improvements or report issues on GitHub
- Pull requests welcome

## License

See the root project LICENSE file
