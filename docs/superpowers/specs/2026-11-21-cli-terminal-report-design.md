# CLI Terminal Report — Design

**Date:** 2026-11-21
**Status:** Draft for approval
**Approach chosen:** A — tiered single report (see [Approaches](#approaches))

## Intent

Bring the browser UI's two most useful analyses — per-block weight magnitude and
effective-rank health — to the `lora-inspector` binary as a first-class terminal
report: fast enough to run casually, legible from 40 to 200 columns, and
machine-readable on demand so an agent can consume it without scraping.

The report should be the thing you reach for before you decide whether a LoRA is
worth loading, and the thing an agent reads instead of being handed a 3000-line
JSON dump.

## What's wrong today

Measured on this machine (Ryzen 7 5800X, 8 cores, release build):

1. **The CLI burns enormous compute and discards most of it.** `parse_block_weights`
   materializes `scale_weight` = `up @ down` as a dense f64 matrix for every layer,
   then computes l1, l2, matrix-norm, max, min and std-dev, keeps them all in a hash
   map, and prints only the L2 norm and per-block averages. On `h3/H3_Motion_BoosterV2`
   that is a 5376×14336 temporary (~616 MB) per layer: the run exceeded **600 s**
   before I killed it. `boo.safetensors` takes 13.3 s. No output consumes those numbers.
2. **Block grouping is close to non-functional.** `parse_block_info` only looks for
   the substring `blocks_`, so on an SD1.5 file the up-block, down-block and mid-block
   with the same index all collapse into one `unknown_N` key — 4 pseudo-blocks for
   264 layers — and all 216 text-encoder layers are printed one per line as
   "non-block weights".
3. **The interesting code is never called.** `inspector` already exposes
   `effective_scale` (Gram-trick Frobenius norm, no materialization),
   `effective_scales_all` (median × 1.5 outlier flag), `rank_metrics` (entropy
   effective rank, balance, top-1 energy, dominance, health) and
   `factorization_balance`. The CLI calls none of them. The frontend's own
   "l2 norm" message is wired straight to `effective_scale`, so the browser already
   uses the cheap path and the CLI does not.
4. **The release profile is misnested.** Root `Cargo.toml` writes `[profile]` with
   `dev = {...}` / `release = {...}` inside it instead of `[profile.release]`. Cargo
   silently ignores the `package = { inspector = { opt-level = 3 } }` override but
   still applies the rest, so `candle-core` and `inspector` are compiled at
   `-C opt-level=z` while the binary gets `3`. Verified by inspecting rustc flags.
5. **The frontend's grouping cannot be ported as-is.** Its `moduleBlocks.js` is ~600
   lines of per-architecture regex. Run against the files in `/mnt/900/lora` it parses
   LTX 0/2880 keys, wan2.2 160/800, qwen-image 1680/2160 — the browser already shows
   an empty block chart for every LTX LoRA on this machine. Copying it into Rust would
   reproduce the gaps in a second language.

## Measured baseline

With `opt-level=3` and `-C target-cpu=x86-64-v3`, single core, using the cheap
`effective_scale` path over every layer:

| file | layers | current CLI | cheap path, opt-level=z | cheap path, opt-level=3 + AVX2 |
| --- | --- | --- | --- | --- |
| `boo.safetensors` (SD1.5, r4) | 264 | 13.3 s | 0.67 s | **0.14 s** |
| `h3/H3_Motion_BoosterV2` (r16) | 208 | >600 s | 0.21 s | 0.16 s |
| `wan2.2/14b/…` (r32) | 400 | — | 0.55 s | 0.41 s |

Rank/SVD pass (hand-rolled f64 Jacobi in `inspector/src/svd.rs`), per layer:

| file | layers | ms/layer | whole file, 1 core |
| --- | --- | --- | --- |
| `wan2.2/14b/…` | 400 | 2.2 | 0.9 s |
| `flux/urban_decay_v20_rank64` | 114 | 9.0 | 1.0 s |
| `qwen-image` (r64) | 720 | 8.0 | 5.8 s |
| `ltx2.3/DR34ML4Y_LT3X_V3` (r128) | 1440 | 51.2 | **74 s** |

CUDA was tested and rejected for this workload: on the rank×rank Gram shapes the
report needs, `cuda` is **4.3× slower** than CPU (84.2 ms vs 19.6 ms for 20 matmuls
including readback). The `cuda` feature does build and run on the Blackwell
`sm_120` card. SVD cannot use the GPU on any candle version, and `inspector` does
not use candle's SVD at all — it is its own code. See [Non-goals](#non-goals).

## Goals

1. `lora-inspector inspect FILE` prints the header and block section in well under a
   second for every file in `/mnt/900/lora`, computing only what it prints.
2. `--rank` adds the effective-rank section, parallelised across layers.
3. Block grouping has one implementation, in `inspector`, reused by the CLI today and
   by wasm later — and it is *better* than the JS version, not a copy of it.
4. Output is legible at 40, 60, 80, 120 and 200 columns; bars, labels and numbers
   stay aligned for CJK and other double-width filenames.
5. `--json` emits a flat, self-describing schema that an agent can interpret from the
   field table alone — no escape codes, no nested-group joins, no absent keys, no float
   noise, and `null` (not computed) cleanly distinguished from `[]` (nothing found).
6. Output is deterministic (sorted, stable ordering) so it can be snapshot-tested.

## Non-goals

- No interactive TUI, no ratatui, no alternate screen. Plain stdout, one shot.
- No new metrics beyond what `inspector` already computes.
- No change to the frontend UI in this project. Only the shared grouper lands in
  `inspector`, ready for wasm to re-export; switching the frontend over is later work.
- No candle upgrade *inside the feature work* — but the upgrade itself is now in scope as
  Phase 0, done first, on this branch (see below). It is a build/dependency change with
  its own gates, kept out of the CLI tasks so neither diff reviews through the other.
- No CUDA acceleration of the default report, and no `--device` flag. The report is
  CPU-only. The `cuda` feature keeps compiling (verified on the Blackwell card) so a
  future GPU path stays reachable, but `inspect` will not probe or select a GPU. The
  current unconditional `cuda_if_available(0)` probe in `main.rs` is removed: on this
  hardware it silently selects the slower device.
- No GPU path for SVD. Not available in this codebase at any candle version.

## Anti-goals

These make the project a failure even if it technically works:

- A default invocation that takes minutes on a file the user owns.
- A second block-grouping implementation living in the CLI crate.
- Output that is pretty on a TTY and broken when piped, redirected or in CI.
- A `--json` schema that changes shape depending on terminal width or TTY.
- JSON that needs the source or a wiki to interpret: nested groups to join, keys that
  vanish, `null` and `[]` used interchangeably, or full float precision noise.
- Metrics that disagree with the browser for the same file, without that gap being
  an explicit, documented decision.

## Phase 0: dependency upgrade

Authorised by the project owner after the design was approved. Current pins are
`candle-core 0.3.2` and `safetensors 0.3.3`; the current releases are `0.11.1` and
`0.8.0`, and the toolchain is rustc 1.95.0.

**Why now rather than later:** the CLI work is about to be written against candle's
`Tensor`/`DType`/`Error` and safetensors' `TensorView`. Writing that code against 0.3.2
and rewriting it after an 8-minor-version jump is the expensive order. Doing the upgrade
first means every later task is written once, against the API that ships.

**What it buys, concretely:**

- `safetensors` 0.8 and `candle` 0.11 are the versions that get fixes and are what the
  ecosystem's tooling assumes.
- candle 0.11 ships `MmapedSafetensors`, which is the zero-copy loader the dropped
  `memmap2` item was looking for. It is **not** adopted in this project (the loader
  abstraction in `inspector` would have to change shape), but it is now reachable.
- The `cuda` feature can be exercised against a current `cudarc`, which matters for the
  Blackwell card even though the report stays CPU-only.

**What it does not buy:** GPU SVD (still CPU in candle, and `inspector`'s SVD is its own
Jacobi implementation), and any change to the metrics.

**Gates — the upgrade is only acceptable if all of these hold:**

1. `cargo test --workspace` green. Snapshot changes are only acceptable when the diff is a
   real numeric change, and each changed value must be explained in the report.
2. `cargo check --target wasm32-unknown-unknown -p lora-inspector-wasm` green, and
   `make build-wasm-dev` produces a wasm no more than 10% larger than the recorded
   baseline (`lora-inspector_bg.wasm` 7 737 686 B, `lora-inspector-simd_bg.wasm`
   7 757 294 B). A bigger jump is reported, not absorbed.
3. `inspector`'s public API does not change shape (same function names and signatures)
   except where a dependency forced it — those cases are listed in the commit message.
4. `cargo build --release -p lora-inspector --features cuda` is attempted; if a current
   `cudarc` cannot build for `sm_120`, that is reported as a finding, not worked around.

**Out of scope for Phase 0:** adopting `MmapedSafetensors`, changing the SVD algorithm,
touching the wasm frontend's JS, or any CLI behaviour.

## Approaches

**A. Tiered single report (chosen).** One `inspect` command. Header and block
section always; rank section only with `--rank`. `--json` emits everything that was
requested, never ANSI. Predictable latency, clean agent contract, pretty part is
instant.

**B. Everything by default.** Rejected: the rank pass is 74 s on the largest LTX
file single-core (~10 s parallel), so a plain inspection becomes a coffee break on
files this user owns.

**C. Summary-first** (~12 lines, `--blocks` / `--rank` to expand). Rejected: most
composable, but reaching what the browser shows in one screen costs three flags, and
the summary-only case is already covered by `--json` for agents.

## Design

### Command surface

```
lora-inspector inspect <FILE> [OPTIONS]

  --rank                 compute and render the effective-rank section (off by default)
  --json                 machine-readable output; no ANSI, no progress, no width logic
  --no-blocks            skip the block section; header/metadata only (header-only load)
  --metadata             include the raw safetensors metadata map in --json output
  --slow                 allow the materialising fallback for networks without an
                         up/down pair (see Degradation); off by default
  --width <N>            override detected terminal width
  --top <N>              rows in the "most loaded blocks" table (default 8)
  --jobs <N>             worker threads for the rank pass (default: available cores)
  --ascii                ASCII glyphs only (░▒▓█ → - = #, ▁▂▃ → .:-)
  --color <auto|always|never>   default auto (TTY and not NO_COLOR)
  --quiet                suppress progress on stderr
```

- `block-weights` is kept and reimplemented on the new pipeline: it renders the block
  section only, and `--output-format json|text` keeps working (its JSON shape changes;
  see [Versioning](#versioning)).
- `compare-metadata` is untouched.
- Progress goes to **stderr** only (it currently shares stdout with the report, which
  corrupts piped output). `indicatif` hides itself when stderr is not a TTY.

### Data flow

```
FILE ──mmap──> LoRAFile
                 ├─ metadata + network info ──────────────> header section
                 ├─ per base_name: effective_scale() ─────> block section   (always)
                 │     (Gram-trick ||up@down||_F, no materialization)
                 └─ per base_name: rank_metrics()    ─────> rank section   (--rank, rayon)
                       (hand-rolled Jacobi SVD, CPU)
```

One pass over `base_names()`, each layer's metrics computed once and reused for both
sections. `scale_weight` is never called on the default path.

### Block grouping (`inspector::blocks`)

New public module. Tokenizer, not regex: normalize the base name into a token stream
(`.`, `_` and `-` are all separators; the original separators are not preserved), then
scan it left to right for the first block-family pattern of 1–3 tokens, and read the
integer that follows it as the block index.

**Match rule: leftmost (outermost) family wins; at the same start position, longest
wins.** Left-to-right position dominates length, because nested block names are real:

```
lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj
         ^^^^^^^^^^                    ^^^^^^^^^^^^^^^^^^
         family: Down (wins, it is leftmost)  → part of the sublayer
```

Taking the inner `transformer_blocks` as the family would move every SDXL attention
layer out of Down/Up/Mid and into `Transformer`, which is not what the browser does and
not what the model's structure is. The same rule keeps Flux PEFT's
`transformer.single_transformer_blocks.N` correct: `single transformer blocks` matches
before any inner pattern, so it groups as `SingleTransformer`.

Everything after the index is the sublayer, joined with `.` (`attn.q`, `mlp.up`,
`attentions.0.transformer_blocks.0.ff_net.0.proj`).

```rust
pub struct BlockKey {
    pub group: BlockGroup,       // Down, Input, Mid, Middle, Up, Output, Double, Single,
                                 // SingleTransformer, Transformer, Refiner, TokenRefiner,
                                 // TxtFusion, Blocks, Layers, Modules, TextEncoder, Other
    pub block_index: Option<usize>,
    pub sublayer: String,        // "attn.q", "mlp.up", "proj_in", …
}
```

Family table, scanned left-to-right (3-token patterns before 2 before 1 at the same
position), checked after text-encoder detection:

| tokens | group |
| --- | --- |
| `double blocks` | Double |
| `single transformer blocks` | SingleTransformer |
| `single blocks` | Single |
| `transformer blocks` | Transformer |
| `down blocks` / `input blocks` | Down / Input |
| `mid block` / `middle block` | Mid / Middle |
| `up blocks` / `output blocks` | Up / Output |
| `txtfusion layerwise blocks` / `txtfusion refiner blocks` | TxtFusion |
| `token refiner refiner blocks`, `token refiner blocks`, `refiner blocks` | TokenRefiner |
| `noise refiner` / `context refiner` | Refiner |
| `blocks` (bare, e.g. `diffusion_model.blocks.3`, `lora_unet_blocks_3`) | Blocks |
| `layers` | Layers |
| no family pattern, but a numeric id | Modules |

`Modules` covers index-less single tensors that still have a numeric id somewhere,
and keys with no family and no id (krea2's `first`, `last_linear`, `txtfusion_projector`,
`tmlp_*`, `tproj_*`) fall to `Other` — one bucket, listed once in the footer rather
than spread across rows that mean nothing.

Text-encoder detection runs first and short-circuits the family match: if the
token stream contains `te`/`teN`, `text model encoder`, `text model`, or
`language model`, the key is `TextEncoder`. The block index is **the integer
immediately following the matched family token** — for
`language_model_model_layers_18_self_attn_o_proj` that is `18`, and for
`lora_te_text_model_encoder_layers_11_self_attn_v_proj` it is `11`; the sublayer is
the remainder (`self_attn.o_proj`). The number is not required to be trailing. This is
what collapses 216 SD1.5 TE layers into 12 TE blocks instead of one, and it is why
detection runs on the *normalized* token stream — PEFT writes `text_model.encoder.layers.11`
with dots, which only matches `text model encoder` after normalization.

Any key that matches nothing becomes `Other` — grouped by its longest common prefix
with siblings, never silently dropped. The report footer states
`N layers ungrouped (prefix…)` when that happens.

`BlockGroup::family()` maps groups to the two display families used by the compact tier:
`UNet` for every block group except `TextEncoder`, `Other` for `TextEncoder`.

Canonical ordering for display: Down/Input → Mid/Middle → Up/Output → Double →
Single → SingleTransformer → Transformer → Refiner → TokenRefiner → TxtFusion →
Blocks → Layers → Modules → Other, then by `block_index`, then by `sublayer`. This
ordering governs the compact/minimal sparkline rows only; the wide tier is one table
of rows sorted by value descending (see Rendering).

### Metrics (reused, not reimplemented)

- **Block weight** = `effective_scale` per base name (Frobenius norm of the scaled
  delta, computed from the rank×rank Gram matrices), averaged per block. The block's
  `max` sublayer value is kept for the outlier annotation.
- **Outlier** = `eff_scale > 1.5 × median across layers`, matching
  `LoRAFile::effective_scales_all`. Note: the JS phase-1 computes
  `max(median, 1e-10) * 1.5` while Rust uses `if median < 1e-10 { 1e-10 } else {1.5×}`;
  these disagree near zero. The Rust formula is authoritative.
- **Rank metrics** = `inspector::svd::RankMetrics` verbatim: `nominal_rank`,
  `effective_rank`, `balance`, `top1_energy`, `dominance`, `health`.

### Rendering

`Terminal` value type carrying `width: usize`, `color: bool`, `unicode: bool`.
Width resolution: `--width` → `COLUMNS` → `TIOCGWINSZ` on stdout → 80.

All padding and truncation uses display width (`unicode-width`), never `chars().count()`,
so `Klein 一edit-consistency` and `retro-comic-illustration-复古风格` align. Truncation is
middle-ellipsis for names (`Klein 一edi…tency`) with `…` or `~` in ASCII mode.

Three width tiers:

**Wide (≥ 100 columns)** — one flat table of the `--top` blocks by value descending,
regardless of family (families do not nest rows; the block label carries the family):

```
block weight   ‖ΔW‖_F scaled at strength 1.0, averaged per base layer
UNet down/up/mid   26 blocks   median 0.41   peak IN04 1.82 (4.4× median)

  IN04  ████████████████████████████  1.82   7 layers
  OUT05 █████████████████████         1.21   7 layers
  MID01 ███████████                   0.63   6 layers
  …
```

**Compact (60–99)** — sparkline per display family (canonical order), then the top 3
rows by value:

```
block weight  median 0.41  peak IN04 1.82 (4.4×)
  UNet  ▁▂▃▄▅▇█▅▃▁▂▃▄▅▃▂▁▂▁   26 blocks
  TE    ▂▃▃▄▄▅▅▄▃▃▂▂▁▁        12 blocks
```

**Minimal (< 60)** — one line per group, no sparkline:

```
UNet 26 blocks  med 0.41  peak 1.82×4.4  TE 12 med 0.33
```

Rank section (only with `--rank`):

```
effective rank   r=64, 114 layers, 2.4 s (8 threads)
GOOD ███████████████░░░  78%    median balance 0.81   6 layers weak or worse

  DB03  r64  eff 51.2  bal 0.80  GOOD
  TE07  r16  eff  3.1  bal 0.19  POOR   ⚠ outlier 3.1×
  …
```

Health colors: `Good` green, `Ok` default, `Weak` yellow, `Poor` red,
`Collapsed` bold red. Color is decoration only — every row carries its health word, so
`--color never` and screenshots remain unambiguous.

### JSON

Designed for an agent to read without a decoder ring: **flat arrays, self-describing
rows, no deep nesting, one number per concept.**

Rules that hold everywhere:

- **Flat arrays, not nested groups.** `blocks` and `rank` are lists of rows. Every row
  repeats its own `group` / `family` / `label`, so
  `jq '.blocks[] | select(.mean > 1)'` works with no join and no second lookup.
- **Stable keys.** A key is always present with the same type. It is never omitted and
  never changes type between files.
- **`null` means "not computed"; `[]` means "computed, nothing found".** `--no-blocks`
  gives `"blocks": null`; a file that grouped nothing gives `"blocks": []`. An agent can
  always tell the difference, which it cannot if the key simply disappears.
- **Numbers are canonical JSON floats rounded to 6 significant digits.** Human output and
  JSON show the same value — no `0.4000000000000001`.
- **Arrays are in display order** (group rank, then index, then name), so printing the
  array as-is is the intended reading order. An agent that wants "top N by value" sorts
  itself; `summary` already carries the peak.
- **`summary` holds the headline numbers** so nobody has to recompute medians, counts
  or elapsed time.
- **Metadata is opt-in.** `--metadata` adds the raw safetensors map; without it the key is
  `null`. That map is frequently hundreds of keys of training-log noise, which is exactly
  what an agent does not need by default.
- **Errors are JSON too**, on stderr, with a non-zero exit. Under `--json` a failure emits
  exactly one object `{"error": {"kind": "…", "message": "…"}}` and no partial report.
  `kind` is one of `io`, `format`, `unsupported`, `usage`.
- **Exit codes:** `0` success, `1` runtime error, `2` usage error (clap's default).
- **`schema_version` only changes when a key is renamed, retyped or removed.** Additive
  fields do not bump it.

```json
{
  "schema_version": 1,
  "file": { "path": "boo.safetensors", "size_bytes": 13164512, "precision": "fp16" },
  "network": {
    "module": "kohya-ss/lora", "type": "LoRA", "family": "UNet",
    "rank": [4], "alpha": [4.0],
    "layers": { "total": 264, "unet": 264, "text_encoder": 216 }
  },
  "metadata": null,
  "blocks": [
    { "group": "Down", "family": "UNet", "label": "IN04", "index": 4, "layers": 7,
      "mean": 1.81524, "max": 3.10422, "median": 1.55002, "outlier": true }
  ],
  "ungrouped": { "layers": 0, "prefixes": [] },
  "skipped": { "layers": 0, "reason": null },
  "rank": null,
  "summary": {
    "block_weight": {
      "median": 0.41,
      "peak": { "label": "IN04", "value": 1.81524, "ratio_to_median": 4.4 },
      "outliers": 3
    }
  }
}
```

With `--rank`, `rank` becomes a flat array and `summary.rank` appears:

```json
  "rank": [
    { "base_name": "lora_unet_double_blocks_3_img_attn_proj", "group": "Double",
      "label": "DB03", "index": 3, "nominal_rank": 64, "effective_rank": 51.2,
      "balance": 0.8, "top1_energy": 0.11, "dominance": 2.3, "health": "Good",
      "eff_scale": 1.02, "outlier": false }
  ],
  "summary": {
    "block_weight": { "median": 0.41, "peak": { "…": 0 }, "outliers": 3 },
    "rank": {
      "layers": 114,
      "by_health": { "Good": 78, "Ok": 30, "Weak": 4, "Poor": 2, "Collapsed": 0 },
      "median_balance": 0.81, "weak_or_worse": 6, "elapsed_ms": 2400
    }
  }
```

Field reference — this table is the contract and is copied verbatim into the CLI README:

| field | type | meaning |
| --- | --- | --- |
| `schema_version` | int | contract version; `1` for this shape |
| `file.path` | string | path as given on the command line |
| `file.size_bytes` | int | file length |
| `file.precision` | string\|null | `fp16`, `bf16`, `fp32`, … of the weight tensors |
| `network.module` | string | resolved module, e.g. `kohya-ss/lora`, `peft` |
| `network.type` | string\|null | `LoRA`, `LoCon`, `LoHA`, `LoKr`, `OFT`, … |
| `network.family` | string | `UNet`, `TextEncoder`, or `Unknown` |
| `network.rank` | int[] | declared ranks, sorted; `[]` when not declared |
| `network.alpha` | float[] | declared alphas, sorted; `[]` when not declared |
| `network.layers.total` | int | base names walked |
| `network.layers.unet` / `.text_encoder` | int | base names per group family, not substring matching |
| `metadata` | object\|null | raw `__metadata__`; only with `--metadata` |
| `blocks[]` | row[] | one row per (group, index), display order; `null` with `--no-blocks` |
| `blocks[].group` | string | `Down`, `Input`, `Mid`, `Middle`, `Up`, `Output`, `Double`, `Single`, `SingleTransformer`, `Transformer`, `Refiner`, `TokenRefiner`, `TxtFusion`, `Blocks`, `Layers`, `Modules`, `TextEncoder`, `Other` |
| `blocks[].family` | string | `UNet` \| `TextEncoder` \| `Other` |
| `blocks[].label` | string | short display tag: `IN04`, `DB03`, `TE11` |
| `blocks[].index` | int\|null | block index taken from the key |
| `blocks[].layers` | int | base names in this block |
| `blocks[].mean` / `.median` / `.max` | float | of the per-layer block weight |
| `blocks[].outlier` | bool | mean exceeds 1.5× the median across all layers |
| `ungrouped` | object | `{layers, prefixes}` for keys matching no family |
| `skipped` | object | `{layers, reason}` for layers without a low-rank factor |
| `rank[]` | row[] | one row per layer, sorted by `base_name`; `null` unless `--rank` |
| `rank[].eff_scale` | float\|null | the same metric as `blocks[].mean`, per layer |
| `rank[].health` | string | `Good`, `Ok`, `Weak`, `Poor`, `Collapsed` |
| `summary.block_weight` | object | `{median, peak: {label, value, ratio_to_median}, outliers}` |
| `summary.rank` | object\|null | `{layers, by_health, median_balance, weak_or_worse, elapsed_ms}` |

### Performance work

1. Fix the profile: real `[profile.release]` section with `opt-level = 3`, `lto = true`,
   and `[profile.release.package.inspector] opt-level = 3` (candle-core inherits 3
   instead of `z`). Keep the `opt-level = "z"` size profile only for the wasm build,
   where it belongs.
2. Add a documented non-portable build for local use:
   `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release` (or a
   `make build-cli-native` target). Default builds stay portable.
3. ~~`memmap2`~~ — dropped. `LoRAFile::new_from_buffer` immediately does
   `buffer.to_vec()` (`file.rs:66-70`) and candle's `BufferedSafetensors::new` only
   accepts a `Vec<u8>`, so an mmap would still produce one full heap copy. Phase 0 makes
   `MmapedSafetensors` reachable, but adopting it would change the loader's shape, so
   `fs::read` stays for this project and zero-copy is a follow-up.
4. `rayon` in the **CLI crate only** (not `inspector`; wasm must not grow). The rank
   pass is a `par_iter` over base names; results are collected and sorted before
   rendering so output stays deterministic. `--jobs` sets the pool.
5. `--no-blocks` uses the header-only load (`LoRAFile::new_from_header_buffer`), so
   `inspect --json --no-blocks` on the 2.7 GB Flux.2 file never reads the payload.
   The normal path always loads tensors — it costs 0.14–0.41 s and is the whole point
   of the command.

### Degradation

- **No materialising fallback, ever.** `effective_scale` falls back to
  `scale_weight` + full L2 when a layer has no `up`/`down` pair (`file.rs:228-246`),
  and for LoHA/LoKr/OFT/BOFT that fallback materialises the dense delta — the exact
  >600 s path this project exists to remove. Those layers return `Some(...)`, so a
  network type check would never fire. The block pass therefore computes the Gram path
  only: a base name whose factors are not a plain 2D `up`/`down` pair is **skipped and
  counted**, not materialised. The section then prints
  `N layers skipped: no low-rank factor (LoHA/LoKr/OFT/BOFT)` plus a hint that
  `--slow` allows the fallback. Header and rank still render.
- Missing `up`/`down` tensors for some layers: those layers are skipped and counted,
  not fatal.
- File smaller than a header, or bad metadata: existing error path, no partial report.
- No TTY: no progress, no color, width 80 unless `--width`/`COLUMNS` says otherwise.

### Versioning

The `block-weights --output-format json` shape changes (it currently dumps `norms`,
`base_names`, `non_block_weights` and a `blocks` map keyed by `unknown_N`). That is a
breaking CLI output change: bump with `scripts/bump_release.py minor` per the project
rules, and note it in the CLI README. The new shape ships as `schema_version: 1` and
that number only changes when a key is renamed, retyped or removed.

## Testing

No `.safetensors` file is committed (all are git-ignored) and every existing test
already depends on `boo.safetensors` being in the working directory. New tests must not
inherit that, so:

- **Grouping**: golden table tests in `inspector` over checked-in **key lists** (plain
  text, one key per line) captured from `/mnt/900/lora` — ~20 keys each for SD1.5, SDXL,
  Flux double/single, Flux PEFT `single_transformer_blocks`, flux2-klein, wan2.2, krea2,
  h3, ltx2.3, qwen-image, moss, gemma3. Key lists, not files: no safetensors dependency
  and no `/mnt` dependency.
- **Rendering and JSON**: a shared test helper writes a tiny synthetic LoRA to a
  `tempfile` via the `safetensors` writer — a handful of layers across two families
  (`down_blocks`, `text_model_encoder.layers`) with known values, so snapshot numbers
  are the helper's, not boo's. Runs in `make test` and CI. `insta` snapshots at widths
  40/60/80/120/200 plus one `--ascii` snapshot.
- **JSON**: `insta` snapshot of the serialized object, plus an assertion that the
  schema is byte-identical between `--width 40` and `--width 200`.
- **Correctness**: a test that the block means the renderer prints equal the means of
  `effective_scale` computed independently in the test, so the renderer cannot drift
  from the metric.
- **Performance**: a smoke test asserting the cheap path never calls `scale_weight`
  (instrument via a counter or by asserting wall-clock bounds on a fixture), so the
  materialized-matrix regression cannot come back silently.
- Existing `make test` and `make wasm-bindgen-test` must stay green; the wasm build
  size is a guard on the `inspector` additions (no new heavy deps: `unicode-width` is
  tiny, no `regex`).

## What's left to the builder

- Exact sparkline resolution (one glyph per block vs. resampled to fit width) and how
  groups with more blocks than columns are downsampled — state the rule you choose.
- How the "peak" annotation is chosen when two blocks tie, and whether the peak line is
  suppressed when the peak is below 2× median.
- The precise `rank.summary` fields.
- Whether `inspect` should accept multiple files, and if so what the separator is.
- Column widths inside each tier.
- Whether `block-weights` emits a deprecation note on stderr.
- Where the synthetic test fixture helper lives (the CLI crate has no `dev-dependencies`
  or `tests/` directory today), and the exact layer values it writes.

## Cleanup

Two throwaway benchmarks I added while measuring must not survive into the branch:
`crates/lora-inspector/examples/bench_tmp.rs` and
`crates/lora-inspector/examples/cudacheck_tmp.rs`.
