# CLI Terminal Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fast, width-adaptive `lora-inspector inspect` terminal report with shared block grouping, per-block weight magnitude, and opt-in effective-rank health.

**Architecture:** Block grouping becomes a new `inspector::blocks` module (tokenizer, not regex) so both the CLI and the wasm frontend can use one implementation. The CLI gains four focused modules — `terminal` (width/color/glyph primitives), `report` (data model, serde-shaped for the JSON contract), `render` (three width tiers), and `main` (clap surface). Metrics come from existing `inspector` APIs; the default path never materializes `up @ down`.

**Tech Stack:** Rust 2021, clap 4, serde/serde_json, `unicode-width`, `terminal_size`, `rayon` (CLI crate only), `indicatif` (stderr progress), `insta` (snapshots).

**Spec:** `docs/superpowers/specs/2026-11-21-cli-terminal-report-design.md`

## Global Constraints

- `inspector` must not gain a heavy dependency (no `regex`, no `rayon`); `unicode-width` in the CLI crate only. wasm build size is a guard.
- `rayon` lives in `crates/lora-inspector` only. `inspector` stays single-threaded.
- Default report path must never call `LoRAFile::scale_weight` (that is the multi-minute materialization path).
- All rendered output is deterministic: sort before render, never iterate a `HashMap`/`HashSet` into display order.
- No ANSI escapes when stdout is not a TTY, when `NO_COLOR` is set, or when `--json` is passed.
- `--json` output must be byte-identical regardless of `--width`.
- Release profile: `opt-level = 3` and `lto = true` in a real `[profile.release]` section; `-C target-cpu` flags stay out of the manifest (documented Makefile target only).
- `make test`, `make wasm-bindgen-test`, `make fmt && dprint fmt` must pass at the end.
- Phase 0 (Task 0) may change dependencies but must not change `inspector`'s public API shape, and the wasm bundle may not grow more than 10% over the recorded baseline.
- JSON contract, exactly as the spec's JSON section and field table: `schema_version: 1`,
  flat `blocks`/`rank` arrays, stable keys that never disappear, `null` = not computed and
  `[]` = computed-and-empty, floats rounded to 6 significant digits, arrays in display
  order, errors as a single `{"error": {"kind", "message"}}` object on stderr.

## Review Focus

1. **Double-width and combining characters in filenames or keys** (`Klein 一edit-consistency`, `retro-comic-illustration-复古风格`) — labels, bars and columns must line up and nothing may panic on a multi-byte grapheme boundary.
2. **Absurd terminal widths** (`COLUMNS=0`, `COLUMNS` unset in CI, `--width 1`, width wider than the content) — must clamp and still render something sensible, never divide by zero or build a negative-width bar.
3. **Files with zero, one, or unparseable block keys** — empty/`Other`-only group sections must print a real message rather than a blank block or a broken table.
4. **Mixed-rank / mixed-alpha files** (`dims()`/`alphas()` return `HashSet` with 2+ values) — header must render all values, not the first, and not panic.
5. **`--json` piped with no TTY while progress is enabled** — progress must not leak into stdout, and stdout must parse as a single JSON object.
6. **A section that was not requested versus one that was requested and is empty** — an agent must be able to tell them apart; `null` and `[]` must never be used interchangeably, and the key must never vanish.

---

### Task 0: Upgrade candle and safetensors

**Files:**
- Modify: `Cargo.toml` (workspace `dependencies`)
- Modify: `Cargo.lock`
- Modify: `crates/inspector/src/*.rs` and `crates/lora-inspector-wasm/src/*.rs` only where the new APIs force it
- Modify: `crates/inspector/src/snapshots/*.snap` only for real value changes, with each change explained

**Interfaces:**
- Consumes: the existing public API of `inspector` (must survive intact).
- Produces: `candle-core 0.11`, `safetensors 0.8`, and a workspace that compiles against them. Later tasks are written once against these APIs.

- [ ] **Step 1: Record the baseline**

Run: `cargo test --workspace 2>&1 | grep -E "^test result" ` and `ls -l crates/lora-inspector-wasm/pkg/*.wasm`
Expected baseline to record in the report: the per-target test counts, plus `lora-inspector_bg.wasm` = 7 737 686 B and `lora-inspector-simd_bg.wasm` = 7 757 294 B.

- [ ] **Step 2: Bump the pins and find the breakage**

Set `candle-core = "0.11"` and `safetensors = "0.8"` in the workspace `dependencies` table, then `cargo check --workspace --all-targets` and collect every error. Expect changes in these areas — do not speculate beyond them, just let the compiler list them: `candle_core::Error` variant shapes (`CannotFindTensor`, `SafeTensor`), `safetensors::tensor::TensorView::new` and `SafeTensors::read_metadata` signatures, `DType`/`WithDType` naming, and anything `wasm_bindgen`-adjacent in the wasm crate.

- [ ] **Step 3: Fix compilation without changing behaviour**

Fix errors one file at a time. The rule for every fix: same public name, same signature, same numeric result. If a dependency forces an `inspector` signature change, make it and list it explicitly in the report and the commit message body.

- [ ] **Step 4: Make the suite green**

Run: `cargo test --workspace`
Expected: PASS. If snapshots differ, run `cargo insta show` (or read the `.snap` diff) and confirm each changed number is a genuine numeric change rather than a formatting artifact. Explain every changed value in the report; revert the upgrade and report BLOCKED if a value changes beyond float noise and you cannot explain why.

- [ ] **Step 5: Verify the wasm target and its size**

Run: `cargo check --target wasm32-unknown-unknown -p lora-inspector-wasm && make build-wasm-dev && ls -l crates/lora-inspector-wasm/pkg/*.wasm`
Expected: check passes, and neither wasm exceeds 10% over the Step 1 baseline (i.e. ≤ 8 511 455 B / ≤ 8 532 023 B). Record the exact new sizes.

- [ ] **Step 6: Attempt the cuda feature**

Run: `cargo build --release -p lora-inspector --features cuda`
Expected: either it builds (report the resulting `cudarc` version) or it fails against the current CUDA toolkit — in which case record the first real error and report it as a finding. Do not add workarounds, and do not remove the feature.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock crates/
git commit -m "build: upgrade candle to 0.11 and safetensors to 0.8" 
```
The commit body lists any `inspector` signature that had to change, the new wasm sizes, and the cuda result.

---

### Task 1: Fix the release profile and record the measurement

**Files:**
- Modify: `Cargo.toml` (root, `[profile]` block at the end)
- Modify: `crates/lora-inspector/Cargo.toml` (keep `cuda` feature as is)
- Modify: `Makefile` — pin the wasm release size profile, which this task removes from the workspace default

**Interfaces:**
- Consumes: nothing.
- Produces: a release build where `candle-core` and `inspector` are compiled at `opt-level=3`, unblocking every later task's performance budget.

- [ ] **Step 1: Confirm the current flag mix**

Run: `cargo clean -p candle-core -p inspector --release && cargo build --release -p lora-inspector -v 2>&1 | grep -oE "\-C opt-level=[a-z0-9]+" | sort | uniq -c`
Expected today: two `opt-level=z` (candle-core, inspector) and one `opt-level=3`.

- [ ] **Step 2: Replace the misnested profile section**

The root manifest has a `[profile]` table containing `dev = {...}` and `release = {...}`. Cargo ignores the `package` override inside it. Rewrite as real sections: `[profile.release]` with `opt-level = 3`, `lto = true`, `codegen-units = 1`, `incremental = false`; `[profile.release.package.inspector] opt-level = 3`; and leave a comment that `opt-level = "z"` belongs to the wasm build only (that build passes its own flags via the Makefile).

- [ ] **Step 3: Re-measure**

Run: `cargo clean --release && cargo build --release -p lora-inspector -v 2>&1 | grep -oE "\-C opt-level=[a-z0-9]+" | sort | uniq -c`
Expected: no `opt-level=z` entries. A full `cargo clean --release` is required here: an incremental build reports a misleading mix because it reuses `z`-built rlibs.

Run: `time ./target/release/lora-inspector block-weights --file boo.safetensors --output-format text >/dev/null`
Expected: well under the 13.3 s baseline recorded in the spec (it still computes the wasteful norms; this task only changes codegen).

- [ ] **Step 3b: Keep the wasm build size-pinned**

The wasm release build inherited `opt-level = "z"` from the workspace profile that this task replaces, and it passes no `opt-level` of its own. wasm-pack cannot be told: `[package.metadata.wasm-pack.profile.*]` accepts only `wasm-bindgen` and `wasm-opt` keys (0.13.1 and 0.15.1), and unknown keys are ignored silently — verified by capturing rustc invocations, which still received `-C opt-level=3`.

So pin it in the Makefile instead: give `build-wasm` `RUSTFLAGS="-C opt-level=z"` and `build-wasm-simd` `RUSTFLAGS="-C target-feature=+simd128 -C opt-level=z"`, leaving the dev targets alone. Verify with a clean `make build-wasm && make build-wasm-simd` and compare against the release baseline `lora-inspector_bg.wasm` 611 266 B / `lora-inspector-simd_bg.wasm` 613 040 B (within ~2%). Never measure `pkg/` after a dev build — those artifacts are ~7.7 MB.

- [ ] **Step 4: Run the existing suite**

Run: `cargo test --workspace`
Expected: PASS. If a snapshot test fails only because of float formatting, stop and report — do not regenerate snapshots in this task.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml
git commit -m "build: use a real [profile.release] so candle and inspector are opt-level 3"
```

---

### Task 2: Key-list fixtures for block grouping

**Files:**
- Create: `crates/inspector/tests/fixtures/blocks/sd15.txt`
- Create: `crates/inspector/tests/fixtures/blocks/sdxl.txt`
- Create: `crates/inspector/tests/fixtures/blocks/flux.txt`
- Create: `crates/inspector/tests/fixtures/blocks/flux_peft.txt`
- Create: `crates/inspector/tests/fixtures/blocks/flux2_klein.txt`
- Create: `crates/inspector/tests/fixtures/blocks/wan22.txt`
- Create: `crates/inspector/tests/fixtures/blocks/krea2.txt`
- Create: `crates/inspector/tests/fixtures/blocks/h3.txt`
- Create: `crates/inspector/tests/fixtures/blocks/ltx23.txt`
- Create: `crates/inspector/tests/fixtures/blocks/qwen_image.txt`
- Create: `crates/inspector/tests/fixtures/blocks/moss.txt`
- Create: `crates/inspector/tests/fixtures/blocks/gemma3.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: plain-text fixture files, one base name per line (suffixes like `.lora_up.weight` already stripped with the existing `inspector::get_base_name`), loaded with `include_str!` by Task 3's tests. No safetensors and no `/mnt` dependency.

- [ ] **Step 1: Capture ~20 base names per architecture**

For each architecture, read the safetensors header from a file under `/mnt/900/lora` (header only — 8-byte length prefix then that many bytes of JSON), take the tensor names, strip the factor suffix, and write ~20 distinct base names per file, covering at least one nested case and one text-encoder case if present. Source files, one per fixture: `sdxl/Aether_Ghost_v1.1_LoRA.safetensors`, `flux/urban_decay_v20_rank64_bf16-step02016.safetensors`, `flux/RetroAnimeFluxV1_r8.safetensors`, `flux2-klein-4b/Klein 一edit-consistency.safetensors`, `wan2.2/14b/23High noise-Cumshot Aesthetics.safetensors`, `krea2/DC Dark Fantasy Style 3 … v1.safetensors`, `h3/H3_Motion_BoosterV2.safetensors`, `ltx2.3/DR34ML4Y_LT3X_V3.safetensors`, `qwen-image/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors`, `mossimage/mj_base_test_w8zz8rd_512plus_moss_v1.2-step00000600.safetensors`, `gemma3/lora_r16/kohya/gemma3_heretic_diff_r16.safetensors`, plus `boo.safetensors` for SD1.5.

- [ ] **Step 2: Verify no fixture leaked a whole file**

Run: `wc -l crates/inspector/tests/fixtures/blocks/*.txt`
Expected: each file 15–25 lines, all under 1 KB total each.

- [ ] **Step 3: Commit**

```bash
git add crates/inspector/tests/fixtures/blocks
git commit -m "test: add per-architecture block-key fixtures"
```

---

### Task 3: `inspector::blocks` grouping

**Files:**
- Create: `crates/inspector/src/blocks.rs`
- Modify: `crates/inspector/src/lib.rs:14-21` (add `pub mod blocks;` beside the other module declarations)

**Interfaces:**
- Consumes: `crate::get_base_name` (existing) only in tests for producing fixture-ready names.
- Produces, for Tasks 5–7:
  - `pub enum BlockGroup { Down, Input, Mid, Middle, Up, Output, Double, Single, SingleTransformer, Transformer, Refiner, TokenRefiner, TxtFusion, Blocks, Layers, Modules, TextEncoder, Other }` — `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]`, serde variant names as written.
  - `pub enum BlockFamily { Unet, TextEncoder, Other }` — `#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]`, with `#[serde(rename = "UNet")]` on `Unet`. Serializes as `"UNet" | "TextEncoder" | "Other"`.
  - `pub struct BlockKey { pub group: BlockGroup, pub block_index: Option<usize>, pub sublayer: String }` with `pub fn family(&self) -> BlockFamily`.
  - `pub fn parse_block_key(base_name: &str) -> BlockKey`.
  - `impl BlockGroup { pub fn tag(&self, index: Option<usize>) -> String; pub fn canonical_rank(&self) -> u8; }` — `tag` yields the two-letter label plus zero-padded 2-digit index (`IN04`, `DB03`, `TE11`, `MID01`), or the bare label when `index` is `None`.

- [ ] **Step 1: Write the failing tests**

Add `#[cfg(test)] mod tests` in `blocks.rs`. The fixture assertion is the point of the task: a table of `(base_name, expected_group, expected_index, expected_sublayer_first_segment)` covering every fixture file via `include_str!`, plus these named cases:

```rust
#[test]
fn leftmost_family_wins_over_inner_transformer_blocks() {
    let k = parse_block_key("lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj");
    assert_eq!(k.group, BlockGroup::Down);
    assert_eq!(k.block_index, Some(0));
    assert_eq!(k.sublayer, "attentions.0.transformer_blocks.0.ff_net.0.proj");
}

#[test]
fn peft_single_transformer_blocks_is_its_own_group() {
    let k = parse_block_key("transformer.single_transformer_blocks.7.linear1");
    assert_eq!(k.group, BlockGroup::SingleTransformer);
    assert_eq!(k.block_index, Some(7));
}

#[test]
fn text_encoder_wins_over_unet_layers() {
    let a = parse_block_key("lora_te_text_model_encoder_layers_11_self_attn_v_proj");
    let b = parse_block_key("language_model_model_layers_18_self_attn_o_proj");
    assert_eq!((a.group, a.block_index), (BlockGroup::TextEncoder, Some(11)));
    assert_eq!((b.group, b.block_index), (BlockGroup::TextEncoder, Some(18)));
}

#[test]
fn dotted_peft_text_encoder_is_normalized() {
    let k = parse_block_key("text_model.encoder.layers.11.self_attn.q_proj");
    assert_eq!(k.group, BlockGroup::TextEncoder);
    assert_eq!(k.block_index, Some(11));
}

#[test]
fn bare_and_refiner_families_are_recognized() {
    assert_eq!(parse_block_key("diffusion_model.blocks.3.cross_attn.k").group, BlockGroup::Blocks);
    assert_eq!(parse_block_key("diffusion_model.token_refiner.refiner_blocks.1.mlp.fc2").group, BlockGroup::TokenRefiner);
    assert_eq!(parse_block_key("diffusion_model.blocks.0noise_refiner_0_adaLN").group, BlockGroup::Other);
    assert_eq!(parse_block_key("diffusion_model.noise_refiner.0.adaLN_modulation").group, BlockGroup::Refiner);
}

#[test]
fn unparseable_key_keeps_its_sublayer_and_is_other() {
    let k = parse_block_key("txtfusion_projector");
    assert_eq!(k.group, BlockGroup::Other);
    assert_eq!(k.block_index, None);
    assert_eq!(k.family(), BlockFamily::Other);
}

#[test]
fn tags_are_stable() {
    assert_eq!(BlockGroup::Input.tag(Some(4)), "IN04");
    assert_eq!(BlockGroup::Double.tag(Some(3)), "DB03");
    assert_eq!(BlockGroup::TextEncoder.tag(Some(11)), "TE11");
    assert_eq!(BlockGroup::Other.tag(None), "OT");
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p inspector blocks -- --nocapture`
Expected: FAIL — `blocks` module / `parse_block_key` not found.

- [ ] **Step 3: Implement the tokenizer**

In `blocks.rs`: normalize the base name by splitting on `.`, `_` and `-` into a `Vec<&str>` of tokens (drop empties), then scan left to right. At each position try 3-, then 2-, then 1-token patterns against the family table from the spec; the first position with any match wins, longest pattern at that position breaking ties. Text-encoder detection (`te`, `te<digit>`, `text`, `model`, `encoder`, `language` markers) is checked first over the whole token stream and short-circuits. The block index is the first token after the matched pattern that parses as `usize`. The sublayer is the remaining tokens joined with `.`. If no pattern matches, `group = Other`, `block_index` = first numeric token if any, `sublayer` = the whole key with separators normalized to `.`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test -p inspector blocks`
Expected: PASS, including all 12 fixture files.

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/src/blocks.rs crates/inspector/src/lib.rs
git commit -m "feat(inspector): add shared block-key grouping"
```

---

### Task 4: Expose the cheap (non-materializing) per-layer norm

**Files:**
- Modify: `crates/inspector/src/file.rs:253` (`fn lora_frobenius_norm`) and `:228` (`pub fn effective_scale`)
- Modify: `crates/inspector/src/file.rs` test module (append one test)

**Interfaces:**
- Consumes: existing private `lora_frobenius_norm(base_name) -> Result<Option<f64>>`, which computes `||up @ down||_F × |alpha / rank|` from rank×rank Gram matrices and returns `Ok(None)` for anything that is not a plain 2D `up`/`down` pair.
- Produces, for Task 5:
  - `pub fn low_rank_effective_scale(&self, base_name: &str) -> Result<Option<f64>>` — the Gram path only, never falls back to `scale_weight`.
  - `pub fn effective_scale(&self, base_name: &str) -> Result<Option<f64>>` keeps its current behaviour (tries the Gram path, then falls back) and delegates its first branch to the new public method.

- [ ] **Step 1: Write the failing test**

In the existing `#[cfg(test)] mod tests` in `file.rs`:

```rust
#[test]
fn low_rank_effective_scale_returns_none_for_hada_layers() {
    let mut f = File::open("edgWar40KAdeptaSororitas.safetensors").unwrap();
    let mut buffer = vec![];
    f.read_to_end(&mut buffer).unwrap();
    let lora_file = LoRAFile::new_from_buffer(&buffer, "edgWar40KAdeptaSororitas.safetensors", &Device::Cpu);
    let hada_base = lora_file
        .base_names()
        .into_iter()
        .find(|b| b.contains("hada"))
        .expect("fixture has hada layers");
    assert_eq!(lora_file.low_rank_effective_scale(&hada_base).unwrap(), None);
}
```

If that fixture has no `hada` layers, use any base name whose keys include `hada_w1_a`; assert `is_none()` either way and state the base name in a comment.

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test -p inspector low_rank_effective_scale -- --nocapture`
Expected: FAIL — no method `low_rank_effective_scale`.

- [ ] **Step 3: Implement**

Rename the private `lora_frobenius_norm` to `pub fn low_rank_effective_scale` (keep the body and doc comment verbatim), and change `effective_scale`'s first branch to call it. Add one sentence to the doc comment: this is the only path that is safe for large layers, because it never materializes `up @ down`.

- [ ] **Step 4: Run it to verify it passes, and that nothing else broke**

Run: `cargo test -p inspector`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/src/file.rs
git commit -m "feat(inspector): expose the non-materializing effective-scale path"
```

---

### Task 5: `Terminal` layout primitives

**Files:**
- Create: `crates/lora-inspector/src/terminal.rs`
- Modify: `crates/lora-inspector/src/main.rs` (add `mod terminal;`)
- Modify: `crates/lora-inspector/Cargo.toml` (add `unicode-width = "0.2"`, `terminal_size = "0.4"`)

**Interfaces:**
- Consumes: nothing.
- Produces, for Tasks 6–8:
  - `pub struct Terminal { pub width: usize, pub color: bool, pub unicode: bool }`
  - `pub enum Style { Good, Ok, Weak, Poor, Collapsed, Dim, Bold }`
  - `pub fn detect(flag_width: Option<usize>, flag_color: ColorChoice, ascii: bool) -> Terminal` where `ColorChoice` is `clap::ValueEnum` with `Auto | Always | Never`.
  - `Terminal::text_width(&str) -> usize` (display width, ANSI- and width-aware), `truncate_middle(&str, usize) -> String`, `pad_right(&str, usize) -> String`, `bar(f64, usize) -> String`, `sparkline(&[f64], usize) -> String`, `paint(Style, &str) -> String`.

- [ ] **Step 1: Write the failing tests**

In `#[cfg(test)] mod tests` of `terminal.rs`:

```rust
#[test]
fn cjk_names_are_measured_by_display_width() {
    let t = Terminal { width: 40, color: false, unicode: true };
    assert_eq!(t.text_width("Klein 一edit-consistency"), 24);
    assert_eq!(t.text_width("复古风格"), 8);
}

#[test]
fn truncation_never_splits_a_grapheme() {
    let t = Terminal { width: 40, color: false, unicode: true };
    let out = t.truncate_middle("retro-comic-illustration-复古风格", 12);
    assert_eq!(t.text_width(&out), 12);
    assert!(out.contains('…'));
}

#[test]
fn width_resolution_clamps_to_a_usable_minimum() {
    assert_eq!(Terminal::detect(Some(0), ColorChoice::Never, false).width, 20);
    assert_eq!(Terminal::detect(Some(1), ColorChoice::Never, false).width, 20);
    assert_eq!(Terminal::detect(Some(200), ColorChoice::Never, false).width, 200);
}

#[test]
fn bars_never_exceed_their_width_even_for_bad_input() {
    let t = Terminal { width: 80, color: false, unicode: true };
    assert_eq!(t.text_width(&t.bar(f64::NAN, 10)), 10);
    assert_eq!(t.text_width(&t.bar(-1.0, 10)), 10);
    assert_eq!(t.text_width(&t.bar(5.0, 10)), 10);
}

#[test]
fn sparkline_resamples_to_exact_width() {
    let t = Terminal { width: 80, color: false, unicode: true };
    let values: Vec<f64> = (0..26).map(|i| i as f64).collect();
    assert_eq!(t.text_width(&t.sparkline(&values, 12)), 12);
    assert_eq!(t.text_width(&t.sparkline(&[], 12)), 12);
}

#[test]
fn color_is_off_when_not_auto_or_not_tty() {
    let t = Terminal::detect(None, ColorChoice::Never, false);
    assert_eq!(t.paint(Style::Good, "x"), "x");
}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p lora-inspector terminal`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement**

Width resolution order: `--width`, then `COLUMNS`, then `terminal_size::terminal_size()`, then 80; clamp to `20..=usize::MAX`. Color enabled only when not `Never`, not `--json`, stdout `IsTerminal`, `NO_COLOR` unset, and `TERM != "dumb"`. `bar` clamps its fraction to `0.0..=1.0` (treating `NaN` as 0), fills floor(width×frac) full blocks and pads the rest with spaces. `sparkline` resamples by nearest-neighbour index into `▁▂▃▄▅▆▇█`, and emits spaces when all values are equal or the slice is empty. ASCII mode substitutes `#`, `-`, `=`, `.`, `:` and `~` for the Unicode glyphs.

- [ ] **Step 4: Run them to verify they pass**

Run: `cargo test -p lora-inspector terminal`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/lora-inspector/src/terminal.rs crates/lora-inspector/src/main.rs crates/lora-inspector/Cargo.toml Cargo.lock
git commit -m "feat(cli): add width- and unicode-aware terminal primitives"
```

---

### Task 6: `report` data model and block aggregation

**Files:**
- Create: `crates/lora-inspector/src/report.rs`
- Modify: `crates/lora-inspector/src/main.rs` (add `mod report;`)

**Interfaces:**
- Consumes: `inspector::blocks::{parse_block_key, BlockGroup, BlockFamily}`, `LoRAFile::{base_names, low_rank_effective_scale, effective_scale, precision, dims, alphas, format}`, `Metadata::{new_from_header_buffer, network_module_display, network_family, network_type}`, `inspector::LayerScale`.
- Produces, for Tasks 7–9: the serde-shaped structs from the spec's JSON section —
  - `pub struct ReportOptions { pub blocks: bool, pub rank: bool, pub top: usize, pub jobs: usize, pub slow: bool }`
  - `pub struct Report { pub schema_version: u32, pub file: FileInfo, pub network: NetworkInfo, pub metadata: Option<HashMap<String, String>>, pub blocks: Option<Vec<BlockStat>>, pub ungrouped: Ungrouped, pub skipped: Skipped, pub rank: Option<Vec<RankLayer>>, pub summary: Summary }`, all `Serialize`. `schema_version` is the constant `1`.
  - `pub struct BlockStat { pub group: BlockGroup, pub family: BlockFamily, pub label: String, pub index: Option<usize>, pub layers: usize, pub mean: f64, pub max: f64, pub median: f64, pub outlier: bool }` — one flat row per (group, index); the group and family repeat on every row.
  - `pub struct Ungrouped { pub layers: usize, pub prefixes: Vec<String> }`, `pub struct Skipped { pub layers: usize, pub reason: Option<String> }`
  - `pub struct RankLayer { pub base_name: String, pub group: BlockGroup, pub label: String, pub index: Option<usize>, pub nominal_rank: usize, pub effective_rank: f64, pub balance: f64, pub top1_energy: f64, pub dominance: Option<f64>, pub health: RankHealth, pub eff_scale: Option<f64>, pub outlier: bool }` — declared here because it is part of the `schema_version: 1` contract; Task 8 only fills it in. Field names and order match `blocks[]` wherever they overlap.
  - `pub struct RankSummary { pub layers: usize, pub by_health: BTreeMap<String, usize>, pub median_balance: f64, pub weak_or_worse: usize, pub elapsed_ms: u128 }` — `by_health` always carries all five health words, including zeros, so an agent can index it safely.
  - `pub struct Summary { pub block_weight: Option<BlockWeightSummary>, pub rank: Option<RankSummary> }`, `pub struct BlockWeightSummary { pub median: f64, pub peak: Peak, pub outliers: usize }`, `pub struct Peak { pub label: String, pub value: f64, pub ratio_to_median: f64 }`
  - `pub fn build_blocks(file: &LoRAFile, opts: &ReportOptions) -> Result<(Vec<BlockStat>, Ungrouped, Skipped, BlockWeightSummary)>` — one pass over `base_names()`, using `low_rank_effective_scale` unless `opts.slow` is set (then `effective_scale`); layers returning `None` increment `skipped.layers` and set `skipped.reason`.
  - `pub fn build_network_info(file: &LoRAFile, meta: &Metadata, total_layers: usize, unet: usize, te: usize) -> NetworkInfo` — `rank`/`alpha` are sorted `Vec<usize>`/`Vec<f64>`.
  - `pub fn build_report(path: &Path, file: &LoRAFile, meta: &Metadata, opts: &ReportOptions) -> Result<Report>` — assembles a `Report`, wiring `blocks: None` when `opts.blocks` is false and `rank: None` unless `opts.rank`. This is the only constructor of `Report`.
  - `pub fn round_sig(x: f64, digits: u32) -> f64` — the single helper every float in the report passes through, so human and JSON output agree and no value exceeds 6 significant digits.

- [ ] **Step 1: Write the failing tests**

Create `crates/lora-inspector/tests/support/mod.rs` (the shared fixture helper, also used by Tasks 7 and 8) and `crates/lora-inspector/tests/report.rs` with `mod support;`. The helper writes a synthetic safetensors file (8-byte LE header length, JSON header with `data_offsets`, little-endian f32 payload) into `std::env::temp_dir()` and deletes it at the end of the test. The synthetic file has 2 `down_blocks` layers with `up`/`down`/`alpha` and 2 `text_model_encoder.layers` layers, values chosen so the block means are round numbers.

```rust
#[test]
fn block_means_average_the_layers_in_each_block() { /* assert BlockStat.mean for IN00 and TE00 against the fixture's known values */ }

#[test]
fn layers_without_low_rank_factors_are_skipped_not_materialized() { /* fixture includes one hada_w1_a/hada_w1_b layer; assert skipped.count == 1 and that no BlockStat carries its value */ }

#[test]
fn ungrouped_layers_are_counted_with_their_prefix() { /* fixture includes one key with no family pattern; assert ungrouped.count == 1 and prefixes == ["txtfusion_projector"] */ }

#[test]
fn multiple_ranks_and_alphas_are_all_reported() { /* fixture declares rank 4 and 8, alpha 4.0 and 8.0; assert network.rank == vec![4, 8], alpha == vec![4.0, 8.0] */ }

#[test]
fn layer_counts_come_from_block_families_not_name_substrings() { /* synthetic file is PEFT-shaped (no lora_unet/lora_te substrings); assert layers.unet == 2, layers.text_encoder == 2, layers.total == 4 */ }

#[test]
fn every_float_is_rounded_to_six_significant_digits() { /* serialize the report and assert no number matches a >6-sig-digit pattern, e.g. 0.4000000000000001 */ }

#[test]
fn rows_repeat_group_and_family_so_no_join_is_needed() { /* assert every BlockStat carries a non-empty group/family even though rows are a flat list */ }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p lora-inspector --test report`
Expected: FAIL — crate has no `tests/` target yet / `report` module missing.

- [ ] **Step 3: Implement**

`build_blocks`: collect `(BlockKey, base_name, scale)` for every base name; group by `(group, block_index)`; compute `mean` (arithmetic), `max`, `median` (sorted copy, even-length average) and `outlier` using the existing Rust formula `if median_of_all < 1e-10 { 1e-10 } else { 1.5 * median_of_all }` compared against the layer's scale. Sort the flat row list by `group.canonical_rank()`, then `index`, then `label` — this is the display order the JSON contract promises. `ungrouped.prefixes` is the sorted, deduplicated first token of each `Other` key's sublayer, cap 10. `skipped` counts layers whose scale was `None`.

- [ ] **Step 4: Run them to verify they pass**

Run: `cargo test -p lora-inspector --test report`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/lora-inspector/src/report.rs crates/lora-inspector/src/main.rs crates/lora-inspector/tests/report.rs
git commit -m "feat(cli): build the block-weight report without materializing weights"
```

---

### Task 7: `render` — three width tiers

**Files:**
- Create: `crates/lora-inspector/src/render.rs`
- Modify: `crates/lora-inspector/src/main.rs` (add `mod render;`)
- Modify: `crates/lora-inspector/Cargo.toml` (`[dev-dependencies] insta = "1.34.0"`)
- Create: `crates/lora-inspector/tests/render_snapshots.rs`

**Interfaces:**
- Consumes: `Report`, `BlockStat`, `Ungrouped`, `Skipped`, `Summary`, `Terminal`, `Style` from Tasks 5–6.
- Produces, for Task 9: `pub fn render(report: &Report, meta: &Metadata, term: &Terminal, opts: &ReportOptions) -> String` (single string with trailing newline; sections separated by one blank line) and `pub fn render_header(report: &Report, meta: &Metadata, term: &Terminal) -> String`. `meta` is passed separately on purpose: the header prints `modelspec.architecture` and `modelspec.trigger_phrase` even when `--metadata` is off, and `Report.metadata` stays `null` in that case.

- [ ] **Step 1: Write the failing tests**

`tests/render_snapshots.rs` starts with `mod support;` and reuses the helper Task 6 created in `tests/support/mod.rs` and, for each width in `[40, 60, 80, 120, 200]`, calls `insta::assert_snapshot!` on `render(&report, &meta, &terminal, &opts)`.

Add one `--ascii` snapshot at width 80, and these named assertions:

```rust
#[test]
fn wide_tier_sorts_rows_by_value_not_by_group() { /* assert the first row of the wide table holds the highest mean across all groups */ }

#[test]
fn compact_tier_keeps_family_order() { /* assert the UNet sparkline row precedes the TextEncoder row even when TE has the larger peak */ }

#[test]
fn empty_block_section_prints_a_message() { /* Report with zero groups: assert output contains "no groupable layers" and has no empty table header */ }

#[test]
fn all_widths_have_identical_json() { /* serde_json::to_string_pretty for width 40 vs 200 → assert_eq */ }

#[test]
fn header_prints_every_rank_and_alpha_when_there_are_several() { /* Report with rank [4,8] and alpha [4.0,8.0]; assert the header line contains "4, 8" and "4, 8" rather than only the first value */ }

#[test]
fn skipped_layers_are_reported_with_the_slow_hint() { /* Report with skipped.layers = 3; assert the footer contains the count and "--slow" */ }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p lora-inspector --test render_snapshots`
Expected: FAIL — `render` not found. Accept the new snapshots with `INSTA_UPDATE=always` only after reading the output and confirming the layout matches the spec's three tiers.

- [ ] **Step 3: Implement the tiers**

Header: filename and size on line 1; `module · type · rank · alpha · precision · layer counts` on line 2; `modelspec.architecture` and `modelspec.trigger_phrase` on line 3 only when present in the metadata (nothing else from metadata is printed; that is what `--metadata` is for). When `network.rank` or `network.alpha` holds more than one value, print them comma-joined rather than one number.

Wide (width ≥ 100): header, `block weight` title line, one flat table of the top `opts.top` blocks sorted by `mean` descending with columns `label | bar | mean | layers`; then a `rank` section when present. Compact (60–99): one sparkline row per **family** (`UNet`, `TextEncoder`), each row's series built from that family's blocks ordered by canonical group rank then index, plus a top-3 table. Minimal (< 60): one line per group (`UNet 26 blocks  med 0.41  peak 1.82×4.4`), no bars. Every string placed in a column goes through `truncate_middle`/`pad_right` with `text_width`. Health words always print, color is additive only.

Footer lines, each printed only when non-zero, directly after the block section:
`N layers ungrouped (prefix, prefix, …)` from `ungrouped`, and
`N layers skipped: no low-rank factor (LoHA/LoKr/OFT/BOFT) — pass --slow to include`
from `skipped`. When `blocks` is an empty array, print `no groupable layers` instead of a
table header.

- [ ] **Step 4: Run them to verify they pass, and review the snapshots as text**

Run: `cargo test -p lora-inspector --test render_snapshots`
Expected: PASS, and reading the `.snap` files for width 40 and 200 shows aligned columns with no lines exceeding the width.

- [ ] **Step 5: Commit**

```bash
git add crates/lora-inspector/src/render.rs crates/lora-inspector/src/main.rs crates/lora-inspector/Cargo.toml crates/lora-inspector/tests/render_snapshots.rs crates/lora-inspector/tests/snapshots
git commit -m "feat(cli): render the report at three width tiers"
```

---

### Task 8: Rank pass with rayon

**Files:**
- Modify: `crates/lora-inspector/src/report.rs` (add `pub fn build_rank` and wire `summary.rank`; the `RankLayer`/`RankSummary` types already exist from Task 6)
- Modify: `crates/lora-inspector/Cargo.toml` (add `rayon = "1.12"`)
- Create: `crates/lora-inspector/tests/rank.rs`

**Interfaces:**
- Consumes: `LoRAFile::rank_metrics(base_name) -> Result<Option<RankMetrics>>`, `inspector::svd::RankMetrics` fields `nominal_rank, effective_rank, balance, top1_energy, dominance, health`.
- Produces, for Tasks 9:
  - `pub fn build_rank(file: &LoRAFile, jobs: usize) -> Result<(Vec<RankLayer>, RankSummary)>` — `group`/`label`/`index` for each row come from `inspector::blocks::parse_block_key(base_name)`, not from the block stats, so the rank pass has no ordering dependency on the block pass. — builds a rayon pool of `jobs` threads, `par_iter` over the sorted base names, collects, sorts by `base_name`, then computes the summary. Deterministic by construction.

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn rank_section_is_identical_for_one_and_four_jobs() { /* build twice, assert serde_json of both is equal */ }

#[test]
fn rank_layers_are_sorted_by_base_name() { /* assert windows(2).all(|w| w[0].base_name <= w[1].base_name) */ }

#[test]
fn health_summary_counts_every_layer() { /* assert by_health values sum to layers */ }

#[test]
fn rank_labels_match_the_block_pass() { /* build blocks and rank from the same fixture; assert every RankLayer.label equals the label the block pass produced for the same base name */ }

#[test]
fn health_summary_always_has_all_five_keys() { /* assert by_health contains Good, Ok, Weak, Poor, Collapsed even when the count is 0 */ }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p lora-inspector --test rank`
Expected: FAIL — `build_rank` not found.

- [ ] **Step 3: Implement**

`jobs == 0` means available cores. Health words come from `RankHealth`'s existing `Serialize`; render them capitalized (`Good`, `Ok`, `Weak`, `Poor`, `Collapsed`). `eff_scale` and `outlier` are taken from the same `LayerScale` logic the block pass uses, so the two sections cannot disagree.

- [ ] **Step 4: Run them to verify they pass**

Run: `cargo test -p lora-inspector --test rank`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/lora-inspector/src/report.rs crates/lora-inspector/tests/rank.rs crates/lora-inspector/Cargo.toml Cargo.lock
git commit -m "feat(cli): parallel effective-rank pass"
```

---

### Task 9: Wire the CLI surface

**Files:**
- Modify: `crates/lora-inspector/src/main.rs` (new `Inspect` subcommand, remove the CUDA probe, progress to stderr)
- Modify: `crates/lora-inspector/README.md`
- Modify: `crates/lora-inspector/src/render.rs` (rank section, if not already in Task 7)

**Interfaces:**
- Consumes: everything from Tasks 3–8.
- Produces: the user-facing command; no new library interfaces.

- [ ] **Step 1: Write the failing CLI tests**

Create `crates/lora-inspector/tests/cli.rs` using `std::process::Command` against `env!("CARGO_BIN_EXE_lora-inspector")`:

```rust
#[test]
fn json_is_a_single_object_and_independent_of_width() { /* run with --json --width 40 and --json --width 200; strip trailing newline; assert both parse to serde_json::Value and are equal */ }

#[test]
fn json_rank_is_null_unless_requested() { /* assert value["rank"].is_null() without --rank; with --rank assert it is an object */ }

#[test]
fn no_blocks_skips_the_block_payload() { /* assert value["blocks"].is_null() and that --json --no-blocks succeeds on the fixture */ }

#[test]
fn a_file_with_nothing_groupable_gives_an_empty_array_not_null() { /* run --json on a fixture whose only keys are ungroupable; assert value["blocks"] == [] and value["ungrouped"]["layers"] > 0 */ }

#[test]
fn progress_never_lands_on_stdout() { /* run --json with stderr captured separately and stdout piped; assert stdout parses as JSON and contains no spinner glyphs */ }

#[test]
fn block_weights_alias_still_works() { /* assert exit status success and output contains "block weight" */ }

#[test]
fn narrow_width_does_not_panic() { /* run with --width 1 and --width 0; assert success and no "panicked" in stderr */ }

#[test]
fn metadata_is_null_unless_requested() { /* assert value["metadata"].is_null(); with --metadata assert it is an object */ }

#[test]
fn schema_version_is_one() { /* assert value["schema_version"] == 1 */ }

#[test]
fn a_missing_file_is_one_json_error_object_with_exit_one() { /* run --json on a nonexistent path; assert exit code 1, stdout empty, stderr parses as {"error": {"kind": "io", "message": ...}} */ }

#[test]
fn a_bad_file_is_an_error_not_a_partial_report() { /* run --json on a truncated fixture; assert exit 1 and stderr is the single error object, never a Report with empty blocks */ }
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cargo test -p lora-inspector --test cli`
Expected: FAIL — no `inspect` subcommand.

- [ ] **Step 3: Implement the surface**

`inspect <FILE>` with `--rank`, `--json`, `--no-blocks`, `--metadata`, `--slow`, `--width`, `--top` (default 8), `--jobs` (default 0 = cores), `--ascii`, `--color`, `--quiet`. Every flag gets a one-line plain-language `help`, and the command gets a worked example in its `long_about`, so `--help` is the API. Under `--json`, all errors go through one path that serializes `{"error": {"kind", "message"}}` to stderr and exits `1`; nothing is printed to stdout on failure. Load rules: `--no-blocks` without `--rank` reads only `8 + header_len` bytes and uses `LoRAFile::new_from_header_buffer`; every other combination reads the whole file. Device is always `Device::Cpu`; delete the `cuda_if_available` probe. Progress (`indicatif`) is constructed on stderr and only when `--rank` runs and stderr is a TTY and `--quiet` is absent. `block-weights` becomes a thin wrapper that calls the same pipeline with `blocks: true, rank: false` and renders the block section only; keep `--output-format json|text` and print a one-line deprecation note to stderr.

- [ ] **Step 4: Run the whole suite and the formatters**

Run: `make test && make fmt && dprint fmt`
Expected: PASS. Then `cargo clippy --workspace --all-targets` and fix warnings introduced by these tasks.

- [ ] **Step 5: Manually verify the visuals at several widths**

Run: `for w in 40 60 80 120 200; do ./target/release/lora-inspector inspect boo.safetensors --width $w; done` and the same with `--rank` on a small file, plus `--ascii` and `--color never`.
Expected: columns aligned, nothing overflows, the 40-column view is still readable.

- [ ] **Step 6: Commit**

```bash
git add crates/lora-inspector
git commit -m "feat(cli): add the inspect command with block and rank sections"
```

---

### Task 10: Docs, version bump, cleanup

**Files:**
- Modify: `crates/lora-inspector/README.md`
- Modify: `README.md` (link line only, if the CLI description changes)
- Delete: `crates/lora-inspector/examples/bench_tmp.rs`, `crates/lora-inspector/examples/cudacheck_tmp.rs`

**Interfaces:**
- Consumes: the finished command.
- Produces: nothing downstream.

- [ ] **Step 1: Remove the throwaway benchmarks**

Run: `git status --short crates/lora-inspector/examples` and confirm both files are untracked before deleting them; they were measurement scratch, not deliverables.

- [ ] **Step 2: Document the command**

README, rewritten as the API reference people and agents both read:

1. **Usage** with two or three worked examples (`inspect FILE`, `inspect FILE --rank`, `inspect FILE --json | jq '.summary'`).
2. **Flag table** — every flag, one plain-language line, default stated.
3. **JSON** — copy the spec's field reference table verbatim, plus the three rules that
   matter most when reading it: flat arrays in display order, `null` = not computed vs
   `[]` = computed and empty, floats rounded to 6 significant digits, `schema_version`
   bumps only on rename/retype/removal, errors are one object on stderr with exit 1.
4. **What the numbers mean** — block weight is the Frobenius norm of the scaled delta at
   strength 1.0; rank health comes from the singular-value energy distribution; `--slow`
   enables the materialising fallback and is much slower.
5. **Local build tip** — `RUSTFLAGS="-C target-cpu=x86-64-v3"`.
6. State the breaking change to `block-weights --output-format json`.

- [ ] **Step 3: Prepare (do not run) the version bump**

`scripts/bump_release.py` commits and creates a git tag — a side effect outside this branch that the controller asks the human about. Do not run it. Instead record in the report: the exact command to run (`python3 scripts/bump_release.py minor`) and why (breaking `block-weights --output-format json` change detailed in the README).

- [ ] **Step 4: Final verification**

Run: `make test && make fmt && dprint fmt && git status --short`
Expected: PASS and a clean tree apart from the bump commit.

- [ ] **Step 5: Report the measured result**

Run the timing table from the spec again on `boo.safetensors`, `h3/H3_Motion_BoosterV2.safetensors` and `wan2.2/14b/…` with `inspect` (default and `--rank`) and record the numbers in the commit message or PR description so the spec's baseline is superseded by real end-to-end figures.
