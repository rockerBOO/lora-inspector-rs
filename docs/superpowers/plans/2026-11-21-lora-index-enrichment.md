# LoRA Index Enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report the structural network format derived from tensor keys, emit one JSON row per layer, and load both into the training-set DuckDB index.

**Architecture:** A new `inspector::structure` module classifies one base name's tensor keys into a `StructureFormat`. `build_report` gains `network.structure` (always) and `report.layers` (behind `--layers`), sharing one per-base-name pass with the existing rank computation. `index.py` gains an `enrich` subcommand that drives the CLI per file, caches the JSON, and writes three typed tables.

**Tech Stack:** Rust (candle, serde, clap, insta), Python 3 + DuckDB.

**Spec:** `docs/superpowers/specs/2026-11-21-lora-index-enrichment-design.md`

## Global Constraints

- `schema_version` stays `1`. Adding keys and flags is additive; only renames, retypes, and removals bump it.
- Existing `--json` output for `--rank`, `blocks`, `summary`, and `network.type` must not change. `network.type` keeps its metadata-derived meaning.
- The terminal report (`render.rs`) is not changed by this plan.
- Every float in JSON goes through `report::round_sig`. `null` means "not computed"; `[]` means "computed, empty".
- Rust formatting and lints: `cargo fmt --all` then `cargo clippy --workspace --all-targets`. Tests: `cargo test --workspace`.
- `/mnt/900/training/sets` is **not** a git repository. Task 6–8 commit nothing.
- Enrichment never deletes rows.

### Plan-time corrections to the spec

Two spec details are superseded here; the spec text was written before these were checked against the code.

1. **`layers[].scale` is dropped.** The spec lists both `scale` and `eff_scale` on a layer row. Both are produced by `LoRAFile::low_rank_effective_scale`, so they are always byte-identical — `build_rank` already emits it as `eff_scale`. The layer row keeps only `eff_scale`, and the index column is `lora_layers.eff_scale`. The invariant becomes "rows with `eff_scale IS NULL` == `skipped.layers`".
2. **The loader's dispatch changes too.** `LoRAFile::scale_weight` (`crates/inspector/src/file.rs:219-244`) today chooses the scaling formula from `metadata.network_type()` — the declared type. That is the exact failure the spec exists to expose: on a file whose metadata disagrees with its keys, the loader applies the wrong formula. Task 4 makes it dispatch on structure first, metadata second.

3. **Fixtures are synthetic, not the repo-root files the spec named.** The spec's verification section names `boo.safetensors`, `ahetableh_9_lokr.safetensors`, and `Klein 一edit-consistency.safetensors`. Those are git-ignored, so a fresh clone has none of them and a snapshot built on them would fail for the next person. Tasks 2–5 build the same shapes in `tests/support/mod.rs`, which is checked in. The factor-free checkpoint shape is covered by Task 5's `write_dense_only` rather than by shipping a 100 MB fixture.

4. **Assertions instead of `insta` snapshots, except one.** The spec asks for `insta` snapshots of `layers`. Exact-value assertions are stronger for the invariants, so those are plain `assert_eq!`. One snapshot is kept, in Task 5, to pin the layer row's field names against accidental renaming — the thing a snapshot is actually good at.

## Review Focus

Inputs and failure modes the spec implies that no single task's happy path exercises. Each is pinned by the test named here.

1. **A file whose declared type disagrees with its keys** — `scale_weight` must use the keys, not the metadata. Pinned in Task 4.
2. **`--layers` on a 2.1 GB full checkpoint** (614 base names, none with factors) — 614 all-null rows, no panic, no allocation proportional to payload. Pinned in Task 5.
3. **`alpha_actual` containing `null`** — observed as `[1.0, 16.0, 32.0, null]` on a LoKr file. `alpha_mismatch` must ignore null entries rather than flag. Pinned in Task 7.
4. **Declared rank 8 against actual `[4, 8, 16]`** — must not flag. Declared 8 against actual `[16]` — must flag. Pinned in Task 7.
5. **A base name carrying two structural marker sets** (`lora_up` and `hada_w1_a` in one layer) — precedence must be deterministic, not `HashMap`-order dependent. Pinned in Task 1.
6. **A file deleted between two runs** — `enrich` records nothing and deletes nothing. Pinned in Task 8.
7. **A truncated or unreadable safetensors** — one error row, the run continues. Pinned in Task 6.
8. **`--layers --no-blocks`** — `blocks: null` while `load: "full"`. Pinned in Task 5.

---

## Task 1: `StructureFormat` classifier

**Files:**
- Create: `crates/inspector/src/structure.rs`
- Modify: `crates/inspector/src/lib.rs` (add `pub mod structure;` after `pub mod statistic;`)
- Test: inline `#[cfg(test)] mod tests` at the bottom of `structure.rs` (this crate's existing pattern — see `metadata.rs`, `weight.rs`)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub enum StructureFormat { LoRa, LoHa, LoKr, Oft, Glora }`
  - `impl StructureFormat { pub fn as_str(self) -> &'static str; pub fn precedence(self) -> u8; }` — `as_str` returns `"LoRA"`, `"LoHa"`, `"LoKr"`, `"OFT"`, `"GLoRA"`; `precedence` returns 0..=4 in that order.
  - `pub fn classify(keys: &[String]) -> Option<StructureFormat>` — `keys` are the full tensor keys belonging to one base name, e.g. `["x.lora_up.weight", "x.lora_down.weight", "x.alpha"]`.

- [ ] **Step 1: Write the failing tests**

In `crates/inspector/src/structure.rs`:

```rust
use crate::structure::StructureFormat;
use StructureFormat::{Glora, LoHa, LoKr, LoRa, Oft};

// Marker sets are matched on the token before any `.weight` suffix, so the
// canonical key shapes from weight.rs are the fixtures here.
fn keys(names: &[&str]) -> Vec<String> { names.iter().map(|s| s.to_string()).collect() }

#[test]
fn classify_kohya_lora() {
    assert_eq!(
        classify(&keys(&["x.lora_up.weight", "x.lora_down.weight", "x.alpha"])),
        Some(LoRa)
    );
}

#[test]
fn classify_peft_lora() {
    assert_eq!(
        classify(&keys(&["x.lora_A.weight", "x.lora_B.weight"])),
        Some(LoRa)
    );
}

#[test]
fn classify_loha() {
    assert_eq!(
        classify(&keys(&[
            "x.hada_w1_a", "x.hada_w1_b", "x.hada_w2_a", "x.hada_w2_b", "x.alpha"
        ])),
        Some(LoHa)
    );
}

#[test]
fn classify_lokr_both_forms() {
    // Fused form.
    assert_eq!(classify(&keys(&["x.lokr_w1", "x.lokr_w2", "x.alpha"])), Some(LoKr));
    // Decomposed form used by decompose_both.
    assert_eq!(
        classify(&keys(&[
            "x.lokr_w1_a", "x.lokr_w1_b", "x.lokr_w2_a", "x.lokr_w2_b", "x.alpha"
        ])),
        Some(LoKr)
    );
}

#[test]
fn classify_oft_forms() {
    assert_eq!(classify(&keys(&["x.oft_diag"])), Some(Oft));
    assert_eq!(classify(&keys(&["x.oft_blocks"])), Some(Oft));
}

#[test]
fn classify_glora() {
    assert_eq!(
        classify(&keys(&["x.a1.weight", "x.a2.weight", "x.b1.weight", "x.b2.weight"])),
        Some(Glora)
    );
}

#[test]
fn classify_returns_none_without_markers() {
    assert_eq!(classify(&keys(&["x.weight", "x.bias"])), None);
    assert_eq!(classify(&[]), None);
}

#[test]
fn classify_precedence_is_deterministic_for_mixed_markers() {
    // A partially merged file can carry both marker sets. The higher-precedence
    // format must win regardless of key order in the input.
    let forward = keys(&["x.hada_w1_a", "x.hada_w1_b", "x.lora_up.weight", "x.lora_down.weight"]);
    let reversed: Vec<String> = forward.iter().rev().cloned().collect();
    assert_eq!(classify(&forward), Some(LoHa));
    assert_eq!(classify(&reversed), Some(LoHa));
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p inspector structure:: 2>&1 | tail -20`
Expected: FAIL — `unresolved import crate::structure` / `cannot find function classify`.

- [ ] **Step 3: Implement `structure.rs`**

Add `pub mod structure;` to `crates/inspector/src/lib.rs` next to `pub mod statistic;`.

In `structure.rs`, above the tests:

```rust
//! Structural format detection from tensor key names.
//!
//! This is what a loader will actually act on: a key set carrying
//! `hada_w1_a`/`hada_w1_b` loads as LoHa whatever `ss_network_module` claims.
//! Kept separate from [`crate::metadata::NetworkType`], which is derived from
//! metadata and is therefore a hint, not evidence.

/// A network format identified purely from tensor key names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StructureFormat { LoRa, LoHa, LoKr, Oft, Glora }

impl StructureFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            StructureFormat::LoRa => "LoRA",
            StructureFormat::LoHa => "LoHa",
            StructureFormat::LoKr => "LoKr",
            StructureFormat::Oft => "OFT",
            StructureFormat::Glora => "GLoRA",
        }
    }

    /// Tie-break and mixed-marker order. Fixed so output never depends on
    /// `HashMap` iteration order.
    pub fn precedence(self) -> u8 {
        match self {
            StructureFormat::LoRa => 0,
            StructureFormat::LoHa => 1,
            StructureFormat::LoKr => 2,
            StructureFormat::Oft => 3,
            StructureFormat::Glora => 4,
        }
    }
}
```

Then `classify`. Implementation notes that the tests do not determine:

- Split each key on `'.'`, and test each token for equality against the marker set (`"hada_w1_a"`, `"hada_w1_b"`, `"hada_w2_a"`, `"hada_w2_b"`, `"lokr_w1"`, `"lokr_w1_a"`, `"lokr_w1_b"`, `"lokr_w2"`, `"lokr_w2_a"`, `"lokr_w2_b"`, `"oft_diag"`, `"oft_blocks"`, `"lora_up"`, `"lora_down"`, `"lora_A"`, `"lora_B"`, `"a1"`, `"a2"`, `"b1"`, `"b2"`). Token equality rather than substring `contains`, because `contains("lokr_w1")` also matches `lokr_w1_a` and would confuse the two forms.
- Collect the matched formats into a `Vec`, take the one with the highest `precedence()`, and return it. Doing it that way makes the mixed-marker test pass for both key orders without relying on scan order.
- `"lora_A"`/`"lora_B"` are case-sensitive; PEFT writes them capitalised and a lowercase match would collide with nothing but should not be loosened.
- An empty `keys` slice returns `None`.

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p inspector structure:: 2>&1 | tail -20`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/src/structure.rs crates/inspector/src/lib.rs
git commit -m "feat(inspector): classify network format from tensor keys"
```

---

## Task 2: `network.structure` in the report

**Files:**
- Modify: `crates/lora-inspector/src/report.rs` (`NetworkInfo`, new `StructureInfo`, new `build_structure`, `build_network_info`)
- Modify: `crates/lora-inspector/tests/support/mod.rs` (new fixture)
- Test: `crates/lora-inspector/tests/cli.rs`

**Interfaces:**
- Consumes: `StructureFormat`, `classify` from Task 1.
- Produces:
  - `pub struct StructureInfo { pub primary: String, pub by_format: BTreeMap<String, usize>, pub formats: Vec<String>, pub has_conv: bool, pub has_dora: bool }`, `Serialize`.
  - `pub fn build_structure(file: &LoRAFile) -> StructureInfo`.
  - `NetworkInfo` gains `pub structure: StructureInfo`, serialized as `structure`.
  - `support::Fixture::write_mixed_structure() -> Fixture` — no `__metadata__`; two plain LoRA layers plus one LoHa layer.

- [ ] **Step 1: Add the fixture**

In `tests/support/mod.rs`, next to `Fixture::write`:

```rust
    /// A file whose layers use two different structural formats, with no
    /// `__metadata__` at all: two plain LoRA layers and one LoHa layer.
    pub fn write_mixed_structure() -> Fixture {
        let path = temp_path();
        fs::write(&path, mixed_structure_fixture_bytes()).expect("write mixed fixture");
        Fixture { path }
    }
```

and the builder:

```rust
fn mixed_structure_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_lora(&mut tensors, "down_blocks.0.attn.proj_in", 4, 4.0, 1.0);
    push_lora(&mut tensors, "down_blocks.0.attn.proj_out", 4, 4.0, 1.0);
    push_hada(&mut tensors, "down_blocks.1.attn.proj_in");
    to_safetensors(tensors)
}
```

- [ ] **Step 2: Write the failing tests**

In `tests/cli.rs`:

```rust
#[test]
fn structure_reports_key_derived_formats() {
    let fixture = support::Fixture::write_mixed_structure();
    let doc = run_json(&fixture, &[]);

    let structure = &doc["network"]["structure"];
    assert_eq!(structure["primary"], "LoRA");
    assert_eq!(structure["by_format"]["LoRA"], 2);
    assert_eq!(structure["by_format"]["LoHa"], 1);
    assert_eq!(structure["formats"], serde_json::json!(["LoHa", "LoRA"]));
    assert_eq!(structure["has_conv"], false);
    assert_eq!(structure["has_dora"], false);
}

#[test]
fn structure_is_present_in_header_only_mode() {
    let fixture = support::Fixture::write_mixed_structure();
    let doc = run_json(&fixture, &["--no-blocks"]);

    assert_eq!(doc["load"], "header_only");
    assert_eq!(doc["network"]["structure"]["primary"], "LoRA");
}

#[test]
fn structure_has_conv_for_conv_factors() {
    let fixture = support::Fixture::write_conv();
    let doc = run_json(&fixture, &[]);
    assert_eq!(doc["network"]["structure"]["has_conv"], true);
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p lora-inspector --test cli structure_ 2>&1 | tail -30`
Expected: FAIL — `no field structure` / `null` where an object is expected.

- [ ] **Step 4: Implement `StructureInfo` and `build_structure`**

In `report.rs`, add `use inspector::structure::{classify, StructureFormat};` and the struct next to `NetworkInfo`. Add the field to `NetworkInfo`:

```rust
pub struct NetworkInfo {
    pub module: String,
    #[serde(rename = "type")]
    pub network_type: Option<NetworkType>,
    pub family: String,
    pub rank: Vec<usize>,
    pub alpha: Vec<f64>,
    pub layers: LayerCounts,
    pub structure: StructureInfo,
}
```

`build_structure` implementation notes:

- Iterate `file.base_names()`; for each, gather the full keys that belong to it. `LoRAFile::tensor_info()` (or `file.shapes()`) provides `(name, dtype, shape)`; a key belongs to a base name when `get_base_name`-style stripping of the known suffixes yields it. If that helper is not public, collect keys by `name.starts_with(base_name)` and pass those — precise stripping is not needed because `classify` matches on whole tokens.
- `by_format` counts base names per `StructureFormat::as_str()`.
- `primary` is the format with the highest count; ties break by lowest `precedence()`. With no classified layers, `primary` is `"Unknown"`, `by_format` is empty, `formats` is `[]`.
- `formats` is the sorted list of keys in `by_format` (a `BTreeMap` gives this for free).
- `has_conv` is true when any key ending in `lora_down` or `lora_up` has more than 2 dimensions (a real convolution kernel; 1×1 convs present as 4-D with singleton trailing dims, which is what the design calls LoCon-ish and is still `true`).
- `has_dora` is true when any key's final token is `dora_scale`.

Call it from `build_network_info` and set the field.

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p lora-inspector --test cli 2>&1 | tail -30`
Expected: PASS, including the existing tests — `network.type` and all other keys unchanged.

- [ ] **Step 6: Commit**

```bash
git add crates/lora-inspector/src/report.rs crates/lora-inspector/tests/support/mod.rs crates/lora-inspector/tests/cli.rs
git commit -m "feat(cli): report key-derived network structure"
```

---

## Task 3: The divergence fixture — declared LoRA, actual LoHa

**Files:**
- Modify: `crates/lora-inspector/tests/support/mod.rs` (new metadata-bearing fixture)
- Test: `crates/lora-inspector/tests/cli.rs`

**Interfaces:**
- Consumes: `to_safetensors_with_metadata`, `push_hada` from support.
- Produces: `support::Fixture::write_declared_lora_actual_loha() -> Fixture`.

This is the case the whole design exists for: metadata says `networks.lora`, keys say LoHa. It is unobservable today.

- [ ] **Step 1: Add the fixture**

```rust
    /// LoHa keys under metadata that declares `networks.lora`. The declared type
    /// and the structural type disagree; nothing in the pre-existing output
    /// reveals it.
    pub fn write_declared_lora_actual_loha() -> Fixture {
        let path = temp_path();
        fs::write(&path, declared_mismatch_fixture_bytes()).expect("write mismatch fixture");
        Fixture { path }
    }
```

```rust
fn declared_mismatch_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_hada(&mut tensors, "down_blocks.0.attn.proj_in");
    push_hada(&mut tensors, "down_blocks.0.attn.proj_out");
    to_safetensors_with_metadata(
        tensors,
        serde_json::json!({
            "ss_network_module": "networks.lora",
            "ss_network_dim": "4",
            "ss_network_alpha": "4.0",
        }),
    )
}
```

- [ ] **Step 2: Write the failing test**

```rust
#[test]
fn declared_type_and_structural_type_can_disagree() {
    let fixture = support::Fixture::write_declared_lora_actual_loha();
    let doc = run_json(&fixture, &[]);

    // The hint.
    assert_eq!(doc["network"]["type"], "LoRA");
    // The truth.
    assert_eq!(doc["network"]["structure"]["primary"], "LoHa");
    assert_eq!(doc["network"]["structure"]["by_format"]["LoHa"], 2);
}
```

- [ ] **Step 3: Run to verify it passes for the right reason**

Run: `cargo test -p lora-inspector --test cli declared_type 2>&1 | tail -30`

If `network.type` is `"LoRA"` and `structure.primary` is `"LoHa"`, this task introduced no new production code — it only proves Task 1 and Task 2 expose the divergence. If `network.type` comes back `null` or `"LoHa"`, the fixture metadata is wrong; fix the fixture, not the assertion.

- [ ] **Step 4: Commit**

```bash
git add crates/lora-inspector/tests/support/mod.rs crates/lora-inspector/tests/cli.rs
git commit -m "test(cli): pin declared-vs-structural type divergence"
```

---

## Task 4: Loader dispatches on structure, not on declared type

**Files:**
- Modify: `crates/inspector/src/file.rs:219-244` (`LoRAFile::scale_weight`)
- Test: `crates/lora-inspector/tests/cli.rs` — deliberately end-to-end, not a unit test

**Interfaces:**
- Consumes: `structure::{classify, StructureFormat}`, `Fixture::write_declared_lora_actual_loha` from Task 3.
- Produces: no new public API. `LoRAFile::scale_weight(&self, base_name: &str) -> Result<Tensor>` keeps its signature; only the formula selection changes.

Today the match is on `metadata.network_type()`. On the mismatched fixture it calls `scale_lora_weight` on LoHa keys, which fails with `TensorNotFound`; `effective_scale` swallows that into `Ok(None)`, so the layer is silently counted as skipped. After this task the same call finds the LoHa markers and returns a real scale.

This is observable from the CLI through `--slow`, which is the path reaching `scale_weight` when `low_rank_effective_scale` declines a layer. Asserting it end-to-end reuses the fixtures Task 3 already built and avoids adding a second hand-rolled safetensors writer to `file.rs` — the existing unit tests there read fixtures off disk via `load_file`, so no writer exists to reuse.

- [ ] **Step 1: Write the failing test**

In `crates/lora-inspector/tests/cli.rs`:

```rust
#[test]
fn slow_path_uses_keys_when_metadata_disagrees() {
    // LoHa keys under metadata declaring networks.lora. Without structure-first
    // dispatch the LoRA formula cannot find lora_up/lora_down, the layer is
    // swallowed into `skipped`, and `blocks` comes back empty.
    let fixture = support::Fixture::write_declared_lora_actual_loha();
    let doc = run_json(&fixture, &["--slow"]);

    assert_eq!(doc["skipped"]["layers"], 0);
    assert_eq!(doc["blocks"].as_array().expect("blocks array").len(), 1);
}
```

Both fixture layers share the `down_blocks.0` block group, so they aggregate to one `BlockStat` row.

- [ ] **Step 2: Run to verify failure**

Run: `cargo test -p lora-inspector --test cli slow_path_uses_keys 2>&1 | tail -20`
Expected: FAIL — `skipped.layers` is 2 and `blocks` is empty.

- [ ] **Step 3: Implement structure-first dispatch**

Replace the `match` on `metadata.network_type()` with:

1. Classify the base name's keys with `classify`, and match `StructureFormat` to the existing `scale_*` calls: `LoRa → scale_lora_weight`, `LoHa → scale_hada_weight`, `LoKr → scale_lokr_weight`, `Oft → scale_diag_oft_weight` if any key ends in `oft_diag` else `scale_boft_weight`, `Glora → scale_glora_weights`.
2. If `classify` returns `None`, fall back to the existing metadata match **unchanged**, including the `Some(_) => Err(InspectorError::UnsupportedNetworkType)` arm and the `None => scale_lora_weight` default.

Do not delete the metadata path: it is the only signal for `GLoRA` variants the classifier does not cover and it is the current default for unmarked keys.

- [ ] **Step 4: Run the full test suite**

Run: `cargo test --workspace 2>&1 | tail -30`
Expected: PASS. Existing tests use metadata that agrees with their keys, so their results are unchanged. Any changed `.snap` file means this was not a dispatch-only change.

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/src/file.rs
git commit -m "fix(inspector): select scaling formula from tensor keys, not declared type"
```

---

## Task 5: `--layers` — one JSON row per layer

**Files:**
- Modify: `crates/lora-inspector/src/report.rs` (new `Layer`, `ReportOptions.layers`, `Report.layers`, refactor `build_rank` into `rank_rows`, new `build_layers`)
- Modify: `crates/lora-inspector/src/main.rs` (`InspectArgs.layers`, the `header_only` rule, `ReportOptions`)
- Test: `crates/lora-inspector/tests/cli.rs`

**Interfaces:**
- Consumes: `structure::classify`; `build_rank`'s existing per-base-name parallel pass.
- Produces:
  - `pub struct Layer { pub base_name: String, pub group: BlockGroup, pub family: BlockFamily, pub label: String, pub index: Option<usize>, pub format: Option<String>, pub nominal_rank: Option<usize>, pub effective_rank: Option<f64>, pub balance: Option<f64>, pub top1_energy: Option<f64>, pub dominance: Option<f64>, pub health: Option<RankHealth>, pub eff_scale: Option<f64>, pub outlier: bool }`, `Serialize`.
  - `ReportOptions.layers: bool`, `Report.layers: Option<Vec<Layer>>`.
  - CLI flag `--layers`.
  - `fn rank_rows(file: &LoRAFile, jobs: usize, slow: bool) -> Result<Vec<Row>>` where `Row { base_name, group, family, index, metrics: Option<RankMetrics>, eff_scale: Option<f64>, format: Option<StructureFormat> }` — private to `report.rs`.
  - `build_rank` keeps its signature and output: rows filtered to `metrics.is_some()`, mapped to `RankLayer`.
  - `build_layers(file, jobs, slow) -> Result<Vec<Layer>>`: every row, mapped to `Layer`.

- [ ] **Step 1: Refactor without changing behaviour, and confirm**

Split `build_rank` into `rank_rows` (all base names, parallel, `metrics: Option<...>`) and a thin `build_rank` that filters and maps. `rank_rows` must keep the existing `raw.sort_by(base_name)`, the median-scale outlier threshold, and the `by_health`/`weak_or_worse` accounting — those read only rows with metrics.

Run: `cargo test -p lora-inspector 2>&1 | tail -20`
Expected: PASS with no snapshot changes. If any `.snap` file changes, the refactor is not behaviour-preserving — fix it before continuing.

- [ ] **Step 2: Write the failing tests**

```rust
#[test]
fn layers_covers_every_base_name() {
    // The shared fixture has 4 LoRA layers plus 1 LoHa layer the cheap Gram path
    // declines, so `skipped.layers` is 1 and `layers` must still have 5 rows.
    let fixture = support::Fixture::write();
    let doc = run_json(&fixture, &["--layers", "--rank"]);

    assert_eq!(doc["network"]["layers"]["total"], 5);
    assert_eq!(doc["layers"].as_array().expect("layers array").len(), 5);
}

#[test]
fn layers_null_scale_matches_skipped_count() {
    let fixture = support::Fixture::write();
    let doc = run_json(&fixture, &["--layers", "--rank"]);

    let null_scale = doc["layers"]
        .as_array()
        .expect("layers array")
        .iter()
        .filter(|row| row["eff_scale"].is_null())
        .count();
    assert_eq!(null_scale as u64, doc["skipped"]["layers"].as_u64().unwrap());
}

#[test]
fn layers_health_present_matches_rank_length() {
    let fixture = support::Fixture::write();
    let doc = run_json(&fixture, &["--layers", "--rank"]);

    let with_health = doc["layers"]
        .as_array()
        .expect("layers array")
        .iter()
        .filter(|row| !row["health"].is_null())
        .count();
    assert_eq!(with_health, doc["rank"].as_array().expect("rank array").len());
}

#[test]
fn layers_records_the_structural_format_per_layer() {
    let fixture = support::Fixture::write_mixed_structure();
    let doc = run_json(&fixture, &["--layers"]);

    let rows = doc["layers"].as_array().expect("layers array");
    let loha = rows.iter().filter(|row| row["format"] == "LoHa").count();
    let lora = rows.iter().filter(|row| row["format"] == "LoRA").count();
    assert_eq!((lora, loha), (2, 1));
}

#[test]
fn layers_is_null_unless_requested_and_does_not_change_rank() {
    let fixture = support::Fixture::write();

    let without = run_json(&fixture, &["--rank"]);
    let with = run_json(&fixture, &["--rank", "--layers"]);

    assert!(without["layers"].is_null());
    assert_eq!(without["rank"], with["rank"]);
    assert_eq!(without["summary"], with["summary"]);
    assert_eq!(without["blocks"], with["blocks"]);
    assert_eq!(doc_schema_version(&with), 1);
}
```

Add `fn doc_schema_version(doc: &serde_json::Value) -> u64` returning `doc["schema_version"].as_u64().unwrap()` if absent from `cli.rs`.

```rust
#[test]
fn layers_with_no_blocks_keeps_full_load() {
    let fixture = support::Fixture::write();
    let doc = run_json(&fixture, &["--layers", "--no-blocks"]);

    assert_eq!(doc["load"], "full");
    assert!(doc["blocks"].is_null());
    assert!(doc["layers"].is_array());
}

#[test]
fn layers_on_a_factorless_file_is_all_nulls_and_does_not_panic() {
    // A dense-only file: no low-rank factors anywhere, so every layer row is
    // null-valued. This is the shape a 2.1 GB full checkpoint presents.
    let fixture = support::Fixture::write_dense_only();
    let doc = run_json(&fixture, &["--layers"]);

    let rows = doc["layers"].as_array().expect("layers array");
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|row| row["nominal_rank"].is_null()));
    assert!(rows.iter().all(|row| row["eff_scale"].is_null()));
}
```

Add the `write_dense_only` fixture in support: two tensors named `0.weight` / `2.weight` with 3-D shapes and no markers, no metadata — the `*-proj` shape from the real tree:

```rust
    /// A dense, factor-free file: the shape the `*-proj.safetensors` artifacts
    /// in the real training tree present (two unmarked tensors, no metadata).
    pub fn write_dense_only() -> Fixture {
        let path = temp_path();
        fs::write(&path, dense_only_fixture_bytes()).expect("write dense fixture");
        Fixture { path }
    }
```

```rust
fn dense_only_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    tensors.push(tensor("0.weight", &[2, 3, 4], 1.0));
    tensors.push(tensor("2.weight", &[2, 3, 4], 1.0));
    to_safetensors(tensors)
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cargo test -p lora-inspector --test cli layers 2>&1 | tail -30`
Expected: FAIL — unknown argument `--layers`.

- [ ] **Step 4: Implement**

In `main.rs`, add to `InspectArgs`:

```rust
    /// Emit one row per layer, including layers with no resolvable factors
    #[clap(long)]
    layers: bool,
```

Change the header-only rule to account for the new flag needing the payload:

```rust
let header_only = args.no_blocks && !args.rank && !args.layers;
```

and set `layers: args.layers` in `ReportOptions`.

In `report.rs`, `build_report` computes `let layers = if opts.layers { Some(build_layers(file, opts.jobs, opts.slow)?) } else { None };` and sets it on `Report`. `build_layers` maps every `rank_rows` entry: `nominal_rank`/`effective_rank`/`balance`/`top1_energy`/`dominance`/`health` come from `metrics` when present and are `None` otherwise; `eff_scale` and `outlier` come from the row (outlier threshold is the layer-median rule already in `build_rank` — factor it so both callers use one threshold); `format` is `row.format.map(|f| f.as_str().to_string())`.

`Layer::family` is `parse_block_key(base_name).family()`, matching how `BlockStat` gets it and how `build_report` counts families.

- [ ] **Step 5: Run to verify pass**

Run: `cargo test -p lora-inspector 2>&1 | tail -30`
Expected: PASS, 7 new tests plus every pre-existing one.

- [ ] **Step 6: Pin the layer row's field names with one snapshot**

Add a test following the `insta` style already used in `crates/lora-inspector/tests/render_snapshots.rs`:

```rust
#[test]
fn layers_row_shape() {
    let fixture = support::Fixture::write_mixed_structure();
    let doc = run_json(&fixture, &["--layers", "--rank"]);
    insta::assert_json_snapshot!("layers_row_shape", &doc["layers"]);
}
```

Run `INSTA_UPDATE=always cargo test -p lora-inspector layers_row_shape`, then read the generated `.snap` and confirm it contains exactly the keys in the `Layer` interface block — no `scale` field, and `health` as a string or null. Commit the `.snap` with the code.

- [ ] **Step 7: Format, lint, commit**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
git add crates/lora-inspector/src/report.rs crates/lora-inspector/src/main.rs crates/lora-inspector/tests/cli.rs crates/lora-inspector/tests/support/mod.rs
git commit -m "feat(cli): add --layers for per-layer rows in --json"
```

---

## Task 6: `index.py enrich` — schema, driver, cache

**Files:**
- Modify: `/mnt/900/training/sets/index.py` (new tables in `get_db`, `cmd_enrich`, `enrich_file`, `fingerprint`, cache read/write, argparse wiring, `cmd_stats` reuse)
- No test file exists in that directory; verification is by command, in Step 6.

**Interfaces:**
- Consumes: the CLI's `inspect --json --layers --rank --quiet` output from Tasks 2 and 5.
- Produces (module-level functions in `index.py`):
  - `INSPECT_DIR: Path` — `SETS_DEFAULT / ".inspect"`
  - `find_inspector() -> str` — binary lookup
  - `inspector_fingerprint(binary: Path) -> str` — `f"{version}|{size}|{mtime_ns}|{SCHEMA_VERSION}"`
  - `enrich_file(binary: str, path: Path) -> tuple[dict, list[str], str | None]` — `(report, warnings, error)`
  - `insert_inspect(con, path: Path, report: dict, fingerprint: str, declared: tuple[int | None, float | None, str | None])` — writes `lora_inspect`, `lora_blocks`, `lora_layers`. `declared` is `(rank, alpha, module)`; Task 6 passes `(None, None, None)` and Task 7 supplies the real parse.
  - `cmd_enrich(args)`

- [ ] **Step 1: Add the three tables to `get_db`**

Append the `CREATE TABLE IF NOT EXISTS` statements for `lora_inspect`, `lora_blocks`, `lora_layers` exactly as given in the spec's Schema section, with one change carried over from the plan-time corrections: `lora_layers` has `eff_scale DOUBLE` and **no** `scale` column.

- [ ] **Step 2: Implement `find_inspector`**

Order: `--inspector` argument, then `os.environ.get("LORA_INSPECTOR")`, then `<index.py's parent>/../lora-inspector-rs/target/release/lora-inspector` resolved, then `shutil.which("lora-inspector")`. Raise `SystemExit` with a clear message naming all four locations when none is executable.

- [ ] **Step 3: Implement `enrich_file`**

- `subprocess.run([binary, "inspect", str(path), "--json", "--layers", "--rank", "--quiet", "--jobs", "1"], capture_output=True, text=True)`.
- On non-zero return, return `({}, stderr_lines, "exit {code}: {first stderr line}")`.
- On empty stdout or a `json.JSONDecodeError`, return `({}, stderr_lines, "unparseable json")`.
- Otherwise `warnings` is the non-empty stderr lines (indicatif is suppressed by `--quiet`, so what remains is parse warnings — observed format: `Error("Parse sequence failed", line: 1, column: 128)`).
- Return `(json.loads(stdout), warnings, None)`.

- [ ] **Step 4: Implement `insert_inspect`**

Write `lora_inspect` from: `file.precision`, `network.structure.primary/formats/by_format/has_conv/has_dora`, `network.type` → `type_declared`, `network.module` → `module_reported`, `network.family` → `family_reported`, `network.layers.*`, `skipped.layers`, `network.rank` → `rank_actual`, `network.alpha` → `alpha_actual`, `summary.block_weight.*`, `summary.rank.*`, `load` → `load_kind`, `schema_version`.

Write `lora_blocks` one row per `blocks[]` entry, and `lora_layers` one row per `layers[]` entry, both with `DELETE FROM <table> WHERE file_path = ?` first so a re-enrich of one file is idempotent without touching other files.

`raw` stores `json.dumps({k: v for k, v in report.items() if k not in ("metadata", "layers", "rank")})`.

`declared` is passed in and written straight to `declared_rank`/`declared_alpha`/`declared_module`. In this task every caller passes `(None, None, None)`; Task 7 supplies the parse.

- [ ] **Step 5: Implement the driver**

- Enumerate `sorted(path for path in Path(args.path).rglob("*.safetensors"))`.
- Load `existing = {file_path: (file_mtime, fingerprint, schema_version)}` from `lora_inspect` left-joined to `loras` on `file_path`.
- Skip a file when its `mtime` is unchanged **and** its stored `fingerprint` equals the current one. `--force` skips this check.
- Concurrency: `concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs)`; sort by size descending before submitting, and run any file over 500 MB on its own (submit, wait, continue) so peak RSS stays bounded.
- Print one line per file: `[i/n] name -> primary/kind` or `[i/n] name ERROR ...`.
- Call `enrich_file` directly in this step; no cache yet.

- [ ] **Step 6: Add the cache and `--from-cache`**

- Cache path: `INSPECT_DIR / path.relative_to(SETS_DEFAULT)` with `.json.gz` appended.
- On a miss, inspect and then write gzip JSON of the full report; pass `--metadata` so the cache alone can rebuild the DB.
- On a hit whose stored fingerprint matches the current fingerprint, read the cache instead of inspecting.
- On a hit whose fingerprint differs, inspect again and overwrite — never replay stale metrics as current.
- `--from-cache` replays any cache entry regardless of fingerprint, and skips inspection entirely when no entry exists.

- [ ] **Step 7: Verify against a small real sample**

```bash
cd /mnt/900/training/sets
uv run --with duckdb index.py enrich --path ./women-flux2-2026-09-03-125944-5bf5e0cb --jobs 4
uv run --with duckdb index.py query "SELECT file_path, type_declared, type_actual, kind, layers_total, skipped_layers, rank_layers FROM lora_inspect"
```

Expected: one row per `.safetensors` in that directory, `type_actual` non-null, `layers_total` equal to the count in `lora_layers` for that file, `skipped_layers` equal to the count of `lora_layers` rows with a null `eff_scale`.

Then confirm the error path:

```bash
truncate -s 1024 /tmp/broken.safetensors
mkdir -p /tmp/enrich-broken && cp /tmp/broken.safetensors /tmp/enrich-broken/broken.safetensors
uv run --with duckdb index.py enrich --path /tmp/enrich-broken
uv run --with duckdb index.py query "SELECT file_path, enrich_error FROM lora_inspect WHERE enrich_error IS NOT NULL"
```

Expected: one row with a non-null `enrich_error`, exit code 0.

- [ ] **Step 8: Commit — none.** `/mnt/900/training/sets` is not a git repository.

---

## Task 7: Divergence verdicts and stats

**Files:**
- Modify: `/mnt/900/training/sets/index.py` (`declared_*` parsing, the four verdicts, `cmd_stats` sections)

**Interfaces:**
- Consumes: `insert_inspect` from Task 6.
- Produces: `declared_scalars(metadata: dict) -> tuple[int | None, float | None, str | None]`, `algo_of_declared(type_str) -> str | None`, `algo_of_actual(primary) -> str | None`, and the verdict columns.

- [ ] **Step 1: Implement `declared_scalars`**

Parse from the raw metadata dict, never from `loras.network_alpha` (that column is `INTEGER` and `to_int("4.0")` yields `NULL`, so alpha is null for nearly every existing row):

- `declared_rank`: `int(float(metadata.get("ss_network_dim")))`, tolerating `"8"`, `"8.0"`, and `None`/`"None"`.
- `declared_alpha`: `float(metadata.get("ss_network_alpha"))`, same tolerance.
- `declared_module`: `metadata.get("ss_network_module")`.
- Any parse failure yields `None`, never an exception.

- [ ] **Step 2: Implement the algorithm normalisation**

Both sides reduce to one of `"lora"`, `"loha"`, `"lokr"`, `"oft"`, `"glora"`, `"glokr"`, else `None`:

- Declared, from `network.type`: `LoRA`/`LoRAC3Lier`/`LoCon`/`LoRAFA`/`DyLoRA` → `"lora"`; `LoHa` → `"loha"`; `LoKr` → `"lokr"`; `OFT`/`DiagOFT`/`BOFT` → `"oft"`; `GLoRA` → `"glora"`; `GLoKr` → `"glokr"`.
- Actual, from `network.structure.primary`: `LoRA` → `"lora"`, `LoHa` → `"loha"`, `LoKr` → `"lokr"`, `OFT` → `"oft"`, `GLoRA` → `"glora"`; `Unknown` → `None`.

- [ ] **Step 3: Compute the verdicts in `insert_inspect`**

- `type_mismatch`: both algos non-null **and** different. Else null.
- `rank_mismatch`: `rank_actual` is non-empty **and** `declared_rank` is not null **and** `declared_rank not in rank_actual`. Else null.
- `alpha_mismatch`: `alpha_actual` has at least one non-null entry **and** `declared_alpha` is not null **and** `declared_alpha` is not in that non-null set. Else null.
- `is_lora`: `len(rank_actual or []) > 0`.
- `kind`: `"lora"` if `is_lora`; else `"checkpoint"` if `layers_total > 100` and the declared architecture has no `/lora` suffix; else `"artifact"` if `layers_total <= 100`; else `"unknown"`.

- [ ] **Step 4: Extend `cmd_stats`**

Add sections for `type_actual`, `kind`, `is_lora`, `precision`, and the health distribution (`SUM(health_good)`…`SUM(health_collapsed)`), in the existing `(title, sql)` list style.

- [ ] **Step 5: Verify the verdicts against real files**

```bash
cd /mnt/900/training/sets
uv run --with duckdb index.py query "SELECT run_name, declared_rank, rank_actual, rank_mismatch FROM lora_inspect ORDER BY rank_mismatch DESC NULLS LAST LIMIT 20"
uv run --with duckdb index.py query "SELECT file_path, kind, layers_total, type_declared, type_actual FROM lora_inspect WHERE kind <> 'lora' LIMIT 20"
```

Expected, based on measurements already taken:
- `women-flux-2025-03-24-174551-*` shows `declared_rank 8` against `rank_actual [16]` with `rank_mismatch = true`.
- No Flux file with `declared_rank 16` and `rank_actual [4, 8, 16]` is flagged.
- Every `*-proj.safetensors` has `kind = 'artifact'`.
- `pov_sd15-2025-10-04-012201-*` has `kind = 'checkpoint'`.

If a Flux file *is* flagged for rank, the membership guard is wrong — fix it, do not relax the test.

- [ ] **Step 6: Verify the alpha guard on a LoKr-style file**

Find a row whose `alpha_actual` contains a null and assert `alpha_mismatch` is not `true` for it unless the declared value is genuinely absent:

```bash
uv run --with duckdb index.py query "SELECT file_path, declared_alpha, alpha_actual, alpha_mismatch FROM lora_inspect WHERE list_contains(alpha_actual, NULL::DOUBLE) LIMIT 10"
```

- [ ] **Step 7: Commit — none.**

---

## Task 8: Views, and the no-prune guarantee

**Files:**
- Modify: `/mnt/900/training/sets/index.py` (`v_runs`, `v_issues`, created in `get_db`)

- [ ] **Step 1: Create the views**

`v_runs`: one row per `run_name` from `lora_inspect` joined to `loras`, with `files` (count), `total_size`, `kinds` (`string_agg(DISTINCT kind)`), `first_date`/`last_date` from `training_date`, and `final_file` chosen by highest step number parsed from `file_name`, falling back to `MAX(file_mtime)`.

`v_issues`: rows where `type_mismatch` or `rank_mismatch` or `alpha_mismatch` is true, or `kind <> 'lora'`, or `block_outliers > 10`, or `len(parse_warnings) > 0`, or `rank_weak_or_worse > rank_layers / 2`, or `enrich_error IS NOT NULL`.

Both as `CREATE OR REPLACE VIEW`, so a re-run of `enrich` needs no migration.

- [ ] **Step 2: Verify the views**

```bash
cd /mnt/900/training/sets
uv run --with duckdb index.py query "SELECT * FROM v_issues LIMIT 20"
uv run --with duckdb index.py query "SELECT COUNT(*) FROM v_runs"
```

Expected: `v_issues` returns the artifacts and the checkpoint, plus any rank divergences. `v_runs` count is plausible against `SELECT COUNT(DISTINCT run_name) FROM loras`.

- [ ] **Step 3: Prove enrichment never deletes**

```bash
cd /mnt/900/training/sets
mkdir -p /tmp/enrich-noprune && cp ./women-flux2-2026-09-03-125944-5bf5e0cb/*.safetensors /tmp/enrich-noprune/
uv run --with duckdb index.py enrich --path /tmp/enrich-noprune
uv run --with duckdb index.py query "SELECT COUNT(*) FROM lora_inspect WHERE file_path LIKE '/tmp/enrich-noprune/%'"
rm /tmp/enrich-noprune/*.safetensors
uv run --with duckdb index.py enrich --path /tmp/enrich-noprune
uv run --with duckdb index.py query "SELECT COUNT(*) FROM lora_inspect WHERE file_path LIKE '/tmp/enrich-noprune/%'"
```

Expected: the second count equals the first. Nothing was deleted. Exit code 0 both times.

- [ ] **Step 4: Prove idempotency**

```bash
uv run --with duckdb index.py enrich --path /tmp/enrich-noprune
uv run --with duckdb index.py query "SELECT COUNT(*) FROM lora_inspect WHERE file_path LIKE '/tmp/enrich-noprune/%'"
```

Re-adding the files and enriching again must leave `lora_layers` row counts unchanged — run the count before and after and compare:

```bash
uv run --with duckdb index.py query "SELECT COUNT(*) FROM lora_layers WHERE file_path LIKE '/tmp/enrich-noprune/%'"
```

Expected: identical before and after, and the second `enrich` reports 0 files inspected only if nothing changed; if it reports them, the rows must still not double.

- [ ] **Step 5: Clean up the temp trees**

```bash
rm -rf /tmp/enrich-noprune /tmp/enrich-broken /tmp/broken.safetensors
```

- [ ] **Step 6: Commit — none.**

---

## Task 9: Full sweep and documentation

**Files:**
- Modify: `/mnt/900/training/sets/train-set-analysis/SKILL.md` (document `enrich`, the three tables, `v_issues`)
- Modify: `/home/rockerboo/code/lora-inspector-rs/README.md` (document `--layers` and `network.structure`)

- [ ] **Step 1: Rebuild the stale binary**

The binary at `target/release/lora-inspector` predates the `load` field. Rebuild before sweeping:

```bash
cd /home/rockerboo/code/lora-inspector-rs
cargo build --release -p lora-inspector
./target/release/lora-inspector --version
```

- [ ] **Step 2: Check disk headroom before the sweep**

```bash
df -h /mnt/900 | tail -1
```

Expected: at least ~2 GB free — the DB grows by roughly 400 MB and the cache by ~200 MB. `/mnt/900` measured 11 GB free at plan time. If it is under 2 GB, stop and report rather than starting a 20-minute sweep that will fail mid-write.

- [ ] **Step 3: Run the full sweep**

Run in the background, since it is a ~20 minute disk-bound job:

```bash
cd /mnt/900/training/sets && uv run --with duckdb index.py enrich --jobs 6
```

- [ ] **Step 4: Verify the sweep**

```bash
uv run --with duckdb index.py query "SELECT kind, COUNT(*) FROM lora_inspect GROUP BY 1 ORDER BY 2 DESC"
uv run --with duckdb index.py query "SELECT COUNT(*) FROM lora_inspect WHERE enrich_error IS NOT NULL"
uv run --with duckdb index.py query "SELECT COUNT(*) FROM lora_layers"
uv run --with duckdb index.py stats
```

Expected: every `.safetensors` on disk has a `lora_inspect` row; `kind` splits into `lora` / `artifact` / `checkpoint`; `enrich_error` count is small and every entry names a real problem; `lora_layers` is in the 500k–1M range.

- [ ] **Step 5: Update both docs**

`train-set-analysis/SKILL.md`: add `enrich` to the Commands block, the three new tables to the schema section, `v_issues` as the "what is wrong" entry point, and a note that declared and actual values are separate columns.

`README.md`: add `--layers` to the `inspect` examples and flags table, and describe `network.structure` as key-derived versus `network.type` as declared.

- [ ] **Step 6: Commit the README change only**

```bash
cd /home/rockerboo/code/lora-inspector-rs
git add README.md
git commit -m "docs: document --layers and network.structure"
```
