# LoRA Architecture/Format Test Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a committed, regenerable test matrix that proves every supported LoRA architecture parses correctly in both its kohya-style and diffusers/PEFT-style key-naming conventions (where both exist), without committing multi-GB safetensors files to git.

**Architecture:** A new `extract-fixture` CLI subcommand reads a safetensors file's header only (no tensor data) and writes a small JSON fixture (key names + shapes + dtypes + raw `__metadata__`). Rust integration tests synthesize a zero-filled safetensors buffer from each fixture and feed it through the real `LoRAFile::new_from_buffer` pipeline. A parallel JS test file feeds the same fixture key lists through `parseSDKey`. One real bug (`unet_keys()`/`text_encoder_keys()` failing on non-kohya-prefixed architectures) gets fixed so the whole matrix passes.

**Tech Stack:** Rust (candle-core, safetensors 0.3.3, serde_json), JS (AVA), existing `lora-inspector` CLI (clap).

## Global Constraints

- Design spec: `docs/superpowers/specs/2026-08-07-lora-architecture-test-matrix-design.md` (gitignored, not committed — reference only).
- Fixtures are committed under `crates/inspector/tests/fixtures/<arch>/<format>.json`; the source safetensors files themselves are never committed.
- Every code change must pass `make test` and `make fmt` before the branch is considered done (per `CLAUDE.md`).
- Source files for extraction live under `/mnt/900/lora` on this machine only — extraction commands in this plan reference absolute paths there; the fixtures produced are what get committed, not the source files.
- 4 matrix cells are intentionally absent (documented gaps): SD1.5-diffusers, SDXL-diffusers, Krea2-kohya, Qwen-Image-diffusers, LTX2.3-kohya. Do not fabricate fixtures for these.

---

### Task 1: Expose `LoRAFormat` publicly from the `inspector` crate

**Files:**
- Modify: `crates/inspector/src/lib.rs:21-22`
- Create: `crates/inspector/tests/public_api_smoke.rs`

**Interfaces:**
- Produces: `inspector::LoRAFormat` (re-export of `crate::weight::LoRAFormat`, variants `Kohya`, `Lycoris`, `Peft`), usable from any file under `crates/inspector/tests/`.

`LoRAFile::format()` (`crates/inspector/src/file.rs:148`) already returns `weight::LoRAFormat` publicly, but `mod weight;` in `lib.rs` has no `pub`, so external integration tests can't name the return type. This task fixes that.

- [ ] **Step 1: Write the failing test**

Create `crates/inspector/tests/public_api_smoke.rs`:

```rust
#[test]
fn lora_format_is_publicly_accessible() {
    assert_ne!(inspector::LoRAFormat::Kohya, inspector::LoRAFormat::Peft);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inspector --test public_api_smoke`
Expected: FAIL to compile with `error[E0433]: failed to resolve: could not find `LoRAFormat` in `inspector``

- [ ] **Step 3: Add the re-export**

In `crates/inspector/src/lib.rs`, change:

```rust
pub use crate::file::LayerScale;
pub use crate::svd::RankMetrics;
mod tensor;
mod weight;
```

to:

```rust
pub use crate::file::LayerScale;
pub use crate::svd::RankMetrics;
pub use crate::weight::LoRAFormat;
mod tensor;
mod weight;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p inspector --test public_api_smoke`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/src/lib.rs crates/inspector/tests/public_api_smoke.rs
git commit -m "Expose LoRAFormat publicly from the inspector crate

Needed so integration tests outside the crate can assert on
LoRAFile::format()'s return type."
```

---

### Task 2: Fix `unet_keys()`/`text_encoder_keys()` prefix gap

**Files:**
- Modify: `crates/inspector/src/weight.rs:314-320` (`BufferedLoRAWeight` impl)
- Modify: `crates/inspector/src/weight.rs:829-835` (`LoRAWeight` impl)
- Modify: `crates/inspector/src/weight.rs:21` area (add helper near `is_peft`)
- Test: `crates/inspector/src/weight.rs` (inline `#[cfg(test)] mod tests`)

**Interfaces:**
- Produces: private fn `is_text_encoder_key(key: &str) -> bool` in `weight.rs`, used by both `WeightKey` impls.
- Consumes: nothing new — `self.keys()` already exists on both impls (`WeightKey::keys`).

Currently `unet_keys()` filters on the literal substring `"lora_unet"` and `text_encoder_keys()` on `"lora_te"`. Non-kohya-prefixed architectures (`diffusion_model.blocks...`, bare `transformer_blocks...`) never match `"lora_unet"`, so `unet_keys()` silently returns empty even though the file has valid unet weights. This backs the "Advanced" panel in the UI (`assets/js/components/analysis/Advanced.jsx`), so it's a real, user-visible bug.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/inspector/src/weight.rs` (near `is_peft_detects_bare_transformer_blocks_keys`, around line 1300):

```rust
#[test]
fn is_text_encoder_key_matches_known_prefixes() {
    assert!(is_text_encoder_key(
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"
    ));
    assert!(is_text_encoder_key(
        "lora_te1_text_model_encoder_layers_5_self_attn_q_proj.alpha"
    ));
    assert!(is_text_encoder_key(
        "text_encoder.encoder.layers.0.self_attn.q_proj.lora_A.weight"
    ));
    assert!(!is_text_encoder_key(
        "lora_unet_blocks_0_cross_attn_k.lora_down.weight"
    ));
    assert!(!is_text_encoder_key(
        "diffusion_model.blocks.0.cross_attn.k.lora_A.weight"
    ));
    assert!(!is_text_encoder_key(
        "transformer_blocks.0.attn.add_k_proj.lora_down.weight"
    ));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inspector is_text_encoder_key_matches_known_prefixes`
Expected: FAIL to compile with `error[E0425]: cannot find function `is_text_encoder_key` in this scope`

- [ ] **Step 3: Add the helper function**

In `crates/inspector/src/weight.rs`, right after `fn is_peft(...)` (around line 25), add:

```rust
fn is_text_encoder_key(key: &str) -> bool {
    key.contains("lora_te") || key.contains("text_encoder") || key.contains("text_model")
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p inspector is_text_encoder_key_matches_known_prefixes`
Expected: PASS (1 passed)

- [ ] **Step 5: Write the failing regression test for `unet_keys()` on a non-kohya-prefixed buffer**

Add to the same test module:

```rust
#[test]
fn unet_keys_matches_diffusion_model_prefixed_and_bare_keys() {
    let buffer = safetensors::serialize(
        vec![
            (
                "diffusion_model.blocks.0.cross_attn.k.lora_down.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![4, 8],
                    &[0u8; 4 * 8 * 4],
                )
                .unwrap(),
            ),
            (
                "transformer_blocks.0.attn.add_k_proj.lora_up.weight".to_string(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    vec![8, 4],
                    &[0u8; 8 * 4 * 4],
                )
                .unwrap(),
            ),
        ],
        &None,
    )
    .unwrap();

    let weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();
    let unet_keys = weight.unet_keys();
    assert_eq!(unet_keys.len(), 2, "expected both keys classified as unet, got {:?}", unet_keys);
    assert!(weight.text_encoder_keys().is_empty());
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p inspector unet_keys_matches_diffusion_model_prefixed_and_bare_keys`
Expected: FAIL — `unet_keys()` returns 0 keys (both filtered out since neither contains `"lora_unet"`)

- [ ] **Step 7: Fix `unet_keys()`/`text_encoder_keys()` in both `WeightKey` impls**

In `crates/inspector/src/weight.rs`, replace the `BufferedLoRAWeight` impl (lines 314-320):

```rust
    fn unet_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_unet")
    }

    fn text_encoder_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_te")
    }
```

with:

```rust
    fn unet_keys(&self) -> Vec<String> {
        self.keys()
            .into_iter()
            .filter(|k| !is_text_encoder_key(k))
            .collect()
    }

    fn text_encoder_keys(&self) -> Vec<String> {
        self.keys()
            .into_iter()
            .filter(|k| is_text_encoder_key(k))
            .collect()
    }
```

And replace the `LoRAWeight` impl (lines 829-835):

```rust
    fn unet_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_unet")
    }

    fn text_encoder_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_te")
    }
```

with the identical pattern:

```rust
    fn unet_keys(&self) -> Vec<String> {
        self.keys()
            .into_iter()
            .filter(|k| !is_text_encoder_key(k))
            .collect()
    }

    fn text_encoder_keys(&self) -> Vec<String> {
        self.keys()
            .into_iter()
            .filter(|k| is_text_encoder_key(k))
            .collect()
    }
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cargo test -p inspector unet_keys_matches_diffusion_model_prefixed_and_bare_keys`
Expected: PASS (1 passed)

- [ ] **Step 9: Verify existing kohya-format tests still pass (no regression on counts)**

Run: `cargo test -p inspector` (full inspector crate test suite)
Expected: all existing tests pass, including `weight.rs`'s own tests

Run: `cd crates/lora-inspector-wasm && cargo test -p lora-inspector-wasm unet_keys text_encoder_keys -- --nocapture`
Expected: `unet_keys` test still asserts length 576, `text_encoder_keys` still asserts length 216, both PASS (these counts must not change since every key in `boo.safetensors` contains exactly one of `lora_unet`/`lora_te`, so inverting the predicate doesn't change which bucket any key lands in)

- [ ] **Step 10: Commit**

```bash
git add crates/inspector/src/weight.rs
git commit -m "Fix unet_keys()/text_encoder_keys() for non-kohya key prefixes

Both previously matched on the literal substrings \"lora_unet\"/\"lora_te\",
so architectures using diffusion_model.* or bare transformer_blocks.*
prefixes (Wan, Krea2, LTX2.3, Qwen-Image, Flux.2/Klein diffusers files)
always returned an empty unet_keys(), even though weight_keys()/
base_names() found their keys fine. This is user-visible: unet_keys()/
text_encoder_keys() back the Advanced panel in the UI.

Fixed by inverting the check: a key is unet unless it looks like a text
encoder key (lora_te/text_encoder/text_model), rather than requiring an
explicit lora_unet allowlist prefix."
```

---

### Task 3: Add `extract-fixture` CLI subcommand

**Files:**
- Modify: `crates/lora-inspector/src/main.rs`

**Interfaces:**
- Produces: `lora-inspector extract-fixture --file <path> --out <path.json>` CLI command, writing a JSON file shaped `{ "source": String, "keys": { <key>: { "dtype": String, "shape": [usize] } }, "metadata": Option<HashMap<String,String>> }`.

This reads only the safetensors header (8-byte length prefix + that many bytes of JSON), following the existing `metadata_from_file` helper's pattern in the same file, so it works even on multi-GB source files without loading tensor data.

- [ ] **Step 1: Add the subcommand variant**

In `crates/lora-inspector/src/main.rs`, in the `Command` enum, after `CompareMetadata`:

```rust
    /// Extract a lightweight JSON fixture (key names, shapes, dtypes, and
    /// raw __metadata__) from a safetensors file's header, without loading
    /// tensor data. Used to build committed test fixtures from large local
    /// LoRA files.
    ExtractFixture {
        /// Path to the safetensors file
        #[clap(short, long)]
        file: PathBuf,

        /// Path to write the JSON fixture to
        #[clap(short, long)]
        out: PathBuf,
    },
```

- [ ] **Step 2: Add the dispatch arm**

In the `match cli.command` block in `fn main()`, after the `Command::CompareMetadata { .. } => { ... }` arm (before the closing `}`), change:

```rust
            Ok(())
        } // Handle other subcommands...
    }
}
```

to:

```rust
            Ok(())
        }

        Command::ExtractFixture { file, out } => extract_fixture(file, out),
    }
}
```

- [ ] **Step 3: Add the `extract_fixture` function and its data types**

Add near `fn metadata_from_file` (after it) in `crates/lora-inspector/src/main.rs`:

```rust
#[derive(Serialize, Deserialize, Debug)]
struct FixtureTensorInfo {
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Fixture {
    source: String,
    keys: HashMap<String, FixtureTensorInfo>,
    metadata: Option<HashMap<String, String>>,
}

fn extract_fixture(file: PathBuf, out: PathBuf) -> Result<()> {
    let mut f = File::open(&file)?;
    f.seek(SeekFrom::Start(0))?;

    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);

    let mut header_data = vec![0u8; header_len as usize];
    f.read_exact(&mut header_data)?;

    #[derive(Deserialize)]
    struct RawTensorInfo {
        dtype: String,
        shape: Vec<usize>,
    }

    let header: HashMap<String, serde_json::Value> = serde_json::from_slice(&header_data)?;

    let mut keys = HashMap::new();
    let mut metadata = None;

    for (key, value) in header {
        if key == "__metadata__" {
            metadata = serde_json::from_value(value).ok();
            continue;
        }
        let info: RawTensorInfo = serde_json::from_value(value)?;
        keys.insert(
            key,
            FixtureTensorInfo {
                dtype: info.dtype,
                shape: info.shape,
            },
        );
    }

    let fixture = Fixture {
        source: file.display().to_string(),
        keys,
        metadata,
    };

    let json = serde_json::to_string_pretty(&fixture)?;
    std::fs::write(&out, json)?;

    println!(
        "Wrote fixture with {} keys to {}",
        fixture.keys.len(),
        out.display()
    );

    Ok(())
}
```

`RawTensorInfo` intentionally omits `data_offsets` — `serde_json::from_value` ignores extra fields it isn't asked to deserialize, so it's fine that the raw header entries also carry a `data_offsets` field.

- [ ] **Step 4: Build and run against a small local test file**

Run: `cargo build -p lora-inspector`
Expected: builds with no errors

Run: `cargo run -p lora-inspector -- extract-fixture --file crates/lora-inspector/boo.safetensors --out /tmp/boo_fixture_smoke.json`
Expected: prints `Wrote fixture with N keys to /tmp/boo_fixture_smoke.json`

Run: `python3 -c "import json; d = json.load(open('/tmp/boo_fixture_smoke.json')); print(len(d['keys']), list(d['keys'])[:2])"`
Expected: prints a key count > 0 and two sample key names shaped like `lora_unet_...` or `lora_te_...`

- [ ] **Step 5: Commit**

```bash
git add crates/lora-inspector/src/main.rs
git commit -m "Add extract-fixture CLI subcommand

Reads a safetensors file's header only (no tensor data) and writes a
JSON fixture of key names, shapes, dtypes, and raw __metadata__. Used
to build the architecture/format test matrix fixtures without
committing multi-GB source files."
```

---

### Task 4: Generate and commit fixture files

**Files:**
- Create: `crates/inspector/tests/fixtures/sd15/kohya.json`
- Create: `crates/inspector/tests/fixtures/sdxl/kohya.json`
- Create: `crates/inspector/tests/fixtures/sdxl/lycoris.json`
- Create: `crates/inspector/tests/fixtures/flux1/kohya.json`
- Create: `crates/inspector/tests/fixtures/flux1/diffusers.json`
- Create: `crates/inspector/tests/fixtures/flux2_klein/kohya.json`
- Create: `crates/inspector/tests/fixtures/flux2_klein/diffusers.json`
- Create: `crates/inspector/tests/fixtures/wan/kohya.json`
- Create: `crates/inspector/tests/fixtures/wan/diffusers.json`
- Create: `crates/inspector/tests/fixtures/krea2/diffusers.json`
- Create: `crates/inspector/tests/fixtures/krea2/lycoris.json`
- Create: `crates/inspector/tests/fixtures/h3/diffusers.json`
- Create: `crates/inspector/tests/fixtures/qwen_image/kohya.json`
- Create: `crates/inspector/tests/fixtures/ltx23/diffusers.json`

**Interfaces:**
- Consumes: `lora-inspector extract-fixture` from Task 3.
- Produces: 14 fixture JSON files, each shaped per Task 3's `Fixture` struct, read by Task 5's support module.

This is mechanical: run the CLI once per source file. Source paths are absolute and specific to this machine (`/mnt/900/lora`); only the JSON output gets committed.

- [ ] **Step 1: Create fixture directories**

```bash
mkdir -p crates/inspector/tests/fixtures/{sd15,sdxl,flux1,flux2_klein,wan,krea2,h3,qwen_image,ltx23}
```

- [ ] **Step 2: Build the release binary once (faster repeated runs)**

```bash
cargo build -p lora-inspector --release
BIN=./target/release/lora-inspector
```

- [ ] **Step 3: Extract all 14 fixtures**

```bash
$BIN extract-fixture --file /mnt/900/lora/add_detail.safetensors \
  --out crates/inspector/tests/fixtures/sd15/kohya.json

$BIN extract-fixture --file /mnt/900/lora/sdxl/anyaXL.safetensors \
  --out crates/inspector/tests/fixtures/sdxl/kohya.json

$BIN extract-fixture --file /mnt/900/lora/sdxl/gwei.safetensors \
  --out crates/inspector/tests/fixtures/sdxl/lycoris.json

$BIN extract-fixture --file /mnt/900/lora/flux/DracoFelis.safetensors \
  --out crates/inspector/tests/fixtures/flux1/kohya.json

$BIN extract-fixture --file /mnt/900/lora/flux/FLUX-daubrez-DB4RZ.safetensors \
  --out crates/inspector/tests/fixtures/flux1/diffusers.json

$BIN extract-fixture --file "/mnt/900/lora/flux2-klein-4b/nipplediffusion-f2-klein-4b.safetensors" \
  --out crates/inspector/tests/fixtures/flux2_klein/kohya.json

$BIN extract-fixture --file "/mnt/900/lora/flux2-klein-9b/dx8152-Klein-Migration.safetensors" \
  --out crates/inspector/tests/fixtures/flux2_klein/diffusers.json

$BIN extract-fixture --file "/mnt/900/lora/wan2.2/wan2.1-i2v-480p-pov-insertion-v1.0.safetensors" \
  --out crates/inspector/tests/fixtures/wan/kohya.json

$BIN extract-fixture --file "/mnt/900/lora/wan2.2/wan2.2_5b_c0wg1rl_72_000002500.safetensors" \
  --out crates/inspector/tests/fixtures/wan/diffusers.json

$BIN extract-fixture --file /mnt/900/lora/krea2/bloomgirls-ultrarealism-krea2_4k.safetensors \
  --out crates/inspector/tests/fixtures/krea2/diffusers.json

$BIN extract-fixture --file /mnt/900/lora/krea2/realism_engine_krea2_v2.safetensors \
  --out crates/inspector/tests/fixtures/krea2/lycoris.json

$BIN extract-fixture --file /mnt/900/lora/h3/minimax_h3_fl2v_turbo_4step_v0.1.safetensors \
  --out crates/inspector/tests/fixtures/h3/diffusers.json

$BIN extract-fixture --file "/mnt/900/lora/qwen-image/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors" \
  --out crates/inspector/tests/fixtures/qwen_image/kohya.json

$BIN extract-fixture --file /mnt/900/lora/ltx2.3/DR34ML4Y_LT3X_V3.safetensors \
  --out crates/inspector/tests/fixtures/ltx23/diffusers.json
```

Expected: 14 "Wrote fixture with N keys to ..." lines, no errors.

- [ ] **Step 4: Sanity-check fixture sizes and shapes**

```bash
du -sh crates/inspector/tests/fixtures/*/*.json
python3 -c "
import json, glob
for path in sorted(glob.glob('crates/inspector/tests/fixtures/*/*.json')):
    d = json.load(open(path))
    print(path, len(d['keys']), 'metadata' if d.get('metadata') else 'no-metadata')
"
```

Expected: all files are well under 1MB; each prints a non-zero key count; `sdxl/lycoris.json` prints `metadata` (it has `ss_network_args`); `krea2/lycoris.json` prints `no-metadata` (confirms the caveat documented in the spec).

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/tests/fixtures/
git commit -m "Add architecture/format test matrix fixtures

14 JSON fixtures (key names, shapes, dtypes, metadata) extracted from
real LoRA files covering SD1.5, SDXL (+LyCORIS), Flux.1, Flux.2/Klein,
Wan, Krea2 (+LyCORIS), MiniMax H3, Qwen-Image, and LTX2.3, in both
kohya-style and diffusers/PEFT-style key naming where both exist.
Source safetensors files are not committed -- only header-derived
metadata."
```

---

### Task 5: Rust fixture-loading and synthetic-safetensors support module

**Files:**
- Create: `crates/inspector/tests/support/mod.rs`

**Interfaces:**
- Consumes: fixture JSON files from Task 4, `safetensors::{serialize, Dtype}`, `safetensors::tensor::TensorView`.
- Produces: `pub fn load_fixture(relative_path: &str) -> Fixture` and `pub fn synthesize_safetensors(fixture: &Fixture) -> Vec<u8>`, used by Task 6's test file via `mod support;`.

Placing this under `tests/support/mod.rs` (not directly under `tests/`) keeps Cargo from treating it as its own test binary — only files directly in `tests/` become separate test crates.

- [ ] **Step 1: Write the failing test (in the same file, testing itself)**

Create `crates/inspector/tests/support/mod.rs`:

```rust
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use safetensors::tensor::TensorView;
use safetensors::Dtype;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct FixtureTensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
}

#[derive(Deserialize)]
pub struct Fixture {
    #[allow(dead_code)]
    pub source: String,
    pub keys: HashMap<String, FixtureTensorInfo>,
    #[allow(dead_code)]
    pub metadata: Option<HashMap<String, String>>,
}

pub fn load_fixture(relative_path: &str) -> Fixture {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(relative_path);
    let data = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {}", path.display(), e));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("failed to parse fixture {}: {}", path.display(), e))
}

fn dtype_from_str(dtype: &str) -> Dtype {
    match dtype {
        "BOOL" => Dtype::BOOL,
        "U8" => Dtype::U8,
        "I8" => Dtype::I8,
        "I16" => Dtype::I16,
        "U16" => Dtype::U16,
        "F16" => Dtype::F16,
        "BF16" => Dtype::BF16,
        "I32" => Dtype::I32,
        "U32" => Dtype::U32,
        "F32" => Dtype::F32,
        "F64" => Dtype::F64,
        "I64" => Dtype::I64,
        "U64" => Dtype::U64,
        other => panic!("unsupported fixture dtype: {other}"),
    }
}

/// Builds a valid, zero-filled safetensors buffer matching a fixture's key/shape/dtype
/// layout. Numeric values don't matter here -- only shapes, dtypes, and key names,
/// since the matrix tests assert on parsing/classification, not on math.
pub fn synthesize_safetensors(fixture: &Fixture) -> Vec<u8> {
    let mut key_order: Vec<&String> = fixture.keys.keys().collect();
    key_order.sort();

    let buffers: Vec<Vec<u8>> = key_order
        .iter()
        .map(|key| {
            let info = &fixture.keys[*key];
            let dtype = dtype_from_str(&info.dtype);
            let n_elements: usize = info.shape.iter().product();
            vec![0u8; n_elements * dtype.size()]
        })
        .collect();

    let views: Vec<(String, TensorView)> = key_order
        .iter()
        .zip(buffers.iter())
        .map(|(key, buf)| {
            let info = &fixture.keys[*key];
            let dtype = dtype_from_str(&info.dtype);
            let view = TensorView::new(dtype, info.shape.clone(), buf)
                .unwrap_or_else(|e| panic!("failed to build tensor view for {key}: {e:?}"));
            ((*key).clone(), view)
        })
        .collect();

    safetensors::serialize(views, &None)
        .unwrap_or_else(|e| panic!("failed to serialize synthetic safetensors buffer: {e:?}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthesize_safetensors_round_trips_through_candle() {
        let fixture = load_fixture("sd15/kohya.json");
        let buffer = synthesize_safetensors(&fixture);

        let loaded = candle_core::safetensors::BufferedSafetensors::new(buffer)
            .expect("synthetic buffer should be a valid safetensors file");
        assert_eq!(loaded.tensors().len(), fixture.keys.len());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p inspector --test architecture_matrix` (this will fail to even find a target — expected at this point, since `tests/support/mod.rs` alone with no `tests/architecture_matrix.rs` referencing it via `mod support;` is not compiled as a test target by Cargo)

Since `tests/support/mod.rs` isn't a standalone Cargo test target, create a temporary throwaway harness to run its inline test: add a one-line `mod support;` at the top of a scratch file `crates/inspector/tests/architecture_matrix.rs` (this file is fully filled in during Task 6; for now it only needs the module declaration to make Task 5's test runnable):

```rust
mod support;
```

Run: `cargo test -p inspector --test architecture_matrix`
Expected: FAIL — `error[E0433]: failed to resolve: use of undeclared crate or module `candle_core`` if `candle_core` isn't in scope, or a straightforward compile/logic failure if the fixture path is wrong. Confirm it's failing for the right reason (fixture not found, or module resolution), not passing by accident.

- [ ] **Step 3: Fix any issues found and run to verify it passes**

Run: `cargo test -p inspector --test architecture_matrix`
Expected: PASS (`synthesize_safetensors_round_trips_through_candle` succeeds)

- [ ] **Step 4: Commit**

```bash
git add crates/inspector/tests/support/mod.rs crates/inspector/tests/architecture_matrix.rs
git commit -m "Add fixture-loading and synthetic-safetensors test support module

Loads a fixture JSON (key/shape/dtype/metadata) and can synthesize a
valid, zero-filled safetensors buffer from it, so the full LoRAFile
load path can be exercised in tests without needing real tensor data."
```

---

### Task 6: Rust architecture matrix tests

**Files:**
- Modify: `crates/inspector/tests/architecture_matrix.rs` (created as a stub in Task 5)

**Interfaces:**
- Consumes: `support::load_fixture`, `support::synthesize_safetensors` (Task 5), `inspector::file::LoRAFile`, `inspector::LoRAFormat` (Task 1).

- [ ] **Step 1: Write the failing tests**

Replace the contents of `crates/inspector/tests/architecture_matrix.rs` with:

```rust
mod support;

use candle_core::Device;
use inspector::file::LoRAFile;
use inspector::LoRAFormat;

use support::Fixture;

fn load_and_check(
    fixture_path: &str,
    expected_format: LoRAFormat,
    check_scale_weight: bool,
) -> LoRAFile {
    let fixture: Fixture = support::load_fixture(fixture_path);
    let buffer = support::synthesize_safetensors(&fixture);
    let file = LoRAFile::new_from_buffer(&buffer, fixture_path, &Device::Cpu);

    assert!(
        file.is_tensors_loaded(),
        "{fixture_path}: expected tensors to load"
    );
    assert_eq!(
        file.format(),
        expected_format,
        "{fixture_path}: unexpected LoRAFormat"
    );

    let base_names = file.base_names();
    assert!(
        !base_names.is_empty(),
        "{fixture_path}: expected non-empty base_names()"
    );

    let dims = file.dims();
    assert!(
        !dims.is_empty(),
        "{fixture_path}: expected non-empty dims()"
    );

    assert!(
        !file.unet_keys().is_empty(),
        "{fixture_path}: expected non-empty unet_keys() (diffusion_model./bare-prefix regression check)"
    );

    if check_scale_weight {
        let sample = &base_names[0];
        let scaled = file.scale_weight(sample);
        assert!(
            scaled.is_ok(),
            "{fixture_path}: scale_weight({sample}) failed: {:?}",
            scaled.err()
        );
    }

    file
}

#[test]
fn sd15_kohya() {
    load_and_check("sd15/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn sdxl_kohya() {
    load_and_check("sdxl/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn sdxl_lycoris() {
    // LoHA, but LoRAFormat::Lycoris is never constructed anywhere in weight.rs --
    // the Kohya/Peft axis is purely about lora_A/B vs lora_up/down naming, so this
    // still reports Kohya. Metadata (ss_network_module=lycoris.kohya, algo=loha) is
    // present, so scale_weight() correctly dispatches to scale_hada_weight.
    load_and_check("sdxl/lycoris.json", LoRAFormat::Kohya, true);
}

#[test]
fn flux1_kohya() {
    load_and_check("flux1/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn flux1_diffusers() {
    load_and_check("flux1/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn flux2_klein_kohya() {
    load_and_check("flux2_klein/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn flux2_klein_diffusers() {
    load_and_check("flux2_klein/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn wan_kohya() {
    load_and_check("wan/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn wan_diffusers() {
    load_and_check("wan/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn krea2_diffusers() {
    load_and_check("krea2/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn krea2_lycoris() {
    // LoKr, no training metadata at all (musubi-tuner didn't write ss_network_args),
    // so LoRAFile::scale_weight can't tell it's LoKr and falls back to plain LoRA
    // scaling, which errors on LoKr-only keys. Metadata-independent network-type
    // detection is a separate, unscoped feature -- this cell stops at unet_keys().
    load_and_check("krea2/lycoris.json", LoRAFormat::Kohya, false);
}

#[test]
fn h3_diffusers() {
    load_and_check("h3/diffusers.json", LoRAFormat::Peft, true);
}

#[test]
fn qwen_image_kohya() {
    load_and_check("qwen_image/kohya.json", LoRAFormat::Kohya, true);
}

#[test]
fn ltx23_diffusers() {
    load_and_check("ltx23/diffusers.json", LoRAFormat::Peft, true);
}
```

- [ ] **Step 2: Run tests to verify current state**

Run: `cargo test -p inspector --test architecture_matrix`
Expected: most tests PASS already (Tasks 1, 2, 5 already fixed the underlying issues); if any fail, read the assertion message -- it names the fixture and exactly which check failed.

- [ ] **Step 3: Diagnose and fix any failures**

For each failure, the assertion message pinpoints the cause category (format mismatch, empty base_names, empty dims, empty unet_keys, or scale_weight error). Given Tasks 1-2 already fixed the known `unet_keys()` gap and Task 4's fixtures were chosen with verified expected formats (see Task 4 Step 4's sanity check and the design spec's per-cell reasoning), no further source fixes are expected here. If a test still fails, treat it as a newly discovered gap: re-verify the expected `LoRAFormat`/flags against the actual fixture content (`python3 -c "import json; d=json.load(open('crates/inspector/tests/fixtures/<path>')); print(list(d['keys'])[:5])"`) before changing test expectations.

- [ ] **Step 4: Run full inspector test suite**

Run: `cargo test -p inspector`
Expected: all tests pass, including `architecture_matrix`, `public_api_smoke`, and all existing inline tests.

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/tests/architecture_matrix.rs
git commit -m "Add Rust architecture/format matrix tests

One test per (architecture, format) fixture cell, asserting format
detection, base_names(), dims(), unet_keys(), and scale_weight() all
work end-to-end through LoRAFile::new_from_buffer on a synthesized
zero-filled safetensors buffer."
```

---

### Task 7: JS architecture matrix tests

**Files:**
- Create: `crates/lora-inspector-wasm/tests/architectureMatrix.test.js`

**Interfaces:**
- Consumes: `parseSDKey` from `crates/lora-inspector-wasm/assets/js/moduleBlocks.js`, fixture JSON from `crates/inspector/tests/fixtures/` (read directly, cross-crate, via relative path).

- [ ] **Step 1: Write the failing test**

Create `crates/lora-inspector-wasm/tests/architectureMatrix.test.js`:

```js
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_ROOT = path.join(
	__dirname,
	"../../inspector/tests/fixtures",
);

const FIXTURE_CELLS = [
	"sd15/kohya.json",
	"sdxl/kohya.json",
	"sdxl/lycoris.json",
	"flux1/kohya.json",
	"flux1/diffusers.json",
	"flux2_klein/kohya.json",
	"flux2_klein/diffusers.json",
	"wan/kohya.json",
	"wan/diffusers.json",
	"krea2/diffusers.json",
	"krea2/lycoris.json",
	"h3/diffusers.json",
	"qwen_image/kohya.json",
	"ltx23/diffusers.json",
];

function loadFixture(relativePath) {
	const raw = fs.readFileSync(path.join(FIXTURES_ROOT, relativePath), "utf8");
	return JSON.parse(raw);
}

// A handful of key suffixes (alpha, dora_scale, etc.) aren't block-structural
// and parseSDKey isn't expected to classify them -- these fixtures cover
// parseSDKey's own supported key shapes (double_blocks/single_blocks/
// transformer_blocks/blocks/te), not every tensor a safetensors file can hold.
const SKIP_SUFFIXES = [".alpha", ".dora_scale"];

for (const cell of FIXTURE_CELLS) {
	test(`parseSDKey handles every key in ${cell}`, (t) => {
		const fixture = loadFixture(cell);
		const keys = Object.keys(fixture.keys).filter(
			(k) => !SKIP_SUFFIXES.some((suffix) => k.endsWith(suffix)),
		);

		t.true(keys.length > 0, `${cell} should have parseable keys`);

		for (const key of keys) {
			t.notThrows(() => parseSDKey(key), `Should parse key: ${key}`);
		}
	});
}
```

- [ ] **Step 2: Run test to verify current state**

Run: `cd crates/lora-inspector-wasm && yarn ava tests/architectureMatrix.test.js`
Expected: run and observe which cells fail; note the failing keys in the AVA output.

- [ ] **Step 3: Diagnose and fix `moduleBlocks.js` gaps, if any**

For each failing key, identify which `parseSDKey` branch should handle it (compare its shape against the existing `FLUX_DOUBLE`/`FLUX_SINGLE`/`FLUX_PEFT`/`H3_BLOCK_RE`/`KREA2_BLOCK_RE` regexes and their guard `key.includes(...)` conditions in `crates/lora-inspector-wasm/assets/js/moduleBlocks.js`). Given the FLUX_PEFT bug from `BUGS.md` was already fixed (verified in the design spec), Wan/LTX2.3/Qwen-Image are new territory for `parseSDKey` and may need new branches -- if so, add a minimal regex + guard following the existing pattern (e.g. a `WAN_BLOCK_RE` for `diffusion_model.blocks.N.<subtype>` and `lora_unet_blocks_N_<subtype>`, mirroring `KREA2_BLOCK_RE`'s structure), scoped to only the key shapes the failing fixture actually contains.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd crates/lora-inspector-wasm && yarn ava tests/architectureMatrix.test.js`
Expected: all cells PASS.

- [ ] **Step 5: Run the full JS test suite**

Run: `cd crates/lora-inspector-wasm && yarn ava`
Expected: all tests pass, including the pre-existing `moduleBlocks.test.js`.

- [ ] **Step 6: Commit**

```bash
git add crates/lora-inspector-wasm/tests/architectureMatrix.test.js
git commit -m "Add JS architecture/format matrix tests

Feeds every non-alpha/dora_scale key from each Rust test matrix
fixture through parseSDKey, so the UI's block-naming logic is proven
against the same real-world key shapes as the Rust matrix."
```

(If Step 3 required changes to `moduleBlocks.js`, include it in this commit and mention the specific gap closed in the commit message instead of the generic one above.)

---

### Task 8: Final full-suite verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `make test`
Expected: all Rust and JS tests pass.

- [ ] **Step 2: Run formatting**

Run: `make fmt && dprint fmt`
Expected: completes with no errors; check `git diff` for any formatting-only changes and include them if present.

- [ ] **Step 3: Review the final diff**

Run: `git status --short` and `git log --oneline -10`
Expected: 7-8 commits from this plan, all fixture/test/source files accounted for, nothing unexpected staged.

- [ ] **Step 4: Commit any formatting fixes**

```bash
git add -u
git commit -m "Apply formatting"
```

(Skip this step if `make fmt`/`dprint fmt` produced no changes.)
