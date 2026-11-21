# Header-Only LoRA Load Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `invalid malloc request` crashes when uploading large (2GB+) LoRA `.safetensors` files by parsing only the safetensors header on initial load, and expose per-tensor name/dtype/shape info to the UI as part of that header parse.

**Architecture:** Add a new `header` module to the `inspector` crate that parses just the safetensors header (8-byte little-endian length prefix + JSON tensor index) without requiring the tensor payload bytes to be present in the buffer. Wire it into `LoRAFile` as an always-resident `header` field (parallel to the existing `metadata` field), and reimplement the key/shape/dtype/format-detection accessors to read from it instead of from the fully-loaded `BufferedLoRAWeight`. On the JS side, read only the file's header bytes via `File.slice()` for the initial upload instead of the whole file with `FileReader.readAsArrayBuffer`.

**Out of scope (known follow-up):** Norms, weight scaling, rank metrics, and alpha values still require the actual tensor bytes and still go through the existing full-buffer `reload_from_buffer` path. For files in the multi-GB range, invoking those features will still hit the same `invalid malloc request` crash — this plan only fixes the crash on *upload*, and makes tensor names/shapes/dtypes available without needing a full load. Making weight-derived stats lazy per-tensor is a separate, much larger project (making the `Weight` trait async end-to-end) that was explicitly deferred.

**Tech Stack:** Rust (`inspector` crate: `safetensors` 0.3.3, `serde_json`), wasm-bindgen (`lora-inspector-wasm` crate), vanilla JS Web Worker (`crates/lora-inspector-wasm/assets/js/worker.js`).

## Global Constraints

- Do not change `LoRAFile::new_from_buffer`'s existing behavior/signature (used by the CLI, native code, full-buffer tests) beyond additionally populating the new `header` field — all existing snapshot/unit tests in `crates/inspector` must keep passing unchanged.
- The wasm-exposed `LoraWorker` constructor keeps its current name and JS call signature (`new worker(buffer, file.name)`); only the *meaning* of `buffer` changes (header-only bytes instead of the whole file).
- `alphas()` stays weight-value-dependent (unchanged) — it reads actual tensor scalars/ranks, which the header does not carry. This is a deliberate scope boundary, not an oversight.
- Run `cargo test` (inspector + lora-inspector-wasm where applicable) and `make fmt` before considering any task done.

---

## File Structure

- Create: `crates/inspector/src/header.rs` — `HeaderIndex` type + `parse_header()`, ported key/shape/dtype logic that today lives inside `BufferedLoRAWeight`.
- Modify: `crates/inspector/src/lib.rs` — declare `mod header;`.
- Modify: `crates/inspector/src/weight.rs` — make `is_peft` `pub(crate)` so `header.rs` can reuse the exact same PEFT-detection logic.
- Modify: `crates/inspector/src/metadata.rs` — add `Metadata::new_from_header_buffer`.
- Modify: `crates/inspector/src/file.rs` — add `header` field + `LoRAFile::new_from_header_buffer`, add `TensorInfo` struct + `tensor_info()`, migrate `keys`/`weight_keys`/`unet_keys`/`text_encoder_keys`/`alpha_keys`/`base_names`/`dims`/`precision`/`format` to read from `header`.
- Modify: `crates/lora-inspector-wasm/src/worker.rs` — constructor now parses header-only bytes; add `tensor_info()` wasm binding.
- Modify: `crates/lora-inspector-wasm/assets/js/worker.js` — add `readHeaderBytes()`, split `ensureLoaded` into `ensureHeaderLoaded`/`ensureWeightsLoaded`, add `withWeights()`, repoint weight-value handlers at it.

---

### Task 1: `header` module — header-only safetensors parsing

**Files:**
- Create: `crates/inspector/src/header.rs`
- Modify: `crates/inspector/src/lib.rs:13-24`
- Modify: `crates/inspector/src/weight.rs:21` (`is_peft` visibility)

**Interfaces:**
- Produces: `pub struct HeaderIndex` with methods `keys()`, `keys_by_key(&str)`, `weight_keys()`, `unet_keys()`, `text_encoder_keys()`, `alpha_keys()`, `base_names()`, `dims() -> HashSet<usize>`, `precision() -> Option<weight::DType>`, `format() -> weight::LoRAFormat`, `tensor_info() -> Vec<(String, weight::DType, Vec<usize>)>`.
- Produces: `pub fn parse_header(buffer: &[u8]) -> crate::Result<(HeaderIndex, Option<HashMap<String, String>>)>` — the second tuple element is the `__metadata__` map, matching `Metadata::metadata`'s type.

- [ ] **Step 1: Write the failing tests**

Create `crates/inspector/src/header.rs` with just the test module first (no implementation yet — this will fail to compile, which is the expected "red" state):

```rust
#[cfg(test)]
mod tests {
    use std::{fs::File, io};
    use std::io::Read;

    use candle_core::Device;

    use crate::weight::BufferedLoRAWeight;

    use super::*;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";
        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;
        Ok(data)
    }

    fn header_only_bytes(buffer: &[u8]) -> &[u8] {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        &buffer[0..8 + header_len]
    }

    #[test]
    fn parse_header_rejects_buffer_smaller_than_length_prefix() {
        let result = parse_header(&[1_u8, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_rejects_truncated_header_json() {
        let buffer = load_test_file().unwrap();
        // 20 bytes is past the 8-byte length prefix but short of the real header JSON.
        let result = parse_header(&buffer[0..20]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_works_with_header_only_buffer() {
        let buffer = load_test_file().unwrap();
        let (header, meta) = parse_header(header_only_bytes(&buffer)).unwrap();

        assert!(!header.keys().is_empty());
        assert!(meta.is_some());
    }

    #[test]
    fn header_index_matches_buffered_weight_for_boo_file() {
        let buffer = load_test_file().unwrap();
        let (header, _meta) = parse_header(&buffer).unwrap();

        let buffered = BufferedLoRAWeight::new(buffer.clone(), &Device::Cpu).unwrap();

        use crate::weight::{Weight, WeightKey};

        let mut header_keys = header.keys();
        let mut buffered_keys = buffered.keys();
        header_keys.sort();
        buffered_keys.sort();
        assert_eq!(header_keys, buffered_keys);

        assert_eq!(header.unet_keys().len(), buffered.unet_keys().len());
        assert_eq!(
            header.text_encoder_keys().len(),
            buffered.text_encoder_keys().len()
        );
        assert_eq!(header.dims(), buffered.dims());
        assert_eq!(header.precision(), buffered.precision());
        assert_eq!(header.format(), buffered.format());

        let mut header_base_names = header.base_names();
        let mut buffered_base_names = buffered.base_names();
        header_base_names.sort();
        buffered_base_names.sort();
        assert_eq!(header_base_names, buffered_base_names);
    }

    #[test]
    fn tensor_info_is_non_empty_and_sorted_by_name() {
        let buffer = load_test_file().unwrap();
        let (header, _meta) = parse_header(&buffer).unwrap();

        let info = header.tensor_info();
        assert!(!info.is_empty());

        let mut names: Vec<&String> = info.iter().map(|(name, _, _)| name).collect();
        let sorted = {
            let mut n = names.clone();
            n.sort();
            n
        };
        assert_eq!(names, sorted);
        names.clear(); // silence unused mut warning path in some rustc versions
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crates/inspector && cargo test header:: 2>&1 | head -50`
Expected: compile error — `parse_header`, `HeaderIndex` not found (module body doesn't exist yet).

- [ ] **Step 3: Make `is_peft` reusable from `header.rs`**

In `crates/inspector/src/weight.rs:21`, change:

```rust
fn is_peft(keys: Vec<String>) -> bool {
```

to:

```rust
pub(crate) fn is_peft(keys: Vec<String>) -> bool {
```

- [ ] **Step 4: Implement `HeaderIndex` and `parse_header`**

Add above the `#[cfg(test)]` block in `crates/inspector/src/header.rs`:

```rust
use std::collections::{HashMap, HashSet};

use crate::weight::{get_base_name, is_peft, DType, LoRAFormat};
use crate::{InspectorError, Result};

/// A tensor's shape/dtype as recorded in the safetensors header, without its
/// payload bytes.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// An index of every tensor's name, dtype and shape parsed from a safetensors
/// header, independent of whether the tensor payload bytes are resident anywhere.
#[derive(Debug, Clone)]
pub struct HeaderIndex {
    tensors: HashMap<String, TensorEntry>,
    format: LoRAFormat,
}

impl HeaderIndex {
    pub fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn keys_by_key(&self, key: &str) -> Vec<String> {
        self.tensors
            .keys()
            .filter(|k| k.contains(key))
            .cloned()
            .collect()
    }

    pub fn weight_keys(&self) -> Vec<String> {
        let hada = &mut self.keys_by_key("hada_w1");
        let lokr = &mut self.keys_by_key("lokr_w1");
        let oft_diag = &mut self.keys_by_key("oft_diag");
        let oft_blocks = &mut self.keys_by_key("oft_block");
        let glora_a1 = &mut self.keys_by_key("a1");
        let glora_b1 = &mut self.keys_by_key("b1");
        let glora_a2 = &mut self.keys_by_key("a2");
        let glora_b2 = &mut self.keys_by_key("b1");
        let mut keys = self.keys_by_key("weight");
        keys.append(hada);
        keys.append(lokr);
        keys.append(oft_diag);
        keys.append(oft_blocks);
        keys.append(glora_a1);
        keys.append(glora_b1);
        keys.append(glora_a2);
        keys.append(glora_b2);

        keys
    }

    pub fn unet_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_unet")
    }

    pub fn text_encoder_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_te")
    }

    pub fn alpha_keys(&self) -> Vec<String> {
        self.keys_by_key("alpha")
    }

    pub fn base_names(&self) -> Vec<String> {
        self.weight_keys()
            .iter()
            .map(|name| get_base_name(name))
            .collect::<HashSet<String>>()
            .into_iter()
            .collect()
    }

    pub fn dims(&self) -> HashSet<usize> {
        self.tensors
            .iter()
            .filter_map(|(k, e)| {
                if k.contains("lora_down")
                    || k.contains("hada_w1_b")
                    || k.contains("lokr_w1")
                    || k.contains("b1.weight")
                    || k.contains("lora_A")
                {
                    e.shape.first().copied()
                } else if k.contains("oft_diag") || k.contains("oft_blocks") {
                    e.shape.last().copied()
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn precision(&self) -> Option<DType> {
        self.tensors
            .iter()
            .filter(|(k, _)| !k.contains("alpha"))
            .collect::<Vec<_>>()
            .first()
            .map(|(_, e)| e.dtype)
    }

    pub fn format(&self) -> LoRAFormat {
        self.format
    }

    /// `(name, dtype, shape)` for every tensor in the file, sorted by name for a
    /// stable UI ordering.
    pub fn tensor_info(&self) -> Vec<(String, DType, Vec<usize>)> {
        let mut info: Vec<_> = self
            .tensors
            .iter()
            .map(|(name, e)| (name.clone(), e.dtype, e.shape.clone()))
            .collect();
        info.sort_by(|a, b| a.0.cmp(&b.0));
        info
    }
}

/// Parses only the safetensors header — the 8-byte little-endian length prefix
/// followed by that many bytes of header JSON — without requiring the tensor
/// payload bytes to be present. `buffer` only needs to contain the first
/// `8 + header_len` bytes of the file; anything beyond that is ignored.
pub fn parse_header(buffer: &[u8]) -> Result<(HeaderIndex, Option<HashMap<String, String>>)> {
    if buffer.len() < 8 {
        return Err(InspectorError::Msg(
            "buffer too small to contain a safetensors header".to_string(),
        ));
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buffer[0..8]);
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    let stop = 8usize
        .checked_add(header_len)
        .ok_or_else(|| InspectorError::Msg("safetensors header length overflow".to_string()))?;
    if stop > buffer.len() {
        return Err(InspectorError::Msg(
            "buffer does not contain the full safetensors header JSON".to_string(),
        ));
    }

    let json_str = std::str::from_utf8(&buffer[8..stop])
        .map_err(|_| InspectorError::Msg("safetensors header is not valid UTF-8".to_string()))?;
    let metadata: safetensors::tensor::Metadata = serde_json::from_str(json_str)?;

    let sample_keys: Vec<String> = metadata.tensors().keys().take(10).cloned().collect();
    let format = if is_peft(sample_keys) {
        LoRAFormat::Peft
    } else {
        LoRAFormat::Kohya
    };

    let tensors = metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| {
            (
                name,
                TensorEntry {
                    dtype: info.dtype.into(),
                    shape: info.shape.clone(),
                },
            )
        })
        .collect();

    Ok((HeaderIndex { tensors, format }, metadata.metadata().clone()))
}
```

- [ ] **Step 5: Declare the module**

In `crates/inspector/src/lib.rs`, add `mod header;` alongside the other module declarations (near `mod parser;` and `mod weight;`, around line 22):

```rust
mod header;
mod parser;
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd crates/inspector && cargo test header:: 2>&1 | tail -30`
Expected: all 5 tests in `header::tests` pass.

- [ ] **Step 7: Commit**

```bash
git add crates/inspector/src/header.rs crates/inspector/src/lib.rs crates/inspector/src/weight.rs
git commit -m "Add header-only safetensors parsing to inspector crate"
```

---

### Task 2: Wire `HeaderIndex` into `LoRAFile` and `Metadata`

**Files:**
- Modify: `crates/inspector/src/metadata.rs:13-20`
- Modify: `crates/inspector/src/file.rs:1-45, 62-153`

**Interfaces:**
- Consumes: `header::{parse_header, HeaderIndex}` from Task 1.
- Produces: `Metadata::new_from_header_buffer(buffer: &[u8]) -> crate::Result<Metadata>`; `LoRAFile::new_from_header_buffer(buffer: &[u8], filename: &str) -> crate::Result<LoRAFile>`; `LoRAFile::tensor_info(&self) -> Vec<TensorInfo>` where `TensorInfo { pub name: String, pub dtype: String, pub shape: Vec<usize> }`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/inspector/src/file.rs` (after the existing `is_tensors_loaded` test, around line 611):

```rust
    fn header_only_bytes(buffer: &[u8]) -> &[u8] {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        &buffer[0..8 + header_len]
    }

    #[test]
    fn header_only_load_exposes_keys_without_weights() -> crate::Result<()> {
        let buffer = load_test_file()?;
        let lora_file =
            LoRAFile::new_from_header_buffer(header_only_bytes(&buffer), "boo.safetensors")?;

        assert!(!lora_file.is_tensors_loaded());
        assert!(!lora_file.keys().is_empty());
        assert!(!lora_file.base_names().is_empty());
        assert!(!lora_file.unet_keys().is_empty());
        assert!(!lora_file.tensor_info().is_empty());

        Ok(())
    }

    #[test]
    fn header_only_load_keys_match_full_buffer_load() -> crate::Result<()> {
        let buffer = load_test_file()?;
        let header_file =
            LoRAFile::new_from_header_buffer(header_only_bytes(&buffer), "boo.safetensors")?;
        let full_file = LoRAFile::new_from_buffer(&buffer, "boo.safetensors", &Device::Cpu);

        let mut header_keys = header_file.keys();
        let mut full_keys = full_file.keys();
        header_keys.sort();
        full_keys.sort();
        assert_eq!(header_keys, full_keys);

        assert_eq!(header_file.dims(), full_file.dims());
        assert_eq!(header_file.format(), full_file.format());
        assert_eq!(header_file.precision(), full_file.precision());

        Ok(())
    }

    #[test]
    fn header_only_load_rejects_truncated_buffer() {
        let result = LoRAFile::new_from_header_buffer(&[1_u8, 2, 3], "boo.safetensors");
        assert!(result.is_err());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crates/inspector && cargo test --lib file:: 2>&1 | tail -40`
Expected: compile error — `new_from_header_buffer` and `tensor_info` not found on `LoRAFile`.

- [ ] **Step 3: Add `Metadata::new_from_header_buffer`**

In `crates/inspector/src/metadata.rs`, add next to `new_from_buffer` (after line 20):

```rust
    /// Builds `Metadata` from just the safetensors header (see
    /// `crate::header::parse_header`), without requiring tensor payload bytes.
    pub fn new_from_header_buffer(buffer: &[u8]) -> crate::Result<Metadata> {
        let (_header, meta_map) = crate::header::parse_header(buffer)?;
        Ok(Metadata { metadata: meta_map })
    }
```

- [ ] **Step 4: Add `header` field and `new_from_header_buffer` to `LoRAFile`**

In `crates/inspector/src/file.rs`, change the struct (lines 22-29):

```rust
/// LoRA file buffer
#[derive(Debug)]
pub struct LoRAFile {
    filename: String,
    weights: Option<BufferedLoRAWeight>,
    scaled_weights: HashMap<String, candle_core::Tensor>,
    metadata: Option<Metadata>,
    header: Option<crate::header::HeaderIndex>,
}
```

Change `new_from_buffer` (lines 34-45) to also populate `header`:

```rust
impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: &str, device: &Device) -> LoRAFile {
        let metadata = Metadata::new_from_buffer(buffer).map_err(|e| e.to_string());
        let header = crate::header::parse_header(buffer).ok().map(|(h, _)| h);

        LoRAFile {
            filename: filename.to_string(),
            weights: BufferedLoRAWeight::new(buffer.to_vec(), device)
                .map(Some)
                .unwrap_or_else(|_| None),
            scaled_weights: HashMap::new(),
            metadata: metadata.map(Some).unwrap_or_else(|_| None),
            header,
        }
    }

    /// Builds a `LoRAFile` from just the safetensors header — no tensor payload
    /// bytes required. `is_tensors_loaded()` will be `false`; call
    /// `reload_weights` with the full file buffer before any weight-value
    /// operation (`scale_weight`, `effective_scale`, `rank_metrics`, ...).
    pub fn new_from_header_buffer(buffer: &[u8], filename: &str) -> Result<LoRAFile> {
        let (header, meta_map) = crate::header::parse_header(buffer)?;

        Ok(LoRAFile {
            filename: filename.to_string(),
            weights: None,
            scaled_weights: HashMap::new(),
            metadata: Some(Metadata { metadata: meta_map }),
            header: Some(header),
        })
    }
```

- [ ] **Step 5: Migrate key/shape/format accessors to read from `header`**

Replace the bodies of these existing methods in `crates/inspector/src/file.rs` (lines 66-126 and 108-112 and 148-153) — leave method signatures unchanged:

```rust
    pub fn unet_keys(&self) -> Vec<String> {
        self.header.as_ref().map(|h| h.unet_keys()).unwrap_or_default()
    }

    pub fn text_encoder_keys(&self) -> Vec<String> {
        self.header
            .as_ref()
            .map(|h| h.text_encoder_keys())
            .unwrap_or_default()
    }

    pub fn weight_keys(&self) -> Vec<String> {
        self.header.as_ref().map(|h| h.weight_keys()).unwrap_or_default()
    }

    pub fn alpha_keys(&self) -> Vec<String> {
        self.header.as_ref().map(|h| h.alpha_keys()).unwrap_or_default()
    }

    pub fn alphas(&self) -> HashSet<weight::Alpha> {
        self.weights
            .as_ref()
            .map(|weights| weights.alphas())
            .unwrap_or_default()
    }

    pub fn dims(&self) -> HashSet<usize> {
        self.header.as_ref().map(|h| h.dims()).unwrap_or_default()
    }

    pub fn precision(&self) -> Option<weight::DType> {
        self.header.as_ref().and_then(|h| h.precision())
    }

    pub fn keys(&self) -> Vec<String> {
        self.header.as_ref().map(|h| h.keys()).unwrap_or_default()
    }

    pub fn base_names(&self) -> Vec<String> {
        self.header.as_ref().map(|h| h.base_names()).unwrap_or_default()
    }
```

(`alphas()` is unchanged from today — shown above only for surrounding context; it must stay `self.weights`-based.)

And `format()` (lines 148-153):

```rust
    pub fn format(&self) -> weight::LoRAFormat {
        self.header
            .as_ref()
            .map(|h| h.format())
            .unwrap_or(weight::LoRAFormat::Kohya)
    }
```

- [ ] **Step 6: Add `TensorInfo` and `tensor_info()`**

Add near `LayerScale` at the top of `crates/inspector/src/file.rs` (after line 20):

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}
```

Add a method on `LoRAFile` (near `base_names`, e.g. after it):

```rust
    pub fn tensor_info(&self) -> Vec<TensorInfo> {
        self.header
            .as_ref()
            .map(|h| {
                h.tensor_info()
                    .into_iter()
                    .map(|(name, dtype, shape)| TensorInfo {
                        name,
                        dtype: dtype.to_string(),
                        shape,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `cd crates/inspector && cargo test 2>&1 | tail -60`
Expected: all tests pass, including every pre-existing `file.rs`/`weight.rs`/`metadata.rs` test (they must be unaffected — same underlying data, just recomputed from the header instead of `BufferedLoRAWeight`) and the new tests from Task 2 Step 1.

- [ ] **Step 8: Commit**

```bash
git add crates/inspector/src/file.rs crates/inspector/src/metadata.rs
git commit -m "Wire header-only parsing into LoRAFile; expose tensor_info"
```

---

### Task 3: wasm `LoraWorker` — header-only constructor + `tensor_info` binding

**Files:**
- Modify: `crates/lora-inspector-wasm/src/worker.rs:26-56, 320-344`

**Interfaces:**
- Consumes: `inspector::file::{LoRAFile, TensorInfo}`, `inspector::metadata::Metadata` from Task 2.
- Produces: `LoraWorker::new_from_buffer(buffer: &[u8], filename: &str)` (unchanged JS-visible name/signature, now expects header-only bytes), `LoraWorker::tensor_info(&self) -> Result<JsValue, serde_wasm_bindgen::Error>`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/lora-inspector-wasm/src/worker.rs` (after `base_names`, around line 439):

```rust
    #[wasm_bindgen_test]
    async fn header_only_constructor_exposes_keys_without_loading_weights() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let header_only = &buffer[0..8 + header_len];

        let worker = LoraWorker::new_from_buffer(header_only, "boo.safetensors")
            .expect("load from header buffer");

        assert!(!worker.is_tensors_loaded());
        assert!(!worker.keys().is_empty());
        assert!(!worker.base_names().is_empty());
        assert!(worker.tensor_info().is_ok());
    }

    #[wasm_bindgen_test]
    async fn reload_from_buffer_loads_weights_after_header_only_construction() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        let header_only = &buffer[0..8 + header_len];

        let mut worker = LoraWorker::new_from_buffer(header_only, "boo.safetensors")
            .expect("load from header buffer");
        assert!(!worker.is_tensors_loaded());

        worker.reload_from_buffer(&buffer);
        assert!(worker.is_tensors_loaded());

        assert!(worker
            .scale_weight("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k")
            .is_ok());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd crates/lora-inspector-wasm && wasm-pack test --headless --firefox -- --lib worker:: 2>&1 | tail -60`
Expected: `header_only_constructor_exposes_keys_without_loading_weights` fails — `worker.keys()` is empty and/or `tensor_info` method doesn't exist yet (compile error until Step 3/4 land).

- [ ] **Step 3: Change the constructor to parse header-only bytes**

In `crates/lora-inspector-wasm/src/worker.rs`, replace `new_from_buffer` (lines 28-36):

```rust
    #[wasm_bindgen(constructor)]
    /// `buffer` must contain at least the safetensors header: the 8-byte
    /// little-endian length prefix plus that many bytes of header JSON. Tensor
    /// payload bytes are not required here — call `reload_from_buffer` with the
    /// full file later to load weight values.
    pub fn new_from_buffer(buffer: &[u8], filename: &str) -> Result<LoraWorker, String> {
        console_error_panic_hook::set_once();
        let metadata = Metadata::new_from_header_buffer(buffer).map_err(|e| e.to_string());
        let file = LoRAFile::new_from_header_buffer(buffer, filename).map_err(|e| e.to_string());

        match (metadata, file) {
            (Ok(metadata), Ok(file)) => Ok(LoraWorker { metadata, file }),
            (Err(e), _) => Err(e),
            (_, Err(e)) => Err(e),
        }
    }
```

- [ ] **Step 4: Add the `tensor_info` binding**

Add near `effective_scales_all` in `crates/lora-inspector-wasm/src/worker.rs` (after line 343):

```rust
    pub fn tensor_info(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
        serde_wasm_bindgen::to_value(&self.file.tensor_info())
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd crates/lora-inspector-wasm && wasm-pack test --headless --firefox -- --lib worker:: 2>&1 | tail -60`
Expected: all `worker::tests` pass, including the two new ones and every pre-existing test (they all construct with the *full* `boo.safetensors` buffer, which still works since `new_from_header_buffer` only reads the header prefix out of whatever buffer it's handed).

- [ ] **Step 6: Commit**

```bash
git add crates/lora-inspector-wasm/src/worker.rs
git commit -m "wasm: parse only the safetensors header on initial LoRA upload"
```

---

### Task 4: worker.js — fetch only header bytes on upload, lazy-load weights on demand

**Files:**
- Modify: `crates/lora-inspector-wasm/assets/js/worker.js:13-97, 378-490, 613-675`

**Interfaces:**
- Consumes: the wasm `LoraWorker` constructor from Task 3 (now expects header-only bytes) and existing `reload_from_buffer`/`is_tensors_loaded`.
- Produces: `readHeaderBytes(file)`, `ensureHeaderLoaded(name)`, `ensureWeightsLoaded(name)`, `withWeights(name, fn)`.

- [ ] **Step 1: Add `readHeaderBytes` and split `ensureLoaded`**

Replace `reloadWorker` and `ensureLoaded` (`crates/lora-inspector-wasm/assets/js/worker.js:35-57`) with:

```js
async function readHeaderBytes(file) {
	const lengthBuffer = await file.slice(0, 8).arrayBuffer();
	const headerLen = Number(new DataView(lengthBuffer).getBigUint64(0, true));
	const headerBuffer = await file.slice(0, 8 + headerLen).arrayBuffer();
	return new Uint8Array(headerBuffer);
}

async function reloadWorker(file) {
	const buffer = await readHeaderBytes(file);
	addWorker(file.name, new LoraWorkerClass(buffer, file.name));
}

// Ensures the LoraWorker object exists (reconstructing it from just the file's
// header if it was GC'd), without requiring tensor weight bytes to be loaded.
async function ensureHeaderLoaded(name) {
	const loraWorker = loraWorkers.get(name);
	const file = files.get(name);

	if (!loraWorker) {
		if (!file) throw new Error(`Cannot reload worker ${name}: file not found`);
		await reloadWorker(file);
	}

	setIdleTimer(name);
}

// Ensures the LoraWorker object exists AND its tensor weight bytes are loaded
// (the full file, via readFile) — required before any weight-value operation
// (scale_weight, norms, rank_metrics, effective_scale, alphas, ...).
async function ensureWeightsLoaded(name) {
	await ensureHeaderLoaded(name);

	const loraWorker = getWorker(name);
	if (!loraWorker.is_tensors_loaded()) {
		const file = files.get(name);
		if (!file)
			throw new Error(`Cannot reload weights for ${name}: file not found`);
		const buffer = await readFile(file);
		loraWorker.reload_from_buffer(buffer);
	}
}
```

- [ ] **Step 2: Add `withWeights` alongside `withWorker`**

Replace `withWorker` (`crates/lora-inspector-wasm/assets/js/worker.js:59-62`) with:

```js
async function withWorker(name, fn) {
	await ensureHeaderLoaded(name);
	return fn(getWorker(name));
}

async function withWeights(name, fn) {
	await ensureWeightsLoaded(name);
	return fn(getWorker(name));
}
```

- [ ] **Step 3: Use header-only bytes for the initial upload**

In `loadWorker` (`crates/lora-inspector-wasm/assets/js/worker.js:392-393`), change:

```js
async function loadWorker(file, worker) {
	const buffer = await readFile(file);
```

to:

```js
async function loadWorker(file, worker) {
	const buffer = await readHeaderBytes(file);
```

(the rest of `loadWorker` is unchanged — it already just passes `buffer` to `new worker(buffer, file.name)`).

- [ ] **Step 4: Repoint weight-value handlers at `withWeights`**

In `crates/lora-inspector-wasm/assets/js/worker.js`, change these functions to call `withWeights` instead of `withWorker` (everything else — `getNetworkModule`, `getWeightKeys`, `getKeys`, `getTextEncoderKeys`, `getUnetKeys`, `getBaseNames`, `getAlphaKeys`, `getWeightDecomposition`, `getRankStabilized`, `getDims`, `getPrecision`, `getNetworkArgs`, `getNetworkType` — stays on `withWorker`, since it's all header/metadata-derived now):

```js
async function getAlphas(e) {
	return withWeights(e.data.name, (w) =>
		Array.from(w.alphas()).sort((a, b) => a - b),
	);
}
```

```js
async function getDoraScales(e) {
	return withWeights(e.data.name, (w) =>
		Array.from(w.doraScales()).sort((a, b) => a > b),
	);
}
```

```js
async function getEffectiveScale(e) {
	const baseName = e.data.baseName;
	return withWeights(e.data.name, (w) => w.effective_scale(baseName));
}
```

```js
async function getEffectiveScalesAll(e) {
	return withWeights(e.data.name, (w) => w.effective_scales_all());
}
```

```js
async function getRankMetrics(e) {
	const baseName = e.data.baseName;
	return withWeights(e.data.name, (w) => {
		try {
			const metrics = w.rank_metrics(baseName);
			return [metrics, undefined];
		} catch (err) {
			console.error(`rank_metrics error for ${baseName}:`, err);
			return [undefined, err];
		}
	});
}
```

And `getNorms` (`crates/lora-inspector-wasm/assets/js/worker.js:469-490`) — change its `withWorker` call to `withWeights`:

```js
async function getNorms(e) {
	const baseName = e.data.baseName;
	return withWeights(e.data.name, (w) => {
		try {
			return [
				w.norms(baseName, [
					"l1_norm",
					"l2_norm",
					"matrix_norm",
					"max",
					"min",
					"std_dev",
					"median",
				]),
				undefined,
			];
		} catch (err) {
			console.error(err);
			return [undefined, err];
		}
	});
}
```

And `getL2Norms` (`crates/lora-inspector-wasm/assets/js/worker.js:492-497`) — change its direct `ensureLoaded` call to `ensureWeightsLoaded`:

```js
async function getL2Norms(e) {
	await ensureWeightsLoaded(e.data.name);
	clearIdleTimer(e.data.name); // prevent idle timeout from firing during long batch loop
	const loraWorker = getWorker(e.data.name); // held across loop — ensureWeightsLoaded already called
```

- [ ] **Step 5: Add a `tensor_info` message handler**

In the `self.onmessage` dispatch (`crates/lora-inspector-wasm/assets/js/worker.js`, alongside the `"weight_keys"` branch around line 193-195), add:

```js
		} else if (e.data.messageType === "tensor_info") {
			getTensorInfo(e).then((tensorInfo) => {
				if (e.data.reply) {
					self.postMessage({
						messageType: "tensor_info",
						tensorInfo,
					});
				}
			});
```

And add the handler function near `getBaseNames` (after line 467):

```js
async function getTensorInfo(e) {
	return withWorker(e.data.name, (w) => w.tensor_info());
}
```

- [ ] **Step 6: Manual verification**

This task has no automated JS test harness in this repo (worker.js has no unit tests). Verify manually:

1. Start the dev server (`cd crates/lora-inspector-wasm && yarn dev`) and rebuild wasm (`make build-wasm-simd DEV=1` from repo root, or the project's usual dev build command).
2. Upload a normal-sized LoRA (e.g. `boo.safetensors`) through the browser UI — confirm metadata, keys, base names, and norms all still work exactly as before.
3. If a large (2GB+) file is available, upload it and confirm the upload itself no longer throws `invalid malloc request` and metadata/keys/tensor list appear. Confirm that subsequently requesting norms/scale-weight on that large file is where the existing (known, out-of-scope) limitation would still surface — do not attempt to fix that in this task.

- [ ] **Step 7: Commit**

```bash
git add crates/lora-inspector-wasm/assets/js/worker.js
git commit -m "worker.js: load only the safetensors header on upload, lazy-load weights on demand"
```

---

## Final Verification

- [ ] Run `cd crates/inspector && cargo test` — all pass.
- [ ] Run `cd crates/lora-inspector-wasm && wasm-pack test --headless --firefox` — all pass.
- [ ] Run `make fmt` from the repo root — no diffs left uncommitted.
- [ ] Manually upload the originally-failing file (`/mnt/900/lora/flux.2-dev/Flux_2-Turbo-LoRA_comfyui.safetensors`) through the browser UI and confirm the upload succeeds and tensor names/shapes are visible, per Task 4 Step 6.
