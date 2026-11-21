//! Shared synthetic-safetensors fixture for the CLI integration tests.
//!
//! The file is hand-rolled (no `safetensors` writer dependency): an 8-byte
//! little-endian header length, then a JSON header with `data_offsets`, then
//! little-endian `f32` payload bytes. `Fixture` removes the temp file on drop.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Handle to a temporary synthetic safetensors file.
///
/// The file is removed when the handle is dropped.
pub struct Fixture {
    path: PathBuf,
}

impl Fixture {
    /// Write the synthetic fixture to a fresh temp file.
    pub fn write() -> Fixture {
        let path = temp_path();
        fs::write(&path, fixture_bytes()).expect("write synthetic fixture");
        Fixture { path }
    }

    /// Write the block-outlier fixture: one heavy block among light blocks,
    /// where the block-mean rule and the layer-median rule disagree.
    pub fn write_outlier() -> Fixture {
        let path = temp_path();
        fs::write(&path, outlier_fixture_bytes()).expect("write outlier fixture");
        Fixture { path }
    }

    /// Path of the written file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file's bytes, re-read from disk so the test exercises the same load
    /// path as `inspect FILE`.
    pub fn bytes(&self) -> Vec<u8> {
        fs::read(&self.path).expect("read synthetic fixture")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn temp_path() -> PathBuf {
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "lora-inspector-report-fixture-{}-{unique}.safetensors",
        std::process::id()
    ))
}

struct Tensor {
    name: String,
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// A tensor whose only non-zero entry is element 0.
///
/// An empty shape is a scalar (`product` is 1), so `first` becomes the single
/// value. For the 2-D `up`/`down` matrices it becomes the `[0][0]` entry, with
/// every other entry zero.
fn tensor(name: &str, shape: &[usize], first: f32) -> Tensor {
    let len = shape.iter().product::<usize>();
    let mut data = vec![0.0f32; len];
    if !data.is_empty() {
        data[0] = first;
    }
    Tensor {
        name: name.to_string(),
        shape: shape.to_vec(),
        data,
    }
}

/// Add a plain 2-D LoRA `up`/`down`/`alpha` layer.
///
/// `up[0][0] == 1` and `down[0][0] == down_value`, so the layer's effective
/// scale is `alpha / rank * down_value` — a round number when `alpha == rank`.
fn push_lora(tensors: &mut Vec<Tensor>, base: &str, rank: usize, alpha: f32, down_value: f32) {
    tensors.push(tensor(
        &format!("{base}.lora_up.weight"),
        &[rank, rank],
        1.0,
    ));
    tensors.push(tensor(
        &format!("{base}.lora_down.weight"),
        &[rank, rank],
        down_value,
    ));
    tensors.push(tensor(&format!("{base}.alpha"), &[], alpha));
}

/// Add a LoHA layer. The cheap Gram path has no plain 2-D up/down pair here,
/// so `low_rank_effective_scale` returns `None` (skipped, never materialized).
fn push_hada(tensors: &mut Vec<Tensor>, base: &str) {
    tensors.push(tensor(&format!("{base}.hada_w1_a"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.hada_w1_b"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.alpha"), &[], 4.0));
}

fn fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();

    // Two down-block layers (IN00): effective scales 2.0 and 10.0, mean 6.0.
    push_lora(&mut tensors, "down_blocks.0.attn.proj_in", 4, 4.0, 2.0);
    push_lora(&mut tensors, "down_blocks.0.attn.proj_out", 4, 4.0, 10.0);

    // Two text-encoder layers (TE00): effective scales 1.0 and 3.0, mean 2.0.
    push_lora(
        &mut tensors,
        "text_model.encoder.layers.0.self_attn.q_proj",
        8,
        8.0,
        1.0,
    );
    push_lora(
        &mut tensors,
        "text_model.encoder.layers.0.self_attn.k_proj",
        8,
        8.0,
        3.0,
    );

    // A LoHA layer the cheap Gram path declines.
    push_hada(&mut tensors, "down_blocks.1.attn.proj_in");

    // A plain LoRA layer whose key matches no family pattern.
    push_lora(&mut tensors, "txtfusion_projector", 4, 4.0, 2.5);

    to_safetensors(tensors)
}

/// One heavy block (mean 4.0) among light blocks (mean 2.0 each).
fn outlier_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();

    // IN00: layers 4.0 and 4.0 -> mean 4.0.
    push_lora(&mut tensors, "down_blocks.0.attn.proj_in", 4, 4.0, 4.0);
    push_lora(&mut tensors, "down_blocks.0.attn.proj_out", 4, 4.0, 4.0);

    // IN01: layers 3.0 and 1.0 -> mean 2.0.
    push_lora(&mut tensors, "down_blocks.1.attn.proj_in", 4, 4.0, 3.0);
    push_lora(&mut tensors, "down_blocks.1.attn.proj_out", 4, 4.0, 1.0);

    // IN02: layers 3.0 and 1.0 -> mean 2.0.
    push_lora(&mut tensors, "down_blocks.2.attn.proj_in", 4, 4.0, 3.0);
    push_lora(&mut tensors, "down_blocks.2.attn.proj_out", 4, 4.0, 1.0);

    to_safetensors(tensors)
}

/// Serialize tensors into the safetensors layout: 8-byte little-endian header
/// length, JSON header with `data_offsets`, then little-endian `f32` payload.
fn to_safetensors(tensors: Vec<Tensor>) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for t in &tensors {
        let end = offset + t.data.len() * 4;
        header.insert(
            t.name.clone(),
            serde_json::json!({
                "dtype": "F32",
                "shape": t.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }

    let header_json = serde_json::to_string(&header).expect("serialize fixture header");
    let mut bytes = Vec::with_capacity(8 + header_json.len() + offset);
    bytes.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(header_json.as_bytes());
    for t in &tensors {
        for value in &t.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}
