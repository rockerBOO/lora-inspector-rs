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

    /// Write the zero-median fixture: one positive block mean and two zero
    /// block means, exercising the floored `ratio_to_median` denominator.
    pub fn write_zero_median() -> Fixture {
        let path = temp_path();
        fs::write(&path, zero_median_fixture_bytes()).expect("write zero-median fixture");
        Fixture { path }
    }

    /// A file whose layers use two different structural formats, with no
    /// `__metadata__` at all: two plain LoRA layers and one LoHa layer.
    pub fn write_mixed_structure() -> Fixture {
        let path = temp_path();
        fs::write(&path, mixed_structure_fixture_bytes()).expect("write mixed fixture");
        Fixture { path }
    }

    /// A dense, factor-free file: the shape the `*-proj.safetensors` artifacts
    /// in the real training tree present (two unmarked tensors, no metadata).
    pub fn write_dense_only() -> Fixture {
        let path = temp_path();
        fs::write(&path, dense_only_fixture_bytes()).expect("write dense fixture");
        Fixture { path }
    }

    /// LoHa keys under metadata that declares `networks.lora`. The declared type
    /// and the structural type disagree; nothing in the pre-existing output
    /// reveals it.
    pub fn write_declared_lora_actual_loha() -> Fixture {
        let path = temp_path();
        fs::write(&path, declared_mismatch_fixture_bytes()).expect("write mismatch fixture");
        Fixture { path }
    }

    /// Write a fixture whose only key matches no block family, so `blocks` is
    /// an empty array rather than `null`.
    pub fn write_ungroupable() -> Fixture {
        let path = temp_path();
        fs::write(&path, ungroupable_fixture_bytes()).expect("write ungroupable fixture");
        Fixture { path }
    }

    /// Write a fixture carrying a `__metadata__` map, so `--metadata` can be
    /// exercised through the CLI.
    pub fn write_metadata() -> Fixture {
        let path = temp_path();
        fs::write(&path, metadata_fixture_bytes()).expect("write metadata fixture");
        Fixture { path }
    }

    /// Write a conv-LoRA fixture whose up/down factors are not 2-D, so the
    /// cheap Gram path declines it and `--slow` materializes it.
    pub fn write_conv() -> Fixture {
        let path = temp_path();
        fs::write(&path, conv_fixture_bytes()).expect("write conv fixture");
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
///
/// Only the `w1` factors are written, so this layer is also *unscalable* even by
/// the materialising path: `scale_hada_weight` needs `w2` as well. That is what
/// makes it a stable "declined" fixture. Use [`push_hada_complete`] when the
/// layer must actually be reconstructible.
fn push_hada(tensors: &mut Vec<Tensor>, base: &str) {
    tensors.push(tensor(&format!("{base}.hada_w1_a"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.hada_w1_b"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.alpha"), &[], 4.0));
}

/// Add a complete LoHa layer: both `w1` and `w2` factor pairs plus alpha, so
/// `scale_hada_weight` can rebuild it. Still not a low-rank up/down pair, so
/// `low_rank_effective_scale` declines it and the materialising path is used.
fn push_hada_complete(tensors: &mut Vec<Tensor>, base: &str) {
    tensors.push(tensor(&format!("{base}.hada_w1_a"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.hada_w1_b"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.hada_w2_a"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.hada_w2_b"), &[4, 4], 1.0));
    tensors.push(tensor(&format!("{base}.alpha"), &[], 4.0));
}

/// Add a conv-LoRA layer with a real 2×2 kernel. The cheap Gram path declines
/// the non-2-D factors; `--slow` materializes the convolution instead.
fn push_conv_lora(tensors: &mut Vec<Tensor>, base: &str) {
    tensors.push(tensor(
        &format!("{base}.lora_down.weight"),
        &[2, 2, 2, 2],
        1.0,
    ));
    tensors.push(tensor(
        &format!("{base}.lora_up.weight"),
        &[2, 2, 1, 1],
        1.0,
    ));
    tensors.push(tensor(&format!("{base}.alpha"), &[], 2.0));
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

    // A noisy f32 scale (0.1 + 0.2) so the rounding guard has real float noise
    // to catch: raw ~0.30000001192092896, rounded to 0.3.
    push_lora(
        &mut tensors,
        "down_blocks.2.attn.proj_in",
        4,
        4.0,
        0.1_f32 + 0.2,
    );

    // A plain LoRA layer whose key matches no family pattern.
    push_lora(&mut tensors, "txtfusion_projector", 4, 4.0, 2.5);

    to_safetensors(tensors)
}

/// One block mean of 1.0 and two zero block means, so the median block mean is
/// exactly 0.0.
fn zero_median_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();

    push_lora(&mut tensors, "down_blocks.0.attn.proj_in", 4, 4.0, 1.0);
    push_lora(&mut tensors, "down_blocks.1.attn.proj_in", 4, 4.0, 0.0);
    push_lora(&mut tensors, "down_blocks.2.attn.proj_in", 4, 4.0, 0.0);

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

/// A fixture whose only key matches no block family (`txtfusion_projector`
/// lands in `BlockGroup::Other`, so `blocks` is empty and `ungrouped` counts it).
fn ungroupable_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_lora(&mut tensors, "txtfusion_projector", 4, 4.0, 2.5);
    to_safetensors(tensors)
}

/// Two plain LoRA layers and one LoHa layer, no `__metadata__`.
fn mixed_structure_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_lora(&mut tensors, "down_blocks.0.attn.proj_in", 4, 4.0, 1.0);
    push_lora(&mut tensors, "down_blocks.0.attn.proj_out", 4, 4.0, 1.0);
    push_hada(&mut tensors, "down_blocks.1.attn.proj_in");
    to_safetensors(tensors)
}

/// Two unmarked tensors with no low-rank factors anywhere.
fn dense_only_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    tensors.push(tensor("0.weight", &[2, 3, 4], 1.0));
    tensors.push(tensor("2.weight", &[2, 3, 4], 1.0));
    to_safetensors(tensors)
}

/// LoHa keys carrying metadata that declares a plain Kohya LoRA.
fn declared_mismatch_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_hada_complete(&mut tensors, "down_blocks.0.attn.proj_in");
    push_hada_complete(&mut tensors, "down_blocks.0.attn.proj_out");
    to_safetensors_with_metadata(
        tensors,
        serde_json::json!({
            "ss_network_module": "networks.lora",
            "ss_network_dim": "4",
            "ss_network_alpha": "4.0",
        }),
    )
}

/// A fixture carrying a `__metadata__` map so the opt-in metadata path is real.
fn metadata_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_lora(&mut tensors, "down_blocks.0.attn.proj_in", 4, 4.0, 1.0);
    to_safetensors_with_metadata(
        tensors,
        serde_json::json!({
            "modelspec.architecture": "flux",
            "modelspec.author": "fixture",
            "modelspec.date": "2026-11-21",
        }),
    )
}

/// A single conv layer for exercising the `--slow` materialising fallback.
fn conv_fixture_bytes() -> Vec<u8> {
    let mut tensors = Vec::new();
    push_conv_lora(&mut tensors, "down_blocks.1.resnets.0.conv1");
    to_safetensors(tensors)
}

/// Serialize tensors into the safetensors layout: 8-byte little-endian header
/// length, JSON header with `data_offsets`, then little-endian `f32` payload.
fn to_safetensors(tensors: Vec<Tensor>) -> Vec<u8> {
    to_safetensors_with_metadata(tensors, serde_json::Value::Null)
}

/// [`to_safetensors`] with an optional `__metadata__` map (omit with `Null`).
fn to_safetensors_with_metadata(tensors: Vec<Tensor>, metadata: serde_json::Value) -> Vec<u8> {
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
    if !metadata.is_null() {
        header.insert("__metadata__".to_string(), metadata);
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
