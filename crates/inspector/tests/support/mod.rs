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
