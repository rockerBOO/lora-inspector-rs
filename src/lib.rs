use candle_core::{DType, Device, Tensor};
use safetensors::tensor::TensorView;

use safetensors::{tensor::TensorInfo, Dtype, SafeTensorError, SafeTensors};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use wasm_bindgen::prelude::*;
use web_sys::console;
use web_sys::js_sys;

/// Helper struct used only for serialization deserialization
#[derive(Serialize, Deserialize)]
struct HashMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "__metadata__")]
    metadata: Option<HashMap<String, String>>,
    #[serde(flatten)]
    tensors: HashMap<String, TensorInfo>,
}

pub fn get_metadata_from_buffer(buffer: &[u8]) -> Result<HashMap<String, String>, SafeTensorError> {
    let (_size, metadata) = SafeTensors::read_metadata(buffer)?;

    metadata.metadata().to_owned().ok_or_else(|| {
        SafeTensorError::IoError(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid metadata",
        ))
    })
}

pub fn compile_metadata(
    buffer: &[u8],
) -> Result<(HashMap<String, String>, f64, f64), Box<dyn std::error::Error>> {
    let metadata = get_metadata_from_buffer(buffer)?;
    let tensors = candle_core::safetensors::load_buffer(buffer, &Device::Cpu)?;

    let average_magnitude = _get_average_magnitude(&tensors)?;
    let average_strength = _get_average_strength(&tensors)?;

    Ok((metadata, average_magnitude, average_strength))
}

#[wasm_bindgen]
pub fn get_metadata(buffer: &[u8]) -> JsValue {
    console_error_panic_hook::set_once();

    match compile_metadata(buffer) {
        Ok(v) => match serde_wasm_bindgen::to_value(&v) {
            Ok(metadata) => metadata,
            Err(e) => {
                console_log(&JsValue::from(format!("Error parsing {e}").as_str()));
                todo!()
            }
        },
        Err(e) => {
            console_log(&JsValue::from(
                format!("Error loading metadata {e}").as_str(),
            ));
            todo!()
        }
    }

    // let mut result: HashMap<String, JsValue> = HashMap::new();
    //
    // result.insert("metadata".to_string(), metadata);
    // result.insert("average_strength".to_string(), average_strength);
    // result.insert("average_magnitude".to_string(), average_magnitude);
    //
    // match serde_wasm_bindgen::to_value(&result) {
    //     Ok(r) => r,
    //     Err(e) => {
    //         console_log(&JsValue::from(
    //             format!("Could not load result {e}").as_str(),
    //         ));
    //
    //         todo!()
    //     }
    // }
}

pub fn into_dtype(dtype: Dtype) -> DType {
    match dtype {
        Dtype::F32 => DType::F32,
        // Dtype::F16 => DType::F16,
        // Dtype::BOOL => DType::BOOL,
        // Dtype::U8 => DType::U8,
        // Dtype::I8 => DType::I8,
        // Dtype::I16 => DType::I16,
        // Dtype::U16 => DType::U16,
        // Dtype::BF16 => DType::,
        // Dtype::I32 => DType::F64,
        // Dtype::U32 => DType::F64,
        Dtype::F64 => DType::F64,
        Dtype::I64 => DType::I64,
        // Dtype::U64 => DType::U64,
        _ => todo!(),
    }
}

pub fn console_log(s: &JsValue) {
    let array = js_sys::Array::new();
    array.push(s);
    console::log(&array);
}

pub fn from_tensor_views(tensors: Vec<TensorView>) -> Vec<Tensor> {
    tensors
        .iter()
        .filter_map(|t| from_tensor_view(t).ok())
        .collect()
}

pub fn from_tensor_view(t: &TensorView) -> Result<Tensor, candle_core::Error> {
    Tensor::from_raw_buffer(t.data(), into_dtype(t.dtype()), t.shape(), &Device::Cpu)
}

pub fn to_tensor_magnitude(t: &Tensor) -> Result<Tensor, candle_core::Error> {
    t.to_dtype(DType::F64)?.powf(2.)?.sum_all()?.sqrt()
}

pub fn to_tensor_strength(t: &Tensor) -> Result<f64, candle_core::Error> {
    Ok(t.to_dtype(DType::F64)?
        .abs()?
        .sum_all()?
        .to_scalar()
        .unwrap_or(0.)
        / t.elem_count() as f64)
}

pub fn _get_average_magnitude(
    tensors: &HashMap<String, Tensor>,
) -> Result<f64, candle_core::Error> {
    let r = tensors
        .iter()
        .filter(|(k, _t)| k.contains("weight"))
        .filter_map(|(_k, t)| to_tensor_magnitude(t).ok())
        .map(|t| t.to_scalar().unwrap_or(0.));

    let count = r.clone().count() as f64;
    Ok(r.sum::<f64>() / count)
    // average_magnitude(.clone().collect())
}

// pub fn _get_average_strength(buffer: &[u8]) -> Result<f64, candle_core::Error> {
pub fn _get_average_strength(tensors: &HashMap<String, Tensor>) -> Result<f64, candle_core::Error> {
    let r = tensors
        .iter()
        .filter(|(k, _t)| k.contains("weight"))
        .filter_map(|(_k, t)| to_tensor_strength(t).ok());

    let count = r.clone().count() as f64;
    Ok(r.sum::<f64>() / count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_get_average_magnitude() {
        let device = candle_core::Device::Cpu;
        let data: Vec<f32> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];
        let a = Tensor::from_vec(data, (1, 1, 4, 4), &device);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("one".to_string(), a.unwrap());
        let result = _get_average_magnitude(&tensors);

        assert_eq!(result.unwrap(), 3.7416573867739413);
    }

    #[test]
    fn test_get_average_strength() {
        let device = candle_core::Device::Cpu;
        let data: Vec<f32> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];
        let a = Tensor::from_vec(data, (1, 1, 4, 4), &device);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("one".to_string(), a.unwrap());
        let result = _get_average_strength(&tensors);

        assert_eq!(result.unwrap(), 0.875);
    }
}

#[wasm_bindgen]
pub fn get_average_magnitude(buffer: &[u8]) -> JsValue {
    console_error_panic_hook::set_once();

    match candle_core::safetensors::load_buffer(buffer, &Device::Cpu) {
        Ok(tensors) => match _get_average_magnitude(&tensors) {
            Ok(v) => match serde_wasm_bindgen::to_value(&v) {
                Ok(v) => v,
                Err(_) => todo!(),
            },
            Err(e) => {
                console_log(&JsValue::from(format!("Error {e}").as_str()));
                todo!()
            }
        },
        Err(_) => todo!(),
    }
}

#[wasm_bindgen]
pub fn get_average_strength(buffer: &[u8]) -> JsValue {
    console_error_panic_hook::set_once();

    match candle_core::safetensors::load_buffer(buffer, &Device::Cpu) {
        Ok(tensors) => match _get_average_strength(&tensors) {
            Ok(v) => match serde_wasm_bindgen::to_value(&v) {
                Ok(v) => v,
                Err(_) => todo!(),
            },
            Err(e) => {
                console_log(&JsValue::from(format!("Error {e}").as_str()));
                todo!()
            }
        },
        Err(_) => todo!(),
    }
}
