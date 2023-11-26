use candle_core::{DType, Device, Tensor};
use regex::Regex;

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
) -> Result<HashMap<String, String>, Box<dyn std::error::Error>> {
    let metadata = get_metadata_from_buffer(buffer)?;
    // let tensors = candle_core::safetensors::load_buffer(buffer, &Device::Cpu)?;

    Ok(metadata)
    // let average_magnitude = _get_average_magnitude(&tensors)?;
    // let average_strength = _get_average_strength(&tensors)?;
    //
    // Ok((metadata, average_magnitude, average_strength))
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
        Dtype::U8 => DType::U8,
        // Dtype::I8 => DType::I8,
        // Dtype::I16 => DType::I16,
        // Dtype::U16 => DType::U16,
        // Dtype::BF16 => DType::,
        Dtype::I32 => DType::I64,
        Dtype::U32 => DType::U32,
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

pub fn to_tensor_magnitude(t: &Tensor) -> Result<f64, candle_core::Error> {
    Ok(t.to_dtype(DType::F64)?
        .powf(2.)?
        .sum_all()?
        .sqrt()?
        .to_scalar()
        .unwrap_or(0.))
}

pub fn to_tensor_strength(t: &Tensor) -> Result<f64, candle_core::Error> {
    Ok(dbg!(t.to_dtype(DType::F64)?.abs()?.sum_all()?)
        .to_scalar()
        .unwrap_or(0.)
        / t.elem_count() as f64)
}

pub fn _get_average_magnitude(
    tensors: &HashMap<String, Tensor>,
) -> Result<f64, candle_core::Error> {
    Ok(dbg!(tensors
        .iter()
        .filter(|(k, _t)| k.contains("weight"))
        .filter_map(|(_k, t)| to_tensor_magnitude(t).ok())
        .sum::<f64>())
        / tensors.len() as f64)
}

// pub fn _get_average_strength(buffer: &[u8]) -> Result<f64, candle_core::Error> {
pub fn _get_average_strength(tensors: &HashMap<String, Tensor>) -> Result<f64, candle_core::Error> {
    Ok(tensors
        .iter()
        .filter(|(k, _t)| k.contains("weight"))
        .filter_map(|(_k, t)| dbg!(to_tensor_strength(t).ok()))
        .sum::<f64>()
        / tensors.len() as f64)

    // let count = r.clone().count() as f64;
    // Ok(r.sum::<f64>() / count)
}

pub fn weights_by_block(
    tensors: HashMap<String, Tensor>,
) -> (HashMap<String, Vec<Tensor>>, HashMap<String, Vec<Tensor>>) {
    let mut text_encoder: HashMap<String, Vec<Tensor>> = HashMap::new();
    let mut text_encoder_alphas: HashMap<String, f64> = HashMap::new();
    let mut unet: HashMap<String, Vec<Tensor>> = HashMap::new();
    let mut unet_alphas: HashMap<String, f64> = HashMap::new();

    let text_encoder_re = Regex::new(r".*layers_(?P<layer>\d+).*").unwrap();
    // let unet_re = Regex::new(
    //     r".*(?P<block_type>up|down|mid)_blocks_.*(?P<block_id>\d+)_attentions_(?P<attn>\d+)_transformer_blocks_(?P<trans>\d+).*",
    // )
    // .unwrap();
    let unet_re =
        Regex::new(r".*(?P<block_type>up|down|mid)_blocks_.*(?P<block_id>\d+).*").unwrap();

    for (key, tensor) in tensors {
        let period_idx = key.find('.').unwrap();

        let name = &key[0..period_idx];

        if key.contains("text_model") {
            // let caps = text_encoder_re.captures(key).unwrap();
            if let Some(caps) = text_encoder_re.captures(name) {
                let block_name = format!(
                    "layer_{:02}",
                    caps[1].to_owned().to_string().parse().unwrap_or(0)
                );
                if key.contains(".alpha") {
                    text_encoder_alphas
                        .entry(block_name)
                        .or_insert(tensor.to_scalar::<f64>().unwrap_or(0.));
                } else {
                    text_encoder
                        .entry(block_name)
                        .and_modify(|v| v.push(tensor.to_owned()))
                        .or_insert(vec![tensor.to_owned()]);
                }
            }
        } else if key.contains("unet") {
            if let Some(caps) = unet_re.captures(name) {
                let block_name = format!(
                    "{:}_{:}",
                    caps["block_type"].to_string(),
                    caps["block_id"].to_string().parse::<u8>().unwrap()
                );

                // + caps["attn"].to_string().parse::<u8>().unwrap()
                // + caps["trans"].to_string().parse::<u8>().unwrap()

                if key.contains(".alpha") {
                    unet_alphas
                        .entry(block_name)
                        .or_insert(tensor.to_scalar::<f64>().unwrap_or(0.));
                } else {
                    unet.entry(block_name)
                        .and_modify(|v| v.push(tensor.to_owned()))
                        .or_insert(vec![tensor.to_owned()]);
                }
            }
        }
    }

    (text_encoder, unet)
}

#[wasm_bindgen]
pub fn get_average_magnitude_by_block(buffer: &[u8]) -> JsValue {
    let (text_encoder, unet) = match candle_core::safetensors::load_buffer(buffer, &Device::Cpu) {
        Ok(tensors) => weights_by_block(tensors),
        Err(e) => {
            console_log(&JsValue::from(format!("Error {e}").as_str()));
            todo!()
        }
    };

    let mut unet_weights: HashMap<String, f64> = HashMap::new();
    for (block, weights) in unet {
        unet_weights.insert(
            block,
            weights
                .iter()
                .filter_map(|tensor| to_tensor_magnitude(tensor).ok())
                .sum::<f64>()
                / (weights.len() as f64),
        );
    }

    let mut text_encoder_weights: HashMap<String, f64> = HashMap::new();
    for (block, weights) in text_encoder {
        text_encoder_weights.insert(
            block,
            weights
                .iter()
                .filter_map(|tensor| to_tensor_magnitude(tensor).ok())
                .sum::<f64>()
                / (weights.len() as f64),
        );
    }

    let mut weights: HashMap<String, HashMap<String, f64>> = HashMap::new();

    weights.insert("text_encoder".to_string(), text_encoder_weights);
    weights.insert("unet".to_string(), unet_weights);

    match serde_wasm_bindgen::to_value(&weights) {
        Ok(v) => v,
        Err(e) => {
            console_log(&JsValue::from(format!("Error {e}").as_str()));
            todo!()
        }
    }
}

#[wasm_bindgen]
pub fn get_average_strength_by_block(buffer: &[u8]) -> JsValue {
    let (text_encoder, unet) = match candle_core::safetensors::load_buffer(buffer, &Device::Cpu) {
        Ok(tensors) => weights_by_block(tensors),
        Err(e) => {
            console_log(&JsValue::from(format!("Error {e}").as_str()));
            todo!()
        }
    };

    let mut unet_weights: HashMap<String, f64> = HashMap::new();
    for (block, weights) in unet {
        unet_weights.insert(
            block,
            weights
                .iter()
                .filter_map(|tensor| to_tensor_strength(tensor).ok())
                .sum::<f64>()
                / (weights.len() as f64),
        );
    }

    let mut text_encoder_weights: HashMap<String, f64> = HashMap::new();
    for (block, weights) in text_encoder {
        text_encoder_weights.insert(
            block,
            weights
                .iter()
                .filter_map(|tensor| to_tensor_strength(tensor).ok())
                .sum::<f64>()
                / (weights.len() as f64),
        );
    }

    let mut weights: HashMap<String, HashMap<String, f64>> = HashMap::new();

    weights.insert("text_encoder".to_string(), text_encoder_weights);
    weights.insert("unet".to_string(), unet_weights);

    match serde_wasm_bindgen::to_value(&weights) {
        Ok(v) => v,
        Err(e) => {
            console_log(&JsValue::from(format!("Error {e}").as_str()));
            todo!()
        }
    }
}

#[cfg(test)]
mod tests {
    // use std::fs::File;

    use super::*;
    use candle_core::Tensor;
    // use memmap2::MmapOptions;

    #[test]
    fn test_weights_by_block() {
        // let device = candle_core::Device::Cpu;
        //
        // let data: Vec<f32> = vec![
        //     1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        // ];

        let filename = "/mnt/900/training/sets/pov-2023-11-25-025803-4cf6f9ce/pov-2023-11-25-025803-4cf6f9ce.safetensors";
        // let file = File::open(filename).unwrap();
        // let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
        //
        // // let a = Tensor::from_vec(data, (1, 1, 4, 4), &device);
        // // let a = Tensor::from_raw_buffer(buffer, DType::F32, );
        let tensors = candle_core::safetensors::load(filename, &Device::Cpu).unwrap();
        // let mut tensors: HashMap<String, Tensor> = HashMap::new();
        // tensors.insert("one".to_string(), a.unwrap());

        let _result = weights_by_block(tensors);

        // dbg!(result);

        // assert_!(result.unwrap(), 1.);
    }

    #[test]
    fn test_to_tensor_magnitude() {
        let device = candle_core::Device::Cpu;
        let data: Vec<f32> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 1, 4, 4), &device).unwrap();

        assert_eq!(to_tensor_magnitude(&tensor).unwrap(), 3.7416573867739413);
    }

    #[test]
    fn test_to_tensor_strength() {
        let device = candle_core::Device::Cpu;
        let data: Vec<f32> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];

        let tensor = Tensor::from_vec(data, (1, 1, 4, 4), &device).unwrap();

        assert_eq!(to_tensor_strength(&tensor).unwrap(), 0.875);
    }
    #[test]
    fn test_get_average_magnitude() {
        let device = candle_core::Device::Cpu;
        let data: Vec<f32> = vec![
            1., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        ];
        let a = Tensor::from_vec(data, (1, 1, 4, 4), &device);
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        tensors.insert("weight".to_string(), a.unwrap());
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
        tensors.insert("weight".to_string(), a.unwrap());
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
