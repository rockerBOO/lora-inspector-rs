use std::fmt::Debug;
// use std::alloc::Global;
use std::string::String;

// use candle_core::safetensors;
// use std::error::Error;
// use std::marker::Send;
// use std::marker::Sync;

mod file;
mod metadata;
mod network;
mod norms;
mod weight;
mod worker;

pub type Result<T> = std::result::Result<T, InspectorError>;

#[derive(Debug)]
pub enum InspectorError {
    Candle(candle_core::Error),
    SafeTensor(safetensors::SafeTensorError),
    Load(String),
    Msg(String),
}

impl InspectorError {
    fn candle(err: candle_core::Error) -> InspectorError {
        InspectorError::Candle(err)
    }

    fn safetensor(err: safetensors::SafeTensorError) -> InspectorError {
        InspectorError::SafeTensor(err)
    }
}

impl From<candle_core::Error> for InspectorError {
    fn from(err: candle_core::Error) -> InspectorError {
        InspectorError::candle(err)
    }
}

impl From<safetensors::SafeTensorError> for InspectorError {
    fn from(err: safetensors::SafeTensorError) -> InspectorError {
        InspectorError::safetensor(err)
    }
}

pub fn get_base_name(name: &str) -> String {
    name.split('.')
        .filter(|part| !matches!(*part, "weight" | "lora_up" | "lora_down" | "alpha"))
        .fold(String::new(), |acc, v| {
            if acc.is_empty() {
                v.to_owned()
            } else {
                format!("{acc}.{v}")
            }
        })
}

#[cfg(test)]
mod tests {
    use crate::get_base_name;

    #[test]
    fn get_base_name_test() {
        let base_name = get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.lora_up.weight");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");

        let base_name =
            get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.lora_down.weight");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");

        let base_name = get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.alpha");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");
    }
}
