use core::fmt;
use std::fmt::Debug;
use std::io;
// use std::alloc::Global;
use std::string::String;

// use pest::Parser;
use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "key.pest"]
pub struct KeyParser;

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
    Io(io::Error),
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

impl From<io::Error> for InspectorError {
    fn from(err: io::Error) -> InspectorError {
        InspectorError::Io(err)
    }
}

impl fmt::Display for InspectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InspectorError::Candle(e) => write!(f, "Candle Error "),
            InspectorError::SafeTensor(_) => write!(f, "SafeTensor Error"),
            InspectorError::Io(_) => write!(f, "IO Error"),
            InspectorError::Load(e) => write!(f, "Load Error {}", e),
            InspectorError::Msg(e) => write!(f, "Error {}", e),
        }
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

    use std::fs;
    use pest::Parser;

    use crate::{get_base_name, KeyParser, Rule};

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

    fn load_keys_json() -> serde_json::Result<Vec<String>> {
        let keys = fs::read_to_string("./keys.json").expect("to read the keys json");
        serde_json::from_str::<Vec<String>>(&keys)
    }

    #[test]
    fn test_key_parsing() {
        let keys: Vec<String> = load_keys_json().unwrap();

        for key in keys {
            let successful_parse = KeyParser::parse(Rule::key, &key);
            assert!(successful_parse.is_ok());
        }
    }
}
