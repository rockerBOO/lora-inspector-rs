use candle_core::{
    safetensors::{load_buffer, BufferedSafetensors, Load},
    Device, Tensor,
};
use std::collections::HashMap;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct BufferedLoRAWeight {
    buffered: BufferedSafetensors,
}

impl Weight for BufferedLoRAWeight {
    fn keys(&self) -> Vec<String> {
        self.buffered
            .tensors()
            .iter()
            .map(|(k, _v)| k.to_owned())
            .map(|v| v.to_owned())
            .collect()
    }

    fn keys_by_key(&self, key: &str) -> Vec<String> {
        self.buffered
            .tensors()
            .iter()
            .map(|(k, _v)| k.to_owned())
            .filter(|k| k.contains(key))
            .map(|k| k.to_owned())
            .collect()
    }

    fn weight_keys(&self) -> Vec<String> {
        self.keys_by_key("weight")
    }

    fn alpha_keys(&self) -> Vec<String> {
        self.keys_by_key("alpha")
    }

    fn up_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_up")
    }

    fn down_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_down")
    }

    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error> {
        self.buffered.load(key, &Device::Cpu)
    }
}

impl BufferedLoRAWeight {
    pub fn load(&self, name: &str, dev: &Device) -> Result<Tensor, candle_core::Error> {
        self.buffered.get(name)?.load(dev)
    }
}

pub trait Weight {
    fn keys(&self) -> Vec<String>;
    fn keys_by_key(&self, key: &str) -> Vec<String>;
    fn up_keys(&self) -> Vec<String>;
    fn weight_keys(&self) -> Vec<String>;
    fn down_keys(&self) -> Vec<String>;
    fn alpha_keys(&self) -> Vec<String>;
    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error>;
}

/// Tensor weights
#[derive(Debug, Clone)]
pub struct LoRAWeight {
    pub tensors: HashMap<String, Tensor>,
}

impl LoRAWeight {
    pub fn new(buffer: Vec<u8>) -> Result<Self, candle_core::Error> {
        Ok(Self {
            tensors: load_buffer(&buffer, &Device::Cpu)?,
        })
    }
}

impl Weight for LoRAWeight {
    fn keys(&self) -> Vec<String> {
        self.tensors.keys().map(|v| v.to_owned()).collect()
    }

    fn keys_by_key(&self, key: &str) -> Vec<String> {
        self.tensors
            .keys()
            .filter(|&k| k.contains(key))
            .map(|k| k.to_owned())
            .collect()
    }

    fn weight_keys(&self) -> Vec<String> {
        self.keys_by_key("weight")
    }

    fn alpha_keys(&self) -> Vec<String> {
        self.keys_by_key("alpha")
    }

    fn up_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_up")
    }

    fn down_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_down")
    }

    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error> {
        self.tensors
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg("Could not load error".to_string()))
            .cloned()
    }
}
