use crate::weight::{LoRAWeight, Weight};
use candle_core::{safetensors::load_buffer, Device};

/// LoRA file buffer
#[derive(Debug, Clone)]
pub struct LoRAFile {
    pub buffer: Vec<u8>,
    pub filename: String,
    weights: Option<LoRAWeight>,
}

impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> LoRAFile {
        LoRAFile {
            buffer: buffer.to_vec(),
            filename,
            weights: None,
        }
    }

    pub fn load_tensors(&mut self) -> Result<(), candle_core::Error> {
        self.weights = Some(LoRAWeight {
            tensors: load_buffer(&self.buffer, &Device::Cpu)?,
        });

        Ok(())
    }

    pub fn filename(&self) -> String {
        self.filename.clone()
    }

    pub fn weight_keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.weight_keys())
            .unwrap_or_default()
    }

    pub fn keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.keys())
            .unwrap_or_default()
    }
}
