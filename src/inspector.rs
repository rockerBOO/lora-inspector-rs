use std::collections::HashMap;

use candle_core::{safetensors::load_buffer, Device, Tensor};

use crate::{metadata::Metadata, Error};

#[derive(Debug)]
pub struct LoRAInspector {
    tensors: HashMap<String, Tensor>,
    metadata: Metadata,
}

impl LoRAInspector {
    pub fn new_from_buffer(buffer: &[u8]) -> Result<LoRAInspector, Error> {
        match load_buffer(buffer, &Device::Cpu) {
            Ok(tensors) => match Metadata::new_from_buffer(buffer) {
                Ok(metadata) => Ok(LoRAInspector { tensors, metadata }),

                Err(_) => Err(Error::Load),
            },
            Err(_) => todo!(),
        }
    }

    pub fn metadata(self) -> Metadata {
        self.metadata
    }

    pub fn keys_by_key(self, key: &str) -> Vec<String> {
        self.tensors
            .keys()
            .filter_map(|k| k.contains(key).then(|| k.to_owned()))
            .collect()
    }

    pub fn weights_keys(self) -> Vec<String> {
        self.keys_by_key("weights")
    }

    pub fn alpha_keys(self) -> Vec<String> {
        self.keys_by_key("alpha")
    }

    pub fn up_keys(self) -> Vec<String> {
        self.keys_by_key("lora_up")
    }

    pub fn down_keys(self) -> Vec<String> {
        self.keys_by_key("lora_down")
    }

    pub fn get(self, key: &str) -> Option<Tensor> {
        self.tensors.get(key).cloned()
    }

    pub fn tensors(self) -> HashMap<String, Tensor> {
        self.tensors
    }
}

#[cfg(test)]
mod tests {
    // use std::fs::File;
    //
    // use candle_core::DType;
    // use memmap2::MmapOptions;
    //
    // use super::*;

    // fn make_buffer() -> Vec<u8> {
    //     let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
    //
    //     let tensors: HashMap<String, Tensor> = [("t1".to_string(), t)].into_iter().collect();
    //
    //     let filename = "buffer.safetensors";
    //     candle_core::safetensors::save(&tensors, filename).unwrap();
    //
    //     let file = File::open(filename).unwrap();
    //     let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    //
    //     buffer.to_owned()
    // }
    //
    // #[test]
    // fn new_from_buffer_valid_data() {
    //     // Arrange
    //     let valid_buffer = make_buffer();
    //
    //     // Act
    //     let result = LoRAInspector::new_from_buffer(&valid_buffer);
    //
    //
    //     // Assert
    //     assert!(dbg!(result).is_ok());
    // }
    //
    // #[test]
    // fn new_from_buffer_invalid_data() {
    //     // Arrange
    //     let invalid_buffer = make_buffer(); // provide an invalid buffer for testing;
    //
    //     // Act
    //     let result = LoRAInspector::new_from_buffer(&invalid_buffer);
    //
    //     // Assert
    //     assert!(result.is_err());
    //     // Add more specific assertions if needed
    // }
    //
    // #[test]
    // fn metadata_returns_correct_value() {
    //     // Arrange
    //     let valid_buffer = make_buffer();// provide a valid buffer for testing;
    //     let inspector = LoRAInspector::new_from_buffer(&valid_buffer).unwrap();
    //
    //     // Act
    //     let metadata = inspector.metadata();
    //
    //     // Assert
    //     // Add assertions to verify that the returned metadata is correct
    // }
    //
    // #[test]
    // fn keys_by_key_returns_matching_keys() {
    //     // Arrange
    //     let valid_buffer = make_buffer();// provide a valid buffer for testing;
    //     let inspector = LoRAInspector::new_from_buffer(&valid_buffer).unwrap();
    //
    //     // Act
    //     let keys = inspector.keys_by_key("weights");
    //
    //     // Assert
    //     // Add assertions to verify that the returned keys contain "weights"
    // }
    //
    // #[test]
    // fn get_returns_correct_tensor() {
    //     // Arrange
    //     let valid_buffer = make_buffer();// provide a valid buffer for testing;
    //     let inspector = LoRAInspector::new_from_buffer(&valid_buffer).unwrap();
    //     let key = "t1"; // choose a key for testing;
    //
    //     // Act
    //     let result = inspector.get(&key);
    //
    //     // Assert
    //     // Add assertions to verify that the returned tensor is correct
    // }
}
