use std::collections::HashMap;

use crate::weight::{BufferedLoRAWeight, WeightKey};
use safetensors::SafeTensors;

/// LoRA file buffer
#[derive(Debug)]
pub struct LoRAFile {
    buffer: Vec<u8>,
    filename: String,
    weights: Option<BufferedLoRAWeight>,
    metadata: Option<HashMap<String, String>>,
}

impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> LoRAFile {
        LoRAFile {
            buffer: buffer.to_vec(),
            filename,
            weights: BufferedLoRAWeight::new(buffer.to_vec())
                .map(Some)
                .unwrap_or_else(|_| None),
            metadata: match SafeTensors::read_metadata(buffer)
                .map(|(_, meta)| meta.metadata().clone())
            {
                Ok(Some(metadata)) => Some(metadata.clone()),
                _ => None,
            },
        }
    }

    pub fn is_tensors_loaded(&self) -> bool {
        self.weights.is_some()
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

    pub fn alpha_keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.alpha_keys())
            .unwrap_or_default()
    }

    pub fn alphas(&self) -> Vec<u32> {
        self.weights
            .as_ref()
            .map(|weights| weights.alphas())
            .unwrap_or_default()
    }

    pub fn keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.keys())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{self, Read},
    };

    use super::LoRAFile;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    // #[test]
    // fn new_from_buffer_creates_instance() {
    //     // Arrange
    //     let buffer = load_test_file().unwrap();
    //     let filename = String::from("boo.safetensors");
    //
    //     // Act
    //     let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
    //
    //
    //
    //     // Assert
    //     // assert_eq!(lora_file.buffer, buffer);
    //     // assert_eq!(lora_file.filename, filename);
    //     // assert!(lora_file.weights.is_none());
    // }

    #[test]
    fn load_tensors_success() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());

        // Act
        // let result = lora_file.load_tensors();

        // Assert
        // assert!(result.is_ok());
        assert!(lora_file.weights.is_some());
    }

    #[test]
    fn filename_returns_correct_value() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());

        // Act
        let result = lora_file.filename();

        // Assert
        assert_eq!(result, filename);
    }

    #[test]
    fn weight_keys_returns_correct_keys() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        // lora_file.load_tensors().unwrap();

        // Act
        let mut result = lora_file.weight_keys();

        // Assert
        insta::assert_json_snapshot!(result.sort());
    }

    #[test]
    fn keys_returns_correct_keys() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        // lora_file.load_tensors().unwrap();

        // Act
        let mut result = lora_file.keys();

        // Assert
        insta::assert_json_snapshot!(result.sort());
    }
}
