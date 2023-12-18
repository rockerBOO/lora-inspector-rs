use std::collections::HashSet;

use candle_core::{DType, Device};
use web_sys::console;

use crate::{
    norms::l1,
    weight::{BufferedLoRAWeight, Weight, WeightKey},
    Error,
};

/// LoRA file buffer
#[derive(Debug)]
pub struct LoRAFile {
    // buffer: Vec<u8>,
    filename: String,
    weights: Option<BufferedLoRAWeight>,
    // metadata: Option<HashMap<String, String>>,
}

impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> LoRAFile {
        LoRAFile {
            // buffer: buffer.to_vec(),
            filename,
            weights: BufferedLoRAWeight::new(buffer.to_vec())
                .map(Some)
                .unwrap_or_else(|_| None),
            // metadata: match SafeTensors::read_metadata(buffer)
            //     .map(|(_, meta)| meta.metadata().clone())
            // {
            //     Ok(Some(metadata)) => Some(metadata.clone()),
            //     _ => None,
            // },
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

    pub fn alphas(&self) -> HashSet<u32> {
        self.weights
            .as_ref()
            .map(|weights| weights.alphas())
            .unwrap_or_default()
    }

    pub fn dims(&self) -> HashSet<u32> {
        self.weights
            .as_ref()
            .map(|weights| weights.dims())
            .unwrap_or_default()
    }

    pub fn keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.keys())
            .unwrap_or_default()
    }

    pub fn base_names(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.base_names())
            .unwrap_or_default()
    }

    pub fn weight_norm(&self, base_name: &str, device: &Device) -> Result<f32, Error> {
        match self.weights.as_ref() {
            Some(weights) => weights
                .scale_weight(base_name, device)
                .map_err(Error::Candle)
                .map(|t| {
                    l1(&t.to_dtype(DType::F32).map_err(Error::Candle)?).map_err(Error::Candle)
                })?,
            None => Err(Error::Msg("no weight found".to_string())),
        }
    }

    // pub fn scale_weight(&self, name: &str, device: Device) -> Vec<String> {
    //     self.weights
    //         .as_ref()
    //         .map(|weights| weights.scale_weight(name, device).ok())
    //         .unwrap_or_default()
    // }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{self, Read},
    };

    use candle_core::Device;

    use crate::weight::LoRAWeight;

    use super::LoRAFile;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

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

        // Act
        let mut result = lora_file.keys();

        // Assert
        insta::assert_json_snapshot!(result.sort());
    }

    #[test]
    fn weight_norm_returns_norm_for_valid_weights() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        let base_name = "lora_unet_up_blocks_1_attentions_0_proj_in";
        let device = &Device::Cpu;

        // Act
        let result = lora_file.weight_norm(base_name, device);

        println!("{:#?}", result);

        // Assert
        assert!(result.is_ok());
        // Add assertions to verify that the norm result is correct
    }

    #[test]
    fn weight_norm_returns_error_for_missing_weights() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        let base_name = "nonexistent_weight";
        let device = &Device::Cpu;

        // Act
        let result = lora_file.weight_norm(base_name, device);

        // Assert
        assert!(result.is_err());
        // Add assertions to verify that the correct error variant is returned
    }

    #[test]
    fn weight_norm_handles_scale_weight_error() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        let base_name = "error_weight";
        let device = &Device::Cpu;

        // Act
        let result = lora_file.weight_norm(base_name, device);

        // Assert
        assert!(result.is_err());
        // Add assertions to verify that the correct error variant is returned
    }

    #[test]
    fn weight_norm_handles_l1_norm_error() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        let base_name = "l1_error_weight";
        let device = &Device::Cpu;

        // Act
        let result = lora_file.weight_norm(base_name, device);

        // Assert
        assert!(result.is_err());
        // Add assertions to verify that the correct error variant is returned
    }
}
