use candle_core::Device;
use std::collections::{HashMap, HashSet};

use crate::{
    metadata::Metadata,
    network::NetworkType,
    norms::{l1, l2, matrix_norm},
    weight::{self, BufferedLoRAWeight, Weight, WeightKey},
    InspectorError, Result,
};

/// LoRA file buffer
#[derive(Debug)]
pub struct LoRAFile {
    filename: String,
    weights: Option<BufferedLoRAWeight>,
    scaled_weights: HashMap<String, candle_core::Tensor>,
    metadata: Option<Metadata>,
}

// const WEIGHT_NOT_LOADED: &str = "Weight not loaded properly";

impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: &str, device: &Device) -> LoRAFile {
        let metadata = Metadata::new_from_buffer(buffer).map_err(|e| e.to_string());

        LoRAFile {
            filename: filename.to_string(),
            weights: BufferedLoRAWeight::new(buffer.to_vec(), device)
                .map(Some)
                .unwrap_or_else(|_| None),
            scaled_weights: HashMap::new(),
            metadata: metadata.map(Some).unwrap_or_else(|_| None),
        }
    }

    pub fn unload(&mut self) {
        self.weights = None;
        self.scaled_weights = HashMap::new();
    }

    pub fn is_tensors_loaded(&self) -> bool {
        self.weights.is_some()
    }

    pub fn filename(&self) -> String {
        self.filename.clone()
    }

    pub fn unet_keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.unet_keys())
            .unwrap_or_default()
    }

    pub fn text_encoder_keys(&self) -> Vec<String> {
        self.weights
            .as_ref()
            .map(|weights| weights.text_encoder_keys())
            .unwrap_or_default()
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

    pub fn alphas(&self) -> HashSet<weight::Alpha> {
        self.weights
            .as_ref()
            .map(|weights| weights.alphas())
            .unwrap_or_default()
    }

    pub fn dims(&self) -> HashSet<usize> {
        self.weights
            .as_ref()
            .map(|weights| weights.dims())
            .unwrap_or_default()
    }

    pub fn precision(&self) -> Option<weight::DType> {
        self.weights
            .as_ref()
            .and_then(|weights| weights.precision())
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

    pub fn l2_norm<T: candle_core::WithDType>(&self, t: &candle_core::Tensor) -> Result<T> {
        l2(&t.to_dtype(candle_core::DType::F64)?)
    }

    pub fn l1_norm<T: candle_core::WithDType>(&self, t: &candle_core::Tensor) -> Result<T> {
        l1(&t.to_dtype(candle_core::DType::F64)?)
    }

    pub fn matrix_norm<T: candle_core::WithDType>(&self, t: &candle_core::Tensor) -> Result<T> {
        matrix_norm(&t.to_dtype(candle_core::DType::F64)?)
    }

    pub fn scaled_capacity(&self) -> usize {
        self.scaled_weights.capacity()
    }

    pub fn shrink_scaled_to_fit(&mut self) {
        self.scaled_weights.shrink_to_fit();
    }

    pub fn format(&self) -> weight::LoRAFormat {
        self.weights
            .as_ref()
            .map(|weights| weights.format())
            .unwrap_or(weight::LoRAFormat::Kohya)
    }

    // pub fn scaled_weight(&self, base_name: &str) -> Option<&candle_core::Tensor> {
    //     self.scaled_weights.get(base_name)
    // }

    pub fn scale_weights(&self) -> Vec<Result<candle_core::Tensor>> {
        self.base_names()
            .iter()
            .map(|base_name| self.scale_weight(base_name))
            .collect()
    }

    pub fn scale_weight(&self, base_name: &str) -> Result<candle_core::Tensor> {
        // if let Some(tensor) = self.scaled_weights.get(base_name) {
        //     return Ok(tensor.clone());
        // }

        match self.weights.as_ref() {
            Some(weights) => match self
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.network_type())
            {
                Some(NetworkType::LoRA) => Ok(weights.scale_lora_weight(base_name)?),
                Some(NetworkType::LoRAFA) => Ok(weights.scale_lora_weight(base_name)?),
                Some(NetworkType::DyLoRA) => Ok(weights.scale_lora_weight(base_name)?),
                Some(NetworkType::GLoRA) => Ok(weights.scale_glora_weights(base_name)?),
                Some(NetworkType::LoKr) => Ok(weights.scale_lokr_weight(base_name)?),
                Some(NetworkType::LoHA) => Ok(weights.scale_hada_weight(base_name)?),
                Some(NetworkType::BOFT) => Ok(weights.scale_boft_weight(base_name)?),
                Some(NetworkType::DiagOFT) => Ok(weights.scale_diag_oft_weight(base_name)?),
                Some(_) => Err(InspectorError::UnsupportedNetworkType),
                None => Ok(weights.scale_lora_weight(base_name)?),
            },
            None => Err(InspectorError::Msg(
                "Weight not loaded. Load the weight first.".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        fs::File,
        io::{self, Read},
    };

    // macro_rules! assert_err {
    //     ($expression:expr, $($pattern:tt)+) => {
    //         match $expression {
    //             $($pattern)+ => (),
    //             ref e => panic!("expected `{}` but got `{:?}`", stringify!($($pattern)+), e),
    //         }
    //     }
    // }

    use candle_core::Device;

    use crate::weight::{self, Alpha};

    use super::LoRAFile;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_file(filename: &str) -> Result<Vec<u8>, io::Error> {
        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    #[test]
    fn load_tensors_success() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let lora_file = LoRAFile::new_from_buffer(&buffer, "boo.safetensors", &Device::Cpu);

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
        let filename = "boo.safetensors";
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename, &Device::Cpu);

        // Act
        let result = lora_file.filename();

        // Assert
        assert_eq!(result, filename);
    }

    #[test]
    fn weight_keys_returns_correct_keys() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = "boo.safetensors";
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename, &Device::Cpu);
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
        let filename = "boo.safetensors";
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename, &Device::Cpu);

        // Act
        let mut result = lora_file.keys();

        // Assert
        insta::assert_json_snapshot!(result.sort());
    }

    #[test]
    fn weight_norm_handles_scale_weight_error() {
        // Arrange
        let buffer = load_test_file().unwrap();
        let filename = "boo.safetensors";
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename, &Device::Cpu);
        let base_name = "error_weight";

        let result = lora_file.scale_weight(base_name);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn load_from_invalid_buffer() {
        // Arrange
        let filename = "boo.safetensors";
        let lora_file = LoRAFile::new_from_buffer(&[1_u8], filename, &Device::Cpu);
        let base_name = "l1_error_weight";

        let result = lora_file.scale_weight(base_name);

        // Assert
        assert!(result.is_err());
    }

    #[test]
    fn weight_load_no_metadata() -> crate::Result<()> {
        let file = "edgWar40KAdeptaSororitas.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);

        let base_name = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";

        let scaled_weight = lora_file
            .scale_weight(base_name)
            .expect("could not scale weight");

        assert_eq!(
            502.32165664434433,
            lora_file.l1_norm::<f64>(&scaled_weight)?
        );

        assert_eq!(
            0.7227786684427061,
            lora_file.l2_norm::<f64>(&scaled_weight)?
        );

        assert_eq!(
            0.7227786684427061,
            lora_file.matrix_norm::<f64>(&scaled_weight)?
        );

        Ok(())
    }

    #[test]
    fn alpha_keys() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);

        let mut alpha_keys = lora_file.alpha_keys();
        alpha_keys.sort_by_key(|a| a.to_lowercase());

        insta::assert_json_snapshot!(alpha_keys);

        Ok(())
    }

    #[test]
    fn alphas() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);
        let mut compare_set = HashSet::new();
        compare_set.insert(Alpha(4.));
        assert_eq!(compare_set, lora_file.alphas());

        Ok(())
    }

    #[test]
    fn dims() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);

        let mut compare_set = HashSet::new();
        compare_set.insert(4);

        assert_eq!(compare_set, lora_file.dims());

        Ok(())
    }

    #[test]
    fn base_names() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        let mut base_names = lora_file.base_names();
        base_names.sort_by_key(|a| a.to_lowercase());

        insta::assert_json_snapshot!(base_names);

        Ok(())
    }

    #[test]
    fn unet_keys() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        assert_eq!(lora_file.unet_keys().len(), 576);

        Ok(())
    }

    #[test]
    fn text_encoder_keys() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        assert_eq!(lora_file.text_encoder_keys().len(), 216);

        Ok(())
    }

    #[test]
    fn precision() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        assert!(lora_file.precision() == Some(weight::DType::F16));

        Ok(())
    }

    #[test]
    fn is_tensors_loaded() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        assert!(lora_file.is_tensors_loaded());

        Ok(())
    }

    #[test]
    #[ignore] // Only run with: cargo test -- --ignored
    fn load_women_flux2_file() -> crate::Result<()> {
        // Regression test for Flux LoRA file parsing
        // This test uses a file path specific to the development machine
        // Run with: cargo test --package inspector load_women_flux2_file -- --ignored --nocapture
        let file = "/mnt/900/training/sets/women-flux2-2026-01-25-013607-046099a7/women-flux2-2026-01-25-013607-046099a7.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        // Verify the file loaded successfully
        assert!(lora_file.is_tensors_loaded());

        // Try to get keys to see if parsing works
        let keys = lora_file.keys();
        println!("Found {} keys in file", keys.len());
        println!("\nFirst 20 keys:");
        for (i, key) in keys.iter().take(20).enumerate() {
            println!("  {}: {}", i + 1, key);
        }

        // Try to get base names
        let base_names = lora_file.base_names();
        println!("\nFound {} base_names in file", base_names.len());
        println!("\nFirst 20 base_names:");
        for (i, name) in base_names.iter().take(20).enumerate() {
            println!("  {}: {}", i + 1, name);
        }

        // Verify it has the expected Flux structure
        assert!(keys.iter().any(|k| k.contains("double_blocks")));
        assert!(keys.iter().any(|k| k.contains("single_blocks")));

        Ok(())
    }

    #[test]
    #[ignore] // Only run with: cargo test -- --ignored
    fn reproduce_flux2_panic() -> crate::Result<()> {
        // Test case to reproduce the panic with specific failing weights
        let file = "/mnt/900/training/sets/women-flux2-2026-01-25-013607-046099a7/women-flux2-2026-01-25-013607-046099a7.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        // These are the weights that panic in the frontend
        let failing_weights = vec![
            "lora_unet_single_blocks_19_linear2",
            "lora_unet_single_blocks_0_linear2",
            "lora_unet_double_blocks_3_img_mlp_0",
        ];

        println!("\nTesting failing weights:");
        for base_name in failing_weights {
            println!("\nTesting: {}", base_name);
            match lora_file.scale_weight(base_name) {
                Ok(tensor) => {
                    println!("  ✓ Scaled successfully. Shape: {:?}", tensor.dims());
                    // Try to compute l2_norm
                    match lora_file.l2_norm::<f64>(&tensor) {
                        Ok(norm) => println!("  ✓ L2 norm: {}", norm),
                        Err(e) => println!("  ✗ L2 norm failed: {:?}", e),
                    }
                }
                Err(e) => {
                    println!("  ✗ Scale weight failed: {:?}", e);
                }
            }
        }

        Ok(())
    }
}
