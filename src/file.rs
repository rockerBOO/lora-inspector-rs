use std::collections::{HashMap, HashSet};

use candle_core::{DType, Device};

use crate::{
    metadata::Metadata,
    network::NetworkType,
    norms::{self, l1, l2, matrix_norm},
    weight::{Alpha, BufferedLoRAWeight, Weight, WeightKey},
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

const WEIGHT_NOT_LOADED: &str = "Weight not loaded properly";

impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> LoRAFile {
        let metadata = Metadata::new_from_buffer(buffer).map_err(|e| e.to_string());
        LoRAFile {
            filename,
            weights: BufferedLoRAWeight::new(buffer.to_vec())
                .map(Some)
                .unwrap_or_else(|_| None),
            scaled_weights: HashMap::new(),
            metadata: metadata.map(Some).unwrap_or_else(|_| None),
        }
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

    pub fn alphas(&self) -> HashSet<Alpha> {
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

    pub fn precision(&self) -> String {
        self.weights
            .as_ref()
            .map(|weights| weights.precision())
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

    pub fn scaled<T: candle_core::WithDType>(
        &mut self,
        base_name: &str,
        collection: Vec<norms::NormFn<T>>,
        device: &Device,
    ) -> Result<HashMap<String, Result<T>>> {
        let scaled = self.scale_weight(base_name, device)?;
        Ok(collection
            .iter()
            .map(|norm| (norm.name.to_owned(), (*norm.function)(scaled.clone())))
            .fold(HashMap::new(), |mut acc, (k, t)| {
                acc.insert(k, t);
                acc
            }))
    }

    pub fn l2_norm<T: candle_core::WithDType>(
        &mut self,
        base_name: &str,
        device: &Device,
    ) -> Result<T> {
        self.scale_weight(base_name, device)
            .and_then(|t| l2(&t.to_dtype(DType::F64)?))
    }

    pub fn l1_norm<T: candle_core::WithDType>(
        &mut self,
        base_name: &str,
        device: &Device,
    ) -> Result<T> {
        self.scale_weight(base_name, device)
            .and_then(|t| l1(&t.to_dtype(DType::F64)?))
    }

    pub fn matrix_norm<T: candle_core::WithDType>(
        &mut self,
        base_name: &str,
        device: &Device,
    ) -> Result<T> {
        self.scale_weight(base_name, device)
            .and_then(|t| matrix_norm(&t.to_dtype(DType::F64)?))
    }

    pub fn scale_weight(
        &mut self,
        base_name: &str,
        device: &Device,
    ) -> Result<candle_core::Tensor> {
        if let Some(tensor) = self.scaled_weights.get(base_name) {
            return Ok(tensor.clone());
        }

        let scaled_weight = match self.weights.as_ref() {
            Some(weights) => match self
                .metadata
                .as_ref()
                .and_then(|metadata| metadata.network_type())
            {
                Some(NetworkType::LoRA) => Ok(weights.scale_lora_weight(base_name, device)?),
                Some(NetworkType::LoRAFA) => Ok(weights.scale_lora_weight(base_name, device)?),
                Some(_) => Err(InspectorError::UnsupportedNetworkType),
                None => Err(InspectorError::Msg(WEIGHT_NOT_LOADED.to_string())),
            },
            None => Err(InspectorError::Msg(WEIGHT_NOT_LOADED.to_string())),
        };

        let _ = scaled_weight.as_ref().is_ok_and(|scaled| {
            self.scaled_weights
                .insert(base_name.to_string(), scaled.clone())
                .is_some()
        });

        scaled_weight

        // match  {
        //     Ok(scaled) => {
        //         self.scaled_weights.insert(base_name.to_string(), scaled.clone());
        //         scaled_weight
        //     }
        //     Err(e) => scaled_weight,
        // };
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        fs::File,
        io::{self, Read},
    };

    macro_rules! assert_err {
        ($expression:expr, $($pattern:tt)+) => {
            match $expression {
                $($pattern)+ => (),
                ref e => panic!("expected `{}` but got `{:?}`", stringify!($($pattern)+), e),
            }
        }
    }

    use candle_core::Device;

    use crate::{weight::Alpha, InspectorError};

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
        let result = lora_file.l1_norm::<f64>(base_name, device);

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
        let result = lora_file.l1_norm::<f64>(base_name, device);

        // Assert
        assert!(result.is_err());

        assert_err!(result, Err(InspectorError::Candle(_)));
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
        let result = lora_file.l1_norm::<f64>(base_name, device);

        // Assert
        assert!(result.is_err());

        assert_err!(result, Err(InspectorError::Candle(_)));
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
        let result = lora_file.l1_norm::<f64>(base_name, device);

        // Assert
        assert!(result.is_err());

        assert_err!(result, Err(InspectorError::Candle(_)));
    }

    #[test]
    fn load_from_invalid_buffer() {
        // Arrange
        let filename = String::from("boo.safetensors");
        let lora_file = LoRAFile::new_from_buffer(&[1_u8], filename.clone());
        let base_name = "l1_error_weight";
        let device = &Device::Cpu;

        // Act
        let result = lora_file.l1_norm::<f64>(base_name, device);

        // Assert
        assert!(result.is_err());
        assert_err!(result, Err(InspectorError::Msg(_)));

        // Act
        let result = lora_file.l2_norm::<f64>(base_name, device);

        // Assert
        assert!(result.is_err());
        assert_err!(result, Err(InspectorError::Msg(_)));

        // Act
        let result = lora_file.matrix_norm::<f64>(base_name, device);

        // Assert
        assert!(result.is_err());
        assert_err!(result, Err(InspectorError::Msg(_)));
    }

    #[test]
    fn weight_load_no_metadata() -> crate::Result<()> {
        let device = &Device::Cpu;
        let file = "edgWar40KAdeptaSororitas.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());

        assert_eq!(
            502.32165664434433,
            lora_file.l1_norm::<f64>(
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj",
                device
            )?
        );

        assert_eq!(
            0.7227786684427061,
            lora_file.l2_norm::<f64>(
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj",
                device
            )?
        );

        assert_eq!(
            0.7227786684427061,
            lora_file.matrix_norm::<f64>(
                "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj",
                device
            )?
        );

        Ok(())
    }

    #[test]
    fn alpha_keys() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());

        let mut alpha_keys = lora_file.alpha_keys();
        alpha_keys.sort_by_key(|a| a.to_lowercase());

        insta::assert_json_snapshot!(alpha_keys);

        Ok(())
    }

    #[test]
    fn alphas() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());
        let mut compare_set = HashSet::new();
        compare_set.insert(Alpha(4.));
        assert_eq!(compare_set, lora_file.alphas());

        Ok(())
    }

    #[test]
    fn dims() -> crate::Result<()> {
        let file = "boo.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());

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
        let lora_file = LoRAFile::new_from_buffer(&buffer, filename.clone());

        let mut base_names = lora_file.base_names();
        base_names.sort_by_key(|a| a.to_lowercase());

        insta::assert_json_snapshot!(base_names);

        Ok(())
    }
}
