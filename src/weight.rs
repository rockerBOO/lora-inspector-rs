use candle_core::{
    safetensors::{load_buffer, BufferedSafetensors, Load},
    DType, Device, Tensor,
};
use std::ops::Mul;
use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Formatter},
};

use wasm_bindgen::prelude::*;

use crate::get_base_name;

#[wasm_bindgen]
pub struct BufferedLoRAWeight {
    buffered: BufferedSafetensors,
}

impl Debug for BufferedLoRAWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (_key, tensor) in self.buffered.tensors() {
            tensor.fmt(f)?
        }

        Ok(())
    }
}

// impl Debug for BufferedLoRAWeight {
//     fn fmt(&self, f: &mut Formatter<'_>) -> Result<String, Error> {
//         // f.debug_struct("BufferedLoRAWeight").field("buffered", &self.buffered).finish()
//
//     }
// }
//
// impl D

impl WeightKey for BufferedLoRAWeight {
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

    fn base_names(&self) -> Vec<String> {
        self.weight_keys()
            .iter()
            .map(|name| get_base_name(name))
            .collect()
    }
}

impl Weight for BufferedLoRAWeight {
    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error> {
        self.buffered.load(key, &Device::Cpu)
    }

    fn scale_weight(&self, base_name: &str, device: &Device) -> Result<Tensor, candle_core::Error> {
        let lora_up = format!("{}.lora_up.weight", base_name);
        let lora_down = format!("{}.lora_down.weight", base_name);
        let lora_alpha = format!("{}.alpha", base_name);

        let up = self.buffered.get(&lora_up)?.load(device)?;
        let down = self.buffered.get(&lora_down)?.load(device)?;
        let alpha = self.buffered.get(&lora_alpha)?.load(device)?;

        let dims = down.dims();

        let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;

        if dims.len() == 2 {
            up.matmul(&down)?.mul(scale)
        } else if dims[2] == 1 && dims[3] == 1 {
            up.squeeze(3)?
                .squeeze(2)?
                .matmul(&down.squeeze(3)?.squeeze(2)?)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .mul(scale)
        } else {
            down.permute((1, 0, 2, 3))?
                .conv2d(&up, 0, 1, 1, 1)?
                .permute((1, 0, 2, 3))?
                .mul(scale)
        }
    }

    fn alphas(&self) -> HashSet<u32> {
        self.buffered
            .tensors()
            .iter()
            .filter(|(k, _v)| k.contains("alpha"))
            .filter_map(|(k, _v)| {
                self.get(k)
                    .unwrap()
                    .to_dtype(DType::U32)
                    .map(|v| v.to_scalar::<u32>())
                    .unwrap()
                    .ok()
            })
            .fold(HashSet::new(), |mut alphas: HashSet<u32>, v| {
                alphas.insert(v);
                alphas
            })
    }

    fn dims(&self) -> HashSet<u32> {
        self.buffered
            .tensors()
            .iter()
            .filter(|(k, _v)| k.contains("lora_down"))
            .filter_map(|(k, _v)| {
                self.get(k)
                    .map(|v| v.to_dtype(DType::U32).map(|v| v.dims2()))
                    .ok()
            })
            .fold(HashSet::new(), |mut dims: HashSet<u32>, res| {
                if let Ok(Ok((v, _))) = res {
                    dims.insert(v as u32);
                };
                dims
            })
    }
}

impl BufferedLoRAWeight {
    pub fn new(buffer: Vec<u8>) -> Result<Self, candle_core::Error> {
        Ok(Self {
            buffered: BufferedSafetensors::new(buffer)?,
        })
    }

    pub fn load(&self, name: &str, device: &Device) -> Result<Tensor, candle_core::Error> {
        self.buffered.get(name)?.load(device)
    }
}

pub trait WeightKey {
    fn keys(&self) -> Vec<String>;
    fn keys_by_key(&self, key: &str) -> Vec<String>;
    fn up_keys(&self) -> Vec<String>;
    fn weight_keys(&self) -> Vec<String>;
    fn down_keys(&self) -> Vec<String>;
    fn alpha_keys(&self) -> Vec<String>;
    fn base_names(&self) -> Vec<String>;
}

pub trait Weight {
    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error>;

    fn scale_weight(&self, base_name: &str, device: &Device) -> Result<Tensor, candle_core::Error>;
    fn alphas(&self) -> HashSet<u32>;
    fn dims(&self) -> HashSet<u32>;
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

impl WeightKey for LoRAWeight {
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

    fn base_names(&self) -> Vec<String> {
        self.weight_keys()
            .iter()
            .map(|name| get_base_name(name))
            .collect()
    }
}

impl Weight for LoRAWeight {
    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error> {
        self.tensors
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg("Could not load error".to_string()))
            .cloned()
    }

    fn scale_weight(
        &self,
        base_name: &str,
        _device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let lora_up = format!("{}.lora_up.weight", base_name);
        let lora_down = format!("{}.lora_up.weight", base_name);
        let lora_alpha = format!("{}.alpha", base_name);

        let up = self
            .tensors
            .get(&lora_up)
            .ok_or_else(|| candle_core::Error::Msg("no lora up".to_string()))?;
        let down = self
            .tensors
            .get(&lora_down)
            .ok_or_else(|| candle_core::Error::Msg("no lora down".to_string()))?;
        let alpha = self
            .tensors
            .get(&lora_alpha)
            .ok_or_else(|| candle_core::Error::Msg("no lora alpha".to_string()))?;

        let dims = down.dims();

        let scale = alpha.to_scalar::<f64>()? / dims[0] as f64;

        if dims.len() == 2 {
            up.matmul(down)?.mul(scale)
        } else if dims[2] == 1 && dims[3] == 1 {
            up.squeeze(3)?
                .squeeze(2)?
                .matmul(&down.squeeze(3)?.squeeze(2)?)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .mul(scale)
        } else {
            down.permute((1, 0, 2, 3))?
                .conv2d(up, 0, 1, 1, 1)?
                .permute((1, 0, 2, 3))?
                .mul(scale)
        }
    }

    fn alphas(&self) -> HashSet<u32> {
        self.tensors
            .iter()
            .filter(|(k, _v)| k.contains("alpha"))
            .filter_map(|(_k, v)| {
                v.to_dtype(DType::U32)
                    .map(|v| v.to_scalar::<u32>())
                    .unwrap()
                    .ok()
            })
            .fold(HashSet::new(), |mut alphas: HashSet<u32>, v| {
                alphas.insert(v);
                alphas
            })
    }

    fn dims(&self) -> HashSet<u32> {
        self.tensors
            .iter()
            .filter(|(k, _v)| k.contains("lora_down"))
            .filter_map(|(k, _v)| {
                self.get(k)
                    .map(|v| v.to_dtype(DType::U32).map(|v| v.dims2()))
                    .ok()
            })
            .fold(HashSet::new(), |mut dims: HashSet<u32>, res| {
                if let Ok(Ok((v, _))) = res {
                    dims.insert(v as u32);
                };
                dims
            })
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashSet,
        fs::File,
        io::{self, Read},
    };

    use super::*;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_conv_file() -> Result<Vec<u8>, io::Error> {
        let filename = "/mnt/900/lora/testing/nsfw/pov-2023-12-18-114530-5d004039.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    #[test]
    fn lora_weight_alphas_returns_unique_values() {
        let buffer = load_test_file().unwrap();

        // Arrange
        let lora_weight = LoRAWeight::new(buffer).unwrap();

        println!("{:#?}", lora_weight);

        // Act
        let result_alphas = lora_weight.alphas();

        // Found a result
        assert_eq!(result_alphas.len(), 1);

        // Assert
        // Add assertions to verify that result_alphas contains unique values
        assert_eq!(
            result_alphas.len(),
            result_alphas.iter().collect::<HashSet<_>>().len()
        );
    }

    #[test]
    fn lora_buffer_weight_alphas_returns_unique_values() {
        let buffer = load_test_file().unwrap();

        // Arrange
        let lora_weight = BufferedLoRAWeight::new(buffer).unwrap();

        // Act
        let result_alphas = lora_weight.alphas();

        // Assert

        // Found a result
        assert_eq!(result_alphas.len(), 1);

        // Add assertions to verify that result_alphas contains unique values
        assert_eq!(
            result_alphas.len(),
            result_alphas.iter().collect::<HashSet<_>>().len()
        );
    }

    #[test]
    fn dims_returns_unique_values() {
        let buffer = load_test_file().unwrap();

        // Arrange
        let lora_weight = LoRAWeight::new(buffer).unwrap();

        // Act
        let result_dims = lora_weight.dims();

        assert_eq!(result_dims.len(), 1);

        // Assert
        // Add assertions to verify that result_dims contains unique values
        assert_eq!(
            result_dims.len(),
            result_dims.iter().collect::<HashSet<_>>().len()
        );
    }

    #[test]
    fn buffered_dims_returns_unique_values() {
        let buffer = load_test_file().unwrap();

        // Arrange
        let lora_weight = BufferedLoRAWeight::new(buffer).unwrap();

        // Act
        let result_dims = lora_weight.dims();

        assert_eq!(result_dims.len(), 1);

        // Assert
        // Add assertions to verify that result_dims contains unique values
        assert_eq!(
            result_dims.len(),
            result_dims.iter().collect::<HashSet<_>>().len()
        );
    }

    #[test]
    fn buffered_dims_scale_weight_conv2d() {
        let buffer = load_test_conv_file().unwrap();

        // Arrange
        let lora_weight = BufferedLoRAWeight::new(buffer).unwrap();

        // Act

        let result = lora_weight
            .scale_weight("lora_unet_down_blocks_1_resnets_1_conv2", &Device::Cpu);

        // Assert
        assert!(result.is_ok());
        let scaled_tensor = result.unwrap();

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[640, 640, 3, 3]);

        // Verify that the tensor values are within an acceptable range
        for value in scaled_tensor.flatten_all().unwrap().to_vec0::<f32>().iter() {
            assert!(*value >= 0. && *value <= 2.);
        }
    }
    //
    // #[test]
    // fn dims_returns_empty_set_when_no_lora_down_tensors() {
    //     // Arrange
    //     let buffer = load_test_file_without_lora_down_tensors().unwrap();
    //     let lora_weight = LoRAWeight::new(buffer).unwrap();
    //
    //     // Act
    //     let result_dims = lora_weight.dims();
    //
    //     // Assert
    //     assert!(result_dims.is_empty());
    // }
    //
    // #[test]
    // fn dims_handles_invalid_tensor_values() {
    //     // Arrange
    //     let buffer = load_test_file_with_invalid_tensors().unwrap();
    //     let lora_weight = LoRAWeight::new(buffer).unwrap();
    //
    //     // Act
    //     let result_dims = lora_weight.dims();
    //
    //     // Assert
    //     // Add assertions to handle cases where invalid tensor values are present
    // }
}
