use candle_core::{
    safetensors::{load_buffer, BufferedSafetensors, Load},
    DType, Device, Tensor,
};

use serde::{Deserialize, Serialize};
use std::ops::Mul;
use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Formatter},
};

use wasm_bindgen::prelude::*;

use crate::get_base_name;

#[derive(Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct Alpha(pub f32);

impl Eq for Alpha {}

impl std::hash::Hash for Alpha {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        self.canonicalize();
    }
}

impl Alpha {
    fn canonicalize(&self) -> i64 {
        (self.0 * 1024.0 * 1024.0).round() as i64
    }
}

impl PartialEq for Alpha {
    fn eq(&self, other: &Alpha) -> bool {
        self.canonicalize() == other.canonicalize()
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct DoRAScale(pub f32);

impl Eq for DoRAScale {}

impl DoRAScale {
    fn canonicalize(&self) -> i64 {
        (self.0 * 1024.0 * 1024.0).round() as i64
    }
}

impl std::hash::Hash for DoRAScale {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        self.canonicalize();
    }
}

impl PartialEq for DoRAScale {
    fn eq(&self, other: &DoRAScale) -> bool {
        self.canonicalize() == other.canonicalize()
    }
}

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
        let hada = &mut self.keys_by_key("hada_w1");
        let lokr = &mut self.keys_by_key("lokr_w1");
        let oft_diag = &mut self.keys_by_key("oft_diag");
        let mut keys = self.keys_by_key("weight");
        keys.append(hada);
        keys.append(lokr);
        keys.append(oft_diag);

        keys
    }

    fn unet_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_unet")
    }

    fn text_encoder_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_te")
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
            .collect::<HashSet<String>>()
            .into_iter()
            .collect()
    }
}

impl Weight for BufferedLoRAWeight {
    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error> {
        self.buffered.load(key, &Device::Cpu)
    }

    fn precision(&self) -> String {
        self.buffered
            .tensors()
            .first()
            .map(|(_key, t)| match t.dtype() {
                safetensors::Dtype::F16 => "fp16".to_string(),
                safetensors::Dtype::BF16 => "bf16".to_string(),
                safetensors::Dtype::F32 => "fp32".to_string(),
                safetensors::Dtype::I64 => "i64".to_string(),
                dtype => panic!("failed to find precision dtype: {:#?}", dtype),
            })
            .unwrap_or_default()
    }

    fn scale_lora_weight(
        &self,
        base_name: &str,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let lora_up = format!("{}.lora_up.weight", base_name);
        let lora_down = format!("{}.lora_down.weight", base_name);
        let lora_alpha = format!("{}.alpha", base_name);
        let lora_dora_scale = format!("{}.dora_scale", base_name);

        let up = self.buffered.get(&lora_up)?.load(device)?.detach()?;
        let down = self.buffered.get(&lora_down)?.load(device)?.detach()?;
        let alpha = self.buffered.get(&lora_alpha)?.load(device)?.detach()?;

        let dims = down.dims();

        let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;

        if dims.len() == 2 {
            self.scale_dora_weight(&up, &lora_dora_scale, device)?
                .matmul(&down)?
                .detach()?
                .mul(scale)?
                .detach()
        } else if dims[2] == 1 && dims[3] == 1 {
            self.scale_dora_weight(&up, &lora_dora_scale, device)?
                .squeeze(3)?
                .squeeze(2)?
                .matmul(&down.squeeze(3)?.squeeze(2)?)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .mul(scale)
        } else {
            down.permute((1, 0, 2, 3))?
                .conv2d(
                    &self.scale_dora_weight(&up, &lora_dora_scale, &device)?,
                    0,
                    1,
                    1,
                    1,
                )?
                .permute((1, 0, 2, 3))?
                .mul(scale)
        }
    }

    fn scale_hada_weight(
        &self,
        base_name: &str,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let hada_w1_a = format!("{}.hada_w1_a", base_name);
        let hada_w1_b = format!("{}.hada_w1_b", base_name);
        let hada_w2_a = format!("{}.hada_w2_a", base_name);
        let hada_w2_b = format!("{}.hada_w2_b", base_name);

        // Tucker
        let hada_t1 = format!("{}.hada_t1", base_name);
        let hada_t2 = format!("{}.hada_t2", base_name);

        let lora_alpha = format!("{}.alpha", base_name);

        let w1_a = self.buffered.get(&hada_w1_a)?.load(device)?;
        let w1_b = self.buffered.get(&hada_w1_b)?.load(device)?;
        let w2_a = self.buffered.get(&hada_w2_a)?.load(device)?;
        let w2_b = self.buffered.get(&hada_w2_b)?.load(device)?;

        let alpha = self.buffered.get(&lora_alpha)?.load(device)?;

        let dims = w1_b.dims();

        let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;

        if let Ok(t1) = self.buffered.get(&hada_t1) {
            let t1 = t1.load(device)?;

            let t2 = self.buffered.get(&hada_t2)?.load(device)?;
            let one = w1_a.matmul(&w1_b)?;
            let two = w2_a.matmul(&w2_b)?;

            let t1_w1 = t1.mul(one)?;
            let t2_w2 = t2.mul(two)?;

            t1_w1.matmul(&t2_w2)?.mul(scale)
        } else {
            let one = w1_a.matmul(&w1_b)?;
            let two = w2_a.matmul(&w2_b)?;

            one.mul(two)?.mul(scale)
        }

        // if dims.len() == 2 {
        //     up.matmul(&down)?.mul(scale)
        // } else if dims[2] == 1 && dims[3] == 1 {
        //     up.squeeze(3)?
        //         .squeeze(2)?
        //         .matmul(&down.squeeze(3)?.squeeze(2)?)?
        //         .unsqueeze(2)?
        //         .unsqueeze(3)?
        //         .mul(scale)
        // } else {
        //     down.permute((1, 0, 2, 3))?
        //         .conv2d(&up, 0, 1, 1, 1)?
        //         .permute((1, 0, 2, 3))?
        //         .mul(scale)
        // }
    }

    fn scale_lokr_weight(
        &self,
        base_name: &str,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let lokr_w1 = format!("{}.lokr_w1", base_name);
        let lokr_w2 = format!("{}.lokr_w2", base_name);

        let lokr_w1_a = format!("{}.lokr_w1_a", base_name);
        let lokr_w1_b = format!("{}.lokr_w1_b", base_name);

        let lokr_w2_a = format!("{}.lokr_w2_a", base_name);
        let lokr_w2_b = format!("{}.lokr_w2_b", base_name);

        let lokr_t2 = format!("{}.lokr_t2", base_name);

        let alpha_key = format!("{}.alpha", base_name);

        let alpha = self.buffered.get(&alpha_key)?.load(device)?;

        if let Ok(w1_a) = self.buffered.get(&lokr_w1_a) {
            let w1_a = w1_a.load(device)?;
            let w1_b = self.buffered.get(&lokr_w1_b)?.load(device)?;
            let w2_a = self.buffered.get(&lokr_w2_a)?.load(device)?;
            let w2_b = self.buffered.get(&lokr_w2_b)?.load(device)?;

            let w1 = w1_a.matmul(&w1_b)?;
            let w2 = w2_a.matmul(&w2_b)?;

            let dims = w1_b.dims();

            let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;
            w1.matmul(&w2)?.mul(scale)
        } else if let Ok(t2) = self.buffered.get(&lokr_t2) {
            let w2_a = self.buffered.get(&lokr_w2_a)?.load(device)?;
            let w2_b = self.buffered.get(&lokr_w2_b)?.load(device)?;
            w2_a.matmul(&w2_b)?.mul(t2.load(device)?)
        } else {
            let w1 = self.buffered.get(&lokr_w1)?.load(device)?;
            let w2 = self.buffered.get(&lokr_w1)?.load(device)?;

            let dims = w1.dims();

            let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;

            w1.matmul(&w2)?.mul(scale)
        }

        // if len(w2.shape) == 4:
        //     w1 = w1.unsqueeze(2).unsqueeze(2)
        // w2 = w2.contiguous()
        // rebuild = torch.kron(w1, w2)
        //
        // return rebuild * scale
    }

    fn scale_dora_weight(
        &self,
        weight: &Tensor,
        lora_dora_scale: &str,
        device: &candle_core::Device,
    ) -> Result<Tensor, candle_core::Error> {

        // self.dora_norm_dims = org_weight.dim() - 1
        // weight_norm = (
        //     weight.transpose(0, 1)
        //     .reshape(weight.shape[1], -1)
        //     .norm(dim=1, keepdim=True)
        //     .reshape(weight.shape[1], *[1] * self.dora_norm_dims)
        //     .transpose(0, 1)
        // )
        // if let Ok(dora_scale) = self.buffered.get(lora_dora_scale) {
        //     weight.matmul(&dora_scale.load(device)?)?.detach()
        // } else {
        //     weight.detach()
        // }
        weight.detach()
    }

    fn alphas(&self) -> HashSet<Alpha> {
        self.buffered
            .tensors()
            .iter()
            .filter(|(k, _v)| k.contains("alpha"))
            .filter_map(|(k, _v)| {
                self.get(k)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .map(|v| v.to_scalar::<f32>())
                    .unwrap()
                    .ok()
            })
            .fold(HashSet::new(), |mut alphas: HashSet<Alpha>, v| {
                alphas.insert(Alpha(v));
                alphas
            })
    }

    fn dora_scales(&self) -> HashSet<DoRAScale> {
        self.buffered
            .tensors()
            .iter()
            .filter(|(k, _v)| k.contains("dora_scale"))
            .filter_map(|(k, _v)| {
                self.get(k)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .map(|v| v.to_scalar::<f32>())
                    .unwrap()
                    .ok()
            })
            .fold(HashSet::new(), |mut scales: HashSet<DoRAScale>, v| {
                scales.insert(DoRAScale(v));
                scales
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

    fn shapes(&self) -> HashMap<String, Vec<usize>> {
        self.buffered
            .tensors()
            .iter()
            .fold(HashMap::new(), |mut acc, (k, t)| {
                acc.insert(k.to_string(), t.shape().to_vec());
                acc
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
    fn unet_keys(&self) -> Vec<String>;
    fn text_encoder_keys(&self) -> Vec<String>;
    fn weight_keys(&self) -> Vec<String>;
    fn down_keys(&self) -> Vec<String>;
    fn alpha_keys(&self) -> Vec<String>;
    fn base_names(&self) -> Vec<String>;
}

pub trait Weight {
    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error>;

    fn precision(&self) -> String;
    fn scale_lora_weight(
        &self,
        base_name: &str,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error>;
    fn scale_hada_weight(
        &self,
        base_name: &str,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error>;
    fn scale_lokr_weight(
        &self,
        base_name: &str,
        device: &Device,
    ) -> Result<Tensor, candle_core::Error>;

    fn scale_dora_weight(
        &self,
        weight: &Tensor,
        lora_dora_scale: &str,
        device: &candle_core::Device,
    ) -> Result<Tensor, candle_core::Error>;
    fn alphas(&self) -> HashSet<Alpha>;
    fn dora_scales(&self) -> HashSet<DoRAScale>;
    fn dims(&self) -> HashSet<u32>;
    fn shapes(&self) -> HashMap<String, Vec<usize>>;
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
        let hada = &mut self.keys_by_key("hada_w1");
        let lokr = &mut self.keys_by_key("lokr_w1");
        let oft_diag = &mut self.keys_by_key("oft_diag");
        let mut keys = self.keys_by_key("weight");
        keys.append(hada);
        keys.append(lokr);
        keys.append(oft_diag);

        keys
    }

    fn unet_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_unet")
    }

    fn text_encoder_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_te")
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
            .collect::<HashSet<String>>()
            .into_iter()
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

    fn precision(&self) -> String {
        self.tensors
            .values()
            .take(1)
            .fold(String::new(), |_acc, t| match t.dtype() {
                DType::BF16 => "bf16".to_string(),
                DType::F16 => "fp16".to_string(),
                DType::F32 => "fp32".to_string(),
                dtype => panic!("Could not get the precision for dtype: {:#?}", dtype),
            })
    }

    fn scale_lora_weight(
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

    fn scale_hada_weight(
        &self,
        base_name: &str,
        _device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let hada_w1_a = format!("{}.hada_w1_a", base_name);
        let hada_w1_b = format!("{}.hada_w1_b", base_name);
        let hada_w2_a = format!("{}.hada_w2_a", base_name);
        let hada_w2_b = format!("{}.hada_w2_b", base_name);

        // Tucker
        let hada_t1 = format!("{}.hada_t1", base_name);
        let hada_t2 = format!("{}.hada_t2", base_name);

        let lora_alpha = format!("{}.alpha", base_name);

        let w1_a = self
            .tensors
            .get(&hada_w1_a)
            .ok_or_else(|| candle_core::Error::Msg("no hada w1_a".to_string()))?;

        let w1_b = self
            .tensors
            .get(&hada_w1_b)
            .ok_or_else(|| candle_core::Error::Msg("no hada w1_b".to_string()))?;

        let w2_a = self
            .tensors
            .get(&hada_w2_a)
            .ok_or_else(|| candle_core::Error::Msg("no hada w2_a".to_string()))?;

        let w2_b = self
            .tensors
            .get(&hada_w2_b)
            .ok_or_else(|| candle_core::Error::Msg("no hada w2_b".to_string()))?;

        let alpha = self
            .tensors
            .get(&lora_alpha)
            .ok_or_else(|| candle_core::Error::Msg("no hada alpha".to_string()))?;

        let dims = w1_b.dims();

        let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;

        if let Some(t1) = self.tensors.get(&hada_t1) {
            let t2 = self
                .tensors
                .get(&hada_t2)
                .ok_or_else(|| candle_core::Error::Msg("no hada tucker t2".to_string()))?;

            let one = w1_a.matmul(w1_b)?;
            let two = w2_a.matmul(w2_b)?;

            let t1_w1 = t1.mul(&one)?;
            let t2_w2 = t2.mul(&two)?;

            t1_w1.matmul(&t2_w2)?.mul(scale)
        } else {
            let one = w1_a.matmul(w1_b)?;
            let two = w2_a.matmul(w2_b)?;

            one.mul(two)?.mul(scale)
        }

        // if dims.len() == 2 {
        //     up.matmul(&down)?.mul(scale)
        // } else if dims[2] == 1 && dims[3] == 1 {
        //     up.squeeze(3)?
        //         .squeeze(2)?
        //         .matmul(&down.squeeze(3)?.squeeze(2)?)?
        //         .unsqueeze(2)?
        //         .unsqueeze(3)?
        //         .mul(scale)
        // } else {
        //     down.permute((1, 0, 2, 3))?
        //         .conv2d(&up, 0, 1, 1, 1)?
        //         .permute((1, 0, 2, 3))?
        //         .mul(scale)
        // }
    }

    fn scale_lokr_weight(
        &self,
        base_name: &str,
        _device: &Device,
    ) -> Result<Tensor, candle_core::Error> {
        let lokr_w1 = format!("{}.lokr_w1", base_name);
        let lokr_w2 = format!("{}.lokr_w2", base_name);

        let lokr_w1_a = format!("{}.lokr_w1_a", base_name);
        let lokr_w1_b = format!("{}.lokr_w1_b", base_name);

        let lokr_w2_a = format!("{}.lokr_w2_a", base_name);
        let lokr_w2_b = format!("{}.lokr_w2_b", base_name);

        let lokr_t2 = format!("{}.lokr_t2", base_name);

        let alpha_key = format!("{}.alpha", base_name);

        let alpha = self
            .tensors
            .get(&alpha_key)
            .ok_or_else(|| candle_core::Error::Msg("no lokr alpha".to_string()))?;

        if let Some(w1_a) = self.tensors.get(&lokr_w1_a) {
            let w1_b = self
                .tensors
                .get(&lokr_w1_b)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w1_b".to_string()))?;
            let w2_a = self
                .tensors
                .get(&lokr_w2_a)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w2_a".to_string()))?;

            let w2_b = self
                .tensors
                .get(&lokr_w2_b)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w2_b".to_string()))?;

            let w1 = w1_a.matmul(w1_b)?;
            let w2 = w2_a.matmul(w2_b)?;

            let dims = w1_b.dims();

            let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;
            w1.matmul(&w2)?.mul(scale)
        } else if let Ok(t2) = self.get(&lokr_t2) {
            let w2_a = self
                .tensors
                .get(&lokr_w2_a)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w2_a".to_string()))?;
            let w2_b = self
                .tensors
                .get(&lokr_w2_b)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w2_b".to_string()))?;
            w2_a.matmul(w2_b)?.mul(t2)
        } else {
            let w1 = self
                .tensors
                .get(&lokr_w1)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w1".to_string()))?;

            let w2 = self
                .tensors
                .get(&lokr_w2)
                .ok_or_else(|| candle_core::Error::Msg("no lokr w2".to_string()))?;

            let dims = w1.dims();

            let scale = alpha.to_dtype(DType::F64)?.to_scalar::<f64>()? / dims[0] as f64;

            w1.matmul(w2)?.mul(scale)
        }

        // if len(w2.shape) == 4:
        //     w1 = w1.unsqueeze(2).unsqueeze(2)
        // w2 = w2.contiguous()
        // rebuild = torch.kron(w1, w2)
        //
        // return rebuild * scale
    }

    fn scale_dora_weight(
        &self,
        weight: &Tensor,
        lora_dora_scale: &str,
        _device: &candle_core::Device,
    ) -> Result<Tensor, candle_core::Error> {
        if let Some(dora_scale) = self.tensors.get(lora_dora_scale) {
            weight.matmul(dora_scale)?.detach()
        } else {
            weight.detach()
        }
    }

    fn alphas(&self) -> HashSet<Alpha> {
        self.tensors
            .iter()
            .filter(|(k, _v)| k.contains("alpha"))
            .filter_map(|(_k, v)| {
                v.to_dtype(DType::F32)
                    .map(|v| v.to_scalar::<f32>())
                    .unwrap()
                    .ok()
            })
            .fold(HashSet::new(), |mut alphas: HashSet<Alpha>, v| {
                alphas.insert(Alpha(v));
                alphas
            })
    }

    fn dora_scales(&self) -> HashSet<DoRAScale> {
        self.tensors
            .iter()
            .filter(|(k, _v)| k.contains("dora_scale"))
            .filter_map(|(k, _v)| {
                self.get(k)
                    .unwrap()
                    .to_dtype(DType::F32)
                    .map(|v| v.to_scalar::<f32>())
                    .unwrap()
                    .ok()
            })
            .fold(HashSet::new(), |mut scales: HashSet<DoRAScale>, v| {
                scales.insert(DoRAScale(v));
                scales
            })
    }

    fn dims(&self) -> HashSet<u32> {
        self.tensors
            .iter()
            // Limiting to a single element for LoRA.
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

    fn shapes(&self) -> HashMap<String, Vec<usize>> {
        self.tensors.keys().fold(HashMap::new(), |mut acc, k| {
            acc.insert(k.to_string(), self.tensors.get(k).unwrap().dims().to_vec());
            acc
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

    fn load_file(filename: &str) -> Result<Vec<u8>, io::Error> {
        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_conv_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./lora_unet_down_blocks_1_resnets_1_conv2.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_hada_block_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./lora_unet_output_blocks_4_1_proj_in.safetensors";

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
    fn buffered_dims_scale_lora_weight_conv2d() {
        let buffer = load_test_conv_file().unwrap();

        // Arrange
        let lora_weight = BufferedLoRAWeight::new(buffer).unwrap();

        // Act

        let result =
            lora_weight.scale_lora_weight("lora_unet_down_blocks_1_resnets_1_conv2", &Device::Cpu);

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

    #[test]
    fn buffered_dims_scale_hada_weight_conv2d() {
        let buffer = load_test_hada_block_file().unwrap();

        // Arrange
        let lora_weight = BufferedLoRAWeight::new(buffer).unwrap();

        // Act

        let result = dbg!(
            lora_weight.scale_hada_weight("lora_unet_output_blocks_4_1_proj_in", &Device::Cpu)
        );

        // Assert
        assert!(result.is_ok());
        let scaled_tensor = result.unwrap();

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[640, 640]);

        // println!(
        //     "max {:#?}",
        //     scaled_tensor
        //         .flatten_all()
        //         .unwrap()
        //         .to_dtype(DType::F64)
        //         .unwrap()
        //         .to_vec0::<f64>()
        //         .iter()
        //         .copied()
        //         .fold(0_f64, |acc, v| { acc.max(dbg!(v)) })
        // );
        // Verify that the tensor values are within an acceptable range
        for value in scaled_tensor
            .flatten_all()
            .iter()
            .take(10)
            .filter_map(|v| v.to_dtype(DType::F64).and_then(|v| v.to_vec0::<i64>()).ok())
        {
            println!("{:#?}", value);
            assert!((0_i64..=8_i64).contains(&value));
            // println!("{:#?}", value);
            // break;
        }
    }

    #[test]
    fn empty_dora_scale_set() {
        let buffer = load_test_file().unwrap();

        // Arrange
        let lora_weight = LoRAWeight::new(buffer).unwrap();

        // Act
        let scales = lora_weight.dora_scales();

        assert_eq!(scales.len(), 0);
    }

    // #[test]
    // fn has_dora_scale() {
    //     let buffer = load_file("/mnt/900/lora/testing/women/women-2023-08-23-122633-56bab2a0.safetensors").unwrap();
    //
    //     // Arrange
    //     let lora_weight = LoRAWeight::new(buffer).unwrap();
    //
    //     // Act
    //     let scales = lora_weight.dora_scales();
    //
    //     assert_eq!(scales.len(), 1000);
    // }

    // #[test]
    // fn weight_norm_handles_scale_weight_lorafa() {
    //     // Arrange
    //     let buffer = load_file(
    //         "/mnt/900/lora/testing/woman.safetensors",
    //     )
    //     .unwrap();
    //
    //     let lora_weight = BufferedLoRAWeight::new(buffer).unwrap();
    //     let base_name = "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_k";
    //     let device = &Device::Cpu;
    //
    //     // Act
    //
    //
    //     // let result = lora_file.l1_norm::<f64>(base_name, device);
    //
    //     // Assert
    //     // assert!(result.is_err());
    //     //
    //     // assert_err!(result, Err(InspectorError::Candle(_)));
    // }
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
