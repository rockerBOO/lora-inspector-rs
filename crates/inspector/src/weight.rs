use candle_core::{
    safetensors::{load_buffer, BufferedSafetensors, Load},
    Device, Tensor,
};
use serde::{Deserialize, Serialize};
use std::ops::Div;
use std::{
    collections::{HashMap, HashSet},
    fmt::{Debug, Formatter},
};
use std::{fmt, ops::Mul};

use wasm_bindgen::prelude::*;

use crate::tensor::kron;

// BufferedLoRAWeight
//   Load weights only when required
// LoRAWeight

fn is_peft(keys: Vec<String>) -> bool {
    keys.into_iter()
        .take(10)
        .any(|k| k.contains("transformer.") || k.contains("diffusion_model."))
}

/// Converts tensor into DType thats compatible with candle
/// We convert BF16 to F32
fn to_compatible_dtype(tensor: &candle_core::Tensor) -> Result<Tensor, candle_core::Error> {
    match tensor.dtype() {
        candle_core::DType::BF16 => tensor.to_dtype(candle_core::DType::F32),
        _ => Ok(tensor.clone()),
    }
}

pub fn get_base_name(name: &str) -> String {
    name.split('.')
        .filter(|part| {
            !matches!(
                *part,
                "weight"
                // Kohya
                    | "lora_up"
                    | "lora_down"
                // PEFT
                    | "lora_A"
                    | "lora_B"
                // Lycoris
                    | "lokr_w1"
                    | "lokr_w2"
                    | "hada_w1_a"
                    | "hada_w1_b"
                    | "hada_w2_a"
                    | "oft_diag"
                    | "hada_w2_b"
                    | "alpha"
                    | "lokr_w1_a"
                    | "lokr_w1_b"
                    | "lokr_w2_a"
                    | "lokr_w2_b"
                    | "oft_blocks"
            )
        })
        .fold(String::new(), |acc, v| {
            if acc.is_empty() {
                v.to_owned()
            } else {
                format!("{acc}.{v}")
            }
        })
}

fn reshape_keep_first_dim(tensor: &Tensor) -> candle_core::Result<Tensor> {
    // Equivalent to tensor.reshape(tensor.size(0), -1)
    let shape = tensor.shape();
    let first_dim = shape.dims()[0];
    let remaining_size: usize = shape.dims()[1..].iter().product();
    tensor.reshape(&[first_dim, remaining_size])
}

fn reshape_keep_last_dim(tensor: &Tensor) -> candle_core::Result<Tensor> {
    // Equivalent to tensor.reshape(-1, tensor.size(-1))
    let shape = tensor.shape();
    let last_dim = shape.dims()[shape.dims().len() - 1];
    let remaining_size: usize = shape.dims()[..shape.dims().len() - 1].iter().product();
    tensor.reshape(&[remaining_size, last_dim])
}

/// Rebuild Tucker decomposition: einsum "i j ..., i p, j r -> p r ..."
/// This contracts tensor t with wa along dimension i and wb along dimension j
fn rebuild_tucker(t: &Tensor, wa: &Tensor, wb: &Tensor) -> candle_core::Result<Tensor> {
    // t shape: [i, j, ...]
    // wa shape: [i, p]
    // wb shape: [j, r]
    // output shape: [p, r, ...]

    // Method 1: Using two sequential matrix multiplications
    // Step 1: Contract wa with t along dimension i
    // wa.T @ t: [p, i] @ [i, j, ...] -> [p, j, ...]
    let wa_t = wa.transpose(0, 1)?; // [p, i]
    let temp = wa_t.matmul(t)?; // [p, j, ...]

    // Step 2: Contract temp with wb along dimension j
    // temp @ wb: [p, j, ...] @ [j, r] -> [p, r, ...]
    let result = temp.matmul(wb)?;

    Ok(result)
}

fn rebuild_lokr_w2(w2_a: &Tensor, w2_b: &Tensor) -> Result<Tensor, candle_core::Error> {
    // r, o, *k = w2b.shape
    let w2b_shape = w2_b.shape();
    let r = w2b_shape.dims()[0];
    let o = w2b_shape.dims()[1];
    let k_dims = &w2b_shape.dims()[2..]; // remaining dimensions

    // w2b.view(r, -1) - flatten all dimensions after r
    let k_total: usize = k_dims.iter().product();
    let w2b_reshaped = w2_b.reshape(&[r, o * k_total])?;

    // w2 = w2a @ w2b.view(r, -1)
    let w2_flat = w2_a.matmul(&w2b_reshaped)?;

    // w2.view(-1, o, *k) - reshape to [w2a_dim0, o, ...k_dims]
    let w2a_out_dim = w2_a.dim(0)?;
    let mut final_shape = vec![w2a_out_dim, o];
    final_shape.extend_from_slice(k_dims);
    w2_flat.reshape(final_shape)
}

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

#[derive(Debug, Copy, Clone, PartialEq, Hash)]
pub enum DType {
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DType::Bool => write!(f, "bool"),
            DType::U8 => write!(f, "u8"),
            DType::I8 => write!(f, "i8"),
            DType::I16 => write!(f, "i16"),
            DType::U16 => write!(f, "u16"),
            DType::F16 => write!(f, "fp16"),
            DType::BF16 => write!(f, "bf16"),
            DType::I32 => write!(f, "i32"),
            DType::U32 => write!(f, "u32"),
            DType::F32 => write!(f, "fp32"),
            DType::F64 => write!(f, "fp64"),
            DType::I64 => write!(f, "i64"),
            DType::U64 => write!(f, "u64"),
        }
    }
}

impl From<safetensors::Dtype> for DType {
    fn from(value: safetensors::Dtype) -> Self {
        match value {
            safetensors::Dtype::BOOL => Self::Bool,
            safetensors::Dtype::U8 => Self::U8,
            safetensors::Dtype::I8 => Self::I8,
            safetensors::Dtype::I16 => Self::I16,
            safetensors::Dtype::U16 => Self::U16,
            safetensors::Dtype::F16 => Self::F16,
            safetensors::Dtype::BF16 => Self::BF16,
            safetensors::Dtype::I32 => Self::I32,
            safetensors::Dtype::U32 => Self::U32,
            safetensors::Dtype::F32 => Self::F32,
            safetensors::Dtype::F64 => Self::F64,
            safetensors::Dtype::I64 => Self::I64,
            safetensors::Dtype::U64 => Self::U64,
            a => panic!("Unsupported Dtype {:?}", a),
        }
    }
}

impl From<candle_core::DType> for DType {
    fn from(value: candle_core::DType) -> Self {
        match value {
            candle_core::DType::U8 => Self::U8,
            candle_core::DType::F16 => Self::F16,
            candle_core::DType::BF16 => Self::BF16,
            candle_core::DType::U32 => Self::U32,
            candle_core::DType::F32 => Self::F32,
            candle_core::DType::F64 => Self::F64,
            candle_core::DType::I64 => Self::I64,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct DoRAScale(pub f32);

#[wasm_bindgen]
pub struct BufferedLoRAWeight {
    buffered: BufferedSafetensors,
    device: Device,
    format: LoRAFormat,
}

impl Debug for BufferedLoRAWeight {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        for (_key, tensor) in self.buffered.tensors() {
            tensor.fmt(f)?
        }

        Ok(())
    }
}

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
        let oft_blocks = &mut self.keys_by_key("oft_block");
        let glora_a1 = &mut self.keys_by_key("a1");
        let glora_b1 = &mut self.keys_by_key("b1");
        let glora_a2 = &mut self.keys_by_key("a2");
        let glora_b2 = &mut self.keys_by_key("b1");
        let mut keys = self.keys_by_key("weight");
        keys.append(hada);
        keys.append(lokr);
        keys.append(oft_diag);
        keys.append(oft_blocks);
        keys.append(glora_a1);
        keys.append(glora_b1);
        keys.append(glora_a2);
        keys.append(glora_b2);

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
        self.buffered.load(key, &self.device)
    }

    fn precision(&self) -> Option<DType> {
        self.buffered
            .tensors()
            .iter()
            .filter(|(k, _)| !k.contains("alpha"))
            .collect::<Vec<_>>()
            .first()
            .map(|(_key, t)| t.dtype().into())
    }

    fn scale_lora_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let up = self.up(base_name)?;
        let down = self.down(base_name)?;
        let alpha = self.alpha(base_name)?;

        let dims = down.dims();

        let scale = (alpha.0 / dims[0] as f32) as f64;

        if dims.len() == 2 {
            to_compatible_dtype(&up)?
                .matmul(&to_compatible_dtype(&down)?)?
                .detach()?
                .mul(scale)?
                .detach()
        } else if dims[2] == 1 && dims[3] == 1 {
            to_compatible_dtype(&up)?
                .squeeze(3)?
                .squeeze(2)?
                .matmul(&to_compatible_dtype(&down)?.squeeze(3)?.squeeze(2)?)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .mul(scale)
        } else {
            to_compatible_dtype(&down)?
                .permute((1, 0, 2, 3))?
                .conv2d(&to_compatible_dtype(&up)?, 0, 1, 1, 1)?
                .permute((1, 0, 2, 3))?
                .mul(scale)
        }
    }

    fn scale_hada_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let hada_w1_a = format!("{}.hada_w1_a", base_name);
        let hada_w1_b = format!("{}.hada_w1_b", base_name);
        let hada_w2_a = format!("{}.hada_w2_a", base_name);
        let hada_w2_b = format!("{}.hada_w2_b", base_name);

        // Tucker
        let hada_t1 = format!("{}.hada_t1", base_name);
        let hada_t2 = format!("{}.hada_t2", base_name);

        let lora_alpha = format!("{}.alpha", base_name);

        let w1_a = self.get(&hada_w1_a)?;
        let w1_b = self.get(&hada_w1_b)?;
        let w2_a = self.get(&hada_w2_a)?;
        let w2_b = self.get(&hada_w2_b)?;

        let alpha = self.get(&lora_alpha)?;

        let dims = w1_b.dims();

        let scale = alpha
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?
            / dims[0] as f64;

        let rebuilt = match (self.get(&hada_t1), self.get(&hada_t2)) {
            (Ok(t1), Ok(t2)) => {
                let rebuild1 = w1_b.transpose(0, 1)?.matmul(&t1.matmul(&w1_a)?)?;
                let rebuild2 = w2_b.transpose(0, 1)?.matmul(&t2.matmul(&w2_a)?)?;

                rebuild1.matmul(&rebuild2)?
            }
            _ => {
                let w1_d = reshape_keep_first_dim(&w1_b)?;
                let w1_u = reshape_keep_last_dim(&w1_a)?;
                let w2_d = reshape_keep_first_dim(&w2_b)?;
                let w2_u = reshape_keep_last_dim(&w2_a)?;

                w1_u.matmul(&w1_d)?.mul(&w2_u.matmul(&w2_d)?)?
            }
        };

        rebuilt.mul(scale)
    }

    fn scale_lokr_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let lokr_w1 = format!("{}.lokr_w1", base_name);
        let lokr_w2 = format!("{}.lokr_w2", base_name);

        let lokr_w1_a = format!("{}.lokr_w1_a", base_name);
        let lokr_w1_b = format!("{}.lokr_w1_b", base_name);

        let lokr_w2_a = format!("{}.lokr_w2_a", base_name);
        let lokr_w2_b = format!("{}.lokr_w2_b", base_name);

        let lokr_t2 = format!("{}.lokr_t2", base_name);

        let alpha_key = format!("{}.alpha", base_name);

        let alpha = self.get(&alpha_key)?;

        let w1 = self.get(&lokr_w1);
        let w2 = self.get(&lokr_w2);

        let (mut w1, w2, scale) = match (w1, w2) {
            (Ok(w1), Ok(w2)) => Ok((w1, w2, 1.0)),
            (
                Ok(w1),
                Err(candle_core::Error::SafeTensor(safetensors::SafeTensorError::TensorNotFound(
                    _,
                ))),
            ) => {
                // need to get w2
                let w2_a = self.get(&lokr_w2_a)?;
                let w2_b = self.get(&lokr_w2_b)?;

                let w2 = if let Ok(t1) = self.get(&lokr_t2) {
                    rebuild_tucker(&t1, &w2_a, &w2_b)?
                } else {
                    rebuild_lokr_w2(&w2_a, &w2_b)?
                };

                let dims = w2_a.dims();

                let scale = alpha
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    / dims[0] as f64;

                Ok((w1, w2, scale))
            }
            (
                Err(candle_core::Error::SafeTensor(safetensors::SafeTensorError::TensorNotFound(
                    _,
                ))),
                Ok(w2),
            ) => {
                let w1_a = self.get(&lokr_w1_a)?;
                let w1 = w1_a.matmul(&self.get(&lokr_w1_b)?)?;
                let dims = w1_a.dims();

                let scale = alpha
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    / dims[0] as f64;
                Ok((w1, w2, scale))
            }
            (
                Err(candle_core::Error::SafeTensor(safetensors::SafeTensorError::TensorNotFound(
                    _,
                ))),
                Err(candle_core::Error::SafeTensor(safetensors::SafeTensorError::TensorNotFound(
                    _,
                ))),
            ) => {
                let w1_a = self.get(&lokr_w1_a)?;
                // need to get w1 and w2
                let w1 = w1_a.matmul(&self.get(&lokr_w1_b)?)?;

                // need to get w2
                let w2_a = self.get(&lokr_w2_a)?;
                let w2_b = self.get(&lokr_w2_b)?;

                let w2 = if let Ok(t1) = self.get(&lokr_t2) {
                    rebuild_tucker(&t1, &w2_a, &w2_b)?
                } else {
                    rebuild_lokr_w2(&w2_a, &w2_b)?
                };

                let dims = w1_a.dims();

                let scale = alpha
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    / dims[0] as f64;

                Ok((w1, w2, scale))
            }
            _ => Err(candle_core::Error::CannotFindTensor {
                path: format!("{lokr_w1_a},{lokr_w1_b},{lokr_w2_a},{lokr_w2_b}"),
            }),
        }?;

        for _ in 0..w2.dims().len() - w1.dims().len() {
            w1 = w1.unsqueeze(candle_core::D::Minus1)?;
        }

        // TODO: Need to handle different size dimensions between w1 and w2
        let rebuild = kron(&w1, &w2.contiguous()?)?;

        if scale != 1.0 {
            rebuild.mul(scale)
        } else {
            Ok(rebuild)
        }
    }

    fn scale_glora_weights(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let w1a_key = format!("{}.a1.weight", base_name);
        let w1b_key = format!("{}.b1.weight", base_name);

        let w2a_key = format!("{}.a2.weight", base_name);
        let w2b_key = format!("{}.b2.weight", base_name);

        let alpha_key = format!("{}.alpha", base_name);

        let alpha = self.get(&alpha_key)?;

        let w1a = self.get(&w1a_key)?;
        let w1b = self.get(&w1b_key)?;

        let w2a = self.get(&w2a_key)?;
        let w2b = self.get(&w2b_key)?;

        let dims = w1b.dims();

        let scale = alpha
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?
            / dims[0] as f64;

        w2b.matmul(&w1b)?.add(&w2a.matmul(&w1a)?)?.mul(scale)
    }

    fn scale_diag_oft_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        // Diag OFT are not scaled by alpha
        let oft_diag_key = format!("{}.oft_diag", base_name);
        self.get(&oft_diag_key)
    }

    fn scale_boft_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        // Diag OFT are not scaled by alpha
        let oft_blocks_key = format!("{}.oft_blocks", base_name);

        self.get(&oft_blocks_key)
    }

    fn alphas(&self) -> HashSet<Alpha> {
        match self.format {
            LoRAFormat::Peft => self
                .buffered
                .tensors()
                .iter()
                .filter(|(k, _v)| k.contains("lora_A"))
                .filter_map(|(k, _v)| self.get(k).map(|v| v.dims().first().cloned()).ok())
                .fold(HashSet::new(), |mut alphas: HashSet<Alpha>, v| {
                    if let Some(alpha) = v {
                        alphas.insert(Alpha(alpha as f32));
                    }
                    alphas
                }),
            _ => self
                .buffered
                .tensors()
                .iter()
                .filter(|(k, _v)| k.contains("alpha"))
                .filter_map(|(k, _v)| {
                    self.get(k)
                        .map(|v| v.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>())
                        .ok()
                })
                .fold(HashSet::new(), |mut alphas: HashSet<Alpha>, v| {
                    if let Ok(v) = v {
                        alphas.insert(Alpha(v));
                    }
                    alphas
                }),
        }
    }

    fn dora_scale(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        self.get(&format!("{base_name}.dora_scale"))
    }

    fn dims(&self) -> HashSet<usize> {
        self.buffered
            .tensors()
            .iter()
            .filter_map(|(k, _v)| {
                if k.contains("lora_down") {
                    self.get(k).map(|v| v.dims()[0]).ok()
                } else if k.contains("hada_w1_b") {
                    self.get(k).map(|v| v.dims()[0]).ok()
                } else if k.contains("lokr_w1") {
                    self.get(k).map(|v| v.dims()[0]).ok()
                } else if k.contains("b1.weight") {
                    self.get(k).map(|v| v.dims()[0]).ok()
                } else if k.contains("oft_diag") {
                    self.get(k).map(|v| v.dims().last().copied()).ok().flatten()
                } else if k.contains("oft_blocks") {
                    self.get(k).map(|v| v.dims().last().copied()).ok().flatten()
                } else {
                    None
                }
            })
            .fold(HashSet::new(), |mut dims: HashSet<usize>, res| {
                dims.insert(res);
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

    fn up(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let lora_up = match self.format {
            LoRAFormat::Peft => format!("{}.lora_B.weight", base_name),
            _ => format!("{}.lora_up.weight", base_name),
        };
        self.get(&lora_up)
    }

    fn down(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let lora_down = match self.format {
            LoRAFormat::Peft => format!("{}.lora_A.weight", base_name),
            _ => format!("{}.lora_down.weight", base_name),
        };
        self.get(&lora_down)
    }

    fn alpha(&self, base_name: &str) -> Result<Alpha, candle_core::Error> {
        match self.format {
            LoRAFormat::Peft => Ok(Alpha(self.rank(base_name)? as f32)),
            _ => {
                let lora_alpha = format!("{}.alpha", base_name);
                let x = self.get(&lora_alpha);

                match x {
                    Ok(x) => Ok(Alpha(
                        x.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?,
                    )),
                    Err(e) => Err(e),
                }
            }
        }
    }

    fn rank(&self, base_name: &str) -> Result<usize, candle_core::Error> {
        self.down(base_name)?
            .dims()
            .first()
            .ok_or_else(|| candle_core::Error::Msg("No rank found".to_string()))
            .cloned()
    }

    fn format(&self) -> LoRAFormat {
        self.format
    }
}

impl BufferedLoRAWeight {
    pub fn new(buffer: Vec<u8>, device: &Device) -> Result<Self, candle_core::Error> {
        let buffered = BufferedSafetensors::new(buffer)?;
        let keys = buffered
            .tensors()
            .iter()
            .take(10)
            .map(|(k, _v)| k.clone())
            .collect();

        let format = if is_peft(keys) {
            LoRAFormat::Peft
        } else {
            LoRAFormat::Kohya
        };

        Ok(Self {
            buffered,
            device: device.clone(),
            format,
        })
    }

    pub fn load(&self, name: &str) -> Result<Tensor, candle_core::Error> {
        self.buffered.get(name)?.load(&self.device)
    }
}

pub trait WeightKey {
    fn keys(&self) -> Vec<String>;
    fn keys_by_key(&self, key: &str) -> Vec<String>;
    #[allow(dead_code)]
    fn up_keys(&self) -> Vec<String>;
    #[allow(dead_code)]
    fn down_keys(&self) -> Vec<String>;
    fn unet_keys(&self) -> Vec<String>;
    fn text_encoder_keys(&self) -> Vec<String>;
    fn weight_keys(&self) -> Vec<String>;
    fn alpha_keys(&self) -> Vec<String>;
    fn base_names(&self) -> Vec<String>;
}

pub trait Weight {
    fn up(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;
    fn down(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;
    fn alpha(&self, base_name: &str) -> Result<Alpha, candle_core::Error>;
    fn rank(&self, base_name: &str) -> Result<usize, candle_core::Error>;

    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error>;

    fn format(&self) -> LoRAFormat;
    fn precision(&self) -> Option<DType>;
    fn scale_lora_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;
    fn scale_glora_weights(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;
    #[allow(dead_code)]
    fn scale_hada_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;
    #[allow(dead_code)]
    fn scale_lokr_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;

    fn scale_diag_oft_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;
    fn scale_boft_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error>;

    fn alphas(&self) -> HashSet<Alpha>;
    #[allow(dead_code)]
    fn dora_scale(&self, key: &str) -> Result<Tensor, candle_core::Error>;
    fn dims(&self) -> HashSet<usize>;
    #[allow(dead_code)]
    fn shapes(&self) -> HashMap<String, Vec<usize>>;
}

/// Tensor weights
#[derive(Debug, Clone)]
pub struct LoRAWeight {
    tensors: HashMap<String, Tensor>,
    format: LoRAFormat,
}

impl LoRAWeight {
    #[allow(dead_code)]
    pub fn new(buffer: Vec<u8>) -> Result<Self, candle_core::Error> {
        let tensors = load_buffer(&buffer, &Device::Cpu)?;

        let keys = tensors.iter().take(10).map(|(k, _v)| k.clone()).collect();

        let format = if is_peft(keys) {
            LoRAFormat::Peft
        } else {
            LoRAFormat::Kohya
        };

        Ok(Self { tensors, format })
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

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum LoRAFormat {
    Kohya,
    Lycoris,
    Peft,
}

impl Weight for LoRAWeight {
    fn up(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let lora_up = match self.format {
            LoRAFormat::Peft => format!("{}.lora_B.weight", base_name),
            _ => format!("{}.lora_up.weight", base_name),
        };
        self.tensors
            .get(&lora_up)
            .ok_or_else(|| candle_core::Error::Msg("lora up not found".to_string()))
            .cloned()
    }

    fn down(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let lora_down = match self.format {
            LoRAFormat::Peft => format!("{}.lora_A.weight", base_name),
            _ => format!("{}.lora_down.weight", base_name),
        };
        self.tensors
            .get(&lora_down)
            .ok_or_else(|| candle_core::Error::Msg("lora down not found".to_string()))
            .cloned()
    }

    fn alpha(&self, base_name: &str) -> Result<Alpha, candle_core::Error> {
        match self.format {
            LoRAFormat::Peft => Ok(Alpha(self.rank(base_name)? as f32)),
            _ => {
                let lora_alpha = format!("{}.alpha", base_name);
                let x = self.tensors.get(&lora_alpha);

                match x {
                    Some(x) => Ok(Alpha(x.to_scalar::<f32>()?)),
                    None => Err(candle_core::Error::Msg("No alpha found".to_string())),
                }
            }
        }
    }

    fn get(&self, key: &str) -> Result<Tensor, candle_core::Error> {
        self.tensors
            .get(key)
            .ok_or_else(|| candle_core::Error::Msg("Could not load error".to_string()))
            .cloned()
    }

    fn format(&self) -> LoRAFormat {
        self.format
    }

    fn precision(&self) -> Option<DType> {
        self.tensors
            .values()
            .take(1)
            .fold(None, |_acc, v| Some(v.dtype().into()))
    }

    fn scale_lora_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let up = self.up(base_name)?;
        let down = self.down(base_name)?;
        let alpha = self.alpha(base_name)?;

        let dims = down.dims();
        let rank = self.rank(base_name)?;

        let scale = alpha.0.div(rank as f32) as f64;

        if dims.len() == 2 {
            to_compatible_dtype(&up)?
                .matmul(&to_compatible_dtype(&down)?)?
                .mul(scale)
        } else if dims[2] == 1 && dims[3] == 1 {
            to_compatible_dtype(&up)?
                .squeeze(3)?
                .squeeze(2)?
                .matmul(&to_compatible_dtype(&down)?.squeeze(3)?.squeeze(2)?)?
                .unsqueeze(2)?
                .unsqueeze(3)?
                .mul(scale)

            // Convolution
        } else {
            to_compatible_dtype(&down)?
                .permute((1, 0, 2, 3))?
                .conv2d(&to_compatible_dtype(&up)?, 0, 1, 1, 1)?
                .permute((1, 0, 2, 3))?
                .mul(scale)
        }
    }

    fn scale_glora_weights(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        let w1a_key = format!("{}.a1.weight", base_name);
        let w1b_key = format!("{}.b1.weight", base_name);

        let w2a_key = format!("{}.a2.weight", base_name);
        let w2b_key = format!("{}.b2.weight", base_name);

        let alpha_key = format!("{}.alpha", base_name);

        let alpha = self
            .tensors
            .get(&alpha_key)
            .ok_or_else(|| candle_core::Error::Msg("no glora alpha".to_string()))?;

        let w1a = self
            .tensors
            .get(&w1a_key)
            .ok_or_else(|| candle_core::Error::Msg("no glora w1a weight".to_string()))?;
        let w1b = self
            .tensors
            .get(&w1b_key)
            .ok_or_else(|| candle_core::Error::Msg("no glora w1b weight".to_string()))?;

        let w2a = self
            .tensors
            .get(&w2a_key)
            .ok_or_else(|| candle_core::Error::Msg("no glora w2a weight".to_string()))?;

        let w2b = self
            .tensors
            .get(&w2b_key)
            .ok_or_else(|| candle_core::Error::Msg("no glora w2b weight".to_string()))?;

        let dims = w1b.dims();

        let scale = alpha
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?
            / dims[0] as f64;

        w2b.matmul(w1b)?.add(&w2a.matmul(w1a)?)?.mul(scale)
    }

    fn scale_hada_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
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

        let scale = alpha
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?
            / dims[0] as f64;

        let rebuilt = match (self.get(&hada_t1), self.get(&hada_t2)) {
            (Ok(t1), Ok(t2)) => {
                let rebuild1 = w1_b.transpose(0, 1)?.matmul(&t1.matmul(w1_a)?)?;
                let rebuild2 = w2_b.transpose(0, 1)?.matmul(&t2.matmul(w2_a)?)?;

                rebuild1.matmul(&rebuild2)?
            }
            _ => {
                let w1_d = reshape_keep_first_dim(&w1_a)?;
                let w1_u = reshape_keep_last_dim(&w1_b)?;
                let w2_d = reshape_keep_first_dim(&w2_a)?;
                let w2_u = reshape_keep_last_dim(&w2_b)?;

                w1_u.matmul(&w1_d)?.mul(&w2_u.matmul(&w2_d)?)?
            }
        };

        rebuilt.mul(scale)
    }

    fn scale_lokr_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
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

        let w1 = self.tensors.get(&lokr_w1);
        let w2 = self.tensors.get(&lokr_w2);

        let (w1, w2, scale) = match (w1, w2) {
            (None, Some(w2)) => {
                let w1_a = self.get(&lokr_w1_a)?;
                let w1 = w1_a.matmul(&self.get(&lokr_w1_b)?)?;
                let dims = w1_a.dims();

                let scale = alpha
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    / dims[0] as f64;
                (w1, w2.clone(), scale)
            }
            (None, None) => {
                let w1_a = self.get(&lokr_w1_a)?;
                // need to get w1 and w2
                let w1 = w1_a.matmul(&self.get(&lokr_w1_b)?)?;

                // need to get w2
                let w2_a = self.get(&lokr_w2_a)?;
                let w2_b = self.get(&lokr_w2_b)?;

                let w2 = if let Ok(t1) = self.get(&lokr_t2) {
                    rebuild_tucker(&t1, &w2_a, &w2_b)?
                } else {
                    rebuild_lokr_w2(&w2_a, &w2_b)?
                };

                let dims = w1_a.dims();

                let scale = alpha
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    / dims[0] as f64;

                (w1, w2, scale)
            }
            (Some(w1), Some(w2)) => (w1.clone(), w2.clone(), 1.0),
            (Some(w1), None) => {
                // need to get w2
                let w2_a = self.get(&lokr_w2_a)?;
                let w2_b = self.get(&lokr_w2_b)?;

                let w2 = if let Ok(t1) = self.get(&lokr_t2) {
                    rebuild_tucker(&t1, &w2_a, &w2_b)?
                } else {
                    rebuild_lokr_w2(&w2_a, &w2_b)?
                };

                let dims = w2_a.dims();

                let scale = alpha
                    .to_dtype(candle_core::DType::F64)?
                    .to_scalar::<f64>()?
                    / dims[0] as f64;

                (w1.clone(), w2, scale)
            }
        };

        // TODO: Need to handle different size dimensions between w1 and w2
        let rebuild = kron(&w1, &w2.contiguous()?)?;

        if scale != 1.0 {
            rebuild.mul(scale)
        } else {
            Ok(rebuild)
        }
    }

    fn scale_diag_oft_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        // Diag OFT are not scaled by alpha
        let oft_diag_key = format!("{}.oft_diag", base_name);

        self.get(&oft_diag_key)
    }

    fn scale_boft_weight(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        // Diag OFT are not scaled by alpha
        let oft_blocks_key = format!("{}.oft_blocks", base_name);

        self.get(&oft_blocks_key)
    }

    fn rank(&self, base_name: &str) -> Result<usize, candle_core::Error> {
        self.down(base_name)?
            .dims()
            .first()
            .ok_or_else(|| candle_core::Error::Msg("No rank found".to_string()))
            .cloned()
    }

    fn alphas(&self) -> HashSet<Alpha> {
        match self.format {
            LoRAFormat::Peft => self
                .tensors
                .iter()
                .filter(|(k, _v)| k.contains("lora_A"))
                // alpha is dim
                .filter_map(|(k, _v)| self.get(k).map(|v| v.dims().first().cloned()).ok())
                .fold(HashSet::new(), |mut alphas: HashSet<Alpha>, v| {
                    if let Some(alpha) = v {
                        alphas.insert(Alpha(alpha as f32));
                    }
                    alphas
                }),
            _ => self
                .tensors
                .iter()
                .filter(|(k, _v)| k.contains("alpha"))
                .filter_map(|(_k, v)| {
                    v.to_dtype(candle_core::DType::F32)
                        .map(|v| v.to_scalar::<f32>())
                        .unwrap()
                        .ok()
                })
                .fold(HashSet::new(), |mut alphas: HashSet<Alpha>, v| {
                    alphas.insert(Alpha(v));
                    alphas
                }),
        }
    }

    fn dora_scale(&self, base_name: &str) -> Result<Tensor, candle_core::Error> {
        self.get(&format!("{base_name}.dora_scale"))
    }

    fn dims(&self) -> HashSet<usize> {
        self.tensors
            .iter()
            .filter_map(|(k, v)| {
                if k.contains("lora_down")
                    || k.contains("hada_w1_b")
                    || k.contains("lokr_w1")
                    || k.contains("b1.weight")
                {
                    v.dims().first()
                } else if k.contains("oft_diag") || k.contains("oft_blocks") {
                    v.dims().last()
                // PEFT
                } else if k.contains("lora_A") {
                    v.dims().first()
                } else {
                    None
                }
            })
            .fold(HashSet::new(), |mut dims: HashSet<usize>, res| {
                dims.insert(*res);
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

    use crate::norms;

    use super::*;

    #[test]
    fn get_base_name_test() {
        let base_name = get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.lora_up.weight");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");

        let base_name =
            get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.lora_down.weight");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");

        let base_name =
            get_base_name("lora_te1_text_model_encoder_layers_5_self_attn_q_proj.hada_w1_a");
        assert_eq!(
            base_name,
            "lora_te1_text_model_encoder_layers_5_self_attn_q_proj"
        );

        let base_name =
            get_base_name("lora_te1_text_model_encoder_layers_5_self_attn_q_proj.lokr_w1");
        assert_eq!(
            base_name,
            "lora_te1_text_model_encoder_layers_5_self_attn_q_proj"
        );

        let base_name =
            get_base_name("lora_te1_text_model_encoder_layers_5_self_attn_q_proj.oft_diag");
        assert_eq!(
            base_name,
            "lora_te1_text_model_encoder_layers_5_self_attn_q_proj"
        );

        let base_name = get_base_name("lora_unet_up_blocks_1_attentions_1_proj_out.alpha");
        assert_eq!(base_name, "lora_unet_up_blocks_1_attentions_1_proj_out");
    }

    // fn load_keys_json() -> serde_json::Result<Vec<String>> {
    //     let keys = fs::read_to_string("./keys.json").expect("to read the keys json");
    //     serde_json::from_str::<Vec<String>>(&keys)
    // }

    // #[test]
    // fn test_key_parsing() {
    //     let keys: Vec<String> = load_keys_json().unwrap();
    //
    //     for key in keys {
    //         let successful_parse = KeyParser::parse(Rule::key, &key);
    //         assert!(successful_parse.is_ok());
    //     }
    // }
    //

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    // fn load_file(filename: &str) -> Result<Vec<u8>, io::Error> {
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }

    fn load_test_conv_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./lora_unet_down_blocks_1_resnets_1_conv2.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    // fn load_test_hada_block_file() -> Result<Vec<u8>, io::Error> {
    //     let filename = "./lora_unet_output_blocks_4_1_proj_in.safetensors";
    //
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }

    // fn load_test_dora_hada_block_file() -> Result<Vec<u8>, io::Error> {
    //     let filename = "dora_lora_unet_down_blocks_1_attentions_0_proj_in.safetensors";
    //
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }

    fn load_test_hada_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./loha.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    // fn load_test_hada_conv_file() -> Result<Vec<u8>, io::Error> {
    //     let filename = "./loha_conv.safetensors";
    //
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }
    //
    fn load_test_lokr_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./women-2024-05-14-200457-2450c2b2.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_diag_oft_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./diag_oft.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_boft_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./boft.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_boft_conv_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./boft_conv.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }
    //
    // fn load_test_glora_file() -> Result<Vec<u8>, io::Error> {
    //     let filename = "./glora.safetensors";
    //
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }
    //
    // fn load_test_glora_conv_file() -> Result<Vec<u8>, io::Error> {
    //     let filename = "./glora_conv.safetensors";
    //
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }

    fn load_test_peft_file() -> Result<Vec<u8>, io::Error> {
        let filename = "./urae_2k_adapter.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    #[test]
    fn lora_buffer_weight_peft() {
        let buffer = load_test_peft_file().unwrap();

        let lora_weight = LoRAWeight::new(buffer).unwrap();

        let dims = lora_weight.dims();

        assert_eq!(dims.len(), 1);
        assert_eq!(dims.get(&16), Some(&16));

        let alphas = lora_weight.alphas();

        assert_eq!(lora_weight.format(), LoRAFormat::Peft);

        assert_eq!(alphas.len(), 1);
        assert_eq!(alphas.get(&Alpha(16.0)), Some(&Alpha(16.0)));
    }

    #[test]
    fn lora_buffer_scale_weight_peft() {
        let buffer = load_test_peft_file().unwrap();

        let lora_weight = LoRAWeight::new(buffer).unwrap();

        let scaled_tensor = lora_weight
            .scale_lora_weight("transformer.transformer_blocks.8.attn.to_q")
            .unwrap();

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[3072, 3072]);

        // Verify that the tensor values are within an acceptable range
        for value in scaled_tensor.flatten_all().unwrap().to_vec0::<f32>().iter() {
            assert!(*value >= 0. && *value <= 2.);
        }
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
        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

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

        dbg!(&result_dims);

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
        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

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
        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

        // Act

        let result = lora_weight.scale_lora_weight("lora_unet_down_blocks_1_resnets_1_conv2");

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

    // HADA
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    #[test]
    fn buffered_scale_hada_weight() -> candle_core::Result<()> {
        let buffer = load_test_hada_file().unwrap();

        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

        let result = lora_weight.scale_hada_weight("lora_unet_up_blocks_1_attentions_0_proj_out");

        assert!(result.is_ok());

        let scaled_tensor = result.unwrap();

        assert_eq!(
            norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
            1.8109415
        );

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[1280, 1280]);

        Ok(())
    }

    // TODO: Need LoHa conv layers working
    // #[test]
    // fn buffered_conv_scale_hada_weight() {
    //     let buffer = load_test_hada_conv_file().unwrap();
    //
    //     let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();
    //
    //     let result =
    //         dbg!(lora_weight.scale_hada_weight("lora_unet_down_blocks_1_resnets_0_conv_shortcut",));
    //
    //     assert!(result.is_ok());
    //
    //     let scaled_tensor = result.unwrap();
    //
    //     assert_eq!(
    //         norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
    //         0.9683933
    //     );
    //
    //     // Verify that the tensor has the correct shape or dimensions
    //     assert_eq!(scaled_tensor.dims(), &[640, 640]);
    // }
    //
    // #[test]
    // fn buffered_dims_scale_hada_weight() {
    //     let buffer = load_test_hada_block_file().unwrap();
    //
    //     // Arrange
    //     let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();
    //
    //     // Act
    //
    //     let result = dbg!(lora_weight.scale_hada_weight("lora_unet_output_blocks_4_1_proj_in"));
    //
    //     assert!(result.is_ok());
    //     let scaled_tensor = result.unwrap();
    //
    //     assert_eq!(
    //         norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
    //         0.9683933
    //     );
    //
    //     // Verify that the tensor has the correct shape or dimensions
    //     assert_eq!(scaled_tensor.dims(), &[640, 640]);
    //
    //     // Verify that the tensor values are within an acceptable range
    //     for value in scaled_tensor.flatten_all().iter().take(10).filter_map(|v| {
    //         v.to_dtype(candle_core::DType::F64)
    //             .and_then(|v| v.to_vec0::<i64>())
    //             .ok()
    //     }) {
    //         println!("{:#?}", value);
    //         assert!((0_i64..=8_i64).contains(&value));
    //         // println!("{:#?}", value);
    //         // break;
    //     }
    // }

    // LoKr
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    #[test]
    fn buffered_scale_lokr_weight() -> candle_core::Result<()> {
        let buffer = load_test_lokr_file().unwrap();

        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

        let result = lora_weight.scale_lokr_weight("lora_unet_up_blocks_1_attentions_0_proj_out");

        assert!(result.is_ok());

        let scaled_tensor = result.unwrap();

        assert_eq!(
            norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
            4.809399
        );

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[1280, 1280, 1, 1]);

        Ok(())
    }

    #[test]
    fn buffered_scale_diag_oft_weight() -> candle_core::Result<()> {
        let buffer = load_test_diag_oft_file().unwrap();

        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

        let result =
            dbg!(lora_weight.scale_diag_oft_weight("lora_unet_up_blocks_1_attentions_0_proj_out"));

        assert!(result.is_ok());

        let scaled_tensor = result.unwrap();

        assert_eq!(
            norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
            35.77691
        );

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[160, 8, 8]);

        Ok(())
    }

    #[test]
    fn buffered_scale_boft_weight() -> candle_core::Result<()> {
        let buffer = load_test_boft_file().unwrap();

        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

        let result = lora_weight.scale_boft_weight(
            "lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_k",
        );

        assert!(result.is_ok());

        let scaled_tensor = result.unwrap();

        assert_eq!(
            norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
            0.116758764
        );

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[6, 32, 10, 10]);

        Ok(())
    }

    #[test]
    fn buffered_scale_boft_conv_weight() -> candle_core::Result<()> {
        let buffer = load_test_boft_conv_file().unwrap();

        let lora_weight = BufferedLoRAWeight::new(buffer, &Device::Cpu).unwrap();

        let result = dbg!(lora_weight.scale_boft_weight("lora_unet_up_blocks_3_resnets_1_conv1"));

        assert!(result.is_ok());

        let scaled_tensor = result.unwrap();

        assert_eq!(
            norms::l2::<f32>(&scaled_tensor.to_dtype(candle_core::DType::F32).unwrap()).unwrap(),
            0.0
        );

        // Verify that the tensor has the correct shape or dimensions
        assert_eq!(scaled_tensor.dims(), &[6, 32, 10, 10]);

        Ok(())
    }

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
