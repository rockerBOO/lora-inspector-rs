use candle_core::Device;
use safetensors::SafeTensorError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::{
    metadata::Metadata,
    network::NetworkType,
    norms::{l1, l2, matrix_norm},
    svd,
    weight::{self, BufferedLoRAWeight, Weight, WeightKey},
    InspectorError, Result,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerScale {
    pub base_name: String,
    pub eff_scale: f64,
    pub is_outlier: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// LoRA file buffer
#[derive(Debug)]
pub struct LoRAFile {
    filename: String,
    weights: Option<BufferedLoRAWeight>,
    scaled_weights: HashMap<String, candle_core::Tensor>,
    metadata: Option<Metadata>,
    header: Option<crate::header::HeaderIndex>,
}

// const WEIGHT_NOT_LOADED: &str = "Weight not loaded properly";

impl LoRAFile {
    pub fn new_from_buffer(buffer: &[u8], filename: &str, device: &Device) -> LoRAFile {
        let metadata = Metadata::new_from_buffer(buffer).map_err(|e| e.to_string());
        let header = crate::header::parse_header(buffer).ok().map(|(h, _)| h);

        LoRAFile {
            filename: filename.to_string(),
            weights: BufferedLoRAWeight::new(buffer.to_vec(), device)
                .map(Some)
                .unwrap_or_else(|_| None),
            scaled_weights: HashMap::new(),
            metadata: metadata.map(Some).unwrap_or_else(|_| None),
            header,
        }
    }

    /// Builds a `LoRAFile` from just the safetensors header — no tensor payload
    /// bytes required. `is_tensors_loaded()` will be `false`; call
    /// `reload_weights` with the full file buffer before any weight-value
    /// operation (`scale_weight`, `effective_scale`, `rank_metrics`, ...).
    pub fn new_from_header_buffer(buffer: &[u8], filename: &str) -> Result<LoRAFile> {
        let (header, meta_map) = crate::header::parse_header(buffer)?;

        Ok(LoRAFile {
            filename: filename.to_string(),
            weights: None,
            scaled_weights: HashMap::new(),
            metadata: Some(Metadata { metadata: meta_map }),
            header: Some(header),
        })
    }

    pub fn unload(&mut self) {
        self.weights = None;
        self.scaled_weights = HashMap::new();
    }

    pub fn reload_weights(&mut self, buffer: &[u8], device: &Device) {
        self.weights = BufferedLoRAWeight::new(buffer.to_vec(), device)
            .map(Some)
            .unwrap_or_else(|_| None);
    }

    pub fn is_tensors_loaded(&self) -> bool {
        self.weights.is_some()
    }

    pub fn filename(&self) -> String {
        self.filename.clone()
    }

    pub fn unet_keys(&self) -> Vec<String> {
        self.header
            .as_ref()
            .map(|h| h.unet_keys())
            .unwrap_or_default()
    }

    pub fn text_encoder_keys(&self) -> Vec<String> {
        self.header
            .as_ref()
            .map(|h| h.text_encoder_keys())
            .unwrap_or_default()
    }

    pub fn weight_keys(&self) -> Vec<String> {
        self.header
            .as_ref()
            .map(|h| h.weight_keys())
            .unwrap_or_default()
    }

    pub fn alpha_keys(&self) -> Vec<String> {
        self.header
            .as_ref()
            .map(|h| h.alpha_keys())
            .unwrap_or_default()
    }

    pub fn alphas(&self) -> HashSet<weight::Alpha> {
        self.weights
            .as_ref()
            .map(|weights| weights.alphas())
            .unwrap_or_default()
    }

    pub fn dims(&self) -> HashSet<usize> {
        self.header.as_ref().map(|h| h.dims()).unwrap_or_default()
    }

    pub fn precision(&self) -> Option<weight::DType> {
        self.header.as_ref().and_then(|h| h.precision())
    }

    pub fn keys(&self) -> Vec<String> {
        self.header.as_ref().map(|h| h.keys()).unwrap_or_default()
    }

    pub fn base_names(&self) -> Vec<String> {
        self.header
            .as_ref()
            .map(|h| h.base_names())
            .unwrap_or_default()
    }

    pub fn tensor_info(&self) -> Vec<TensorInfo> {
        self.header
            .as_ref()
            .map(|h| {
                h.tensor_info()
                    .into_iter()
                    .map(|(name, dtype, shape)| TensorInfo {
                        name,
                        dtype: dtype.to_string(),
                        shape,
                    })
                    .collect()
            })
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
        self.header
            .as_ref()
            .map(|h| h.format())
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

    pub fn effective_scale(&self, base_name: &str) -> Result<Option<f64>> {
        match self.weights.as_ref() {
            None => Ok(None),
            Some(_) => {
                if let Some(norm) = self.lora_frobenius_norm(base_name)? {
                    return Ok(Some(norm));
                }

                match self.scale_weight(base_name) {
                    Ok(t) => Ok(Some(self.l2_norm::<f64>(&t)?)),
                    Err(InspectorError::UnsupportedNetworkType) => Ok(None),
                    Err(InspectorError::Candle(candle_core::Error::SafeTensor(
                        SafeTensorError::TensorNotFound(_),
                    ))) => Ok(None),
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Frobenius norm of the reconstructed `up @ down` delta weight, computed from the
    /// low-rank factors directly instead of materializing the full `m x n` product.
    ///
    /// `||up @ down||_F^2 == trace((up^T up)(down down^T))`, which only requires the
    /// `rank x rank` Gram matrices of `up` and `down`. For layers with a huge output
    /// dimension (e.g. DiT modulation/projection layers), the full product can be
    /// hundreds of MB to reconstruct just to throw away everything but its norm, which
    /// can exhaust the wasm heap. Returns `Ok(None)` when the layer isn't a plain 2D
    /// LoRA up/down pair (conv/tucker/other decompositions fall back to full
    /// materialization since this identity doesn't directly apply to them).
    fn lora_frobenius_norm(&self, base_name: &str) -> Result<Option<f64>> {
        let weights = match self.weights.as_ref() {
            Some(weights) => weights,
            None => return Ok(None),
        };

        let up = match weights.up(base_name) {
            Ok(t) => t,
            Err(candle_core::Error::SafeTensor(SafeTensorError::TensorNotFound(_))) => {
                return Ok(None)
            }
            Err(e) => return Err(InspectorError::from(e)),
        };
        let down = match weights.down(base_name) {
            Ok(t) => t,
            Err(candle_core::Error::SafeTensor(SafeTensorError::TensorNotFound(_))) => {
                return Ok(None)
            }
            Err(e) => return Err(InspectorError::from(e)),
        };

        if up.dims().len() != 2 || down.dims().len() != 2 {
            return Ok(None);
        }

        let alpha = weights.alpha(base_name)?;
        let rank = down.dims()[0] as f64;
        let scale = alpha.0 as f64 / rank;

        let up = up.to_dtype(candle_core::DType::F64)?;
        let down = down.to_dtype(candle_core::DType::F64)?;

        let gram_u = up.t()?.matmul(&up)?; // rank x rank
        let gram_v = down.matmul(&down.t()?)?; // rank x rank

        let frob_sq: f64 = gram_u.mul(&gram_v)?.sum_all()?.to_scalar()?;

        Ok(Some(frob_sq.max(0.0).sqrt() * scale.abs()))
    }

    pub fn factorization_balance(&self, base_name: &str) -> Result<Option<f64>> {
        match self.weights.as_ref() {
            None => Ok(None),
            Some(weights) => {
                let up = match weights.up(base_name) {
                    Ok(t) => t,
                    Err(candle_core::Error::SafeTensor(SafeTensorError::TensorNotFound(_))) => {
                        return Ok(None)
                    }
                    Err(e) => return Err(InspectorError::from(e)),
                };
                let down = match weights.down(base_name) {
                    Ok(t) => t,
                    Err(candle_core::Error::SafeTensor(SafeTensorError::TensorNotFound(_))) => {
                        return Ok(None)
                    }
                    Err(e) => return Err(InspectorError::from(e)),
                };
                let up_norm = self.matrix_norm::<f64>(&up)?;
                let down_norm = self.matrix_norm::<f64>(&down)?;
                if down_norm < f64::EPSILON {
                    return Ok(None);
                }
                Ok(Some(up_norm / down_norm))
            }
        }
    }

    pub fn rank_metrics(&self, base_name: &str) -> Result<Option<svd::RankMetrics>> {
        match self.weights.as_ref() {
            None => Ok(None),
            Some(weights) => {
                let up = match weights.up(base_name) {
                    Ok(t) => t,
                    Err(candle_core::Error::SafeTensor(SafeTensorError::TensorNotFound(_))) => {
                        return Ok(None)
                    }
                    Err(e) => return Err(InspectorError::from(e)),
                };
                let down = match weights.down(base_name) {
                    Ok(t) => t,
                    Err(candle_core::Error::SafeTensor(SafeTensorError::TensorNotFound(_))) => {
                        return Ok(None)
                    }
                    Err(e) => return Err(InspectorError::from(e)),
                };
                Ok(Some(svd::rank_metrics(&up, &down)?))
            }
        }
    }

    pub fn effective_scales_all(&self) -> Vec<LayerScale> {
        let scales: Vec<(String, f64)> = self
            .base_names()
            .into_iter()
            .filter_map(|name| {
                self.effective_scale(&name)
                    .ok()
                    .flatten()
                    .map(|s| (name, s))
            })
            .collect();

        if scales.is_empty() {
            return vec![];
        }

        let mut sorted_vals: Vec<f64> = scales.iter().map(|(_, s)| *s).collect();
        sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = {
            let n = sorted_vals.len();
            if n % 2 == 0 {
                (sorted_vals[n / 2 - 1] + sorted_vals[n / 2]) / 2.0
            } else {
                sorted_vals[n / 2]
            }
        };
        // If median is near zero, use a small absolute floor to avoid flagging everything
        let threshold = if median < 1e-10 { 1e-10 } else { 1.5 * median };

        scales
            .into_iter()
            .map(|(base_name, eff_scale)| LayerScale {
                is_outlier: eff_scale > threshold,
                base_name,
                eff_scale,
            })
            .collect()
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
    fn effective_scale_is_l2_of_scaled_weight() -> crate::Result<()> {
        let file = "edgWar40KAdeptaSororitas.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);
        let base_name = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";

        let eff = lora_file.effective_scale(base_name)?.unwrap();
        let scaled = lora_file.scale_weight(base_name)?;
        let l2 = lora_file.l2_norm::<f64>(&scaled)?;
        // `effective_scale` takes a low-rank shortcut (trace of small Gram matrices)
        // instead of materializing the full up @ down product, so it sums the same
        // quantity in a different order than `l2_norm` and only agrees up to f32
        // rounding, not bit-for-bit.
        assert!((eff - l2).abs() / l2.abs() < 1e-5, "eff={eff} l2={l2}");
        Ok(())
    }

    #[test]
    fn factorization_balance_reasonable() -> crate::Result<()> {
        let file = "edgWar40KAdeptaSororitas.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);
        let base_name = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";
        let bal = lora_file.factorization_balance(base_name)?.unwrap();
        assert!(bal > 0.1 && bal < 10.0, "balance={}", bal);
        Ok(())
    }

    #[test]
    fn rank_metrics_returns_valid_health() -> crate::Result<()> {
        let file = "edgWar40KAdeptaSororitas.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);
        let base_name = "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj";
        let metrics = lora_file.rank_metrics(base_name)?.unwrap();
        assert!(metrics.balance > 0.0 && metrics.balance <= 1.0);
        assert!(metrics.top1_energy > 0.0 && metrics.top1_energy <= 1.0);
        assert!(
            metrics.effective_rank >= 1.0,
            "effective_rank={}",
            metrics.effective_rank
        );
        Ok(())
    }

    #[test]
    fn effective_scales_all_returns_one_per_base_name() -> crate::Result<()> {
        let file = "edgWar40KAdeptaSororitas.safetensors";
        let buffer = load_file(file)?;
        let lora_file = LoRAFile::new_from_buffer(&buffer, file, &Device::Cpu);

        let results = lora_file.effective_scales_all();
        // Must have at most one entry per base_name (some may not resolve)
        assert!(results.len() <= lora_file.base_names().len());
        // For a uniform LoRA file, all layers should resolve
        assert!(!results.is_empty(), "expected at least some layers");
        // All eff_scale values must be non-negative
        assert!(results.iter().all(|r| r.eff_scale >= 0.0));
        // is_outlier field must be consistent: outliers have higher eff_scale than non-outliers
        // (at minimum the field is present and boolean)
        let max_non_outlier = results
            .iter()
            .filter(|r| !r.is_outlier)
            .map(|r| r.eff_scale)
            .fold(0.0f64, f64::max);
        let min_outlier = results
            .iter()
            .filter(|r| r.is_outlier)
            .map(|r| r.eff_scale)
            .fold(f64::MAX, f64::min);
        // If there are outliers, all outliers must be above all non-outliers
        if results.iter().any(|r| r.is_outlier) {
            assert!(
                min_outlier > max_non_outlier,
                "outlier min={} should be > non-outlier max={}",
                min_outlier,
                max_non_outlier
            );
        }
        Ok(())
    }

    #[test]
    fn outlier_threshold_logic() {
        // Synthetic: scales [1.0, 1.0, 1.0, 1.0, 5.0]
        // Median = 1.0, threshold = 1.5 * 1.0 = 1.5
        // Only 5.0 > 1.5 → 1 outlier
        let scales = vec![1.0f64, 1.0, 1.0, 1.0, 5.0];
        let mut sorted = scales.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        let threshold = if median < 1e-10 { 1e-10 } else { 1.5 * median };
        let outliers: Vec<bool> = scales.iter().map(|&s| s > threshold).collect();
        assert_eq!(outliers, vec![false, false, false, false, true]);
        assert!((median - 1.0).abs() < 1e-10);
        assert!((threshold - 1.5).abs() < 1e-10);
    }

    fn header_only_bytes(buffer: &[u8]) -> &[u8] {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        &buffer[0..8 + header_len]
    }

    #[test]
    fn header_only_load_exposes_keys_without_weights() -> crate::Result<()> {
        let buffer = load_test_file()?;
        let lora_file =
            LoRAFile::new_from_header_buffer(header_only_bytes(&buffer), "boo.safetensors")?;

        assert!(!lora_file.is_tensors_loaded());
        assert!(!lora_file.keys().is_empty());
        assert!(!lora_file.base_names().is_empty());
        assert!(!lora_file.unet_keys().is_empty());
        assert!(!lora_file.tensor_info().is_empty());

        Ok(())
    }

    #[test]
    fn header_only_load_keys_match_full_buffer_load() -> crate::Result<()> {
        let buffer = load_test_file()?;
        let header_file =
            LoRAFile::new_from_header_buffer(header_only_bytes(&buffer), "boo.safetensors")?;
        let full_file = LoRAFile::new_from_buffer(&buffer, "boo.safetensors", &Device::Cpu);

        let mut header_keys = header_file.keys();
        let mut full_keys = full_file.keys();
        header_keys.sort();
        full_keys.sort();
        assert_eq!(header_keys, full_keys);

        assert_eq!(header_file.dims(), full_file.dims());
        assert_eq!(header_file.format(), full_file.format());
        assert_eq!(header_file.precision(), full_file.precision());

        Ok(())
    }

    #[test]
    fn header_only_load_rejects_truncated_buffer() {
        let result = LoRAFile::new_from_header_buffer(&[1_u8, 2, 3], "boo.safetensors");
        assert!(result.is_err());
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

    #[test]
    #[ignore] // Only run with: cargo test -- --ignored
    fn load_minimax_h3_file() -> crate::Result<()> {
        // Regression test for a diffusers-native PEFT LoRA (lora_A/lora_B with a
        // ".default." adapter-name segment, bare "transformer_blocks.N" and
        // "token_refiner.refiner_blocks.N" keys) that currently parses as 0 layers.
        let file = "/mnt/900/lora/h3/minimax_h3_fl2v_turbo_4step_v0.1.safetensors";
        let buffer = load_file(file)?;
        let filename = String::from(file);
        let lora_file = LoRAFile::new_from_buffer(&buffer, &filename, &Device::Cpu);

        assert!(lora_file.is_tensors_loaded());
        assert_eq!(lora_file.format(), weight::LoRAFormat::Peft);

        let base_names = lora_file.base_names();
        assert!(!base_names.is_empty(), "expected some base names");
        assert!(
            base_names.contains(&"transformer_blocks.0.attn.to_q".to_string()),
            "base names should not retain the adapter name segment: {:?}",
            base_names.iter().take(5).collect::<Vec<_>>()
        );

        let scaled = lora_file
            .scale_weight("transformer_blocks.0.attn.to_q")
            .expect("could not scale weight");
        assert_eq!(scaled.dims(), &[7168, 5376]);

        let scales = lora_file.effective_scales_all();
        assert!(!scales.is_empty(), "expected at least some layers");

        assert!(!lora_file.dims().is_empty(), "expected some LoRA ranks");

        Ok(())
    }
}
