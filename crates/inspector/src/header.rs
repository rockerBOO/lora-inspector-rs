use std::collections::{HashMap, HashSet};

use crate::weight::{get_base_name, is_peft, DType, LoRAFormat};
use crate::InspectorError;

/// A tensor's shape/dtype as recorded in the safetensors header, without its
/// payload bytes.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub dtype: DType,
    pub shape: Vec<usize>,
}

/// An index of every tensor's name, dtype and shape parsed from a safetensors
/// header, independent of whether the tensor payload bytes are resident anywhere.
#[derive(Debug, Clone)]
pub struct HeaderIndex {
    tensors: HashMap<String, TensorEntry>,
    format: LoRAFormat,
}

impl HeaderIndex {
    pub fn keys(&self) -> Vec<String> {
        self.tensors.keys().cloned().collect()
    }

    pub fn keys_by_key(&self, key: &str) -> Vec<String> {
        self.tensors
            .keys()
            .filter(|k| k.contains(key))
            .cloned()
            .collect()
    }

    pub fn weight_keys(&self) -> Vec<String> {
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

    pub fn unet_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_unet")
    }

    pub fn text_encoder_keys(&self) -> Vec<String> {
        self.keys_by_key("lora_te")
    }

    pub fn alpha_keys(&self) -> Vec<String> {
        self.keys_by_key("alpha")
    }

    pub fn base_names(&self) -> Vec<String> {
        self.weight_keys()
            .iter()
            .map(|name| get_base_name(name))
            .collect::<HashSet<String>>()
            .into_iter()
            .collect()
    }

    pub fn dims(&self) -> HashSet<usize> {
        self.tensors
            .iter()
            .filter_map(|(k, e)| {
                if k.contains("lora_down")
                    || k.contains("hada_w1_b")
                    || k.contains("lokr_w1")
                    || k.contains("b1.weight")
                    || k.contains("lora_A")
                {
                    e.shape.first().copied()
                } else if k.contains("oft_diag") || k.contains("oft_blocks") {
                    e.shape.last().copied()
                } else {
                    None
                }
            })
            .collect()
    }

    pub fn precision(&self) -> Option<DType> {
        self.tensors
            .iter()
            .filter(|(k, _)| !k.contains("alpha"))
            .collect::<Vec<_>>()
            .first()
            .map(|(_, e)| e.dtype)
    }

    pub fn format(&self) -> LoRAFormat {
        self.format
    }

    /// `(name, dtype, shape)` for every tensor in the file, sorted by name for a
    /// stable UI ordering.
    pub fn tensor_info(&self) -> Vec<(String, DType, Vec<usize>)> {
        let mut info: Vec<_> = self
            .tensors
            .iter()
            .map(|(name, e)| (name.clone(), e.dtype, e.shape.clone()))
            .collect();
        info.sort_by(|a, b| a.0.cmp(&b.0));
        info
    }
}

/// Parses only the safetensors header — the 8-byte little-endian length prefix
/// followed by that many bytes of header JSON — without requiring the tensor
/// payload bytes to be present. `buffer` only needs to contain the first
/// `8 + header_len` bytes of the file; anything beyond that is ignored.
pub fn parse_header(
    buffer: &[u8],
) -> crate::Result<(HeaderIndex, Option<HashMap<String, String>>)> {
    if buffer.len() < 8 {
        return Err(InspectorError::Msg(
            "buffer too small to contain a safetensors header".to_string(),
        ));
    }

    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buffer[0..8]);
    let header_len = u64::from_le_bytes(len_bytes) as usize;

    let stop = 8usize
        .checked_add(header_len)
        .ok_or_else(|| InspectorError::Msg("safetensors header length overflow".to_string()))?;
    if stop > buffer.len() {
        return Err(InspectorError::Msg(
            "buffer does not contain the full safetensors header JSON".to_string(),
        ));
    }

    let json_str = std::str::from_utf8(&buffer[8..stop])
        .map_err(|_| InspectorError::Msg("safetensors header is not valid UTF-8".to_string()))?;
    let metadata: safetensors::tensor::Metadata = serde_json::from_str(json_str)?;

    let sample_keys: Vec<String> = metadata.tensors().keys().take(10).cloned().collect();
    let format = if is_peft(sample_keys) {
        LoRAFormat::Peft
    } else {
        LoRAFormat::Kohya
    };

    let tensors = metadata
        .tensors()
        .into_iter()
        .map(|(name, info)| {
            (
                name,
                TensorEntry {
                    dtype: info.dtype.into(),
                    shape: info.shape.clone(),
                },
            )
        })
        .collect();

    Ok((HeaderIndex { tensors, format }, metadata.metadata().clone()))
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::{fs::File, io};

    use candle_core::Device;

    use crate::weight::BufferedLoRAWeight;

    use super::*;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";
        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;
        Ok(data)
    }

    fn header_only_bytes(buffer: &[u8]) -> &[u8] {
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&buffer[0..8]);
        let header_len = u64::from_le_bytes(len_bytes) as usize;
        &buffer[0..8 + header_len]
    }

    #[test]
    fn parse_header_rejects_buffer_smaller_than_length_prefix() {
        let result = parse_header(&[1_u8, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_rejects_truncated_header_json() {
        let buffer = load_test_file().unwrap();
        // 20 bytes is past the 8-byte length prefix but short of the real header JSON.
        let result = parse_header(&buffer[0..20]);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_works_with_header_only_buffer() {
        let buffer = load_test_file().unwrap();
        let (header, meta) = parse_header(header_only_bytes(&buffer)).unwrap();

        assert!(!header.keys().is_empty());
        assert!(meta.is_some());
    }

    #[test]
    fn header_index_matches_buffered_weight_for_boo_file() {
        let buffer = load_test_file().unwrap();
        let (header, _meta) = parse_header(&buffer).unwrap();

        let buffered = BufferedLoRAWeight::new(buffer.clone(), &Device::Cpu).unwrap();

        use crate::weight::{Weight, WeightKey};

        let mut header_keys = header.keys();
        let mut buffered_keys = buffered.keys();
        header_keys.sort();
        buffered_keys.sort();
        assert_eq!(header_keys, buffered_keys);

        assert_eq!(header.unet_keys().len(), buffered.unet_keys().len());
        assert_eq!(
            header.text_encoder_keys().len(),
            buffered.text_encoder_keys().len()
        );
        assert_eq!(header.dims(), buffered.dims());
        assert_eq!(header.precision(), buffered.precision());
        assert_eq!(header.format(), buffered.format());

        let mut header_base_names = header.base_names();
        let mut buffered_base_names = buffered.base_names();
        header_base_names.sort();
        buffered_base_names.sort();
        assert_eq!(header_base_names, buffered_base_names);
    }

    #[test]
    fn tensor_info_is_non_empty_and_sorted_by_name() {
        let buffer = load_test_file().unwrap();
        let (header, _meta) = parse_header(&buffer).unwrap();

        let info = header.tensor_info();
        assert!(!info.is_empty());

        let mut names: Vec<&String> = info.iter().map(|(name, _, _)| name).collect();
        let sorted = {
            let mut n = names.clone();
            n.sort();
            n
        };
        assert_eq!(names, sorted);
        names.clear(); // silence unused mut warning path in some rustc versions
    }
}
