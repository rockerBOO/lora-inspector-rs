use std::collections::HashMap;

use safetensors::{SafeTensorError, SafeTensors};
use serde::{Deserialize, Serialize};

use crate::network::{NetworkArgs, NetworkModule, NetworkType};

#[derive(Debug, Clone, Serialize, Deserialize)]
// #[wasm_bindgen]
pub struct Metadata {
    size: usize,
    pub metadata: HashMap<String, String>,
}

impl Metadata {
    pub fn new_from_buffer(buffer: &[u8]) -> Result<Metadata, SafeTensorError> {
        let (size, metadata) = SafeTensors::read_metadata(buffer)?;

        match metadata.metadata().to_owned() {
            Some(metadata) => Ok(Metadata { size, metadata }),
            None => Err(SafeTensorError::MetadataIncompleteBuffer),
        }
    }

    pub fn get(&self, key: &str) -> Option<String> {
        self.metadata.get(key).to_owned().cloned()
    }

    pub fn insert(&mut self, key: &str, value: String) -> Option<String> {
        self.metadata.insert(key.to_string(), value)
    }

    pub fn network_args(&self) -> Option<NetworkArgs> {
        match &self.metadata.get("ss_network_args") {
            Some(network_args) => match serde_json::from_str::<NetworkArgs>(network_args) {
                Ok(network_args) => Some(network_args),
                Err(_) => None,
            },
            None => None,
        }
    }

    pub fn network_type(&self) -> Option<NetworkType> {
        // try to discover the network type
        match self.network_module() {
            Some(NetworkModule::KohyaSSLoRA) => match self.network_args() {
                Some(network_args) => match network_args.conv_dim {
                    // We need to make the name for LoCon/Lo-Curious
                    Some(_) => Some(NetworkType::LoRA),
                    None => Some(NetworkType::LoRA),
                },
                None => None,
            },
            Some(NetworkModule::Lycoris) => {
                self.network_args()
                    .and_then(|network_args| match network_args.algo().as_str() {
                        "diag-oft" => Some(NetworkType::DiagOFT),
                        "loha" => Some(NetworkType::LoHA),
                        "lokr" => Some(NetworkType::LoKr),
                        "glora" => Some(NetworkType::GLora),
                        "glokr" => Some(NetworkType::GLoKr),
                        _ => None,
                    })
            }
            Some(NetworkModule::KohyaSSLoRAFA) => Some(NetworkType::LoRAFA),
            Some(NetworkModule::KohyaSSDyLoRA) => Some(NetworkType::DyLoRA),
            None => None,
        }
    }

    pub fn network_module(&self) -> Option<NetworkModule> {
        match self.metadata.get("ss_network_module") {
            Some(network_module) => {
                match network_module.as_str() {
                    "networks.lora" => {
                        // this is a Kohya network, probably
                        Some(NetworkModule::KohyaSSLoRA)
                    }
                    "networks.lora_fa" => {
                        // this is a Kohya network, probably
                        Some(NetworkModule::KohyaSSLoRAFA)
                    }
                    "networks.dylora" => {
                        // this is a Kohya network, probably
                        Some(NetworkModule::KohyaSSDyLoRA)
                    }
                    "lycoris.kohya" => {
                        // lycoris network module
                        Some(NetworkModule::Lycoris)
                    }
                    _ => None,
                }
            }
            None => None,
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use std::fs::File;
//
//     use super::*;
//     use memmap2::MmapOptions;
//
//     fn new_from_file(filename: &str) -> Result<Metadata, SafeTensorError> {
//         let file = File::open(filename).unwrap();
//         let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
//         Metadata::new_from_buffer(&buffer)
//     }
//
//     #[test]
//     fn test_metadata() {
//         let filename = "/mnt/900/lora/booscapes_v2.safetensors";
//
//         safetensors::SafeTensors::load()
//     }
// }
