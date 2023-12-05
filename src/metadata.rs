use std::collections::HashMap;

use safetensors::{SafeTensorError, SafeTensors};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
#[derive(Debug, Serialize, Deserialize)]
pub struct Metadata {
    size: usize,
    metadata: HashMap<String, String>,
}

impl Metadata {
    pub fn new_from_buffer(buffer: &[u8]) -> Result<Metadata, SafeTensorError> {
        let (size, metadata) = SafeTensors::read_metadata(buffer)?;

        match metadata.metadata().to_owned() {
            Some(metadata) => Ok(Metadata { size, metadata }),
            None => Err(SafeTensorError::MetadataIncompleteBuffer),
        }
    }

    pub fn get(self, key: &str) -> Option<String> {
        self.metadata.get(key).to_owned().cloned()
    }

    pub fn insert(mut self, key: &str, value: String) -> Option<String> {
        self.metadata.insert(key.to_string(), value)
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
