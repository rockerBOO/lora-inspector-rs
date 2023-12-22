use std::fmt;

use candle_core::Device;
use pest::Parser;
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::file::LoRAFile;
use crate::metadata::Metadata;
use crate::network::NetworkModule;
use crate::{KeyParser, Rule};

#[wasm_bindgen]
pub struct LoraWorker {
    metadata: Metadata,
    file: LoRAFile,
}

impl fmt::Display for LoraWorker {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LoRAWorker: {}", self.filename())
    }
}

#[wasm_bindgen]
impl LoraWorker {
    #[wasm_bindgen(constructor)]
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> Result<LoraWorker, String> {
        let metadata = Metadata::new_from_buffer(buffer).map_err(|e| e.to_string());
        let file = LoRAFile::new_from_buffer(buffer, filename.clone());

        metadata.map(|metadata| LoraWorker { metadata, file })
    }

    pub fn metadata(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.metadata.metadata)
            .unwrap_or_else(|_v| serde_wasm_bindgen::to_value("invalid metadata").unwrap())
    }

    pub fn filename(&self) -> String {
        self.file.filename()
    }

    pub fn is_tensors_loaded(&self) -> bool {
        self.file.is_tensors_loaded()
    }

    pub fn weight_keys(&self) -> Vec<String> {
        self.file.weight_keys()
    }

    pub fn alpha_keys(&self) -> Vec<String> {
        self.file.alpha_keys()
    }

    pub fn alphas(&self) -> Vec<u32> {
        self.file.alphas().into_iter().collect()
    }

    pub fn dims(&self) -> Vec<u32> {
        self.file.dims().into_iter().collect()
    }

    pub fn keys(&self) -> Vec<String> {
        self.file.keys()
    }

    pub fn base_names(&self) -> Vec<String> {
        self.file.base_names()
    }

    pub fn parse_key(&self, parse_key: &str) {
        let successful_parse = KeyParser::parse(Rule::key, parse_key);
        if let Ok(pairs) = successful_parse {
            console::log_1(&format!("{:#?}", pairs).into());
        }
    }

    pub fn l1_norm(&self, base_name: &str) -> Option<f64> {
        self.file
            .l1_norm(base_name, &Device::Cpu)
            .map_err(|e| {
                console::error_1(&format!("L1 norm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn l2_norm(&self, base_name: &str) -> Option<f64> {
        self.file
            .l2_norm(base_name, &Device::Cpu)
            .map_err(|e| {
                console::error_1(&format!("L2 norm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn matrix_norm(&self, base_name: &str) -> Option<f64> {
        self.file
            .matrix_norm(base_name, &Device::Cpu)
            .map_err(|e| {
                console::error_1(&format!("Matrix orm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn network_module(&self) -> String {
        match self.metadata.network_module() {
            Some(NetworkModule::Lycoris) => "lycoris".to_owned(),
            Some(NetworkModule::KohyaSSLoRA) => "kohya-ss/lora".to_owned(),
            Some(NetworkModule::KohyaSSLoRAFA) => "kohya-ss/lora_fa".to_owned(),
            Some(NetworkModule::KohyaSSDyLoRA) => "kohya-ss/dylora".to_owned(),
            None => "no_module_found".to_owned(),
        }
    }

    pub fn network_args(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
        serde_wasm_bindgen::to_value(&self.metadata.network_args())
    }

    pub fn network_type(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
        serde_wasm_bindgen::to_value(&self.metadata.network_type())
    }
}

// #[wasm_bindgen]
// pub fn l1_norm(worker: LoraWorker) -> Vec<f32> {
//     worker.norm(l1)
// }
//
// #[wasm_bindgen]
// pub fn l2_norm(worker: LoraWorker) -> Vec<f32> {
//     worker.norm(l2)
// }
//
// #[wasm_bindgen]
// pub fn spectral_norm(worker: LoraWorker) -> Vec<f32> {
//     worker.norm(spectral)
// }

// #[wasm_bindgen]
// pub fn metadata(worker: LoraWorker) -> Metadata {
//     worker.metadata()
// }
