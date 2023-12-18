extern crate console_error_panic_hook;
use std::fmt;

use wasm_bindgen::prelude::*;

use crate::file::LoRAFile;
use crate::metadata::Metadata;
use crate::network::NetworkModule;

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
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> Result<LoraWorker, JsError> {
        console_error_panic_hook::set_once();
        Ok(LoraWorker {
            metadata: Metadata::new_from_buffer(buffer)?,
            file: LoRAFile::new_from_buffer(buffer, filename),
        })
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
