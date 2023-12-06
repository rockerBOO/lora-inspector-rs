#![allow(unused)]
extern crate console_error_panic_hook;
use wasm_bindgen::prelude::*;

use crate::inspector::LoRAInspector;
use crate::metadata::Metadata;
// use crate::norms::{l1, l2, spectral};

#[wasm_bindgen]
pub struct LoraWorker {
    inspector: LoRAInspector,
    filename: String,
    metadata: Metadata,
}

#[wasm_bindgen]
impl LoraWorker {
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> Result<LoraWorker, JsError> {
        console_error_panic_hook::set_once();
        match LoRAInspector::new_from_buffer(buffer) {
            Ok(inspector) => match Metadata::new_from_buffer(buffer) {
                Ok(metadata) => Ok(LoraWorker {
                    inspector,
                    filename,
                    metadata,
                }),
                Err(e) => Err(JsError::new("Could not load metadata")),
            },
            Err(e) => Err(JsError::new(
                &format!("Could not load inspector. {:?}", e).to_owned(),
            )),
        }
    }

    pub fn metadata(self) -> Metadata {
        self.metadata
    }

    pub fn filename(self) -> String {
        self.filename
    }

    // pub fn norm<T: WithDType>(
    //     self,
    //     norm_fn: impl Fn(&Tensor) -> Result<T, candle_core::Error>,
    // ) -> Vec<T> {
    //     self.inspector
    //         .tensors()
    //         .values()
    //         .filter_map(|v| norm_fn(v).ok())
    //         .collect()
    // }
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

#[wasm_bindgen]
pub fn metadata(worker: LoraWorker) -> Metadata {
    worker.metadata()
}
