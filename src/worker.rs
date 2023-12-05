use candle_core::{Tensor, WithDType};
use wasm_bindgen::prelude::*;

use crate::inspector::LoRAInspector;
use crate::metadata::Metadata;
use crate::norms::{l1, l2, spectral};
use crate::Error;

#[wasm_bindgen]
pub struct LoraWorker {
    inspector: LoRAInspector,
    filename: String,
    metadata: Metadata,
}

impl LoraWorker {
    pub fn new_from_buffer(buffer: &[u8], filename: String) -> Result<LoraWorker, Error> {
        match LoRAInspector::new_from_buffer(buffer) {
            Ok(inspector) => match Metadata::new_from_buffer(buffer) {
                Ok(metadata) => Ok(LoraWorker {
                    inspector,
                    filename,
                    metadata,
                }),
                Err(e) => Err(Error::SafeTensor(e)),
            },
            Err(e) => Err(e),
        }
    }

    pub fn metadata(self) -> Metadata {
        self.metadata
    }

    pub fn filename(self) -> String {
        self.filename
    }

    pub fn norm<T: WithDType>(
        self,
        norm_fn: impl Fn(&Tensor) -> Result<T, candle_core::Error>,
    ) -> Vec<T> {
        self.inspector
            .tensors()
            .values()
            .filter_map(|v| norm_fn(v).ok())
            .collect()
    }
}

#[wasm_bindgen]
pub fn l1_norm(worker: LoraWorker) -> Vec<f32> {
    worker.norm(l1)
}

#[wasm_bindgen]
pub fn l2_norm(worker: LoraWorker) -> Vec<f32> {
    worker.norm(l2)
}

#[wasm_bindgen]
pub fn spectral_norm(worker: LoraWorker) -> Vec<f32> {
    worker.norm(spectral)
}

#[wasm_bindgen]
pub fn metadata(worker: LoraWorker) -> Metadata {
    worker.metadata()
}
