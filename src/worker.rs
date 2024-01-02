use std::collections::HashMap;
use std::fmt;

use candle_core::Device;
use pest::Parser;
use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::file::LoRAFile;
use crate::metadata::Metadata;
use crate::network::NetworkModule;
use crate::{norms, statistic, InspectorError, KeyParser, Rule};

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

    pub fn unet_keys(&self) -> Vec<String> {
        self.file.unet_keys()
    }

    pub fn text_encoder_keys(&self) -> Vec<String> {
        self.file.text_encoder_keys()
    }

    pub fn weight_keys(&self) -> Vec<String> {
        self.file.weight_keys()
    }

    pub fn alpha_keys(&self) -> Vec<String> {
        self.file.alpha_keys()
    }

    pub fn alphas(&self) -> Vec<JsValue> {
        self.file
            .alphas()
            .into_iter()
            .map(|alpha| {
                serde_wasm_bindgen::to_value(&alpha)
                    .unwrap_or_else(|_v| serde_wasm_bindgen::to_value("invalid alphas").unwrap())
            })
            .collect()
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

    pub fn precision(&self) -> String {
        self.file.precision()
    }

    pub fn parse_key(&self, parse_key: &str) {
        let successful_parse = KeyParser::parse(Rule::key, parse_key);
        if let Ok(pairs) = successful_parse {
            console::log_1(&format!("{:#?}", pairs).into());
        }
    }

    // TODO: Rename this function please
    pub fn scaled(&mut self, base_name: &str, scaled_funcs: Vec<String>) -> crate::Result<JsValue> {
        let scaled = self.file.scaled::<f64>(
            base_name,
            scaled_funcs
                .iter()
                .map(|v| match v.as_str() {
                    "l1_norm" => norms::NormFn {
                        name: "l1_norm".to_string(),
                        function: Box::new(|t| {
                            norms::l1::<f64>(&t.to_dtype(candle_core::DType::F64)?)
                        }),
                    },
                    "l2_norm" => norms::NormFn {
                        name: "l2_norm".to_string(),
                        function: Box::new(|t| {
                            norms::l2::<f64>(&t.to_dtype(candle_core::DType::F64)?)
                        }),
                    },
                    "matrix_norm" => norms::NormFn {
                        name: "matrix_norm".to_string(),
                        function: Box::new(|t| {
                            norms::matrix_norm::<f64>(&t.to_dtype(candle_core::DType::F64)?)
                        }),
                    },
                    "sparsity" => norms::NormFn {
                        name: "sparsity".to_string(),
                        function: Box::new(|t| norms::sparsity(&t)),
                    },
                    "max" => norms::NormFn {
                        name: "max".to_string(),
                        function: Box::new(|t| norms::max(&t)),
                    },
                    "min" => norms::NormFn {
                        name: "min".to_string(),
                        function: Box::new(|t| norms::min(&t)),
                    },
                    "std_dev" => norms::NormFn {
                        name: "std_dev".to_string(),
                        function: Box::new(|t| {
                            statistic::std_deviation(&t.to_dtype(candle_core::DType::F64)?)?
                                .ok_or_else(|| {
                                    InspectorError::Msg(
                                        "Could not get the standard deviation calculation"
                                            .to_string(),
                                    )
                                })
                        }),
                    },
                    "median" => norms::NormFn {
                        name: "median".to_string(),
                        function: Box::new(|t| {
                            statistic::median::<f64>(&t.to_dtype(candle_core::DType::F64)?)
                        }),
                    },
                    norm_type => norms::NormFn {
                        name: norm_type.to_string(),
                        function: Box::new(|_| {
                            Err(InspectorError::Msg("invalid norm type".to_string()))
                        }),
                    },
                    // "max" => ("l1_norm".to_string(), self.file.l1_norm),
                    // "min" => ("l1_norm".to_string(), self.file.l1_norm),
                    // "mean" => ("l1_norm".to_string(), self.file.l1_norm),
                })
                .collect::<Vec<norms::NormFn<f64>>>(),
            &Device::Cpu,
        );

        // console::log_1(&format!("scaled: {:#?}", scaled).into());

        match scaled {
            Ok(s) => Ok(serde_wasm_bindgen::to_value(
                &s.iter()
                    .map(|(k, v)| (k.to_owned(), v.as_ref().unwrap()))
                    .collect::<HashMap<String, &f64>>(),
            )?),
            Err(_) => Err(InspectorError::Msg(
                "Invalid response for norms...".to_string(),
            )),
        }
    }

    pub fn l1_norm(&mut self, base_name: &str) -> Option<f64> {
        self.file
            .l1_norm(base_name, &Device::Cpu)
            .map_err(|e| {
                console::error_1(&format!("L1 norm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn l2_norm(&mut self, base_name: &str) -> Option<f64> {
        self.file
            .l2_norm(base_name, &Device::Cpu)
            .map_err(|e| {
                console::error_1(&format!("L2 norm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn matrix_norm(&mut self, base_name: &str) -> Option<f64> {
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
