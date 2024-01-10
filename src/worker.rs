use pest::Parser;
use std::collections::HashMap;
use std::fmt;
use wasm_bindgen::prelude::*;
use web_sys::console;

extern crate console_error_panic_hook;

use crate::file::LoRAFile;
use crate::metadata::Metadata;
use crate::network::NetworkModule;
use crate::{norms, statistic, InspectorError, KeyParser, Rule};
use std::panic;

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
    pub fn new_from_buffer(buffer: &[u8], filename: &str) -> Result<LoraWorker, String> {
        panic::set_hook(Box::new(console_error_panic_hook::hook));
        let metadata = Metadata::new_from_buffer(buffer).map_err(|e| e.to_string());
        let file = LoRAFile::new_from_buffer(buffer, filename);

        metadata.map(|metadata| LoraWorker { metadata, file })
    }

    pub fn metadata(&self) -> Result<JsValue, serde_wasm_bindgen::Error> {
        serde_wasm_bindgen::to_value(&self.metadata.metadata)
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
        panic::set_hook(Box::new(console_error_panic_hook::hook));
        self.file.precision()
    }

    pub fn parse_key(&self, parse_key: &str) {
        let successful_parse = KeyParser::parse(Rule::key, parse_key);
        if let Ok(pairs) = successful_parse {
            console::log_1(&format!("{:#?}", pairs).into());
        }
    }

    pub fn scale_weights(&mut self) -> usize {
        console_error_panic_hook::set_once();

        self.file
            .base_names()
            .iter()
            .filter_map(|base_name| {
                match self.file.scale_weight(base_name, &candle_core::Device::Cpu) {
                    Ok(ok) => Some(ok),
                    Err(e) => {
                        console::error_1(&format!("scale weight error: {:#?}", e).into());
                        None
                    }
                }
            })
            .count()
    }

    pub fn scale_weight(&mut self, base_name: &str) -> Result<bool, JsValue> {
        self.file
            .scale_weight(base_name, &candle_core::Device::Cpu)
            .map(|_t| true)
            .map_err(|e| JsValue::from_str(e.to_string().as_str()))
    }

    // TODO: Rename this function please
    pub fn norms(&self, base_name: &str, scaled_funcs: Vec<String>) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        let normative_funcs = scaled_funcs
            .iter()
            .map(|v| match v.as_str() {
                "l1_norm" => norms::NormFn {
                    name: "l1_norm".to_string(),
                    function: Box::new(|t| norms::l1::<f64>(&t.to_dtype(candle_core::DType::F64)?)),
                },
                "l2_norm" => norms::NormFn {
                    name: "l2_norm".to_string(),
                    function: Box::new(|t| norms::l2::<f64>(&t.to_dtype(candle_core::DType::F64)?)),
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
                        statistic::std_deviation(&t.to_dtype(candle_core::DType::F64)?)?.ok_or_else(
                            || {
                                InspectorError::Msg(
                                    "Could not get the standard deviation calculation".to_string(),
                                )
                            },
                        )
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
            })
            .collect::<Vec<norms::NormFn<f64>>>();

        match self.file.scaled_weight(base_name) {
            Some(scaled_weight) => Ok(serde_wasm_bindgen::to_value(
                &normative_funcs
                    .iter()
                    .map(|norm_fn| {
                        (
                            norm_fn.name.to_owned(),
                            (norm_fn.function)(scaled_weight.clone()).unwrap(),
                        )
                    })
                    .collect::<HashMap<String, f64>>(),
            )?),
            None => Err(JsValue::from_str(&format!(
                "could not get scaled weights for {}",
                base_name
            ))),
        }
    }

    pub fn l1_norm(&self, base_name: &str) -> Option<f64> {
        self.file
            .l1_norm(base_name)
            .map_err(|e| {
                console::error_1(&format!("L1 norm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn l2_norm(&self, base_name: &str) -> Option<f64> {
        self.file
            .l2_norm(base_name)
            .map_err(|e| {
                console::error_1(&format!("L2 norm for {} Error: {:#?}", base_name, e).into());
                e
            })
            .ok()
    }

    pub fn matrix_norm(&self, base_name: &str) -> Option<f64> {
        self.file
            .matrix_norm(base_name)
            .map_err(|e| {
                console::error_1(&format!("Matrix norm for {} Error: {:#?}", base_name, e).into());
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

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test_configure;
    use wasm_bindgen_test::*;

    // use crate::network::{BlockUsizeSeq, NetworkArgs};

    use super::LoraWorker;

    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::{js_sys::Uint8Array, JsFuture};
    use web_sys::{Request, RequestInit, Response};

    const ENDPOINT: &str = "https://lora-inspector-test-files.us-east-1.linodeobjects.com";

    fn file(file: &str) -> String {
        format!("{}/{}", ENDPOINT, file)
    }

    async fn load_test_file(url: &str) -> Result<Vec<u8>, JsValue> {
        let mut opts = RequestInit::new();
        opts.method("GET");

        let request = Request::new_with_str_and_init(url, &opts)?;

        let window = web_sys::window().unwrap();
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

        // `resp_value` is a `Response` object.
        assert!(resp_value.is_instance_of::<Response>());
        let resp: Response = resp_value.dyn_into().unwrap();

        let vec = JsFuture::from(resp.array_buffer()?).await?;

        let uint8 = Uint8Array::new(&vec).to_vec();

        Ok(uint8)
    }

    #[wasm_bindgen_test]
    async fn conv_scale_weight() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer =
            load_test_file(file("lora_unet_down_blocks_1_resnets_1_conv2.safetensors").as_str())
                .await
                .unwrap();

        let mut worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        let _ = worker
            .scale_weight("lora_unet_down_blocks_1_resnets_1_conv2")
            .unwrap();

        assert_eq!(
            worker.l2_norm("lora_unet_down_blocks_1_resnets_1_conv2"),
            Some(0.40116092442056206)
        );
    }

    #[wasm_bindgen_test]
    async fn base_names() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.base_names().len(), 264);
    }

    #[wasm_bindgen_test]
    async fn weight_keys() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.weight_keys().len(), 528);
    }

    #[wasm_bindgen_test]
    async fn alpha_keys() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.weight_keys().len(), 528);
    }

    #[wasm_bindgen_test]
    async fn filename() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.filename(), "boo.safetensors");
    }

    #[wasm_bindgen_test]
    async fn unet_keys() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.unet_keys().len(), 576);
    }

    #[wasm_bindgen_test]
    async fn text_encoder_keys() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.text_encoder_keys().len(), 216);
    }

    #[wasm_bindgen_test]
    async fn alphas() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.alphas().len(), 1);
    }

    #[wasm_bindgen_test]
    async fn dims() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.alphas().len(), 1);
    }

    #[wasm_bindgen_test]
    async fn keys() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.keys().len(), 792);
    }

    #[wasm_bindgen_test]
    async fn precision() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.precision(), "fp16");
    }

    #[wasm_bindgen_test]
    async fn l1_norm() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let mut worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert!(worker
            .scale_weight("lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k")
            .unwrap());

        assert_eq!(
            worker.l1_norm("lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k"),
            Some(58.98886960744858)
        );
    }

    #[wasm_bindgen_test]
    async fn l2_norm() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let mut worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert!(worker
            .scale_weight("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k")
            .unwrap());

        assert_eq!(
            worker.l2_norm("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k"),
            Some(0.42464358477734687)
        );
    }

    #[wasm_bindgen_test]
    async fn matrix_norm() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let mut worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert!(worker
            .scale_weight("lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k")
            .unwrap());

        assert_eq!(
            worker.matrix_norm(
                "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn2_to_k"
            ),
            Some(0.42464358477734687)
        );
    }

    #[wasm_bindgen_test]
    async fn network_module() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.network_module(), "kohya-ss/lora");
    }

    #[wasm_bindgen_test]
    async fn network_args() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("booscapes_v2.safetensors").as_str())
            .await
            .unwrap();

        let worker = LoraWorker::new_from_buffer(&buffer, "booscapes_v2.safetensors")
            .expect("load from buffer");

        assert!(worker.network_args().unwrap().is_object());
        // assert_eq!(
        //     worker.network_args().unwrap(),
        //     serde_wasm_bindgen::to_value(&NetworkArgs {
        //         dropout: Some(0.3),
        //         block_dims: Some(BlockUsizeSeq(vec![
        //             4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8
        //         ])),
        //         block_alphas: Some(BlockUsizeSeq(vec![
        //             16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32,
        //             32, 32, 32, 32, 32
        //         ])),
        //         ..Default::default()
        //     })
        //     .unwrap()
        // );
    }

    #[wasm_bindgen_test]
    async fn network_type() {
        wasm_bindgen_test_configure!(run_in_browser);
        let buffer = load_test_file(file("boo.safetensors").as_str())
            .await
            .unwrap();

        let worker =
            LoraWorker::new_from_buffer(&buffer, "boo.safetensors").expect("load from buffer");

        assert_eq!(worker.network_type().unwrap(), "LoRA");
    }
}
