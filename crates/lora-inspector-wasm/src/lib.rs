use core::fmt;
use std::fmt::Debug;
use std::io;
// use std::alloc::Global;
use std::string::String;

use web_sys::wasm_bindgen::JsValue;

extern crate wee_alloc;

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

// use pest::Parser;
// use pest_derive::Parser;
// use wasm_bindgen::JsValue;
//
// #[derive(Parser)]
// #[grammar = "key.pest"]
// pub struct KeyParser;

// use candle_core::safetensors;
// use std::error::Error;
// use std::marker::Send;
// use std::marker::Sync;

// pub mod file;
// pub mod metadata;
// mod network;
// mod norms;
// mod parser;
// mod statistic;
// mod weight;
mod worker;

// pub use wasm_bindgen_rayon::init_thread_pool;

pub type Result<T> = std::result::Result<T, InspectorError>;

#[derive(Debug)]
pub enum InspectorError {
    Candle(candle_core::Error),
    SafeTensor(safetensors::SafeTensorError),
    Io(io::Error),
    Load(String),
    Msg(String),
    NotFound,
    UnsupportedNetworkType,
    SerdeWasmBindgenError(serde_wasm_bindgen::Error),
}

impl InspectorError {
    fn candle(err: candle_core::Error) -> InspectorError {
        InspectorError::Candle(err)
    }

    fn safetensor(err: safetensors::SafeTensorError) -> InspectorError {
        InspectorError::SafeTensor(err)
    }
}

impl From<candle_core::Error> for InspectorError {
    fn from(err: candle_core::Error) -> InspectorError {
        InspectorError::candle(err)
    }
}

impl From<safetensors::SafeTensorError> for InspectorError {
    fn from(err: safetensors::SafeTensorError) -> InspectorError {
        InspectorError::safetensor(err)
    }
}

impl From<io::Error> for InspectorError {
    fn from(err: io::Error) -> InspectorError {
        InspectorError::Io(err)
    }
}

impl fmt::Display for InspectorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InspectorError::Candle(e) => write!(f, "Candle Error {:#?}", e),
            InspectorError::SafeTensor(e) => write!(f, "SafeTensor Error {:#?}", e),
            InspectorError::SerdeWasmBindgenError(e) => {
                write!(f, "Serde WASM Bindgen error: {}", e)
            }
            InspectorError::Io(e) => write!(f, "IO Error: {:#?}", e),
            InspectorError::Load(e) => write!(f, "Load Error {}", e),
            InspectorError::Msg(e) => write!(f, "Error {}", e),
            InspectorError::UnsupportedNetworkType => write!(f, "Unsupported network type"),
            InspectorError::NotFound => write!(f, "Not found"),
        }
    }
}

impl From<JsValue> for InspectorError {
    fn from(value: JsValue) -> Self {
        InspectorError::Msg(value.as_string().unwrap())
    }
}

impl From<serde_wasm_bindgen::Error> for InspectorError {
    fn from(value: serde_wasm_bindgen::Error) -> Self {
        InspectorError::SerdeWasmBindgenError(value)
    }
}

