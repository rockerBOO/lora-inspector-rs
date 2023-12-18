mod file;
mod metadata;
mod network;
mod norms;
mod weight;
mod worker;

#[derive(Debug)]
pub enum Error {
    Candle(candle_core::Error),
    SafeTensor(safetensors::SafeTensorError),
    Load(String),
    Msg(String)
}
