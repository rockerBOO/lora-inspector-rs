use serde::{de, Deserialize, Deserializer, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Deserialize, Serialize, Clone, Debug)]
pub enum NetworkModule {
    KohyaSSLoRA,
    KohyaSSLoRAFA,
    KohyaSSDyLoRA,
    Lycoris,
}

#[wasm_bindgen]
#[derive(Deserialize, Serialize, Clone, Debug)]
pub enum NetworkType {
    LoRA,
    LoRAFA,
    LoHA,
    LoKr,
    IA3,
    DyLoRA,
    GLora,
    GLoKr,
    DiagOFT,
}

#[wasm_bindgen]
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct NetworkArgs {
    algo: Option<String>,

    #[serde(deserialize_with = "de_optional_bool_from_str")]
    pub rescale: Option<bool>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    pub dropout: Option<f64>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    pub rank_dropout: Option<f64>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    pub module_dropout: Option<f64>,

    #[serde(deserialize_with = "de_optional_usize_from_str")]
    pub conv_dim: Option<usize>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    pub conv_alpha: Option<f64>,
}

#[wasm_bindgen]
impl NetworkArgs {
    #[wasm_bindgen(getter)]
    pub fn algo(&self) -> String {
        self.algo.clone().unwrap_or_default()
    }

    #[wasm_bindgen(setter)]
    pub fn set_algo(&mut self, algo: &str) {
        self.algo = Some(algo.to_string());
    }
}
fn de_optional_f64_from_str<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<f64>().map(Some).map_err(de::Error::custom)
}

fn de_optional_usize_from_str<'de, D>(deserializer: D) -> Result<Option<usize>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<usize>().map(Some).map_err(de::Error::custom)
}

fn de_optional_bool_from_str<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "True" => Ok(Some(true)),
        "False" => Ok(Some(false)),
        _ => Err(de::Error::custom("Could not parse to bool")),
    }
}
