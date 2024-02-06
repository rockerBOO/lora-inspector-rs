use std::str::FromStr;

use pest::Parser;
use serde::{
    de::{self},
    ser::{SerializeSeq, SerializeTuple},
    Deserialize, Deserializer, Serialize,
};
use serde_json::Value;
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
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq)]
pub enum NetworkType {
    LoRA,
    LoRAFA,
    LoCon,
    LoHA,
    LoKr,
    IA3,
    DyLoRA,
    GLora,
    GLoKr,
    DiagOFT,
    BOFT
}

#[derive(Deserialize, Serialize, Clone, Debug, Default)]
pub struct NetworkArgs {
    pub algo: Option<String>,

    #[serde(deserialize_with = "de_optional_f64_from_str_or_f64")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dropout: Option<f64>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank_dropout: Option<f64>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub module_dropout: Option<f64>,

    #[serde(deserialize_with = "de_optional_usize_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_dim: Option<usize>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_alpha: Option<f64>,

    #[serde(deserialize_with = "de_optional_vec_sequence_usize_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_dims: Option<BlockUsizeSeq>,

    #[serde(deserialize_with = "de_optional_vec_sequence_usize_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub block_alphas: Option<BlockUsizeSeq>,

    // #[serde(deserialize_with = "de_optional_vec_sequence_f64_from_str")]
    // #[serde(default, skip_serializing_if = "Option::is_none")]
    pub down_lr_weight: Option<LrWeight>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mid_lr_weight: Option<f64>,

    // #[serde(deserialize_with = "de_optional_vec_sequence_f64_from_str")]
    // #[serde(default, skip_serializing_if = "Option::is_none")]
    pub up_lr_weight: Option<LrWeight>,

    // block_lr_zero_threshold=0.1
    pub drop_keys: Option<String>,

    #[serde(deserialize_with = "de_optional_bool_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub use_cp: Option<bool>,

    #[serde(deserialize_with = "de_optional_bool_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rescale: Option<bool>,

    #[serde(deserialize_with = "de_optional_f64_from_str")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub constrain: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum Op {
    Plus,
    Minus,
}

impl From<&str> for Op {
    fn from(value: &str) -> Self {
        match value {
            "+" => Op::Plus,
            "-" => Op::Minus,
            v => panic!("Could not convert {} to Op", v),
        }
    }
}

#[derive(Debug, Clone)]
pub enum LrWeight {
    // sine, cosine, linear, reverse_linear, zeros
    Sine(Op, f64),
    Cosine(Op, f64),
    Linear(Op, f64),
    ReverseLinear(Op, f64),
    Zeros(Op, f64),
    Block(f64),
    BlockF64Seq(BlockF64Seq),
    BlockUsizeSeq(BlockUsizeSeq),
}

impl LrWeight {
    fn serialize_op<S: serde::Serializer>(
        serializer: S,
        curve: &str,
        op: &Op,
        amount: &f64,
    ) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_tuple(3)?;
        seq.serialize_element(curve)?;
        seq.serialize_element(match op {
            Op::Plus => "+",
            Op::Minus => "-",
        })?;
        seq.serialize_element(&amount)?;

        seq.end()
    }
}

use pest_derive::Parser;

#[derive(Parser)]
#[grammar = "lr_weight.pest"]
pub struct LrWeightParser;

impl Serialize for LrWeight {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            LrWeight::Sine(op, amount) => LrWeight::serialize_op(serializer, "sine", op, amount),
            LrWeight::Cosine(op, amount) => {
                LrWeight::serialize_op(serializer, "cosine", op, amount)
            }
            LrWeight::Linear(op, amount) => {
                LrWeight::serialize_op(serializer, "linear", op, amount)
            }
            LrWeight::ReverseLinear(op, amount) => {
                LrWeight::serialize_op(serializer, "reverse_linear", op, amount)
            }
            LrWeight::Zeros(op, amount) => LrWeight::serialize_op(serializer, "zeros", op, amount),
            LrWeight::Block(amount) => serializer.serialize_f64(*amount),
            LrWeight::BlockF64Seq(amounts) => {
                let mut seq = serializer.serialize_seq(Some(amounts.0.len()))?;
                for e in &amounts.0 {
                    seq.serialize_element(&e)?;
                }
                seq.end()
            }
            LrWeight::BlockUsizeSeq(amounts) => {
                let mut seq = serializer.serialize_seq(Some(amounts.0.len()))?;
                for e in &amounts.0 {
                    seq.serialize_element(&e)?;
                }
                seq.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for LrWeight {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // let s = String::deserialize(deserializer)?;
        let v = Value::deserialize(deserializer)?;
        Ok(match v {
            Value::String(s) => match s.split('+').collect::<Vec<&str>>() {
                parts if parts.len() > 1 => {
                    let pairs = LrWeightParser::parse(Rule::f, &s).unwrap();
                    let mut curve = "";
                    let mut op = "";
                    let mut amount = "";
                    for pair in pairs {
                        match pair.as_rule() {
                            Rule::curve => curve = pair.as_str(),
                            Rule::op => op = pair.as_str(),
                            Rule::amount => amount = pair.as_str(),
                            _ => continue,
                        };
                    }

                    match curve {
                        "cosine" => {
                            LrWeight::Cosine(op.into(), amount.parse().map_err(de::Error::custom)?)
                        }
                        "sine" => {
                            LrWeight::Sine(op.into(), amount.parse().map_err(de::Error::custom)?)
                        }
                        "linear" => {
                            LrWeight::Linear(op.into(), amount.parse().map_err(de::Error::custom)?)
                        }
                        "reverse_linear" => LrWeight::ReverseLinear(
                            op.into(),
                            amount.parse().map_err(de::Error::custom)?,
                        ),
                        "zeros" => {
                            LrWeight::Zeros(op.into(), amount.parse().map_err(de::Error::custom)?)
                        }
                        curve => panic!("Invalid curve found for {}", curve),
                    }
                }
                single => LrWeight::BlockF64Seq(
                    single.first().unwrap().parse().map_err(de::Error::custom)?,
                ),
            },
            Value::Number(num) => LrWeight::Block(
                num.as_f64()
                    .ok_or_else(|| de::Error::custom("Invalid number"))?,
            ),
            value => panic!("Could not extract value: {:#?}", value),
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ParseSeqError;

impl std::fmt::Display for ParseSeqError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parse sequence failed")
    }
}

// TODO: Probably could merge these 2 unto 1 struct
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BlockF64Seq(Vec<f64>);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct BlockUsizeSeq(pub Vec<usize>);

impl FromStr for BlockF64Seq {
    type Err = ParseSeqError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // let s = String::deserialize(deserializer)?;
        let mut collection: Vec<f64> = vec![];
        for part in s.split(',') {
            collection.push(part.parse::<f64>().map_err(|_| ParseSeqError)?);
        }

        Ok(Self(collection))
    }
}

impl FromStr for BlockUsizeSeq {
    type Err = ParseSeqError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // let s = String::deserialize(deserializer)?;
        let mut collection: Vec<usize> = vec![];
        for part in s.split(',') {
            collection.push(part.parse::<usize>().map_err(|_| ParseSeqError)?);
        }

        Ok(Self(collection))
    }
}

// fn de_optional_vec_sequence_f64_from_str<'de, D>(
//     deserializer: D,
// ) -> Result<Option<BlockF64Seq>, D::Error>
// where
//     D: Deserializer<'de>,
// {
//     let s = String::deserialize(deserializer)?;
//     BlockF64Seq::from_str(&s)
//         .map(Some)
//         .map_err(de::Error::custom)
// }

fn de_optional_vec_sequence_usize_from_str<'de, D>(
    deserializer: D,
) -> Result<Option<BlockUsizeSeq>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    BlockUsizeSeq::from_str(&s)
        .map(Some)
        .map_err(de::Error::custom)
}

fn de_optional_f64_from_str<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    s.parse::<f64>().map(Some).map_err(de::Error::custom)
}

fn de_optional_f64_from_str_or_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
where
    D: Deserializer<'de>,
{
    // let str = serde_json::Value::deserialize_str(deserializer);
    // match deserializer.deserialize_str() {
    //     Ok(_) => todo!(),
    //     Err(_) => todo!(),
    // }
    Ok(match Value::deserialize(deserializer)? {
        Value::String(s) => Some(s.parse().map_err(de::Error::custom)?),
        Value::Number(num) => Some(
            num.as_f64()
                .ok_or_else(|| de::Error::custom("Invalid number"))?,
        ),
        _ => return Err(de::Error::custom("wrong type")),
    })
}

// fn de_optional_lr_weight_from_str_or_f64<'de, D>(deserializer: D) -> Result<Option<f64>, D::Error>
// where
//     D: Deserializer<'de>,
// {
//     // let str = serde_json::Value::deserialize_str(deserializer);
//     // match deserializer.deserialize_str() {
//     //     Ok(_) => todo!(),
//     //     Err(_) => todo!(),
//     // }
//     Ok(match Value::deserialize(deserializer)? {
//         Value::String(s) => Some(s.parse().map_err(de::Error::custom)?),
//         Value::Number(num) => Some(
//             num.as_f64()
//                 .ok_or_else(|| de::Error::custom("Invalid number"))?,
//         ),
//         _ => return Err(de::Error::custom("wrong type")),
//     })
// }

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

#[cfg(test)]
mod tests {
    // use std::{
    //     fs::File,
    //     io::{self, Read},
    // };

    // macro_rules! assert_err {
    //     ($expression:expr, $($pattern:tt)+) => {
    //         match $expression {
    //             $($pattern)+ => (),
    //             ref e => panic!("expected `{}` but got `{:?}`", stringify!($($pattern)+), e),
    //         }
    //     }
    // }

    use super::*;

    // fn load_test_file() -> Result<Vec<u8>, io::Error> {
    //     let filename = "boo.safetensors";
    //
    //     let mut f = File::open(filename)?;
    //     let mut data = vec![];
    //     f.read_to_end(&mut data)?;
    //
    //     Ok(data)
    // }

    #[test]
    fn network_args() -> crate::Result<()> {
        let json = r#"{
  "dropout": "0.5",
  "rank_dropout": "0.4",
  "module_dropout": "0.1",
  "block_dims": "2,2,2,2,4,4,4,4,8,8,8,8,8,8,8,8,8,4,4,4,4,2,2,2,2",
  "block_alphas": "8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16",
  "drop_keys": "to_v",
  "conv_block_dims": "1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2",
  "conv_block_alphas": "4,4,4,4,4,4,4,4,4,4,4,4,4,8,8,8,8,8,8,8,8,8,8,8,8",
  "down_lr_weight": "0,0,0,0,0,0,0,1,1,1,1,1",
  "up_lr_weight": "1,1,1,1,0,0,0,0,0,0,0,0",
  "mid_lr_weight": "1",
    "conv_dim": "8", "conv_alpha": "4", "use_cp": "True", "algo": "loha"
}"#;

        let network_args: Result<NetworkArgs, serde_json::Error> = serde_json::from_str(json);

        println!("{:#?}", network_args);

        assert!(network_args.is_ok());
        Ok(())
    }
}
