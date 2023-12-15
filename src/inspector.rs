

// use crate::file::LoRAFile;

// use candle_core::{safetensors::load_buffer, Device, Tensor};
// use serde::{de, Deserialize, Deserializer, Serialize};
//
// use crate::metadata::Metadata;

// /// LoRA file buffer
// #[derive(Debug, Clone)]
// pub struct LoRAFile {
//     buffer: Vec<u8>,
//     filename: String,
// }
//
// /// Tensor weights
// #[derive(Debug, Clone)]
// pub struct LoRAWeight {
//     tensors: HashMap<String, Tensor>,
// }
//
// /// LoRA Metadata
// #[derive(Debug, Clone)]
// pub struct LoRAMetadata {
//     metadata: Metadata,
// }

// #[derive(Clone)]
// pub struct LoRAInspector {}

// impl LoRAInspector {
//     pub fn new_from_buffer(buffer: &[u8], filename: String) -> LoRAFile {
//         LoRAFile {
//             buffer: buffer.to_vec(),
//             filename,
//         }
//
//         // match load_buffer(buffer, &Device::Cpu) {
//         //     Ok(tensors) => match Metadata::new_from_buffer(buffer) {
//         //         Ok(metadata) => Ok(LoRAInspector { tensors, metadata }),
//         //
//         //         Err(_) => Err(Error::Load("Failed to load metadata".to_owned())),
//         //     },
//         //     Err(e) => Err(Error::Load(format!("Failed to load buffer {:?}", e))),
//         // }
//     }
//
//     // pub fn load_tensors(&mut self) -> Result<&Option<HashMap<String, Tensor>>, candle_core::Error> {
//     //     self.tensors = Some(load_buffer(&self.buffer, &Device::Cpu)?);
//     //
//     //     Ok(&self.tensors)
//     // }
//
//     // pub fn tensors(&mut self) {
//     //     self.load_tensors()
//     //
//     //     self.tensors
//     // }
//
//     // pub fn metadata(&self) -> &Option<Metadata> {
//     //     &self.metadata
//     // }
//     //
//     // pub fn network_args(&self) -> Option<NetworkArgs> {
//     //     //
//     //     match &self.metadata {
//     //         Some(metadata) => match metadata.get("ss_network_args") {
//     //             Some(network_args) => match serde_json::from_str::<NetworkArgs>(&network_args) {
//     //                 Ok(network_args) => Some(network_args),
//     //                 Err(_) => None,
//     //             },
//     //             None => todo!(),
//     //         },
//     //         None => None,
//     //     }
//     // }
//     //
//     // pub fn network_type(&self) -> Option<NetworkType> {
//     //     // try to discover the network type
//     //     match self.network_module() {
//     //         Some(NetworkModule::KohyaSSLoRA) => match self
//     //             .network_args()
//     //             .map(|network_args| network_args.conv_dim)
//     //         {
//     //             // We need to make the name for LoCon/Lo-Curious
//     //             Some(_) => Some(NetworkType::LoRA),
//     //             None => Some(NetworkType::LoRA),
//     //         },
//     //         Some(NetworkModule::Lycoris) => {
//     //             self.network_args()
//     //                 .and_then(|network_args| match network_args.algo {
//     //                     Some(algo) => match algo.as_str() {
//     //                         "diag-oft" => Some(NetworkType::DiagOFT),
//     //                         "loha" => Some(NetworkType::LoHA),
//     //                         "lokr" => Some(NetworkType::LoKr),
//     //                         "glora" => Some(NetworkType::GLora),
//     //                         "glokr" => Some(NetworkType::GLoKr),
//     //                         _ => None,
//     //                     },
//     //                     None => None,
//     //                 })
//     //         }
//     //         Some(NetworkModule::KohyaSSLoRAFA) => Some(NetworkType::LoRAFA),
//     //         Some(NetworkModule::KohyaSSDyLoRA) => Some(NetworkType::DyLoRA),
//     //         None => None,
//     //     }
//     // }
//     //
//     // pub fn network_module(&self) -> Option<NetworkModule> {
//     //     match self
//     //         .metadata
//     //         .map(|metadata| metadata.get("ss_network_module"))
//     //     {
//     //         Some(Some(network_module)) => {
//     //             match network_module.as_str() {
//     //                 "networks.lora" => {
//     //                     // this is a Kohya network, probably
//     //                     Some(NetworkModule::KohyaSSLoRA)
//     //                 }
//     //                 "networks.lora_fa" => {
//     //                     // this is a Kohya network, probably
//     //                     Some(NetworkModule::KohyaSSLoRAFA)
//     //                 }
//     //                 "networks.dylora" => {
//     //                     // this is a Kohya network, probably
//     //                     Some(NetworkModule::KohyaSSDyLoRA)
//     //                 }
//     //                 "lycoris.kohya" => {
//     //                     // lycoris network module
//     //                     Some(NetworkModule::Lycoris)
//     //                 }
//     //                 _ => None,
//     //             }
//     //         }
//     //         Some(None) => None,
//     //         None => None,
//     //     }
//     // }
// }

#[cfg(test)]
mod tests {
    // use std::fs::File;
    //
    // use candle_core::DType;
    // use memmap2::MmapOptions;
    //
    // use super::*;

    // fn make_buffer() -> Vec<u8> {
    //     let t = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
    //
    //     let tensors: HashMap<String, Tensor> = [("t1".to_string(), t)].into_iter().collect();
    //
    //     let filename = "buffer.safetensors";
    //     candle_core::safetensors::save(&tensors, filename).unwrap();
    //
    //     let file = File::open(filename).unwrap();
    //     let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    //
    //     buffer.to_owned()
    // }
    //
    // #[test]
    // fn new_from_buffer_valid_data() {
    //     // Arrange
    //     let valid_buffer = make_buffer();
    //
    //     // Act
    //     let result = LoRAInspector::new_from_buffer(&valid_buffer);
    //
    //
    //     // Assert
    //     assert!(dbg!(result).is_ok());
    // }
    //
    // #[test]
    // fn new_from_buffer_invalid_data() {
    //     // Arrange
    //     let invalid_buffer = make_buffer(); // provide an invalid buffer for testing;
    //
    //     // Act
    //     let result = LoRAInspector::new_from_buffer(&invalid_buffer);
    //
    //     // Assert
    //     assert!(result.is_err());
    //     // Add more specific assertions if needed
    // }
    //
    // #[test]
    // fn metadata_returns_correct_value() {
    //     // Arrange
    //     let valid_buffer = make_buffer();// provide a valid buffer for testing;
    //     let inspector = LoRAInspector::new_from_buffer(&valid_buffer).unwrap();
    //
    //     // Act
    //     let metadata = inspector.metadata();
    //
    //     // Assert
    //     // Add assertions to verify that the returned metadata is correct
    // }
    //
    // #[test]
    // fn keys_by_key_returns_matching_keys() {
    //     // Arrange
    //     let valid_buffer = make_buffer();// provide a valid buffer for testing;
    //     let inspector = LoRAInspector::new_from_buffer(&valid_buffer).unwrap();
    //
    //     // Act
    //     let keys = inspector.keys_by_key("weights");
    //
    //     // Assert
    //     // Add assertions to verify that the returned keys contain "weights"
    // }
    //
    // #[test]
    // fn get_returns_correct_tensor() {
    //     // Arrange
    //     let valid_buffer = make_buffer();// provide a valid buffer for testing;
    //     let inspector = LoRAInspector::new_from_buffer(&valid_buffer).unwrap();
    //     let key = "t1"; // choose a key for testing;
    //
    //     // Act
    //     let result = inspector.get(&key);
    //
    //     // Assert
    //     // Add assertions to verify that the returned tensor is correct
    // }
}
