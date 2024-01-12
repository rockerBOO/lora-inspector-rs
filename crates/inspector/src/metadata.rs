use std::collections::HashMap;

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::network::{NetworkArgs, NetworkModule, NetworkType};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Metadata {
    size: usize,
    pub metadata: Option<HashMap<String, String>>,
}

impl Metadata {
    pub fn new_from_buffer(buffer: &[u8]) -> crate::Result<Metadata> {
        Ok(
            SafeTensors::read_metadata(buffer).map(|(size, metadata)| Metadata {
                size,
                metadata: metadata.metadata().clone(),
            })?,
        )
    }

    // pub fn get(&self, key: &str) -> Option<String> {
    //     self.metadata.get(key).to_owned().cloned()
    // }
    //
    // pub fn insert(&mut self, key: &str, value: String) -> Option<String> {
    //     self.metadata.insert(key.to_string(), value)
    // }

    pub fn network_args(&self) -> Option<NetworkArgs> {
        match &self.metadata.as_ref().map(|v| v.get("ss_network_args")) {
            Some(Some(network_args)) => match serde_json::from_str::<NetworkArgs>(network_args) {
                Ok(network_args) => Some(network_args),
                Err(e) => {
                    println!("{:#?}", e);
                    None
                }
            },
            _ => None,
        }
    }

    pub fn network_type(&self) -> Option<NetworkType> {
        // try to discover the network type
        match self.network_module() {
            Some(NetworkModule::KohyaSSLoRA) => match dbg!(self.network_args()) {
                Some(network_args) => match network_args.conv_dim {
                    // We need to make the name for LoCon/Lo-Curious
                    Some(_) => Some(NetworkType::LoRA),
                    None => Some(NetworkType::LoRA),
                },
                None => Some(NetworkType::LoRA),
            },
            Some(NetworkModule::Lycoris) => self.network_args().map(|network_args| {
                match network_args.algo.as_ref().map(|algo| algo.as_ref()) {
                    Some("diag-oft") => NetworkType::DiagOFT,
                    Some("loha") => NetworkType::LoHA,
                    Some("lokr") => NetworkType::LoKr,
                    Some("glora") => NetworkType::GLora,
                    Some("glokr") => NetworkType::GLoKr,
                    Some("locon") => NetworkType::LoCon,
                    Some(algo) => panic!("Invalid algo {}", algo),
                    None => panic!("No algo found for lycoris"),
                }
            }),
            Some(NetworkModule::KohyaSSLoRAFA) => Some(NetworkType::LoRAFA),
            Some(NetworkModule::KohyaSSDyLoRA) => Some(NetworkType::DyLoRA),
            None => None,
        }
    }

    pub fn network_module(&self) -> Option<NetworkModule> {
        match self.metadata.as_ref().map(|v| v.get("ss_network_module")) {
            Some(Some(network_module)) => {
                match network_module.as_str() {
                    "networks.lora" => {
                        // this is a Kohya network, probably
                        Some(NetworkModule::KohyaSSLoRA)
                    }
                    "networks.lora_fa" => {
                        // this is a Kohya network, probably
                        Some(NetworkModule::KohyaSSLoRAFA)
                    }
                    "networks.dylora" => {
                        // this is a Kohya network, probably
                        Some(NetworkModule::KohyaSSDyLoRA)
                    }
                    "lycoris.kohya" => {
                        // lycoris network module
                        Some(NetworkModule::Lycoris)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        fs::File,
        io::{self, Read},
    };

    macro_rules! assert_err {
        ($expression:expr, $($pattern:tt)+) => {
            match $expression {
                $($pattern)+ => (),
                ref e => panic!("expected `{}` but got `{:?}`", stringify!($($pattern)+), e),
            }
        }
    }

    use super::*;

    fn load_test_file() -> Result<Vec<u8>, io::Error> {
        let filename = "boo.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_new_file() -> Result<Vec<u8>, io::Error> {
        let filename = "booscapes_v2.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_no_type() -> Result<Vec<u8>, io::Error> {
        let filename = "/mnt/900/lora/sdxl_vanelreup.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    fn load_test_no_meta_file() -> Result<Vec<u8>, io::Error> {
        let filename = "edgWar40KAdeptaSororitas.safetensors";

        let mut f = File::open(filename)?;
        let mut data = vec![];
        f.read_to_end(&mut data)?;

        Ok(data)
    }

    #[test]
    fn load_from_invalid_buffer() -> crate::Result<()> {
        let metadata = Metadata::new_from_buffer(&[1_u8]);

        // Act

        assert!(metadata.is_err());
        assert_err!(metadata, Err(_));

        Ok(())
    }

    #[test]
    fn load_from_buffer() -> crate::Result<()> {
        let buffer = load_test_file()?;
        let metadata = Metadata::new_from_buffer(&buffer);

        // Act

        assert!(metadata.is_ok());
        assert_err!(metadata, Ok(_));

        Ok(())
    }

    #[test]
    fn no_network_args() -> crate::Result<()> {
        // let buffer = load_test_file()?;
        let buffer = load_test_no_type()?;
        let metadata = Metadata::new_from_buffer(&buffer)?;

        // println!("{:#?}", metadata.metadata.clone().unwrap());
        //
        // let network_args = .unwrap();
        // let compare_network_args = NetworkArgs::new(None);

        assert!(metadata.network_args().is_none());

        Ok(())
    }

    #[test]
    fn network_args() -> crate::Result<()> {
        let buffer = load_test_new_file()?;
        let metadata = Metadata::new_from_buffer(&buffer)?;

        println!(
            "{:#?}",
            metadata.metadata.clone().unwrap().get("ss_network_args")
        );

        // let network_args = metadata.network_args().unwrap();
        // let compare_network_args = NetworkArgs::new(None);

        assert!(metadata.network_args().is_some());

        Ok(())
    }

    #[test]
    fn network_type() -> crate::Result<()> {
        let buffer = load_test_new_file()?;
        let metadata = Metadata::new_from_buffer(&buffer)?;

        println!(
            "{:#?}",
            metadata.metadata.clone().unwrap().get("ss_network_module")
        );

        // let network_args = metadata.network_args().unwrap();
        // let compare_network_args = NetworkArgs::new(None);

        assert!(metadata.network_type().is_some());
        assert_eq!(metadata.network_type().unwrap(), NetworkType::LoRA);

        Ok(())
    }

    #[test]
    fn no_network_type() -> crate::Result<()> {
        let buffer = load_test_no_type()?;
        let metadata = Metadata::new_from_buffer(&buffer)?;

        println!(
            "{:#?}",
            metadata.metadata.clone().unwrap().get("ss_network_module")
        );

        assert!(metadata.network_type().is_some());
        assert_eq!(metadata.network_type().unwrap(), NetworkType::LoRA);

        Ok(())
    }

    #[test]
    fn no_meta_network_type() -> crate::Result<()> {
        let buffer = load_test_no_meta_file()?;
        let metadata = Metadata::new_from_buffer(&buffer)?;

        assert!(metadata.network_type().is_none());

        Ok(())
    }
}
