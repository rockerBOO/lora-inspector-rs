use std::collections::HashMap;

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::network::{NetworkArgs, NetworkModule, NetworkType, WeightDecomposition};

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Metadata {
    pub metadata: Option<HashMap<String, String>>,
}

impl Metadata {
    pub fn new_from_buffer(buffer: &[u8]) -> crate::Result<Metadata> {
        Ok(
            SafeTensors::read_metadata(buffer).map(|(_size, metadata)| Metadata {
                metadata: metadata.metadata().clone(),
            })?,
        )
    }

    pub fn metadata_size(&self) -> usize {
        self.metadata.as_ref().map_or(0, |m| m.len())
    }
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
            Some(NetworkModule::KohyaSSLoRA) => match self.network_args() {
                Some(network_args) => match network_args.conv_dim {
                    // We need to make the name for LoCon/Lo-Curious
                    Some(_) => Some(NetworkType::LoRAC3Lier),
                    None => Some(NetworkType::LoRA),
                },
                None => Some(NetworkType::LoRA),
            },
            Some(NetworkModule::Lycoris) => self.network_args().map(|network_args| {
                match network_args.algo.as_ref().map(|algo| algo.as_ref()) {
                    Some("diag-oft") => NetworkType::DiagOFT,
                    Some("boft") => NetworkType::BOFT,
                    Some("loha") => NetworkType::LoHA,
                    Some("lokr") => NetworkType::LoKr,
                    Some("glora") => NetworkType::GLoRA,
                    Some("glokr") => NetworkType::GLoKr,
                    Some("locon") => NetworkType::LoCon,
                    Some("lora") => NetworkType::LoRA,
                    Some(algo) => panic!("Invalid algo {}", algo),
                    // Defaults to LoRA
                    None => NetworkType::LoRA,
                }
            }),
            Some(NetworkModule::KohyaSSLoRAFA) => Some(NetworkType::LoRAFA),
            Some(NetworkModule::KohyaSSDyLoRA) => Some(NetworkType::DyLoRA),
            Some(NetworkModule::KohyaSSOFT) => Some(NetworkType::OFT),
            Some(NetworkModule::KohyaSSLoRAFlux) => Some(NetworkType::LoRA),
            Some(NetworkModule::KohyaSSLoRALumina) => Some(NetworkType::LoRA),
            Some(NetworkModule::KohyaSSLoRASD3) => Some(NetworkType::LoRA),
            None => None,
        }
    }

    pub fn weight_decomposition(&self) -> Option<WeightDecomposition> {
        self.network_args()
            .map(|network_args| match network_args.dora_wd {
                Some(true) => WeightDecomposition::DoRA,
                _ => WeightDecomposition::None,
            })
    }

    pub fn rank_stabilized(&self) -> Option<bool> {
        self.network_args().map(|network_args| {
            let rs_lora = network_args.rs_lora.unwrap_or(false);
            let rank_stabilized = network_args.rank_stabilized.unwrap_or(false);
            rs_lora || rank_stabilized
        })
    }

    pub fn network_module(&self) -> Option<NetworkModule> {
        match self.metadata.as_ref().map(|v| v.get("ss_network_module")) {
            Some(Some(network_module)) => match network_module.as_str() {
                "networks.lora" => Some(NetworkModule::KohyaSSLoRA),
                "networks.lora_flux" => Some(NetworkModule::KohyaSSLoRAFlux),
                "networks.lora_sd3" => Some(NetworkModule::KohyaSSLoRASD3),
                "networks.lora_fa" => Some(NetworkModule::KohyaSSLoRAFA),
                "networks.dylora" => Some(NetworkModule::KohyaSSDyLoRA),
                "networks.oft" => Some(NetworkModule::KohyaSSOFT),
                "lycoris.kohya" => Some(NetworkModule::Lycoris),
                _ => None,
            },
            _ => None,
        }
    }
}

#[derive(Debug, Default)]
pub struct DiffChanged {
    pub old: String,
    pub new: String,
}

#[derive(Debug, Default)]
pub struct Diff {
    pub added: HashMap<String, String>,
    pub removed: HashMap<String, String>,
    pub changed: HashMap<String, DiffChanged>,
}

pub fn compare_metadata(m1: &Metadata, m2: &Metadata) -> Diff {
    // Only removed since nothing remains
    if let (Some(m1_meta), None) = (&m1.metadata, &m2.metadata) {
        let mut removed = HashMap::new();
        for (k, v) in m1_meta.iter() {
            removed.insert(k.clone(), v.clone());
        }

        Diff {
            removed,
            added: HashMap::new(),
            changed: HashMap::new(),
        }
    } else if let (Some(m1_meta), Some(m2_meta)) = (&m1.metadata, &m2.metadata) {
        let mut added = HashMap::new();
        let mut removed = HashMap::new();
        let mut changed: HashMap<String, DiffChanged> = HashMap::new();

        for (k, v) in m1_meta.iter() {
            if let Some(v2) = m2_meta.get(k) {
                if v != v2 {
                    changed.insert(
                        k.clone(),
                        DiffChanged {
                            old: v.clone(),
                            new: v2.clone(),
                        },
                    );
                }
            } else {
                println!("Removed {}", k);
                removed.insert(k.clone(), v.clone());
            }
        }

        for (k, v) in m2_meta.iter() {
            if !m1_meta.contains_key(k) {
                added.insert(k.clone(), v.clone());
            }
        }
        if !added.is_empty() || !removed.is_empty() || !changed.is_empty() {
            Diff {
                added,
                removed,
                changed,
            }
        } else {
            Diff::default()
        }
    } else {
        Diff::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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

    #[test]
    fn no_weight_decomposition() -> crate::Result<()> {
        let buffer = load_test_file()?;
        let metadata = Metadata::new_from_buffer(&buffer)?;

        assert!(metadata.weight_decomposition().is_none());

        Ok(())
    }

    #[test]
    fn test_compare_metadata_no_changes() {
        let m1 = Metadata {
            metadata: Some(HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ])),
        };

        let m2 = Metadata {
            metadata: Some(HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ])),
        };

        let diff = compare_metadata(&m1, &m2);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_compare_metadata_added() {
        let m1 = Metadata {
            metadata: Some(HashMap::from([("key1".to_string(), "value1".to_string())])),
        };

        let m2 = Metadata {
            metadata: Some(HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ])),
        };

        let diff = compare_metadata(&m1, &m2);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added.get("key2"), Some(&"value2".to_string()));
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_compare_metadata_removed() {
        let m1 = Metadata {
            metadata: Some(HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ])),
        };

        let m2 = Metadata {
            metadata: Some(HashMap::from([("key1".to_string(), "value1".to_string())])),
        };

        let diff = compare_metadata(&m1, &m2);
        assert!(diff.added.is_empty());
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed.get("key2"), Some(&"value2".to_string()));
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_compare_metadata_changed() {
        let m1 = Metadata {
            metadata: Some(HashMap::from([("key1".to_string(), "value1".to_string())])),
        };

        let m2 = Metadata {
            metadata: Some(HashMap::from([(
                "key1".to_string(),
                "new_value1".to_string(),
            )])),
        };

        let diff = compare_metadata(&m1, &m2);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.changed.len(), 1);
        let changed = diff.changed.get("key1").unwrap();
        assert_eq!(changed.old, "value1");
        assert_eq!(changed.new, "new_value1");
    }

    #[test]
    fn test_compare_metadata_multiple_changes() {
        let m1 = Metadata {
            metadata: Some(HashMap::from([
                ("key1".to_string(), "value1".to_string()),
                ("key2".to_string(), "value2".to_string()),
            ])),
        };

        let m2 = Metadata {
            metadata: Some(HashMap::from([
                ("key1".to_string(), "new_value1".to_string()),
                ("key3".to_string(), "value3".to_string()),
            ])),
        };

        let diff = compare_metadata(&m1, &m2);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added.get("key3"), Some(&"value3".to_string()));
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed.get("key2"), Some(&"value2".to_string()));
        assert_eq!(diff.changed.len(), 1);
        let changed = diff.changed.get("key1").unwrap();
        assert_eq!(changed.old, "value1");
        assert_eq!(changed.new, "new_value1");
    }
    #[test]
    fn test_compare_metadata_none() {
        let m1 = Metadata { metadata: None };

        let m2 = Metadata { metadata: None };

        let diff = compare_metadata(&m1, &m2);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_compare_metadata_some_none() {
        let m1 = Metadata {
            metadata: Some(HashMap::from([("key1".to_string(), "value1".to_string())])),
        };

        let m2 = Metadata { metadata: None };

        let diff = compare_metadata(&m1, &m2);
        assert!(diff.added.is_empty());
        println!("{:?}", diff);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed.get("key1"), Some(&"value1".to_string()));
        assert!(diff.changed.is_empty());
    }

    #[test]
    fn test_compare_metadata_empty() {
        let m1 = Metadata {
            metadata: Some(HashMap::new()),
        };

        let m2 = Metadata {
            metadata: Some(HashMap::new()),
        };

        let diff = compare_metadata(&m1, &m2);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert!(diff.changed.is_empty());
    }
}
