use std::collections::HashMap;

use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::network::{NetworkArgs, NetworkFamily, NetworkModule, NetworkType, WeightDecomposition};
use crate::weight::LoRAFormat;

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

    /// Builds `Metadata` from just the safetensors header (see
    /// `crate::header::parse_header`), without requiring tensor payload bytes.
    pub fn new_from_header_buffer(buffer: &[u8]) -> crate::Result<Metadata> {
        let (_header, meta_map) = crate::header::parse_header(buffer)?;
        Ok(Metadata { metadata: meta_map })
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
            Some(NetworkModule::KohyaSSLoRAGeneric) => Some(NetworkType::LoRA),
            Some(NetworkModule::MusubiTunerLoRAFlux2) => Some(NetworkType::LoRA),
            Some(NetworkModule::KohyaSSLoRALumina) => Some(NetworkType::LoRA),
            Some(NetworkModule::KohyaSSLoRASD3) => Some(NetworkType::LoRA),
            Some(NetworkModule::MusubiTunerLoRAKrea2) => Some(NetworkType::LoRA),
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
                "networks.lora_generic" => Some(NetworkModule::KohyaSSLoRAGeneric),
                "networks.lora_flux" => Some(NetworkModule::KohyaSSLoRAFlux),
                "networks.lora_flux_2" => Some(NetworkModule::MusubiTunerLoRAFlux2),
                "networks.lora_sd3" => Some(NetworkModule::KohyaSSLoRASD3),
                "networks.lora_krea2" => Some(NetworkModule::MusubiTunerLoRAKrea2),
                "networks.lora_fa" => Some(NetworkModule::KohyaSSLoRAFA),
                "networks.dylora" => Some(NetworkModule::KohyaSSDyLoRA),
                "networks.oft" => Some(NetworkModule::KohyaSSOFT),
                "lycoris.kohya" => Some(NetworkModule::Lycoris),
                _ => None,
            },
            _ => None,
        }
    }

    /// Reads a metadata value, trimmed, treating blank values as absent.
    fn metadata_value(&self, key: &str) -> Option<&str> {
        self.metadata
            .as_ref()?
            .get(key)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
    }

    /// The module as the trainer names it, without its vendor prefix:
    /// `networks.lora_generic` -> `lora_generic`, `lycoris.kohya` -> `kohya`.
    pub fn network_module_name(&self) -> Option<String> {
        let raw = self.metadata_value("ss_network_module")?;

        Some(match raw.split_once('.') {
            Some((_vendor, module)) if !module.is_empty() => module.to_string(),
            _ => raw.to_string(),
        })
    }

    /// Which trainer produced this file, from metadata alone.
    /// Key-shape evidence is layered on top by [`Self::resolve_network_family`].
    pub fn network_family(&self) -> NetworkFamily {
        if let Some(module) = self.metadata_value("ss_network_module") {
            if let Some(name) = module.strip_prefix("networks.") {
                return match name {
                    "lora_flux_2" | "lora_krea2" => NetworkFamily::MusubiTuner,
                    _ => NetworkFamily::Kohya,
                };
            }

            if module.starts_with("lycoris.") {
                return NetworkFamily::Lycoris;
            }
        }

        if let Some(version) = self.metadata_value("modelspec.implementation_version") {
            if version.starts_with("sd-scripts/") {
                return NetworkFamily::Kohya;
            }
        }

        if let Some(implementation) = self.metadata_value("modelspec.implementation") {
            if implementation.contains("kohya-ss/sd-scripts")
                || implementation.contains("OpenMOSS/MOSS-Image")
            {
                return NetworkFamily::Kohya;
            }

            if implementation.contains("diffusers") {
                return NetworkFamily::Diffusers;
            }
        }

        NetworkFamily::Unknown
    }

    /// Family resolution for a loaded file: metadata first, then the key shape
    /// the loader detected. `LoRAFormat::Kohya` is the loader's fallback and is
    /// deliberately not treated as evidence.
    pub fn resolve_network_family(&self, format: &LoRAFormat) -> NetworkFamily {
        match self.network_family() {
            NetworkFamily::Unknown => match format {
                LoRAFormat::Peft => NetworkFamily::Peft,
                LoRAFormat::Lycoris => NetworkFamily::Lycoris,
                LoRAFormat::Kohya => NetworkFamily::Unknown,
            },
            family => family,
        }
    }

    /// The string shown to a user for this file's trainer module. Table-known
    /// modules keep their established names; anything else is attributed to the
    /// family that produced it, and only a file with no evidence reads `unknown`.
    pub fn network_module_display(&self, format: &LoRAFormat) -> String {
        if let Some(module) = self.network_module() {
            return match module {
                NetworkModule::KohyaSSLoRA => "kohya-ss/lora".to_string(),
                NetworkModule::KohyaSSLoRAGeneric => "kohya-ss/lora_generic".to_string(),
                NetworkModule::KohyaSSLoRAFlux => "kohya-ss/lora_flux".to_string(),
                NetworkModule::MusubiTunerLoRAFlux2 => "musubi-tuner/lora_flux_2".to_string(),
                NetworkModule::KohyaSSLoRALumina => "kohya-ss/lora_lumina".to_string(),
                NetworkModule::KohyaSSLoRASD3 => "kohya-ss/lora_sd3".to_string(),
                NetworkModule::MusubiTunerLoRAKrea2 => "musubi-tuner/lora_krea2".to_string(),
                NetworkModule::KohyaSSLoRAFA => "kohya-ss/lora_fa".to_string(),
                NetworkModule::KohyaSSDyLoRA => "kohya-ss/dylora".to_string(),
                NetworkModule::KohyaSSOFT => "kohya-ss/oft".to_string(),
                NetworkModule::Lycoris => "lycoris".to_string(),
            };
        }

        let family = self.resolve_network_family(format);

        match (family, self.network_module_name()) {
            (NetworkFamily::Unknown, _) => "unknown".to_string(),
            (family, Some(module)) => format!("{}/{}", family.display(), module),
            (family, None) => family.display().to_string(),
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
    fn musubi_tuner_flux2_network_module() -> crate::Result<()> {
        use std::collections::HashMap;

        // Create metadata with networks.lora_flux_2 module
        let mut meta_map = HashMap::new();
        meta_map.insert(
            "ss_network_module".to_string(),
            "networks.lora_flux_2".to_string(),
        );

        let metadata = Metadata {
            metadata: Some(meta_map),
        };

        // Verify it's recognized as MusubiTunerLoRAFlux2
        assert_eq!(
            metadata.network_module(),
            Some(NetworkModule::MusubiTunerLoRAFlux2)
        );

        // Verify it maps to LoRA network type
        assert_eq!(metadata.network_type(), Some(NetworkType::LoRA));

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

    fn meta(pairs: &[(&str, &str)]) -> Metadata {
        Metadata {
            metadata: Some(
                pairs
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.to_string()))
                    .collect(),
            ),
        }
    }

    #[test]
    fn network_family_from_module_string() {
        use crate::network::NetworkFamily;

        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_generic")]).network_family(),
            NetworkFamily::Kohya
        );
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_anima")]).network_family(),
            NetworkFamily::Kohya
        );
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_flux_2")]).network_family(),
            NetworkFamily::MusubiTuner
        );
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_krea2")]).network_family(),
            NetworkFamily::MusubiTuner
        );
        assert_eq!(
            meta(&[("ss_network_module", "lycoris.kohya")]).network_family(),
            NetworkFamily::Lycoris
        );
    }

    #[test]
    fn network_family_from_implementation() {
        use crate::network::NetworkFamily;

        assert_eq!(
            meta(&[(
                "modelspec.implementation_version",
                "sd-scripts/11d8cedd0c67b93a10f8b2d75f43554f6f12c794",
            )])
            .network_family(),
            NetworkFamily::Kohya
        );
        assert_eq!(
            meta(&[(
                "modelspec.implementation",
                "https://github.com/OpenMOSS/MOSS-Image",
            )])
            .network_family(),
            NetworkFamily::Kohya
        );
        assert_eq!(
            meta(&[(
                "modelspec.implementation",
                "https://github.com/huggingface/diffusers",
            )])
            .network_family(),
            NetworkFamily::Diffusers
        );
    }

    #[test]
    fn network_family_unknown_cases() {
        use crate::network::NetworkFamily;

        assert_eq!(meta(&[]).network_family(), NetworkFamily::Unknown);
        assert_eq!(
            meta(&[("ss_network_module", "")]).network_family(),
            NetworkFamily::Unknown
        );
        assert_eq!(
            meta(&[("ss_network_module", "foo.bar")]).network_family(),
            NetworkFamily::Unknown
        );
        assert_eq!(
            meta(&[(
                "modelspec.implementation",
                "https://github.com/nobody/some-trainer",
            )])
            .network_family(),
            NetworkFamily::Unknown
        );
        assert_eq!(
            Metadata { metadata: None }.network_family(),
            NetworkFamily::Unknown
        );
    }

    #[test]
    fn network_module_name_strips_vendor_prefix() {
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_generic")]).network_module_name(),
            Some("lora_generic".to_string())
        );
        assert_eq!(
            meta(&[("ss_network_module", "lycoris.kohya")]).network_module_name(),
            Some("kohya".to_string())
        );
        assert_eq!(meta(&[]).network_module_name(), None);
        assert_eq!(meta(&[("ss_network_module", "")]).network_module_name(), None);
    }

    #[test]
    fn lora_generic_maps_to_kohya_module_and_lora_type() {
        use crate::network::{NetworkModule, NetworkType};

        let m = meta(&[("ss_network_module", "networks.lora_generic")]);
        assert_eq!(m.network_module(), Some(NetworkModule::KohyaSSLoRAGeneric));
        assert_eq!(m.network_type(), Some(NetworkType::LoRA));
    }

    #[test]
    fn unmapped_kohya_module_is_not_typed() {
        let m = meta(&[("ss_network_module", "networks.lora_anima")]);
        assert_eq!(m.network_module(), None);
        assert_eq!(m.network_type(), None);
    }

    #[test]
    fn resolve_family_uses_format_only_as_last_resort() {
        use crate::network::NetworkFamily;
        use crate::weight::LoRAFormat;

        assert_eq!(
            meta(&[]).resolve_network_family(&LoRAFormat::Peft),
            NetworkFamily::Peft
        );
        assert_eq!(
            meta(&[]).resolve_network_family(&LoRAFormat::Lycoris),
            NetworkFamily::Lycoris
        );
        // LoRAFormat::Kohya is the fallback format, not evidence.
        assert_eq!(
            meta(&[]).resolve_network_family(&LoRAFormat::Kohya),
            NetworkFamily::Unknown
        );
        // Metadata outranks the format.
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_generic")])
                .resolve_network_family(&LoRAFormat::Peft),
            NetworkFamily::Kohya
        );
    }

    #[test]
    fn module_display_keeps_known_strings() {
        use crate::weight::LoRAFormat;

        assert_eq!(
            meta(&[("ss_network_module", "networks.lora")]).network_module_display(&LoRAFormat::Kohya),
            "kohya-ss/lora"
        );
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_krea2")])
                .network_module_display(&LoRAFormat::Kohya),
            "musubi-tuner/lora_krea2"
        );
        assert_eq!(
            meta(&[("ss_network_module", "lycoris.kohya")]).network_module_display(&LoRAFormat::Lycoris),
            "lycoris"
        );
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_generic")])
                .network_module_display(&LoRAFormat::Kohya),
            "kohya-ss/lora_generic"
        );
    }

    #[test]
    fn module_display_never_says_no_module_found() {
        use crate::weight::LoRAFormat;

        // Unmapped module under a known vendor prefix keeps the family prefix.
        assert_eq!(
            meta(&[("ss_network_module", "networks.lora_anima")])
                .network_module_display(&LoRAFormat::Kohya),
            "kohya-ss/lora_anima"
        );
        // Family known, module unknown.
        assert_eq!(
            meta(&[(
                "modelspec.implementation",
                "https://github.com/huggingface/diffusers"
            )])
            .network_module_display(&LoRAFormat::Kohya),
            "diffusers"
        );
        // Nothing known, but the key shape is PEFT.
        assert_eq!(
            Metadata { metadata: None }.network_module_display(&LoRAFormat::Peft),
            "peft"
        );
        // Nothing known and the format is the fallback.
        assert_eq!(
            Metadata { metadata: None }.network_module_display(&LoRAFormat::Kohya),
            "unknown"
        );
        // Empty and unrecognised vendor prefixes are unknown, not "kohya-ss/".
        assert_eq!(
            meta(&[("ss_network_module", "")]).network_module_display(&LoRAFormat::Kohya),
            "unknown"
        );
        assert_eq!(
            meta(&[("ss_network_module", "foo.bar")]).network_module_display(&LoRAFormat::Kohya),
            "unknown"
        );
    }
}
