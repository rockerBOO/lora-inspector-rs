//! Structural format detection from tensor key names.
//!
//! This is what a loader will actually act on: a key set carrying
//! `hada_w1_a`/`hada_w1_b` loads as LoHa whatever `ss_network_module` claims.
//! Kept separate from [`crate::metadata::NetworkType`], which is derived from
//! metadata and is therefore a hint, not evidence.

/// A network format identified purely from tensor key names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StructureFormat {
    LoRa,
    LoHa,
    LoKr,
    Oft,
    Glora,
}

impl StructureFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            StructureFormat::LoRa => "LoRA",
            StructureFormat::LoHa => "LoHa",
            StructureFormat::LoKr => "LoKr",
            StructureFormat::Oft => "OFT",
            StructureFormat::Glora => "GLoRA",
        }
    }

    /// Lower wins. A layer carrying more than one marker set takes the lowest,
    /// and a file whose formats tie on layer count takes the lowest, so both
    /// call sites agree and neither depends on `HashMap` iteration order.
    pub fn precedence(self) -> u8 {
        match self {
            StructureFormat::LoRa => 0,
            StructureFormat::LoHa => 1,
            StructureFormat::LoKr => 2,
            StructureFormat::Oft => 3,
            StructureFormat::Glora => 4,
        }
    }
}

/// Classify one base name's tensor keys.
///
/// Markers are matched as whole dot-separated tokens rather than substrings:
/// `contains("lokr_w1")` also matches `lokr_w1_a`, which would make the fused
/// and decomposed LoKr forms indistinguishable.
///
/// A partially merged file can carry markers for more than one format at once,
/// so every match is collected and the highest [`StructureFormat::precedence`]
/// wins. That keeps the result independent of key order.
pub fn classify(keys: &[String]) -> Option<StructureFormat> {
    let mut best: Option<StructureFormat> = None;

    for key in keys {
        for token in key.split('.') {
            let Some(format) = marker_format(token) else {
                continue;
            };
            best = Some(match best {
                Some(current) if current.precedence() >= format.precedence() => current,
                _ => format,
            });
        }
    }

    best
}

/// The format a single key token is a marker for, if any.
fn marker_format(token: &str) -> Option<StructureFormat> {
    match token {
        "lora_up" | "lora_down" | "lora_A" | "lora_B" => Some(StructureFormat::LoRa),
        "hada_w1_a" | "hada_w1_b" | "hada_w2_a" | "hada_w2_b" => Some(StructureFormat::LoHa),
        "lokr_w1" | "lokr_w1_a" | "lokr_w1_b" | "lokr_w2" | "lokr_w2_a" | "lokr_w2_b" => {
            Some(StructureFormat::LoKr)
        }
        "oft_diag" | "oft_blocks" => Some(StructureFormat::Oft),
        "a1" | "a2" | "b1" | "b2" => Some(StructureFormat::Glora),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::classify;
    use super::StructureFormat::{Glora, LoHa, LoKr, LoRa, Oft};

    fn keys(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn classify_kohya_lora() {
        assert_eq!(
            classify(&keys(&[
                "x.lora_up.weight",
                "x.lora_down.weight",
                "x.alpha"
            ])),
            Some(LoRa)
        );
    }

    #[test]
    fn classify_peft_lora() {
        assert_eq!(
            classify(&keys(&["x.lora_A.weight", "x.lora_B.weight"])),
            Some(LoRa)
        );
    }

    #[test]
    fn classify_loha() {
        assert_eq!(
            classify(&keys(&[
                "x.hada_w1_a",
                "x.hada_w1_b",
                "x.hada_w2_a",
                "x.hada_w2_b",
                "x.alpha"
            ])),
            Some(LoHa)
        );
    }

    #[test]
    fn classify_lokr_both_forms() {
        assert_eq!(
            classify(&keys(&["x.lokr_w1", "x.lokr_w2", "x.alpha"])),
            Some(LoKr)
        );
        assert_eq!(
            classify(&keys(&[
                "x.lokr_w1_a",
                "x.lokr_w1_b",
                "x.lokr_w2_a",
                "x.lokr_w2_b",
                "x.alpha"
            ])),
            Some(LoKr)
        );
    }

    #[test]
    fn classify_oft_forms() {
        assert_eq!(classify(&keys(&["x.oft_diag"])), Some(Oft));
        assert_eq!(classify(&keys(&["x.oft_blocks"])), Some(Oft));
    }

    #[test]
    fn classify_glora() {
        assert_eq!(
            classify(&keys(&[
                "x.a1.weight",
                "x.a2.weight",
                "x.b1.weight",
                "x.b2.weight"
            ])),
            Some(Glora)
        );
    }

    #[test]
    fn classify_returns_none_without_markers() {
        assert_eq!(classify(&keys(&["x.weight", "x.bias"])), None);
        assert_eq!(classify(&[]), None);
    }

    #[test]
    fn classify_precedence_is_deterministic_for_mixed_markers() {
        let forward = keys(&[
            "x.hada_w1_a",
            "x.hada_w1_b",
            "x.lora_up.weight",
            "x.lora_down.weight",
        ]);
        let reversed: Vec<String> = forward.iter().rev().cloned().collect();
        assert_eq!(classify(&forward), Some(LoHa));
        assert_eq!(classify(&reversed), Some(LoHa));
    }
}
