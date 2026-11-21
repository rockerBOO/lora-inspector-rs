//! Block-key grouping for LoRA base names.
//!
//! A base name (factor suffixes already stripped) is normalized into a token
//! stream — `.`, `_` and `-` are all separators — then scanned left to right for
//! the first block-family pattern. Text-encoder detection runs first and
//! short-circuits the family scan.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockGroup {
    Down,
    Input,
    Mid,
    Middle,
    Up,
    Output,
    Double,
    Single,
    SingleTransformer,
    Transformer,
    Refiner,
    TokenRefiner,
    TxtFusion,
    Blocks,
    Layers,
    TextEncoder,
    Other,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockFamily {
    #[serde(rename = "UNet")]
    Unet,
    TextEncoder,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockKey {
    pub group: BlockGroup,
    pub block_index: Option<usize>,
    pub sublayer: String,
}

impl BlockKey {
    pub fn family(&self) -> BlockFamily {
        match self.group {
            BlockGroup::TextEncoder => BlockFamily::TextEncoder,
            BlockGroup::Other => BlockFamily::Other,
            _ => BlockFamily::Unet,
        }
    }
}

impl BlockGroup {
    /// Two-letter display label plus a zero-padded 2-digit index, or the bare
    /// label when `index` is `None`.
    pub fn tag(&self, index: Option<usize>) -> String {
        let label = self.tag_label();
        match index {
            Some(i) => format!("{label}{i:02}"),
            None => label.to_string(),
        }
    }

    /// Canonical display order (lower sorts first).
    pub fn canonical_rank(&self) -> u8 {
        match self {
            BlockGroup::Down => 0,
            BlockGroup::Input => 1,
            BlockGroup::Mid => 2,
            BlockGroup::Middle => 3,
            BlockGroup::Up => 4,
            BlockGroup::Output => 5,
            BlockGroup::Double => 6,
            BlockGroup::Single => 7,
            BlockGroup::SingleTransformer => 8,
            BlockGroup::Transformer => 9,
            BlockGroup::Refiner => 10,
            BlockGroup::TokenRefiner => 11,
            BlockGroup::TxtFusion => 12,
            BlockGroup::Blocks => 13,
            BlockGroup::Layers => 14,
            BlockGroup::TextEncoder => 15,
            BlockGroup::Other => 16,
        }
    }

    fn tag_label(&self) -> &'static str {
        match self {
            BlockGroup::Down | BlockGroup::Input => "IN",
            BlockGroup::Mid | BlockGroup::Middle => "MID",
            BlockGroup::Up | BlockGroup::Output => "OUT",
            BlockGroup::Double => "DB",
            BlockGroup::Single | BlockGroup::SingleTransformer => "SB",
            BlockGroup::Transformer | BlockGroup::Blocks | BlockGroup::Layers => "TB",
            BlockGroup::Refiner => "RF",
            BlockGroup::TokenRefiner => "TR",
            BlockGroup::TxtFusion => "TF",
            BlockGroup::TextEncoder => "TE",
            BlockGroup::Other => "OT",
        }
    }
}

fn is_separator(c: char) -> bool {
    matches!(c, '.' | '_' | '-')
}

fn is_numeric(token: &str) -> bool {
    token.parse::<usize>().is_ok()
}

/// Split a base name on `.`, `_` and `-`, returning the tokens and the separator
/// that precedes each token. `seps[0]` is unused (there is no preceding token);
/// `seps[i]` for `i > 0` is the separator between `tokens[i - 1]` and `tokens[i]`.
fn tokenize(name: &str) -> (Vec<&str>, Vec<char>) {
    let mut tokens = Vec::new();
    let mut seps = Vec::new();
    let mut token_start: Option<usize> = None;
    let mut pending_sep = '\0';

    for (byte_index, ch) in name.char_indices() {
        if is_separator(ch) {
            if let Some(start) = token_start.take() {
                tokens.push(&name[start..byte_index]);
                seps.push(pending_sep);
            }
            pending_sep = ch;
        } else if token_start.is_none() {
            token_start = Some(byte_index);
        }
    }
    if let Some(start) = token_start {
        tokens.push(&name[start..]);
        seps.push(pending_sep);
    }
    (tokens, seps)
}

/// Rejoin `tokens[start..]` with `.`, preserving the original `_` only when it
/// joins two non-numeric tokens (a word-internal separator). Numeric-adjacent
/// separators, `.` and `-` all become `.`.
fn join_sublayer(tokens: &[&str], seps: &[char], start: usize) -> String {
    let mut out = String::new();
    for i in start..tokens.len() {
        if i > start {
            let keep_underscore =
                seps[i] == '_' && !is_numeric(tokens[i - 1]) && !is_numeric(tokens[i]);
            out.push(if keep_underscore { '_' } else { '.' });
        }
        out.push_str(tokens[i]);
    }
    out
}

fn matches_at(tokens: &[&str], pos: usize, pattern: &[&str]) -> bool {
    tokens.len() >= pos + pattern.len() && &tokens[pos..pos + pattern.len()] == pattern
}

fn first_index_pos(tokens: &[&str], start: usize) -> Option<usize> {
    (start..tokens.len()).find(|&i| is_numeric(tokens[i]))
}

const PATTERNS_4: &[(&[&str], BlockGroup)] = &[(
    &["token", "refiner", "refiner", "blocks"],
    BlockGroup::TokenRefiner,
)];

const PATTERNS_3: &[(&[&str], BlockGroup)] = &[
    (
        &["single", "transformer", "blocks"],
        BlockGroup::SingleTransformer,
    ),
    (&["txtfusion", "layerwise", "blocks"], BlockGroup::TxtFusion),
    (&["txtfusion", "refiner", "blocks"], BlockGroup::TxtFusion),
    (&["token", "refiner", "blocks"], BlockGroup::TokenRefiner),
];

const PATTERNS_2: &[(&[&str], BlockGroup)] = &[
    (&["double", "blocks"], BlockGroup::Double),
    (&["single", "blocks"], BlockGroup::Single),
    (&["transformer", "blocks"], BlockGroup::Transformer),
    (&["down", "blocks"], BlockGroup::Down),
    (&["input", "blocks"], BlockGroup::Input),
    (&["middle", "block"], BlockGroup::Middle),
    (&["up", "blocks"], BlockGroup::Up),
    (&["output", "blocks"], BlockGroup::Output),
    (&["refiner", "blocks"], BlockGroup::TokenRefiner),
    (&["noise", "refiner"], BlockGroup::Refiner),
    (&["context", "refiner"], BlockGroup::Refiner),
];

const PATTERNS_1: &[(&[&str], BlockGroup)] = &[
    (&["blocks"], BlockGroup::Blocks),
    (&["layers"], BlockGroup::Layers),
];

/// Scan left to right for the first family match. At a position the longest
/// pattern wins. Returns `(group, pattern_end, block_index_position)`.
fn scan_family(tokens: &[&str]) -> Option<(BlockGroup, usize, Option<usize>)> {
    for pos in 0..tokens.len() {
        for (pattern, group) in PATTERNS_4 {
            if matches_at(tokens, pos, pattern) {
                return Some((*group, pos + 4, first_index_pos(tokens, pos + 4)));
            }
        }
        for (pattern, group) in PATTERNS_3 {
            if matches_at(tokens, pos, pattern) {
                return Some((*group, pos + 3, first_index_pos(tokens, pos + 3)));
            }
        }
        for (pattern, group) in PATTERNS_2 {
            if matches_at(tokens, pos, pattern) {
                return Some((*group, pos + 2, first_index_pos(tokens, pos + 2)));
            }
        }
        // kohya's `mid_block` has no block index; the integers that follow are
        // sub-block ids that belong in the sublayer, so the index stays `None`.
        if matches_at(tokens, pos, &["mid", "block"]) {
            return Some((BlockGroup::Mid, pos + 2, None));
        }
        for (pattern, group) in PATTERNS_1 {
            if matches_at(tokens, pos, pattern) {
                // Bare `blocks`/`layers` are only a family when the block index
                // follows immediately; a fused token like `0noise` means the word
                // belongs to some other (malformed) structure and must not group.
                if pos + 1 < tokens.len() && is_numeric(tokens[pos + 1]) {
                    return Some((*group, pos + 2, Some(pos + 1)));
                }
            }
        }
    }
    None
}

fn is_te_token(token: &str) -> bool {
    token == "te"
        || (token.len() > 2
            && token.starts_with("te")
            && token[2..].bytes().all(|b| b.is_ascii_digit()))
}

/// Text-encoder detection runs first. Returns the end position of the matched
/// marker (leftmost, longest phrase at a position).
fn detect_text_encoder(tokens: &[&str]) -> Option<usize> {
    for i in 0..tokens.len() {
        if is_te_token(tokens[i]) {
            return Some(i + 1);
        }
        if i + 3 <= tokens.len()
            && tokens[i] == "text"
            && tokens[i + 1] == "model"
            && tokens[i + 2] == "encoder"
        {
            return Some(i + 3);
        }
        if i + 2 <= tokens.len()
            && ((tokens[i] == "text" && tokens[i + 1] == "model")
                || (tokens[i] == "language" && tokens[i + 1] == "model"))
        {
            return Some(i + 2);
        }
    }
    None
}

pub fn parse_block_key(base_name: &str) -> BlockKey {
    let (tokens, seps) = tokenize(base_name);

    if let Some(marker_end) = detect_text_encoder(&tokens) {
        let index_pos = first_index_pos(&tokens, marker_end);
        let block_index = index_pos.map(|p| tokens[p].parse::<usize>().unwrap());
        let sublayer_start = index_pos.map_or(marker_end, |p| p + 1);
        return BlockKey {
            group: BlockGroup::TextEncoder,
            block_index,
            sublayer: join_sublayer(&tokens, &seps, sublayer_start),
        };
    }

    if let Some((group, pattern_end, index_pos)) = scan_family(&tokens) {
        let block_index = index_pos.map(|p| tokens[p].parse::<usize>().unwrap());
        let sublayer_start = index_pos.map_or(pattern_end, |p| p + 1);
        return BlockKey {
            group,
            block_index,
            sublayer: join_sublayer(&tokens, &seps, sublayer_start),
        };
    }

    let block_index = first_index_pos(&tokens, 0).map(|p| tokens[p].parse::<usize>().unwrap());
    BlockKey {
        group: BlockGroup::Other,
        block_index,
        sublayer: join_sublayer(&tokens, &seps, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leftmost_family_wins_over_inner_transformer_blocks() {
        let k = parse_block_key(
            "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_ff_net_0_proj",
        );
        assert_eq!(k.group, BlockGroup::Down);
        assert_eq!(k.block_index, Some(0));
        assert_eq!(
            k.sublayer,
            "attentions.0.transformer_blocks.0.ff_net.0.proj"
        );
    }

    #[test]
    fn peft_single_transformer_blocks_is_its_own_group() {
        let k = parse_block_key("transformer.single_transformer_blocks.7.linear1");
        assert_eq!(k.group, BlockGroup::SingleTransformer);
        assert_eq!(k.block_index, Some(7));
    }

    #[test]
    fn text_encoder_wins_over_unet_layers() {
        let a = parse_block_key("lora_te_text_model_encoder_layers_11_self_attn_v_proj");
        let b = parse_block_key("language_model_model_layers_18_self_attn_o_proj");
        assert_eq!(
            (a.group, a.block_index),
            (BlockGroup::TextEncoder, Some(11))
        );
        assert_eq!(
            (b.group, b.block_index),
            (BlockGroup::TextEncoder, Some(18))
        );
    }

    #[test]
    fn dotted_peft_text_encoder_is_normalized() {
        let k = parse_block_key("text_model.encoder.layers.11.self_attn.q_proj");
        assert_eq!(k.group, BlockGroup::TextEncoder);
        assert_eq!(k.block_index, Some(11));
    }

    #[test]
    fn bare_and_refiner_families_are_recognized() {
        assert_eq!(
            parse_block_key("diffusion_model.blocks.3.cross_attn.k").group,
            BlockGroup::Blocks
        );
        assert_eq!(
            parse_block_key("diffusion_model.token_refiner.refiner_blocks.1.mlp.fc2").group,
            BlockGroup::TokenRefiner
        );
        assert_eq!(
            parse_block_key("diffusion_model.blocks.0noise_refiner_0_adaLN").group,
            BlockGroup::Other
        );
        assert_eq!(
            parse_block_key("diffusion_model.noise_refiner.0.adaLN_modulation").group,
            BlockGroup::Refiner
        );
    }

    #[test]
    fn unparseable_key_keeps_its_sublayer_and_is_other() {
        let k = parse_block_key("txtfusion_projector");
        assert_eq!(k.group, BlockGroup::Other);
        assert_eq!(k.block_index, None);
        assert_eq!(k.family(), BlockFamily::Other);
    }

    #[test]
    fn tags_are_stable() {
        assert_eq!(BlockGroup::Input.tag(Some(4)), "IN04");
        assert_eq!(BlockGroup::Double.tag(Some(3)), "DB03");
        assert_eq!(BlockGroup::TextEncoder.tag(Some(11)), "TE11");
        assert_eq!(BlockGroup::Other.tag(None), "OT");
    }

    #[test]
    fn parses_base_name_after_suffix_stripping() {
        let base =
            crate::get_base_name("lora_unet_down_blocks_1_attentions_0_proj_out.lora_up.weight");
        let k = parse_block_key(&base);
        assert_eq!(k.group, BlockGroup::Down);
        assert_eq!(k.block_index, Some(1));
        assert_eq!(k.sublayer, "attentions.0.proj_out");
    }

    fn check_fixture(content: &str, expected: &[(&str, BlockGroup, Option<usize>, &str)]) {
        let lines: Vec<&str> = content
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(lines.len(), expected.len(), "fixture line count");
        for (line, exp) in lines.iter().zip(expected) {
            assert_eq!(*line, exp.0, "fixture line changed");
            let key = parse_block_key(line);
            assert_eq!(key.group, exp.1, "group for {line}");
            assert_eq!(key.block_index, exp.2, "block_index for {line}");
            let first_segment = key.sublayer.split('.').next().unwrap_or("");
            assert_eq!(first_segment, exp.3, "sublayer first segment for {line}");
        }
    }

    #[test]
    fn fixture_driven_grouping_matches_golden_table() {
        check_fixture(
            include_str!("../tests/fixtures/blocks/flux.txt"),
            &[
                (
                    "lora_unet_double_blocks_0_img_attn_proj",
                    BlockGroup::Double,
                    Some(0),
                    "img_attn_proj",
                ),
                (
                    "lora_unet_double_blocks_0_img_attn_qkv",
                    BlockGroup::Double,
                    Some(0),
                    "img_attn_qkv",
                ),
                (
                    "lora_unet_double_blocks_0_img_mlp_0",
                    BlockGroup::Double,
                    Some(0),
                    "img_mlp",
                ),
                (
                    "lora_unet_double_blocks_0_txt_attn_proj",
                    BlockGroup::Double,
                    Some(0),
                    "txt_attn_proj",
                ),
                (
                    "lora_unet_double_blocks_0_txt_mod_lin",
                    BlockGroup::Double,
                    Some(0),
                    "txt_mod_lin",
                ),
                (
                    "lora_unet_double_blocks_5_img_attn_qkv",
                    BlockGroup::Double,
                    Some(5),
                    "img_attn_qkv",
                ),
                (
                    "lora_unet_double_blocks_10_txt_mlp_2",
                    BlockGroup::Double,
                    Some(10),
                    "txt_mlp",
                ),
                (
                    "lora_unet_double_blocks_18_img_attn_proj",
                    BlockGroup::Double,
                    Some(18),
                    "img_attn_proj",
                ),
                (
                    "lora_unet_single_blocks_0_linear1",
                    BlockGroup::Single,
                    Some(0),
                    "linear1",
                ),
                (
                    "lora_unet_single_blocks_0_linear2",
                    BlockGroup::Single,
                    Some(0),
                    "linear2",
                ),
                (
                    "lora_unet_single_blocks_0_modulation_lin",
                    BlockGroup::Single,
                    Some(0),
                    "modulation_lin",
                ),
                (
                    "lora_unet_single_blocks_5_linear1",
                    BlockGroup::Single,
                    Some(5),
                    "linear1",
                ),
                (
                    "lora_unet_single_blocks_10_linear1",
                    BlockGroup::Single,
                    Some(10),
                    "linear1",
                ),
                (
                    "lora_unet_single_blocks_10_modulation_lin",
                    BlockGroup::Single,
                    Some(10),
                    "modulation_lin",
                ),
                (
                    "lora_unet_single_blocks_20_linear2",
                    BlockGroup::Single,
                    Some(20),
                    "linear2",
                ),
                (
                    "lora_unet_single_blocks_30_linear1",
                    BlockGroup::Single,
                    Some(30),
                    "linear1",
                ),
                (
                    "lora_unet_single_blocks_37_linear1",
                    BlockGroup::Single,
                    Some(37),
                    "linear1",
                ),
                (
                    "lora_unet_single_blocks_37_modulation_lin",
                    BlockGroup::Single,
                    Some(37),
                    "modulation_lin",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/flux_peft.txt"),
            &[
                (
                    "transformer.single_transformer_blocks.0.attn.to_k",
                    BlockGroup::SingleTransformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer.single_transformer_blocks.0.attn.to_q",
                    BlockGroup::SingleTransformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer.single_transformer_blocks.0.norm.linear",
                    BlockGroup::SingleTransformer,
                    Some(0),
                    "norm",
                ),
                (
                    "transformer.single_transformer_blocks.0.proj_mlp",
                    BlockGroup::SingleTransformer,
                    Some(0),
                    "proj_mlp",
                ),
                (
                    "transformer.single_transformer_blocks.1.proj_out",
                    BlockGroup::SingleTransformer,
                    Some(1),
                    "proj_out",
                ),
                (
                    "transformer.single_transformer_blocks.18.attn.to_v",
                    BlockGroup::SingleTransformer,
                    Some(18),
                    "attn",
                ),
                (
                    "transformer.single_transformer_blocks.19.proj_mlp",
                    BlockGroup::SingleTransformer,
                    Some(19),
                    "proj_mlp",
                ),
                (
                    "transformer.single_transformer_blocks.29.proj_out",
                    BlockGroup::SingleTransformer,
                    Some(29),
                    "proj_out",
                ),
                (
                    "transformer.single_transformer_blocks.37.attn.to_q",
                    BlockGroup::SingleTransformer,
                    Some(37),
                    "attn",
                ),
                (
                    "transformer.transformer_blocks.0.attn.add_k_proj",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer.transformer_blocks.0.attn.to_q",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer.transformer_blocks.0.ff.net.0.proj",
                    BlockGroup::Transformer,
                    Some(0),
                    "ff",
                ),
                (
                    "transformer.transformer_blocks.0.ff_context.net.2",
                    BlockGroup::Transformer,
                    Some(0),
                    "ff_context",
                ),
                (
                    "transformer.transformer_blocks.0.norm1.linear",
                    BlockGroup::Transformer,
                    Some(0),
                    "norm1",
                ),
                (
                    "transformer.transformer_blocks.4.attn.to_out.0",
                    BlockGroup::Transformer,
                    Some(4),
                    "attn",
                ),
                (
                    "transformer.transformer_blocks.4.ff.net.2",
                    BlockGroup::Transformer,
                    Some(4),
                    "ff",
                ),
                (
                    "transformer.transformer_blocks.10.attn.add_q_proj",
                    BlockGroup::Transformer,
                    Some(10),
                    "attn",
                ),
                (
                    "transformer.transformer_blocks.18.norm1_context.linear",
                    BlockGroup::Transformer,
                    Some(18),
                    "norm1_context",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/flux2_klein.txt"),
            &[
                (
                    "diffusion_model.double_blocks.0.img_attn.proj",
                    BlockGroup::Double,
                    Some(0),
                    "img_attn",
                ),
                (
                    "diffusion_model.double_blocks.0.img_attn.qkv",
                    BlockGroup::Double,
                    Some(0),
                    "img_attn",
                ),
                (
                    "diffusion_model.double_blocks.0.img_mlp.0",
                    BlockGroup::Double,
                    Some(0),
                    "img_mlp",
                ),
                (
                    "diffusion_model.double_blocks.0.txt_attn.proj",
                    BlockGroup::Double,
                    Some(0),
                    "txt_attn",
                ),
                (
                    "diffusion_model.double_blocks.0.txt_mlp.2",
                    BlockGroup::Double,
                    Some(0),
                    "txt_mlp",
                ),
                (
                    "diffusion_model.double_blocks.1.txt_attn.qkv",
                    BlockGroup::Double,
                    Some(1),
                    "txt_attn",
                ),
                (
                    "diffusion_model.double_blocks.2.img_attn.qkv",
                    BlockGroup::Double,
                    Some(2),
                    "img_attn",
                ),
                (
                    "diffusion_model.double_blocks.4.img_mlp.2",
                    BlockGroup::Double,
                    Some(4),
                    "img_mlp",
                ),
                (
                    "diffusion_model.double_blocks.7.txt_mlp.0",
                    BlockGroup::Double,
                    Some(7),
                    "txt_mlp",
                ),
                (
                    "diffusion_model.single_blocks.0.linear1",
                    BlockGroup::Single,
                    Some(0),
                    "linear1",
                ),
                (
                    "diffusion_model.single_blocks.0.linear2",
                    BlockGroup::Single,
                    Some(0),
                    "linear2",
                ),
                (
                    "diffusion_model.single_blocks.5.linear1",
                    BlockGroup::Single,
                    Some(5),
                    "linear1",
                ),
                (
                    "diffusion_model.single_blocks.11.linear1",
                    BlockGroup::Single,
                    Some(11),
                    "linear1",
                ),
                (
                    "diffusion_model.single_blocks.17.linear2",
                    BlockGroup::Single,
                    Some(17),
                    "linear2",
                ),
                (
                    "diffusion_model.single_blocks.23.linear2",
                    BlockGroup::Single,
                    Some(23),
                    "linear2",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/gemma3.txt"),
            &[
                (
                    "language_model_model_layers_18_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(18),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_19_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(19),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_20_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(20),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_21_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(21),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_22_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(22),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_23_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(23),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_24_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(24),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_25_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(25),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_26_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(26),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_27_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(27),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_28_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(28),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_29_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(29),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_30_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(30),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_31_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(31),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_32_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(32),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_33_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(33),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_34_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(34),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_35_self_attn_o_proj",
                    BlockGroup::TextEncoder,
                    Some(35),
                    "self_attn_o_proj",
                ),
                (
                    "language_model_model_layers_43_mlp_down_proj",
                    BlockGroup::TextEncoder,
                    Some(43),
                    "mlp_down_proj",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/h3.txt"),
            &[
                (
                    "diffusion_model.blocks.0.attn.out_proj",
                    BlockGroup::Blocks,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.0.attn.qkv_proj",
                    BlockGroup::Blocks,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.0.mlp.fc1",
                    BlockGroup::Blocks,
                    Some(0),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.0.mlp.fc2",
                    BlockGroup::Blocks,
                    Some(0),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.1.attn.qkv_proj",
                    BlockGroup::Blocks,
                    Some(1),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.1.mlp.fc2",
                    BlockGroup::Blocks,
                    Some(1),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.10.attn.out_proj",
                    BlockGroup::Blocks,
                    Some(10),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.25.mlp.fc1",
                    BlockGroup::Blocks,
                    Some(25),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.49.attn.qkv_proj",
                    BlockGroup::Blocks,
                    Some(49),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.49.mlp.fc2",
                    BlockGroup::Blocks,
                    Some(49),
                    "mlp",
                ),
                (
                    "diffusion_model.token_refiner.blocks.0.attn.out_proj",
                    BlockGroup::TokenRefiner,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.token_refiner.blocks.0.attn.qkv_proj",
                    BlockGroup::TokenRefiner,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.token_refiner.blocks.0.mlp.fc1",
                    BlockGroup::TokenRefiner,
                    Some(0),
                    "mlp",
                ),
                (
                    "diffusion_model.token_refiner.blocks.0.mlp.fc2",
                    BlockGroup::TokenRefiner,
                    Some(0),
                    "mlp",
                ),
                (
                    "diffusion_model.token_refiner.blocks.1.attn.qkv_proj",
                    BlockGroup::TokenRefiner,
                    Some(1),
                    "attn",
                ),
                (
                    "diffusion_model.token_refiner.blocks.1.mlp.fc2",
                    BlockGroup::TokenRefiner,
                    Some(1),
                    "mlp",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/krea2.txt"),
            &[
                (
                    "diffusion_model.blocks.0.attn.wo",
                    BlockGroup::Blocks,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.0.attn.wq",
                    BlockGroup::Blocks,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.0.mlp.up",
                    BlockGroup::Blocks,
                    Some(0),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.1.attn.wk",
                    BlockGroup::Blocks,
                    Some(1),
                    "attn",
                ),
                (
                    "diffusion_model.blocks.1.mlp.gate",
                    BlockGroup::Blocks,
                    Some(1),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.11.mlp.down",
                    BlockGroup::Blocks,
                    Some(11),
                    "mlp",
                ),
                (
                    "diffusion_model.blocks.27.attn.wq",
                    BlockGroup::Blocks,
                    Some(27),
                    "attn",
                ),
                (
                    "diffusion_model.txtfusion.layerwise_blocks.0.attn.wq",
                    BlockGroup::TxtFusion,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.txtfusion.layerwise_blocks.1.mlp.up",
                    BlockGroup::TxtFusion,
                    Some(1),
                    "mlp",
                ),
                (
                    "diffusion_model.txtfusion.refiner_blocks.0.attn.wo",
                    BlockGroup::TxtFusion,
                    Some(0),
                    "attn",
                ),
                (
                    "diffusion_model.txtfusion.refiner_blocks.1.mlp.down",
                    BlockGroup::TxtFusion,
                    Some(1),
                    "mlp",
                ),
                (
                    "lora_unet_first",
                    BlockGroup::Other,
                    None,
                    "lora_unet_first",
                ),
                (
                    "lora_unet_last_linear",
                    BlockGroup::Other,
                    None,
                    "lora_unet_last_linear",
                ),
                (
                    "lora_unet_tmlp_0",
                    BlockGroup::Other,
                    Some(0),
                    "lora_unet_tmlp",
                ),
                (
                    "lora_unet_tproj_1",
                    BlockGroup::Other,
                    Some(1),
                    "lora_unet_tproj",
                ),
                (
                    "lora_unet_txtfusion_projector",
                    BlockGroup::Other,
                    None,
                    "lora_unet_txtfusion_projector",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/ltx23.txt"),
            &[
                (
                    "diffusion_model.transformer_blocks.0.attn1.to_gate_logits",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn1.to_k",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn1.to_out.0",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn1.to_q",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn1.to_v",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn2.to_out.0",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn2",
                ),
                (
                    "diffusion_model.transformer_blocks.0.attn2.to_q",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn2",
                ),
                (
                    "diffusion_model.transformer_blocks.0.audio_attn1.to_k",
                    BlockGroup::Transformer,
                    Some(0),
                    "audio_attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.0.audio_attn2.to_v",
                    BlockGroup::Transformer,
                    Some(0),
                    "audio_attn2",
                ),
                (
                    "diffusion_model.transformer_blocks.0.audio_to_video_attn.to_gate_logits",
                    BlockGroup::Transformer,
                    Some(0),
                    "audio_to_video_attn",
                ),
                (
                    "diffusion_model.transformer_blocks.1.attn1.to_out.0",
                    BlockGroup::Transformer,
                    Some(1),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.1.attn1.to_q",
                    BlockGroup::Transformer,
                    Some(1),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.5.attn1.to_v",
                    BlockGroup::Transformer,
                    Some(5),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.17.attn1.to_k",
                    BlockGroup::Transformer,
                    Some(17),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.17.attn2.to_q",
                    BlockGroup::Transformer,
                    Some(17),
                    "attn2",
                ),
                (
                    "diffusion_model.transformer_blocks.31.attn1.to_q",
                    BlockGroup::Transformer,
                    Some(31),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.47.attn1.to_gate_logits",
                    BlockGroup::Transformer,
                    Some(47),
                    "attn1",
                ),
                (
                    "diffusion_model.transformer_blocks.47.attn1.to_out.0",
                    BlockGroup::Transformer,
                    Some(47),
                    "attn1",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/moss.txt"),
            &[
                (
                    "lora_unet_layers_0_proj_ffn_down",
                    BlockGroup::Layers,
                    Some(0),
                    "proj_ffn_down",
                ),
                (
                    "lora_unet_layers_0_proj_ffn_up_gate",
                    BlockGroup::Layers,
                    Some(0),
                    "proj_ffn_up_gate",
                ),
                (
                    "lora_unet_layers_0_proj_k",
                    BlockGroup::Layers,
                    Some(0),
                    "proj_k",
                ),
                (
                    "lora_unet_layers_0_proj_o",
                    BlockGroup::Layers,
                    Some(0),
                    "proj_o",
                ),
                (
                    "lora_unet_layers_0_proj_q",
                    BlockGroup::Layers,
                    Some(0),
                    "proj_q",
                ),
                (
                    "lora_unet_layers_0_proj_v",
                    BlockGroup::Layers,
                    Some(0),
                    "proj_v",
                ),
                (
                    "lora_unet_layers_1_proj_ffn_down",
                    BlockGroup::Layers,
                    Some(1),
                    "proj_ffn_down",
                ),
                (
                    "lora_unet_layers_1_proj_q",
                    BlockGroup::Layers,
                    Some(1),
                    "proj_q",
                ),
                (
                    "lora_unet_layers_10_proj_ffn_up_gate",
                    BlockGroup::Layers,
                    Some(10),
                    "proj_ffn_up_gate",
                ),
                (
                    "lora_unet_layers_10_proj_k",
                    BlockGroup::Layers,
                    Some(10),
                    "proj_k",
                ),
                (
                    "lora_unet_layers_20_proj_o",
                    BlockGroup::Layers,
                    Some(20),
                    "proj_o",
                ),
                (
                    "lora_unet_layers_20_proj_v",
                    BlockGroup::Layers,
                    Some(20),
                    "proj_v",
                ),
                (
                    "lora_unet_layers_30_proj_ffn_down",
                    BlockGroup::Layers,
                    Some(30),
                    "proj_ffn_down",
                ),
                (
                    "lora_unet_layers_30_proj_q",
                    BlockGroup::Layers,
                    Some(30),
                    "proj_q",
                ),
                (
                    "lora_unet_layers_41_proj_ffn_up_gate",
                    BlockGroup::Layers,
                    Some(41),
                    "proj_ffn_up_gate",
                ),
                (
                    "lora_unet_layers_41_proj_k",
                    BlockGroup::Layers,
                    Some(41),
                    "proj_k",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/qwen_image.txt"),
            &[
                (
                    "transformer_blocks.0.attn.add_k_proj",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.attn.add_q_proj",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.attn.to_add_out",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.attn.to_k",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.attn.to_out.0",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.attn.to_q",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.attn.to_v",
                    BlockGroup::Transformer,
                    Some(0),
                    "attn",
                ),
                (
                    "transformer_blocks.0.img_mlp.net.0.proj",
                    BlockGroup::Transformer,
                    Some(0),
                    "img_mlp",
                ),
                (
                    "transformer_blocks.0.img_mlp.net.2",
                    BlockGroup::Transformer,
                    Some(0),
                    "img_mlp",
                ),
                (
                    "transformer_blocks.0.txt_mlp.net.0.proj",
                    BlockGroup::Transformer,
                    Some(0),
                    "txt_mlp",
                ),
                (
                    "transformer_blocks.1.attn.to_q",
                    BlockGroup::Transformer,
                    Some(1),
                    "attn",
                ),
                (
                    "transformer_blocks.1.img_mlp.net.2",
                    BlockGroup::Transformer,
                    Some(1),
                    "img_mlp",
                ),
                (
                    "transformer_blocks.20.attn.to_v",
                    BlockGroup::Transformer,
                    Some(20),
                    "attn",
                ),
                (
                    "transformer_blocks.20.txt_mlp.net.2",
                    BlockGroup::Transformer,
                    Some(20),
                    "txt_mlp",
                ),
                (
                    "transformer_blocks.59.attn.to_out.0",
                    BlockGroup::Transformer,
                    Some(59),
                    "attn",
                ),
                (
                    "transformer_blocks.59.img_mlp.net.0.proj",
                    BlockGroup::Transformer,
                    Some(59),
                    "img_mlp",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/sd15.txt"),
            &[
                (
                    "lora_te_text_model_encoder_layers_0_mlp_fc1",
                    BlockGroup::TextEncoder,
                    Some(0),
                    "mlp_fc1",
                ),
                (
                    "lora_te_text_model_encoder_layers_0_self_attn_q_proj",
                    BlockGroup::TextEncoder,
                    Some(0),
                    "self_attn_q_proj",
                ),
                (
                    "lora_te_text_model_encoder_layers_5_self_attn_k_proj",
                    BlockGroup::TextEncoder,
                    Some(5),
                    "self_attn_k_proj",
                ),
                (
                    "lora_te_text_model_encoder_layers_11_self_attn_out_proj",
                    BlockGroup::TextEncoder,
                    Some(11),
                    "self_attn_out_proj",
                ),
                (
                    "lora_unet_down_blocks_0_attentions_0_proj_in",
                    BlockGroup::Down,
                    Some(0),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q",
                    BlockGroup::Down,
                    Some(0),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_1_attentions_0_proj_out",
                    BlockGroup::Down,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_ff_net_0_proj",
                    BlockGroup::Down,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_2_attentions_1_transformer_blocks_0_attn2_to_v",
                    BlockGroup::Down,
                    Some(2),
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_proj_in",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_out_0",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_2",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn2_to_v",
                    BlockGroup::Up,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0",
                    BlockGroup::Up,
                    Some(2),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_3_attentions_0_transformer_blocks_0_attn2_to_out_0",
                    BlockGroup::Up,
                    Some(3),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_3_attentions_1_transformer_blocks_0_attn1_to_q",
                    BlockGroup::Up,
                    Some(3),
                    "attentions",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/sdxl.txt"),
            &[
                (
                    "lora_unet_down_blocks_1_attentions_0_proj_in",
                    BlockGroup::Down,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q",
                    BlockGroup::Down,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_1_attentions_1_transformer_blocks_0_attn2_to_v",
                    BlockGroup::Down,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_2_attentions_0_proj_out",
                    BlockGroup::Down,
                    Some(2),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0",
                    BlockGroup::Down,
                    Some(2),
                    "attentions",
                ),
                (
                    "lora_unet_down_blocks_2_attentions_1_transformer_blocks_1_ff_net_2",
                    BlockGroup::Down,
                    Some(2),
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_proj_in",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn1_to_k",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_transformer_blocks_0_attn2_to_out_0",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_mid_block_attentions_0_transformer_blocks_0_ff_net_0_proj",
                    BlockGroup::Mid,
                    None,
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q",
                    BlockGroup::Up,
                    Some(0),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_0_attentions_0_transformer_blocks_0_ff_net_2",
                    BlockGroup::Up,
                    Some(0),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0",
                    BlockGroup::Up,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_1_attentions_2_proj_in",
                    BlockGroup::Up,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_1_attentions_2_transformer_blocks_1_attn2_to_out_0",
                    BlockGroup::Up,
                    Some(1),
                    "attentions",
                ),
                (
                    "lora_unet_up_blocks_1_attentions_2_transformer_blocks_1_ff_net_0_proj",
                    BlockGroup::Up,
                    Some(1),
                    "attentions",
                ),
            ],
        );

        check_fixture(
            include_str!("../tests/fixtures/blocks/wan22.txt"),
            &[
                (
                    "diffusion_model.blocks.0.cross_attn.k",
                    BlockGroup::Blocks,
                    Some(0),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.0.cross_attn.o",
                    BlockGroup::Blocks,
                    Some(0),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.0.cross_attn.q",
                    BlockGroup::Blocks,
                    Some(0),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.0.cross_attn.v",
                    BlockGroup::Blocks,
                    Some(0),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.0.ffn.0",
                    BlockGroup::Blocks,
                    Some(0),
                    "ffn",
                ),
                (
                    "diffusion_model.blocks.0.ffn.2",
                    BlockGroup::Blocks,
                    Some(0),
                    "ffn",
                ),
                (
                    "diffusion_model.blocks.0.self_attn.k",
                    BlockGroup::Blocks,
                    Some(0),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.0.self_attn.o",
                    BlockGroup::Blocks,
                    Some(0),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.0.self_attn.q",
                    BlockGroup::Blocks,
                    Some(0),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.0.self_attn.v",
                    BlockGroup::Blocks,
                    Some(0),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.1.cross_attn.k",
                    BlockGroup::Blocks,
                    Some(1),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.1.cross_attn.q",
                    BlockGroup::Blocks,
                    Some(1),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.1.self_attn.o",
                    BlockGroup::Blocks,
                    Some(1),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.1.self_attn.v",
                    BlockGroup::Blocks,
                    Some(1),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.20.cross_attn.o",
                    BlockGroup::Blocks,
                    Some(20),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.20.self_attn.q",
                    BlockGroup::Blocks,
                    Some(20),
                    "self_attn",
                ),
                (
                    "diffusion_model.blocks.39.cross_attn.v",
                    BlockGroup::Blocks,
                    Some(39),
                    "cross_attn",
                ),
                (
                    "diffusion_model.blocks.39.self_attn.k",
                    BlockGroup::Blocks,
                    Some(39),
                    "self_attn",
                ),
            ],
        );
    }
}
