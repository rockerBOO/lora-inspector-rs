// TODO: This is getting a bit messy but needs to work at the moment.

// Handle parsing of the keys

const SDRE =
	/.*(?<block_type>up|down|mid)_blocks?_(?<block_id>\d+).*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;

const MID_SDRE =
	/.*(?<block_type>up|down|mid)_block_.*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<block_id>\d+)_.*(?<subblock_id>\d+)?.*/;
const TE_SDRE =
	/te(?<encoder>\d+)?_.*_(?<block_id>\d+).*(?<block_type>self_attn|mlp)/;
const NUM_OF_BLOCKS = 12;

const FLUX_DOUBLE =
	/(lora_unet|diffusion_model)[._](?<block_type>double_blocks)[._](?<block_id>\d+)[._](?<modality>txt|img)[._](?<subblock_type>attn[._]proj|attn[._]qkv|mlp[._]0|mlp[._]2|mod[._]lin)/;

const FLUX_SINGLE =
	/(lora_unet|diffusion_model)[._](?<block_type>single_blocks)[._](?<block_id>\d+)[._](?<subblock_type>linear1|linear2|modulation[._]lin)/;

const FLUX_PEFT =
	/transformer\.(?<block_type>single_transformer_blocks|transformer_blocks)\.(?<block_id>\d+)(\.(?<type>\w+))?\.?(?<subtype>\w+)?/;

// MiniMax H3 (diffusers-native PEFT, no "transformer." prefix): bare
// transformer_blocks.N.* plus a token_refiner.refiner_blocks.N.* block group.
// Also covers the ai-toolkit trained variant, which prefixes keys with
// "diffusion_model." and uses "blocks"/"token_refiner.blocks" with
// attn.qkv_proj/attn.out_proj/mlp.fc1/mlp.fc2 sub-blocks. Qwen-Image reuses
// this layout but names its feed-forward halves txt_mlp/img_mlp.
const H3_BLOCK_RE =
	/(?:diffusion_model\.)?(?<block_type>transformer_blocks|token_refiner\.refiner_blocks|blocks|token_refiner\.blocks)\.(?<block_id>\d+)\.(?<subblock_type>attn(?:\.\w+)?|txt_mlp(?:\.\w+)?|img_mlp(?:\.\w+)?|mlp(?:\.\w+)?|ff)/;

// Wan 2.x video DiT block naming: kohya-style lora_unet_blocks_N_(self_attn|
// cross_attn)_* / lora_unet_blocks_N_ffn_N, and diffusers-style
// diffusion_model.blocks.N.<self_attn|cross_attn|ffn>.*
const WAN_BLOCK_RE =
	/(lora_unet|diffusion_model)[._]blocks[._](?<block_id>\d+)[._](?<subblock_type>self_attn|cross_attn|ffn)/;

// LTX 2.3 audio/video multimodal DiT (diffusers-native, "diffusion_model."
// prefix): diffusion_model.transformer_blocks.N.<attn1|attn2|audio_attn1|
// audio_attn2|audio_to_video_attn|video_to_audio_attn>.*
const LTX_BLOCK_RE =
	/diffusion_model\.transformer_blocks\.(?<block_id>\d+)\.(?<subblock_type>audio_attn\d+|audio_to_video_attn|video_to_audio_attn|attn\d+)/;

const LUMINA_TRANSFORMER =
	/.*unet.*(?<block_type>layers|noise_refiner|context_refiner).*_(?<block_id>\d+)_(?<type>adaLN_modulation|feed_forward|attention_out|attention_qkv)(?<subblock_type>_w\d+)?/;

// MOSS-Image (sd-scripts networks.lora_generic) also names its DiT blocks
// "layers.N", but the sub-modules are proj_q/proj_k/proj_v/proj_o for
// attention and proj_ffn_up_gate/proj_ffn_down for the SwiGLU FFN. This has to
// be checked before LUMINA_TRANSFORMER, whose type group does not cover them.
const MOSS_IMAGE_BLOCK_RE =
	/(?:lora_unet_|diffusion_model\.)layers[._](?<block_id>\d+)[._](?<subblock_type>proj_q|proj_k|proj_v|proj_o|proj_ffn_up_gate|proj_ffn_down)/;

const MOSS_IMAGE_ATTENTION_MODULES = new Set([
	"proj_q",
	"proj_k",
	"proj_v",
	"proj_o",
]);

const GEMMA =
	/.*te.*(?<block_type>layers).*_(?<block_id>\d+)_(?<type>self_attn_q_proj|self_attn_k_proj|self_attn_v_proj|self_attn_o_proj|mlp_down_proj|mlp_up_proj|mlp_gate_proj)/;

// Stable Diffusion XL / SGM (Stability) U-Net naming used by kohya and
// LyCORIS checkpoints: input_blocks_{0-8}, middle_block_{0-2}, output_blocks_{0-8}.
// The block name follows the outer index only; the inner index addresses
// resnet/attention sub-blocks within it.
const SDXL_RE =
	/(?<block_type>input|output|middle)_blocks?_(?<block_id>\d+)(?:_(?<subblock_id>\d+))?_(?<subblock_type>transformer_blocks|in_layers|emb_layers|out_layers|skip_connection|op|conv|proj_in|proj_out)/;

// Resnet bodies (the conv side of a block) versus its attention half.
const SDXL_RESNET_MODULES = new Set([
	"in_layers",
	"emb_layers",
	"out_layers",
	"skip_connection",
]);

const SDXL_NUM_OF_INPUT_BLOCKS = 9;
// MID follows IN00-IN08, then OUT00-OUT08.
const SDXL_MID_BLOCK_IDX = SDXL_NUM_OF_INPUT_BLOCKS + 1;

// Krea 2 (musubi-tuner networks.lora_krea2) DiT block naming. Covers both
// the underscore-joined "lora_unet_"-prefixed keys and the dot-separated
// "diffusion_model."-prefixed keys musubi-tuner also saves (e.g. rank_64
// checkpoints), which use "." instead of "_" between segments.
const KREA2_BLOCK_RE =
	/(?:lora_unet_|diffusion_model\.)(?<block_type>blocks|txtfusion[._]layerwise_blocks|txtfusion[._]refiner_blocks)[._](?<block_id>\d+)[._](?<subblock_type>attn[._]\w+|mlp[._]\w+)/;

const KREA2_NUM_OF_BLOCKS = 28;
const KREA2_NUM_OF_LAYERWISE_BLOCKS = 2;

/**
 * @typedef {Object} Module
 * @property {number} idx - The index of the module
 * @property {number} blockIdx - Block index between 1 and 48
 * @property {string} name - Common name (e.g., "IN01")
 * @property {string|number} blockId - ID of the block (e.g., "0", "1")
 * @property {string|number} subBlockId - ID of the subblock
 * @property {string} type - Type of the block (e.g., "resnets", "attentions")
 * @property {string} blockType - The type of block (e.g., "up", "down", "mid")
 * @property {boolean} isConv - Whether this is a convolution key
 * @property {boolean} isAttention - Whether this is an attention key
 * @property {boolean} isSampler - Whether this is a upscaler/downscaler
 * @property {string} key - The original key
 */

/**
 * Parses a key string from a Stable Diffusion model to extract information about the block
 * @param {string} key - The key string to parse
 * @returns {Module} - Object containing information about the parsed key
 * @throws {Error} - If the key does not match any of the regex patterns
 */
function parseSDKey(key) {
	// Initialize default values
	/** @readonly */
	const result = {
		idx: -1,
		blockIdx: -1,
		name: "",
		blockId: "",
		subBlockId: "",
		type: "",
		blockType: "",
		isConv: false,
		isAttention: false,
		isSampler: false,
		key: key,
	};

	// Handle input keys
	if (key.includes("txt_in")) {
		return {
			...result,
			idx: 0,
			blockIdx: 0,
			name: "TXT_IN",
			type: "embedder",
		};
	}
	if (key.includes("vector_in")) {
		return {
			...result,
			idx: 0,
			blockIdx: 0,
			name: "VEC_IN",
			type: "in",
		};
	}
	if (key.includes("guidance_in")) {
		return {
			...result,
			idx: 0,
			blockIdx: 0,
			name: "GUI_IN",
			type: "embedder",
		};
	}
	if (key.includes("time_in")) {
		return {
			...result,
			idx: 0,
			blockIdx: 0,
			name: "TIME_IN",
			type: "embedder",
		};
	}

	if (key.includes("img_in")) {
		return {
			...result,
			idx: 0,
			blockIdx: 0,
			name: "IMG_IN",
			type: "embedder",
		};
	}

	// Wan 2.x DiT blocks (self_attn/cross_attn/ffn) and LTX 2.3 audio/video
	// blocks. Both must be matched ahead of the H3 and Krea 2 block branches
	// below: H3 also claims the "diffusion_model.blocks." prefix (ai-toolkit
	// variant) and Krea 2 claims "lora_unet_blocks_", so a Wan key would
	// otherwise be parsed with the wrong block type.
	if (
		(key.includes("_blocks_") || key.includes(".blocks.")) &&
		(key.includes("self_attn") ||
			key.includes("cross_attn") ||
			key.includes("ffn"))
	) {
		const matches = key.match(WAN_BLOCK_RE);
		if (!matches) {
			throw new Error(`Wan: Did not match on key: ${key} ${WAN_BLOCK_RE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);
		const isAttention = groups.subblock_type !== "ffn";

		return {
			...result,
			type: isAttention ? "attentions" : "mlp",
			idx: idx,
			blockId: `${idx}`,
			blockType: "blocks",
			blockIdx: idx,
			name: `TB${padTwo(idx)}`,
			isAttention: isAttention,
		};
	}

	// LTX 2.3 audio/video multimodal DiT blocks
	if (key.includes("diffusion_model.transformer_blocks.")) {
		const matches = key.match(LTX_BLOCK_RE);
		if (!matches) {
			throw new Error(`LTX: Did not match on key: ${key} ${LTX_BLOCK_RE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);

		return {
			...result,
			type: "attentions",
			idx: idx,
			blockId: `${idx}`,
			blockType: groups.subblock_type,
			name: `TB${padTwo(idx)}`,
			isAttention: true,
		};
	}

	// MiniMax H3 bare transformer_blocks / token_refiner.refiner_blocks,
	// and the ai-toolkit variant's diffusion_model.blocks / diffusion_model.token_refiner.blocks
	if (
		key.startsWith("transformer_blocks.") ||
		key.startsWith("token_refiner.refiner_blocks.") ||
		key.startsWith("diffusion_model.blocks.") ||
		key.startsWith("diffusion_model.token_refiner.blocks.")
	) {
		const matches = key.match(H3_BLOCK_RE);
		if (!matches) {
			throw new Error(`H3: Did not match on key: ${key} ${H3_BLOCK_RE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);
		const isRefiner =
			groups.block_type === "token_refiner.refiner_blocks" ||
			groups.block_type === "token_refiner.blocks";
		const namePrefix = isRefiner ? "TR" : "TB";
		const isAttention = groups.subblock_type.startsWith("attn");

		return {
			...result,
			type: isAttention ? "attentions" : "mlp",
			idx: idx,
			blockId: idx,
			blockType: groups.block_type,
			name: `${namePrefix}${padTwo(idx)}`,
			isAttention: isAttention,
		};
	}

	// Parse Flux PEFT transformer blocks
	if (key.includes("transformer.")) {
		const matches = key.match(FLUX_PEFT);
		if (!matches) {
			throw new Error(`Did not match on key: ${key} ${FLUX_PEFT}`);
		}

		const groups = matches.groups;
		const blockId = Number.parseInt(groups.block_id);
		const blockType = groups.block_type;

		const namePrefix = blockType === "single_transformer_blocks" ? "SB" : "DB";

		return {
			...result,
			type: "transformer",
			blockId: blockId,
			blockType: blockType,
			name: `${namePrefix}${padTwo(blockId)}`,
			isAttention: true,
			subtype: groups.subtype || "",
		};
	}

	// LUMINA2 Parse final layer
	if (key.includes("unet_final_layer")) {
		return {
			...result,
			type: "mlp",
			blockId: 0,
			blockType: "linear",
			name: "FLB00",
		};
	}

	// LUMINA2  Parse time embedder
	if (key.includes("unet_t_embedder")) {
		return {
			...result,
			type: "embedder",
			blockId: 0,
			blockType: "linear",
			name: "TE00",
		};
	}

	// LUMINA2 Parse cap embedder
	if (key.includes("unet_cap_embedder")) {
		return {
			...result,
			type: "embedder",
			blockId: 0,
			blockType: "linear",
			name: "CE00",
		};
	}

	// LUMINA2 Parse x embedder
	if (key.includes("unet_x_embedder")) {
		return {
			...result,
			type: "embedder",
			blockId: 0,
			blockType: "linear",
			name: "XE00",
		};
	}

	// Parse MOSS-Image DiT layers
	if (key.includes("unet_layers")) {
		const matches = key.match(MOSS_IMAGE_BLOCK_RE);

		if (matches) {
			const groups = matches.groups;
			const idx = Number.parseInt(groups.block_id);
			const isAttention = MOSS_IMAGE_ATTENTION_MODULES.has(
				groups.subblock_type,
			);

			return {
				...result,
				type: isAttention ? "attentions" : "mlp",
				idx: idx,
				blockId: idx,
				blockType: "layers",
				name: `TB${padTwo(idx)}`,
				isAttention: isAttention,
			};
		}
	}

	// Parse Lumina transformer
	if (
		key.includes("unet_layers") ||
		key.includes("unet_noise_refiner") ||
		key.includes("unet_context_refiner")
	) {
		const matches = key.match(LUMINA_TRANSFORMER);
		if (!matches) {
			throw new Error(`Did not match on key: ${key} ${LUMINA_TRANSFORMER}`);
		}

		const groups = matches.groups;
		const blockId = Number.parseInt(groups.block_id);
		const blockType = groups.block_type;
		const blockKey = `${blockType[0].toUpperCase()}B`;

		return {
			...result,
			type: "transformer",
			blockId: blockId,
			blockType: blockType,
			name: `${blockKey}${padTwo(blockId)}`,
			isAttention: true,
		};
	}

	// Parse GEMMA 2 for Lumina
	if (key.includes("te_layers")) {
		const matches = key.match(GEMMA);
		if (!matches) {
			throw new Error(`Did not match on key: ${key} ${LUMINA_TRANSFORMER}`);
		}

		const groups = matches.groups;
		const blockId = Number.parseInt(groups.block_id);

		return {
			...result,
			type: "transformer",
			blockId: blockId,
			blockType: groups.block_type,
			name: `TE${padTwo(blockId)}`,
			isAttention: true,
		};
	}

	// Krea 2: single-tensor modules (no block index)
	if (
		key.includes("lora_unet_first") ||
		key.includes("lora_unet_last_linear") ||
		key.includes("lora_unet_tmlp_") ||
		key.includes("lora_unet_tproj_") ||
		key.includes("lora_unet_txtfusion_projector") ||
		key.includes("lora_unet_txtmlp_") ||
		key.includes("diffusion_model.first") ||
		key.includes("diffusion_model.last_linear") ||
		key.includes("diffusion_model.last.linear") ||
		key.includes("diffusion_model.tmlp.") ||
		key.includes("diffusion_model.tproj.") ||
		key.includes("diffusion_model.txtfusion.projector") ||
		key.includes("diffusion_model.txtmlp.")
	) {
		return {
			...result,
			idx: 0,
			blockIdx: 0,
			name: "KREA2_IN",
			type: "embedder",
		};
	}

	// Krea 2 DiT blocks (main stream + txtfusion layerwise/refiner blocks)
	if (key.includes("lora_unet_blocks_") || key.includes("txtfusion")) {
		const matches = key.match(KREA2_BLOCK_RE);
		if (!matches) {
			throw new Error(`Krea2: Did not match on key: ${key} ${KREA2_BLOCK_RE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);

		let namePrefix = "TB";
		let blockIdxOffset = 0;
		if (groups.block_type.includes("layerwise_blocks")) {
			namePrefix = "TFL";
			blockIdxOffset = KREA2_NUM_OF_BLOCKS;
		} else if (groups.block_type.includes("refiner_blocks")) {
			namePrefix = "TFR";
			blockIdxOffset = KREA2_NUM_OF_BLOCKS + KREA2_NUM_OF_LAYERWISE_BLOCKS;
		}

		return {
			...result,
			type: groups.subblock_type.startsWith("attn") ? "attentions" : "mlp",
			idx: idx,
			blockId: `${idx}`,
			blockType: groups.block_type,
			blockIdx: blockIdxOffset + idx,
			name: `${namePrefix}${padTwo(idx)}`,
			isAttention: groups.subblock_type.startsWith("attn"),
		};
	}

	// Parse Flux double blocks
	if (key.includes("double_blocks")) {
		const matches = key.match(FLUX_DOUBLE);
		if (!matches) {
			throw new Error(`Did not match on key: ${key} ${FLUX_DOUBLE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);

		return {
			...result,
			type: "transformer",
			blockType: groups.block_type,
			idx: idx,
			blockId: `${idx}`,
			name: `DB${padTwo(idx)}`,
			isAttention: true,
		};
	}

	// Parse Flux single blocks
	if (key.includes("single_blocks")) {
		const matches = key.match(FLUX_SINGLE);
		if (!matches) {
			throw new Error(`Did not match on key: ${key} ${FLUX_SINGLE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);

		return {
			...result,
			type: "transformer",
			blockType: groups.block_type,
			idx: idx,
			blockId: `${idx}`,
			subBlockId: "0",
			name: `SB${padTwo(idx)}`,
			isAttention: true,
		};
	}

	// Parse SDXL blocks
	if (
		key.includes("input_blocks") ||
		key.includes("output_blocks") ||
		key.includes("middle_block")
	) {
		const matches = key.match(SDXL_RE);
		if (!matches) {
			throw new Error(`UNet: Did not match on key: ${key} ${SDXL_RE}`);
		}

		const groups = matches.groups;
		const outerId = Number.parseInt(groups.block_id);
		const subblockType = groups.subblock_type;

		result.blockType = groups.block_type;
		result.blockId = `${outerId}`;
		result.subBlockId = groups.subblock_id ?? "";

		if (subblockType === "transformer_blocks") {
			result.type = "attentions";
			result.isAttention = true;
		} else if (SDXL_RESNET_MODULES.has(subblockType)) {
			result.type = "resnets";
			result.isConv = true;
		} else if (subblockType === "op") {
			// Downsampler convolution: input_blocks_{3,6}_0_op
			result.type = "downsamplers";
			result.isSampler = true;
		} else if (subblockType === "conv") {
			// Upsampler convolution: output_blocks_{2,5}_*_conv
			result.type = "upsamplers";
			result.isSampler = true;
		} else {
			// proj_in / proj_out attention projections
			result.type = "attentions";
			result.isAttention = true;
		}

		result.idx = outerId;

		if (groups.block_type === "input") {
			result.blockIdx = 1 + outerId;
			result.name = `IN${padTwo(outerId)}`;
		} else if (groups.block_type === "output") {
			result.blockIdx = SDXL_MID_BLOCK_IDX + 1 + outerId;
			result.name = `OUT${padTwo(outerId)}`;
		} else {
			// middle_block_0-2 are all one chart point
			result.blockIdx = SDXL_MID_BLOCK_IDX;
			result.name = "MID00";
		}

		return result;
	}

	// Parse text encoder blocks
	if (key.includes("lora_te")) {
		const matches = key.match(TE_SDRE);
		if (!matches) {
			throw new Error(`Did not match on key: ${key}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);
		// SDXL ships two text encoders (CLIP-L and CLIP-G), so keys carry a
		// lora_te1_/lora_te2_ prefix. Keep them apart instead of collapsing
		// both into one TE00-TE31 series; SD1.x has a single lora_te_ encoder.
		const encoder = groups.encoder;
		const name = encoder ? `TE${encoder}-${padTwo(idx)}` : `TE${padTwo(idx)}`;

		return {
			...result,
			type: "encoder",
			idx: idx,
			blockId: `${idx}`,
			blockType: groups.block_type,
			name: name,
			isAttention: groups.block_type === "self_attn",
		};
	}

	// Parse mid blocks
	if (key.includes("mid_block_")) {
		const matches = key.match(MID_SDRE);
		if (!matches) {
			throw new Error(`Mid: Did not match on key: ${key}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);

		return {
			...result,
			type: groups.type,
			idx: idx,
			blockType: groups.block_type,
			blockId: `${idx}`,
			subBlockId: Number.parseInt(groups.subblock_id),
			name: `MID${padTwo(idx)}`,
			isAttention: groups.type === "attentions",
			isConv: groups.type === "resnets",
			blockIdx: NUM_OF_BLOCKS,
		};
	}

	// Parse standard SD blocks
	const matches = key.match(SDRE);
	if (!matches) {
		throw new Error(`UNet: Did not match on key: ${key} ${SDRE}`);
	}

	const groups = matches.groups;
	const blockId = Number.parseInt(groups.block_id);
	const subBlockId = Number.parseInt(groups.subblock_id);
	let idx = -1;

	// Update result with extracted values
	result.type = groups.type;
	result.blockType = groups.block_type;
	result.blockId = blockId;
	result.subBlockId = subBlockId;

	// Set idx and flags based on type
	if (groups.type === "attentions") {
		idx = 3 * blockId + subBlockId;
		result.isAttention = true;
	} else if (groups.type === "resnets") {
		idx = 3 * blockId + subBlockId;
		result.isConv = true;
	} else if (groups.type === "upsamplers" || groups.type === "downsamplers") {
		idx = 3 * blockId + 2;
		result.isSampler = true;
	}

	// Set blockIdx and name based on block_type
	if (groups.block_type === "down") {
		result.blockIdx = 1 + idx;
		result.name = `IN${padTwo(idx)}`;
	} else if (groups.block_type === "up") {
		result.blockIdx = NUM_OF_BLOCKS + 1 + idx;
		result.name = `OUT${padTwo(idx)}`;
	} else if (groups.block_type === "mid") {
		result.blockIdx = NUM_OF_BLOCKS;
	}

	result.idx = idx;
	return result;
}

/**
 * Helper function to pad a number with leading zeros
 * @param {number|string} num - Number to pad
 * @returns {string} - Padded number as string
 */
function padTwo(num) {
	return String(num).padStart(2, "0");
}

export { parseSDKey, SDRE, MID_SDRE, TE_SDRE, NUM_OF_BLOCKS };
