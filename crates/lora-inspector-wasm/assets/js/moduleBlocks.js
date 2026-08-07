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

// MiniMax H3 / Qwen-Image (diffusers-native PEFT, no "transformer." prefix):
// bare transformer_blocks.N.* plus a token_refiner.refiner_blocks.N.* block
// group. Qwen-Image adds txt_mlp/img_mlp feed-forward subtypes alongside H3's
// generic "ff".
const H3_BLOCK_RE =
	/(?<block_type>transformer_blocks|token_refiner\.refiner_blocks)\.(?<block_id>\d+)\.(?<subblock_type>attn|ff|txt_mlp|img_mlp)/;

// LTX 2.3 audio/video multimodal DiT (diffusers-native, "diffusion_model."
// prefix): diffusion_model.transformer_blocks.N.<attn1|attn2|audio_attn1|
// audio_attn2|audio_to_video_attn|video_to_audio_attn>.*
const LTX_BLOCK_RE =
	/diffusion_model\.transformer_blocks\.(?<block_id>\d+)\.(?<subblock_type>audio_attn\d+|audio_to_video_attn|video_to_audio_attn|attn\d+)/;

// Wan 2.x video DiT block naming: kohya-style lora_unet_blocks_N_(self_attn|
// cross_attn)_* / lora_unet_blocks_N_ffn_N, and diffusers-style
// diffusion_model.blocks.N.<self_attn|cross_attn|ffn>.*
const WAN_BLOCK_RE =
	/(lora_unet|diffusion_model)[._]blocks[._](?<block_id>\d+)[._](?<subblock_type>self_attn|cross_attn|ffn)/;

const LUMINA_TRANSFORMER =
	/.*unet.*(?<block_type>layers|noise_refiner|context_refiner).*_(?<block_id>\d+)_(?<type>adaLN_modulation|feed_forward|attention_out|attention_qkv)(?<subblock_type>_w\d+)?/;

const GEMMA =
	/.*te.*(?<block_type>layers).*_(?<block_id>\d+)_(?<type>self_attn_q_proj|self_attn_k_proj|self_attn_v_proj|self_attn_o_proj|mlp_down_proj|mlp_up_proj|mlp_gate_proj)/;

const SDXL_RE =
	/.*(?<block_type>input|output|middle)_blocks?_(?<block_id>\d+).*_(\d+_)?((?<type>transformer_blocks)_(?<subblock_id>\d+)_(?<subtype>attn\d+|ff)?_(?<subblock_type>to_k|to_out_0|to_q|to_v|net_0_proj|net_2).*|proj_in|proj_out)/;

const SDXL_NUM_OF_BLOCKS = 26;

// LyCORIS SDXL/SD1.x (old-style webui block numbering) ResBlock/Downsample/
// Upsample modules within input_blocks/output_blocks/middle_block: in_layers/
// out_layers/emb_layers (ResBlock conv/norm/embedding projections),
// skip_connection (ResBlock shortcut conv), and op/conv (downsampler/
// upsampler convolutions). LyCORIS trains these alongside attention, unlike
// kohya-ss LoRA which is attention-only, so they don't match SDXL_RE.
const SDXL_RESNET_RE =
	/(?<block_type>input|output|middle)_blocks?_(?<block_id>\d+)(_(?<subblock_id>\d+))?_(?<subtype>in_layers|out_layers|emb_layers|skip_connection|op|conv)/;

// Krea 2 DiT block naming: kohya-style (musubi-tuner networks.lora_krea2)
// lora_unet_blocks_N_attn_wq / lora_unet_txtfusion_layerwise_blocks_N_mlp_gate,
// and diffusers-style diffusion_model.blocks.N.attn.wq / diffusion_model.
// txtfusion.layerwise_blocks.N.mlp.down. The [._] separator classes let one
// regex cover both naming conventions.
const KREA2_BLOCK_RE =
	/(lora_unet|diffusion_model)[._](?<block_type>blocks|txtfusion[._]layerwise_blocks|txtfusion[._]refiner_blocks)[._](?<block_id>\d+)[._](?<subblock_type>attn\w*|mlp\w*)/;

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

	// MiniMax H3 bare transformer_blocks / token_refiner.refiner_blocks
	if (
		key.startsWith("transformer_blocks.") ||
		key.startsWith("token_refiner.refiner_blocks.")
	) {
		const matches = key.match(H3_BLOCK_RE);
		if (!matches) {
			throw new Error(`H3: Did not match on key: ${key} ${H3_BLOCK_RE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);
		const namePrefix =
			groups.block_type === "token_refiner.refiner_blocks" ? "TR" : "TB";

		return {
			...result,
			type: groups.subblock_type === "attn" ? "attentions" : "mlp",
			idx: idx,
			blockId: idx,
			blockType: groups.block_type,
			name: `${namePrefix}${padTwo(idx)}`,
			isAttention: groups.subblock_type === "attn",
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

	// Wan 2.x DiT blocks (self_attn/cross_attn/ffn). Checked ahead of the Krea 2
	// "lora_unet_blocks_"/"diffusion_model.blocks." guards below since both
	// architectures can share the same "blocks_N_" prefix.
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

	// Krea 2: single-tensor modules (no block index)
	if (
		key.includes("lora_unet_first") ||
		key.includes("lora_unet_last_linear") ||
		key.includes("lora_unet_tmlp_") ||
		key.includes("lora_unet_tproj_") ||
		key.includes("lora_unet_txtfusion_projector") ||
		key.includes("lora_unet_txtmlp_")
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
	if (
		key.includes("lora_unet_blocks_") ||
		key.includes("txtfusion_") ||
		key.includes("diffusion_model.blocks.") ||
		key.includes("diffusion_model.txtfusion.")
	) {
		const matches = key.match(KREA2_BLOCK_RE);
		if (!matches) {
			throw new Error(`Krea2: Did not match on key: ${key} ${KREA2_BLOCK_RE}`);
		}

		const groups = matches.groups;
		const idx = Number.parseInt(groups.block_id);
		// Normalize block_type since the diffusers-style match captures a
		// literal "." separator (e.g. "txtfusion.layerwise_blocks") while the
		// kohya-style match captures "_" (e.g. "txtfusion_layerwise_blocks").
		const blockType = groups.block_type.replace(/\./g, "_");

		let namePrefix = "TB";
		let blockIdxOffset = 0;
		if (blockType === "txtfusion_layerwise_blocks") {
			namePrefix = "TFL";
			blockIdxOffset = KREA2_NUM_OF_BLOCKS;
		} else if (blockType === "txtfusion_refiner_blocks") {
			namePrefix = "TFR";
			blockIdxOffset = KREA2_NUM_OF_BLOCKS + KREA2_NUM_OF_LAYERWISE_BLOCKS;
		}

		return {
			...result,
			type: groups.subblock_type.startsWith("attn") ? "attentions" : "mlp",
			idx: idx,
			blockId: `${idx}`,
			blockType: blockType,
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
		if (matches) {
			const groups = matches.groups;
			let idx = -1;
			const blockId = groups.block_id;
			const subBlockId = groups.subblock_id;

			// Update result with extracted values
			result.type = groups.type;
			result.blockType = groups.block_type;
			result.blockId = blockId;
			result.subBlockId = subBlockId;

			// Set idx and isAttention based on subtype
			if (
				groups.subtype === "attn1" ||
				groups.subtype === "attn2" ||
				groups.subtype === "ff"
			) {
				idx = 3 * Number.parseInt(blockId) + Number.parseInt(subBlockId);
				result.isAttention = true;
			}

			// Set blockIdx and name based on block_type
			if (groups.block_type === "input") {
				result.blockIdx = 1 + idx;
				result.name = `IN${padTwo(idx)}`;
			} else if (groups.block_type === "output") {
				result.blockIdx = SDXL_NUM_OF_BLOCKS + 1 + idx;
				result.name = `OUT${padTwo(idx)}`;
			} else if (groups.block_type === "middle") {
				result.blockIdx = SDXL_NUM_OF_BLOCKS;
			}

			result.idx = idx;
			return result;
		}

		// Fall back to ResBlock/Downsample/Upsample naming (LyCORIS trains these
		// too, unlike attention-only kohya-ss LoRA)
		const resnetMatches = key.match(SDXL_RESNET_RE);
		if (!resnetMatches) {
			throw new Error(`UNet: Did not match on key: ${key} ${SDXL_RE}`);
		}

		const groups = resnetMatches.groups;
		const blockId = Number.parseInt(groups.block_id);
		const subBlockId = groups.subblock_id
			? Number.parseInt(groups.subblock_id)
			: 0;
		const isSampler = groups.subtype === "op" || groups.subtype === "conv";
		const idx = 3 * blockId + subBlockId;

		result.type = isSampler
			? groups.block_type === "input"
				? "downsamplers"
				: "upsamplers"
			: "resnets";
		result.blockType = groups.block_type;
		result.blockId = `${blockId}`;
		result.subBlockId = `${subBlockId}`;
		result.isConv = !isSampler;
		result.isSampler = isSampler;
		result.idx = idx;

		if (groups.block_type === "input") {
			result.blockIdx = 1 + idx;
			result.name = `IN${padTwo(idx)}`;
		} else if (groups.block_type === "output") {
			result.blockIdx = SDXL_NUM_OF_BLOCKS + 1 + idx;
			result.name = `OUT${padTwo(idx)}`;
		} else if (groups.block_type === "middle") {
			result.blockIdx = SDXL_NUM_OF_BLOCKS;
			result.name = `MID${padTwo(blockId)}`;
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

		return {
			...result,
			type: "encoder",
			idx: idx,
			blockId: `${idx}`,
			blockType: groups.block_type,
			name: `TE${padTwo(idx)}`,
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
