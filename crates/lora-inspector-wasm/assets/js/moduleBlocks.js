// Handle parsing of the keys

const SDRE =
  /.*(?<block_type>up|down|mid)_blocks?_(?<block_id>\d+).*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;

const MID_SDRE =
  /.*(?<block_type>up|down|mid)_block_.*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<block_id>\d+)_.*(?<subblock_id>\d+)?.*/;
const TE_SDRE =
  /te(?<encoder>\d+)?_.*_(?<block_id>\d+).*(?<block_type>self_attn|mlp)/;
const NUM_OF_BLOCKS = 12;

const FLUX_DOUBLE =
  /lora_unet_(?<block_type>double_blocks)_(?<block_id>\d+)_(?<modality>txt|img)_(?<subblock_type>attn_proj|attn_qkv|mlp_0|mlp_2|mod_lin)/;

const FLUX_SINGLE =
  /lora_unet_(?<block_type>single_blocks)_(?<block_id>\d+)_(?<subblock_type>linear1|linear2|modulation_lin)/;

const FLUX_PEFT =
  /transformer\.(?<block_type>single_transformer_blocks|transformer_blocks)\.(?<block_id>\d+)\.(?<type>\w+)\.(?<subtype>\w+)/;

const LUMINA_TRANSFORMER =
  /.*unet.*(?<block_type>layers|noise_refiner|context_refiner).*_(?<block_id>\d+)_(?<type>adaLN_modulation|feed_forward|attention_out|attention_qkv)(?<subblock_type>_w\d+)?/;

const GEMMA =
  /.*te.*(?<block_type>layers).*_(?<block_id>\d+)_(?<type>self_attn_q_proj|self_attn_k_proj|self_attn_v_proj|self_attn_o_proj|mlp_down_proj|mlp_up_proj|mlp_gate_proj)/;

const SDXL_RE =
  /.*(?<block_type>input|output|middle)_blocks?_(?<block_id>\d+).*_(\d+_)?((?<type>transformer_blocks)_(?<subblock_id>\d+)_(?<subtype>attn\d+|ff)?_(?<subblock_type>to_k|to_out_0|to_q|to_v|net_0_proj|net_2).*|proj_in|proj_out)/;

const SDXL_NUM_OF_BLOCKS = 26;

function parseSDKey(key) {
  let blockIdx = -1;
  let idx;

  let isConv = false;
  let isAttention = false;
  let isSampler = false;
  // const isProjection = false;
  // const isFeedForward = false;

  let type;
  let blockType;
  let blockId;
  let subBlockId;
  let name;

  console.log("parsing keys");

  if (key.includes("single_transformer_blocks")) {
    const matches = key.match(FLUX_PEFT);
    if (!matches) {
      throw new Error(`Did not match on key: ${key} ${FLUX_PEFT}`);
    }
    const groups = matches.groups;
    type = "transformer";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    const blockKey = "SB";

    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = true;
  } else if (key.includes("transformer_blocks")) {
    const matches = key.match(FLUX_PEFT);
    if (!matches) {
      throw new Error(`Did not match on key: ${key} ${FLUX_PEFT}`);
    }
    const groups = matches.groups;
    type = "transformer";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    const blockKey = "DB";

    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = true;
  } else if (key.includes("unet_final_layer")) {
    type = "mlp";
    blockId = 0;
    blockType = "linear";
    const blockKey = "FLB";
    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = false;
  } else if (key.includes("unet_t_embedder")) {
    type = "embedder";
    blockId = 0;
    blockType = "linear";
    const blockKey = "TE";
    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = false;
  } else if (key.includes("unet_cap_embedder")) {
    type = "embedder";
    blockId = 0;
    blockType = "linear";
    const blockKey = "CE";
    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = false;
  } else if (key.includes("unet_x_embedder")) {
    type = "embedder";
    blockId = 0;
    blockType = "linear";
    const blockKey = "XE";
    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = false;
  } else if (
    key.includes("unet_layers") ||
    key.includes("unet_noise_refiner") ||
    key.includes("unet_context_refiner")
  ) {
    // LUMINA
    const matches = key.match(LUMINA_TRANSFORMER);
    if (!matches) {
      throw new Error(`Did not match on key: ${key} ${LUMINA_TRANSFORMER}`);
    }
    const groups = matches.groups;
    type = "transformer";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    const blockKey = `${blockType[0].toUpperCase()}B`;

    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = true;
  } else if (key.includes("te_layers")) {
    // GEMMA 2 for Lumina
    const matches = key.match(GEMMA);
    if (!matches) {
      throw new Error(`Did not match on key: ${key} ${LUMINA_TRANSFORMER}`);
    }
    const groups = matches.groups;
    type = "transformer";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    const blockKey = "TE";

    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = true;
    // Flux
  } else if (key.includes("double_blocks")) {
    const matches = key.match(FLUX_DOUBLE);
    if (!matches) {
      throw new Error(`Did not match on key: ${key} ${FLUX}`);
    }
    const groups = matches.groups;
    type = "transformer";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    const blockKey = `${blockType[0].toUpperCase()}B`;

    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = true;
  } else if (key.includes("single_blocks")) {
    const matches = key.match(FLUX_SINGLE);
    if (!matches) {
      throw new Error(`Did not match on key: ${key} ${FLUX}`);
    }
    const groups = matches.groups;
    type = "transformer";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    const blockKey = `${blockType[0].toUpperCase()}B`;

    name = `${blockKey}${padTwo(blockId)}`;

    isAttention = true;
  } else if (
    key.includes("input_blocks") ||
    key.includes("output_blocks") ||
    key.includes("middle_block")
  ) {
    const matches = key.match(SDXL_RE);
    if (!matches) {
      throw new Error(`UNet: Did not match on key: ${key} ${SDXL_RE}`);
    }
    const groups = matches.groups;

    type = groups.type;
    blockType = groups.block_type;
    blockId = Number.parseInt(groups.block_id);
    subBlockId = Number.parseInt(groups.subblock_id);

    if (
      groups.subtype === "attn1" ||
      groups.subtype === "attn2" ||
      groups.subtype === "ff"
    ) {
      idx = 3 * blockId + subBlockId;
      isAttention = true;
    }
    // } else if (groups.type === "resnets") {
    //   idx = 3 * blockId + subBlockId;
    //   isConv = true;
    // } else if (groups.type === "upsamplers" || groups.type === "downsamplers") {
    //   idx = 3 * blockId + 2;
    //   isSampler = true;
    // }

    if (groups.block_type === "input") {
      blockIdx = 1 + idx;
      name = `IN${padTwo(idx)}`;
    } else if (groups.block_type === "output") {
      blockIdx = SDXL_NUM_OF_BLOCKS + 1 + idx;
      name = `OUT${padTwo(idx)}`;
    } else if (groups.block_type === "middle") {
      blockIdx = SDXL_NUM_OF_BLOCKS;
    }
  } else if (key.includes("lora_te")) {
    const matches = key.match(TE_SDRE);
    if (!matches) {
      throw new Error(`Did not match on key: ${key}`);
    }
    const groups = matches.groups;
    type = "encoder";
    blockId = Number.parseInt(groups.block_id);
    blockType = groups.block_type;

    name = `TE${padTwo(blockId)}`;

    if (blockType === "self_attn") {
      isAttention = true;
    }
  } else if (key.includes("mid_block_")) {
    const matches = key.match(MID_SDRE);
    name = "MID";

    if (!matches) {
      throw new Error(`Mid: Did not match on key: ${key}`);
    }
    const groups = matches.groups;

    type = groups.type;
    blockType = groups.block_type;
    blockId = Number.parseInt(groups.block_id);
    subBlockId = Number.parseInt(groups.subblock_id);

    name = `MID${padTwo(blockId)}`;

    if (groups.type === "attentions") {
      isAttention = true;
    } else if (groups.type === "resnets") {
      isConv = true;
    }

    blockIdx = NUM_OF_BLOCKS;
  } else {
    const matches = key.match(SDRE);
    if (!matches) {
      throw new Error(`UNet: Did not match on key: ${key} ${SDRE}`);
    }
    const groups = matches.groups;

    type = groups.type;
    blockType = groups.block_type;
    blockId = Number.parseInt(groups.block_id);
    subBlockId = Number.parseInt(groups.subblock_id);

    if (groups.type === "attentions") {
      idx = 3 * blockId + subBlockId;
      isAttention = true;
    } else if (groups.type === "resnets") {
      idx = 3 * blockId + subBlockId;
      isConv = true;
    } else if (groups.type === "upsamplers" || groups.type === "downsamplers") {
      idx = 3 * blockId + 2;
      isSampler = true;
    }

    if (groups.block_type === "down") {
      blockIdx = 1 + idx;
      name = `IN${padTwo(idx)}`;
    } else if (groups.block_type === "up") {
      blockIdx = NUM_OF_BLOCKS + 1 + idx;
      name = `OUT${padTwo(idx)}`;
    } else if (groups.block_type === "mid") {
      blockIdx = NUM_OF_BLOCKS;
    }
  }

  const results = {
    // Used in commmon format IN01
    idx,
    // Block index between 1 and 24
    blockIdx,
    // Common name IN01
    name,
    // name of the block up, down, mid
    // id of the block (up_0, down_1)
    blockId,
    // id of the subblock (resnet, attentions)
    subBlockId,
    // resnets, attentions, upscalers, downscalers
    type,
    //
    blockType,
    // is a convolution key
    isConv,
    // is an attention key
    isAttention,
    // is a upscaler/downscaler
    isSampler,
    key,
  };

  console.log(key, results);

  return results;
}

function padTwo(number, padWith = "0") {
  if (number < 10) {
    return `${padWith}${number}`;
  }

  return `${number}`;
}

export { parseSDKey, SDRE, MID_SDRE, TE_SDRE, NUM_OF_BLOCKS };
