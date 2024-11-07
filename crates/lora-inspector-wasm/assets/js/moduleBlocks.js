// Handle parsing of the keys

const SDRE =
  /.*(?<block_type>up|down|mid)_blocks?_(?<block_id>\d+).*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;

const MID_SDRE =
  /.*(?<block_type>up|down|mid)_block_.*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<block_id>\d+)_.*(?<subblock_id>\d+)?.*/;
const TE_SDRE =
  /te(?<encoder>\d+)?_.*_(?<block_id>\d+).*(?<block_type>self_attn|mlp)/;
const NUM_OF_BLOCKS = 12;

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

  // Handle the text encoder
  if (key.includes("lora_te")) {
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
  // } else if (key.includes("te1") || key.includes("te2")) {
  //   const matches = key.match(TE_SDRE);
  //
  //   if (!matches) {
  //     throw new Error(`Did not match on key: ${key}`);
  //   }
  //
  //   const groups = matches.groups;
  //   type = "encoder";
  //   blockId = Number.parseInt(groups.block_id);
  //   blockType = groups.block_type;
  //
  //   name = `TE${padTwo(blockId)}`;
  //
  //   if (blockType === "self_attn") {
  //     isAttention = true;
  //   }
    // Handling the UNet values
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

  return results;
}

function padTwo(number, padWith = "0") {
  if (number < 10) {
    return `${padWith}${number}`;
  }

  return `${number}`;
}

export { parseSDKey, SDRE, MID_SDRE, TE_SDRE, NUM_OF_BLOCKS };
