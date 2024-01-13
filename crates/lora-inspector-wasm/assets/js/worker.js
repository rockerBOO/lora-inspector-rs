// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts("/pkg/lora_inspector_wasm.js");

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { LoraWorker } = wasm_bindgen;

// let clients = [];
// let wasms = [];

// loraWorkers are specific objects that manage the LoRA file, buffer, and inspection.
let loraWorkers = new Map();

function addWorker(name, worker) {
  // console.log("Adding worker ", name);
  loraWorkers.set(name, worker);

  return loraWorkers.get(name);
}

function removeWorker(workerName) {
  // console.log("Removing worker ", workerName);
  loraWorkers.remove(workerName);
}

function getWorker(workerName) {
  const loraWorker = loraWorkers.get(workerName);

  if (!loraWorker) {
    throw new Error(`Could not find worker ${workerName}`);
  }

  return loraWorker;
}

function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  wasm_bindgen("/pkg/lora_inspector_wasm_bg.wasm").then(() => {
    onerror = (event) => {
      console.log("There is an error inside your worker!", event);
    };

    onmessage = async (e) => {
      if (e.data.messageType === "file_upload") {
        // unload old workers for now...
        // console.log("Clearing workers");
        // loraWorkers.clear();
        fileUploadHandler(e);
      } else if (e.data.messageType === "is_available") {
        if (e.data.reply) {
          self.postMessage({
            messageType: "is_available",
          });
        }
      } else if (e.data.messageType === "network_module") {
        getNetworkModule(e).then((networkModule) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "network_module",
              networkModule,
            });
          }
        });
      } else if (e.data.messageType === "network_args") {
        getNetworkArgs(e).then((networkArgs) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "network_args",
              networkArgs,
            });
          }
        });
      } else if (e.data.messageType === "network_type") {
        getNetworkType(e).then((networkType) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "network_type",
              networkType,
            });
          }
        });
      } else if (e.data.messageType === "weight_keys") {
        getWeightKeys(e);
      } else if (e.data.messageType === "keys") {
        getKeys(e).then((keys) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "keys",
              keys,
            });
          }
        });
      } else if (e.data.messageType === "text_encoder_keys") {
        getTextEncoderKeys(e).then((textEncoderKeys) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "text_encoder_keys",
              textEncoderKeys,
            });
          }
        });
      } else if (e.data.messageType === "unet_keys") {
        getUnetKeys(e).then((unetKeys) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "unet_keys",
              unetKeys,
            });
          }
        });
      } else if (e.data.messageType === "base_names") {
        getBaseNames(e).then((baseNames) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "base_names",
              baseNames,
            });
          }
        });
      } else if (e.data.messageType === "scale_weights") {
        scaleWeights(e).then((baseNames) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "scale_weights",
            });
          }
        });
      } else if (e.data.messageType === "scale_weights_with_progress") {
        iterScaleWeights(e).then((baseNames) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "scale_weights_with_progress",
            });
          }
        });
      } else if (e.data.messageType === "scale_weight") {
        scaleWeight(e).then((baseNames) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "scale_weight",
            });
          }
        });
      } else if (e.data.messageType === "l2_norm") {
        // We must lock if we are getting scaled weights

        getL2Norms(e).then((norms) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "l2_norm",
              norms,
            });
          }
        });
      } else if (e.data.messageType === "norms") {
        // We must lock if we are getting scaled weights
        getNorms(e).then((norms) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "norms",
              norms,
              baseName: e.data.baseName,
            });
          }
        });
      } else if (e.data.messageType === "alpha_keys") {
        getAlphaKeys(e);
      } else if (e.data.messageType === "dims") {
        getDims(e).then((dims) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "dims",
              dims,
            });
          }
        });
      } else if (e.data.messageType === "precision") {
        getPrecision(e).then((precision) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "precision",
              precision,
            });
          }
        });
      } else if (e.data.messageType === "alphas") {
        getAlphas(e).then((alphas) => {
          if (e.data.reply) {
            self.postMessage({
              messageType: "alphas",
              alphas,
            });
          }
        });
      }
    };
  });
}

init_wasm_in_worker();

// FUNCTIONS
// ============================

async function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      const buffer = new Uint8Array(e.target.result);
      resolve(buffer);
    };
    reader.readAsArrayBuffer(file);
  });
}

async function fileUploadHandler(e) {
  console.time("file_upload");
  const file = e.data.file;
  const buffer = await readFile(file);

  try {
    const loraWorker = addWorker(file.name, new LoraWorker(buffer, file.name));

    console.timeEnd("file_upload");
    self.postMessage({
      messageType: "metadata",
      filename: file.name,
      metadata: loraWorker.metadata(),
    });
  } catch (err) {
    console.error("Could not upload the LoRA", err);
    self.postMessage({
      messageType: "metadata_error",
      message: "could not parse the LoRA",
      errorMessage: err,
      errorCode: 1,
    });
  }
}

async function getNetworkModule(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.network_module();
}

async function getWeightKeys(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.weight_keys();
}

async function getKeys(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.keys();
}

async function scaleWeights(e) {
  console.log("scaling weights...");
  console.time("scale_weights");
  // console.log(performance.memory);
  await navigator.locks.request(`scale-weights`, async (lock) => {
    const name = e.data.name;

    const loraWorker = loraWorkers.get(name);
    loraWorker.scale_weights();
    console.timeEnd("scale_weights");
  });
}

async function iterScaleWeights(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  await navigator.locks.request(`scale-weights`, async (lock) => {
    const baseNames = loraWorker.base_names();
    const totalCount = baseNames.length;

    let currentCount = 0;

    console.time("scale_weights");
    await Promise.allSettled(
      baseNames.map((baseName) => {
        currentCount += 1;

        try {
          loraWorker.scale_weight(baseName);

          self.postMessage({
            messageType: "scale_weight_progress",
            currentCount,
            totalCount,
            baseName: baseName,
          });
        } catch (e) {
          console.error(e);
          self.postMessage({
            messageType: "scale_weight_progress",
            currentCount,
            totalCount,
            baseName: baseName,
          });
        }
      }),
    ).then(() => {
      console.log("Finished scaled weight progress");
      self.postMessage({
        messageType: "scale_weight_progress_finished",
      });

      console.time("scale_weights");
    });
  });
}

async function scaleWeight(e) {
  console.log("scaling weight...");
  console.time("scale_weight");
  // console.log(performance.memory);
  await navigator.locks.request(`scale-weights`, async (lock) => {
    const name = e.data.name;
    const baseName = e.data.baseName;

    const loraWorker = loraWorkers.get(name);
    loraWorker.scale_weight(baseName);

    console.timeEnd("scale_weight");
  });
}

async function getTextEncoderKeys(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.text_encoder_keys();
}

async function getUnetKeys(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.unet_keys();
}

async function getBaseNames(e) {
  const loraWorker = getWorker(e.data.name);

  const baseNames = loraWorker.base_names();
  // baseNames.forEach(baseName => loraWorker.parse_key(baseName));

  return baseNames;
}

async function getNorms(e) {
  // console.log("getting norms", e.data);
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);
  const baseName = e.data.baseName;

  console.log("Getting norm for ", baseName);

  const scaled = loraWorker.norms(baseName, [
    "l1_norm",
    "l2_norm",
    "matrix_norm",
    "max",
    "min",
    // "std_dev",
    // "median",
  ]);

  return scaled;
}

async function getL2Norms(e) {
  const loraWorker = getWorker(e.data.name);

  // await navigator.locks.request(`scaled-weights`, async (lock) => {
  //   const name = e.data.name;
  //
  //   const loraWorker = loraWorkers.get(name);
  //   loraWorker.scale_weights();
  // });

  console.time("Calculating norms");
  console.log("Calculating l2 norms...");

  const baseNames = loraWorker.base_names();
  const totalCount = baseNames.length;

  let currentCount = 0;

  let l2Norms = baseNames
    .map((base_name) => {
      currentCount += 1;

      self.postMessage({
        messageType: "l2_norms_progress",
        currentCount,
        totalCount,
        baseName: base_name,
      });

      try {
        return [base_name, loraWorker.l2_norm(base_name)];
      } catch (e) {
        console.error(e);
        return [base_name, undefined];
      }
    })
    .reduce(
      (acc, [base_name, norm]) => {
        if (norm === undefined) {
          return acc;
        }

        // loraWorker.parse_key(key);
        const parts = parseSDKey(base_name);

        const blockName = parts.name;

        acc["block"].set(blockName, (acc["block"].get(blockName) ?? 0) + norm);
        acc["block_count"].set(
          blockName,
          (acc["block_count"].get(blockName) ?? 0) + 1,
        );
        acc["block_mean"].set(
          blockName,
          acc["block"].get(blockName) / acc["block_count"].get(blockName),
        );

        acc["metadata"].set(blockName, parts);

        return acc;
      },
      {
        block: new Map(),
        block_count: new Map(),
        block_mean: new Map(),
        metadata: new Map(),
      },
    );

  self.postMessage({
    messageType: "l2_norms_progress_finished",
  });

  // console.log(
  //   "weight_norms block",
  //   Array.from(l2Norms["block"]).sort(([k, _], [k2, _v]) => {
  //     return k > k2;
  //   }),
  // );
  // console.log(
  //   "weight_norms count",
  //   Array.from(l2Norms["block_count"]).sort(([k, _], [k2, _v]) => {
  //     return k > k2;
  //   }),
  // );
  const norms = Array.from(l2Norms["block_mean"]).sort(([k, _], [k2, _v]) => {
    return k > k2;
  });

  // console.log("weight_norms mean", norms);
  console.timeEnd("Calculating norms");

  // Split between TE and UNet
  const splitNorms = norms.reduce(
    (acc, [k, v]) => {
      if (!k) {
        debugger;
      }
      if (k.includes("TE")) {
        acc.te.set(k, { mean: v, metadata: l2Norms["metadata"].get(k) });
      } else {
        acc.unet.set(k, { mean: v, metadata: l2Norms["metadata"].get(k) });
      }
      return acc;
    },
    { te: new Map(), unet: new Map() },
  );

  return splitNorms;
}

async function getAlphaKeys(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  return loraWorker.alpha_keys();
}

async function getAlphas(e) {
  const loraWorker = getWorker(e.data.name);

  return Array.from(loraWorker.alphas()).sort((a, b) => a > b);
}

async function getDims(e) {
  const loraWorker = getWorker(e.data.name);

  return Array.from(loraWorker.dims()).sort((a, b) => a > b);
}

async function getPrecision(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.precision();
}

async function getNetworkArgs(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.network_args();
}

async function getNetworkType(e) {
  const loraWorker = getWorker(e.data.name);

  return loraWorker.network_type();
}

function sendClientMessage(message) {
  clients.forEach((client) => {
    client.postMessage(message);
  });
}

function sendWASMMessage(message) {
  wasms.forEach((wasm) => {
    wasm.postMessage(message);
  });
}

// Handle parsing of the keys

const SDRE =
  /.*(?<block_type>up|down|mid)_blocks?_.*(?<block_id>\d+).*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;

const MID_SDRE =
  /.*(?<block_type>up|down|mid)_block_.*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<block_id>\d+)_.*(?<subblock_id>\d+)?.*/;
const TE_SDRE = /(?<block_id>\d+).*(?<block_type>self_attn|mlp)/;
const NUM_OF_BLOCKS = 12;

function parseSDKey(key) {
  let blockIdx = -1;
  let idx;

  let isConv = false;
  let isAttention = false;
  let isSampler = false;
  let isProjection = false;
  let isFeedForward = false;

  let type;
  let blockType;
  let blockId;
  let subBlockId;
  let name;

  // Handle the text encoder
  if (key.includes("te_text_model")) {
    const matches = key.match(TE_SDRE);
    if (matches) {
      const groups = matches.groups;
      type = "encoder";
      blockId = parseInt(groups["block_id"]);
      blockType = groups["block_type"];

      name = `TE${padTwo(blockId)}`;

      if (blockType === "self_attn") {
        isAttention = true;
      }
    }
    // Handling the UNet values
  } else {
    const matches = key.match(SDRE);
    if (matches) {
      const groups = matches.groups;

      type = groups["type"];
      blockType = groups["block_type"];
      blockId = parseInt(groups["block_id"]);
      subBlockId = parseInt(groups["subblock_id"]);

      // console.log(groups["block_id"]);

      if (groups["type"] === "attentions") {
        idx = 3 * blockId + subBlockId;
        isAttention = true;
      } else if (groups["type"] === "resnets") {
        idx = 3 * blockId + subBlockId;
        isConv = true;
      } else if (
        groups["type"] === "upsamplers" ||
        groups["type"] === "downsamplers"
      ) {
        idx = 3 * blockId + 2;
        isSampler = true;
      }

      // console.log("block_type", groups["block_type"]);

      if (groups["block_type"] === "down") {
        blockIdx = 1 + idx;
        name = `IN${padTwo(idx)}`;
      } else if (groups["block_type"] === "up") {
        blockIdx = NUM_OF_BLOCKS + 1 + idx;
        name = `OUT${padTwo(idx)}`;
      } else if (groups["block_type"] === "mid") {
        blockIdx = NUM_OF_BLOCKS;
      }
      // Handle the mid block
    } else if (key.includes("mid_block_")) {
      const midMatch = key.match(MID_SDRE);
      name = `MID`;

      if (midMatch) {
        const groups = midMatch.groups;

        type = groups["type"];
        blockType = groups["block_type"];
        blockId = parseInt(groups["block_id"]);
        subBlockId = parseInt(groups["subblock_id"]);

        name = `MID${padTwo(blockId)}`;

        if (groups.type == "attentions") {
          isAttention = true;
        } else if (groups.type === "resnets") {
          isConv = true;
        }
      }

      blockIdx = NUM_OF_BLOCKS;
    }
  }

  return {
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
}

function padTwo(number, padWith = "0") {
  if (number < 10) {
    return `${padWith}${number}`;
  }

  return `${number}`;
}
