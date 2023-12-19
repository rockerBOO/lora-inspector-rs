// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts("/pkg/lora_inspector_rs.js");

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { LoraWorker } = wasm_bindgen;

// let clients = [];
// let wasms = [];

// loraWorkers are specific objects that manage the LoRA file, buffer, and inspection.
let loraWorkers = new Map();

function addWorker(name, worker) {
  loraWorkers.insert(name, worker);
}

function removeWorker(workerName) {
  loraWorkers.remove(workerName);
}

function getWorker(workerName) {
  return loraWorkers.get(workerName);
}

function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  wasm_bindgen("/pkg/lora_inspector_rs_bg.wasm").then(() => {
    onerror = (event) => {
      console.log("There is an error inside your worker!", event);
    };

    onmessage = async (e) => {
      if (e.data.messageType === "file_upload") {
        // unload old workers for now...
        loraWorkers.clear();
        fileUploadHandler(e);
      } else if (e.data.messageType === "network_module") {
        getNetworkModule(e);
      } else if (e.data.messageType === "network_args") {
        getNetworkArgs(e);
      } else if (e.data.messageType === "network_type") {
        getNetworkType(e);
      } else if (e.data.messageType === "weight_keys") {
        getWeightKeys(e);
      } else if (e.data.messageType === "keys") {
        getKeys(e);
      } else if (e.data.messageType === "base_names") {
        getBaseNames(e);
      } else if (e.data.messageType === "weight_norms") {
        getWeightNorms(e);
      } else if (e.data.messageType === "alpha_keys") {
        getAlphaKeys(e);
      } else if (e.data.messageType === "dims") {
        getDims(e);
      } else if (e.data.messageType === "alphas") {
        getAlphas(e);
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
  console.log("Reading file...");
  const file = e.data.file;
  const buffer = await readFile(file);

  console.log("Loading file...");
  try {
    loraWorkers.set(file.name, new LoraWorker(buffer, file.name));
    const loraWorker = loraWorkers.get(file.name);

    console.log("Getting metadata...");

    console.timeEnd("file_upload");
    self.postMessage({
      messageType: "metadata",
      metadata: loraWorker.metadata(),
    });
  } catch (err) {
    self.postMessage({
      messageType: "metadata_error",
      message: "could not parse the LoRA",
      errorMessage: err,
      errorCode: 1,
    });
  }
}

async function getNetworkModule(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.network_module());
}

async function getWeightKeys(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.weight_keys());
}

async function getKeys(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log("keys", loraWorker.keys());
}

async function getBaseNames(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  const baseNames = loraWorker.base_names();

  console.log("base names", baseNames.map(parseSDKey));
}

async function getWeightNorms(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.time("Calculating norms");

  let l2Norms = loraWorker
    .base_names()
    .map((base_name) => [base_name, loraWorker.l2_norm(base_name)])
		.map(([base_name, norm]) => {

		console.log([base_name, norm]);
		return [base_name, norm];
	})
    .reduce(
      (acc, [base_name, norm]) => {
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

        return acc;
      },
      {
        block: new Map(),
        block_count: new Map(),
        block_mean: new Map(),
      },
    );

  console.log(
    "weight_norms block",
    Array.from(l2Norms["block"]).sort(([k, _], [k2, _v]) => {
      return k > k2;
    }),
  );
  console.log(
    "weight_norms count",
    Array.from(l2Norms["block_count"]).sort(([k, _], [k2, _v]) => {
      return k > k2;
    }),
  );
  console.log(
    "weight_norms mean",
    Array.from(l2Norms["block_mean"]).sort(([k, _], [k2, _v]) => {
      return k > k2;
    }),
  );
  console.timeEnd("Calculating norms");
}

async function getAlphaKeys(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log("alpha keys", loraWorker.alpha_keys());
}

async function getAlphas(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log("alphas", loraWorker.alphas());
}

async function getDims(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log("dims", loraWorker.dims());
}

async function getNetworkArgs(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.network_args());
}

async function getNetworkType(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.network_type());
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

const SDRE =
  /.*(?<block_type>up|down|mid)_blocks?_.*(?<block_id>\d+).*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;

const MID_SDRE =
  /.*(?<block_type>up|down|mid)_block_.*(?<type>resnets|attentions|upsamplers|downsamplers)_(?<subblock_id>\d+).*/;
const TE_SDRE = /(?<block_id>\d+).*(?<block_type>self_attn|mlp)/;
const NUM_OF_BLOCKS = 12;

function parseSDKey(key) {
  let blockIdx = -1;
  let idx;

  let isConv = false;
  let isAttention = false;
  let isSampler = false;

  let type;
  let blockType;
  let blockId;
  let subBlockId;
  let name;

  if (key.includes("te_text_model")) {
    const matches = key.match(TE_SDRE);
    if (matches) {
      const groups = matches.groups;
      blockId = parseInt(groups["block_id"]);
      blockType = groups["block_type"];

      if (blockType === "self_attn") {
        isAttention = true;
      }
    }
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

			console.log("block_type", groups['block_type'])

      if (groups["block_type"] === "down") {
        blockIdx = 1 + idx;
        name = `IN${padTwo(idx)}`;
      } else if (groups["block_type"] === "up") {
        blockIdx = NUM_OF_BLOCKS + 1 + idx;
        name = `OUT${padTwo(idx)}`;
      } else if (groups["block_type"] === "mid") {
        blockIdx = NUM_OF_BLOCKS;
   
      }
    } else if (key.includes("mid_block_")) {
      const midMatch = key.match(MID_SDRE);
			name = `MID`;

      if (midMatch) {
        const groups = midMatch.groups;

        type = groups["type"];
        blockType = groups["block_type"];
        subBlockId = parseInt(groups["subblock_id"]);

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
