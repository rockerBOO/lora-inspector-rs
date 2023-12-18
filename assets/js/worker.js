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

  console.log("base names", loraWorker.base_names());
}

async function getWeightNorms(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

	console.time("Calculating norms")
  console.log(
    "weight_norms",
    loraWorker.base_names().map((name) => [name, loraWorker.weight_norm(name)]),
  );
	console.timeEnd("Calculating norms")
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

init_wasm_in_worker();
