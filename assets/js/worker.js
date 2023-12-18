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
        fileUploadHandler(e);
      } else if (e.data.messageType === "network_module") {
        getNetworkModule(e);
      } else if (e.data.messageType === "network_args") {
        getNetworkArgs(e);
      } else if (e.data.messageType === "network_type") {
        getNetworkType(e);
      } else if (e.data.messageType === "weight_keys") {
        ensureTensorsLoaded(e)
          .then(getWeightkeys)
          .catch(handleTensorsLoadedError);
      
      } else if (e.data.messageType === "alphas") {
        ensureTensorsLoaded(e)
          .then(getAlphas)
          .catch(handleTensorsLoadedError);
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

function ensureTensorsLoaded(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  return new Promise((resolve) => {
    if (!loraWorker.is_tensors_loaded()) {
      if (!loraWorker.load_tensors()) {
        reject("Could not load the tensors");
      }
    }

    resolve(e);
  });
}

function handleTensorsLoadedError(err) {
  self.postMessage({
    messageType: "ensure_tensors_loaded_error",
    message: "Could not load the tensors",
    errorMessage: err,
    errorCode: 3,
  });
}

async function loadTensors(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  if (!loraWorker.load_tensors()) {
    self.postMessage({
      messageType: "load_tensor_error",
      message: "Failed to load the tensor",
      errorCode: 2,
    });
  }
}

async function getNetworkModule(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.network_module());
}

async function getWeightkeys(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.weight_keys());
}

async function getAlphas(e) {
  const name = e.data.name;
  const loraWorker = loraWorkers.get(name);

  console.log(loraWorker.alphas());
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
