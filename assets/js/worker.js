// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts("/pkg/lora_inspector_rs.js");

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { LoraWorker } = wasm_bindgen;

// let clients = [];
// let wasms = [];

function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  wasm_bindgen("/pkg/lora_inspector_rs_bg.wasm").then(() => {
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

    let processing = false;

    onerror = (event) => {
      console.log("There is an error inside your worker!", event);
    };

    onmessage = async (e) => {
      if (e.data.messageType === "file_upload") {
        console.time("file_upload");
        console.log("Reading file...");
        const file = e.data.file;
        const buffer = await readFile(file);

        console.log("Loading file...");
        try {
          const loraWorker = new LoraWorker(buffer, file.name);

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
    };
  });
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
