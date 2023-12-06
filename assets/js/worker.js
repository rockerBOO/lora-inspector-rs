// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts("/pkg/lora_inspector_rs.js");

console.log("Initializing worker");

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { LoraWorker } = wasm_bindgen;

// let clients = [];
// let wasms = [];

function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  wasm_bindgen("/pkg/lora_inspector_rs_bg.wasm").then(() => {
    // Create a new object of the `NumberEval` struct.
    // var num_eval = NumberEval.new();

    console.log("Worker setup...");

    // onconnect = function (event) {
    //   console.log("WORKER WASMS? ports", event.ports);
    //   //   const port = event.ports[0];
    //   //
    //   //   // Set callback to handle messages passed to the worker.
    //     onmessage = async (e) => {
    //       console.log("WORKER WASMS? ports", ports["wasm"]);
    //       console.log(e.data);
    //       // By using methods of a struct as reaction to messages passed to the
    //       // worker, we can preserve our state between messages.
    //       // var worker_result = num_eval.is_even(event.data);
    //
    //       // Send response back to be handled by callback in main thread.
    //   sendClientMessage(e.data)
    //     };
    // };
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

onmessage = async (e) => {
  console.log("got a message on the worker", e.data);
  const buffer = await readFile(e.data);
  const loraWorker = LoraWorker.new_from_buffer(buffer, e.data.name);
  console.log("got filename", loraWorker.filename());
};

// onconnect = function (event) {
//   console.log("onconnect::connected", event.name);
//   console.log(
//     event.ports,
//     event.name,
//     event.origin,
//     event.source,
//     event.lastEventId,
//     event.data,
//     Object.getOwnPropertyNames(event),
//   );
//
//   const port = event.ports[0];
//   // ports["client"].push(port);
//   port.onmessage = async (event) => {
//     console.log("onmessage::message", event.data);
//     // console.log(
//     //   "WORKER",
//     //   "ports",
//     //   event["ports"],
//     //   "origin:",
//     //   event.origin,
//     //   event.isTrusted,
//     //   Object.getOwnPropertyNames(event),
//     // );
//     // console.log(event.ports, event.name, event.origin, event.source, event.lastEventId);
//     // console.log("client ports", ports["client"]);
//     console.log("WORKER", event.data);
//
//     if (
//       event.data["origin"] == "main" &&
//       event.data["message_type"] == "register"
//     ) {
//       console.log("WORKER", "ADD CLIENT");
//       clients.push(port);
//     }
//
//     let data = event.data;
//     if (event.data instanceof Map) {
//       data = Object.fromEntries(event.data);
//     }
//     if (data["origin"] === "WASM" && data["message_type"] === "register") {
//       console.log("WORKER", "ADD WASMS");
//       wasms.push(port);
//     }
//
//     // if (event.data.get("origin") == "WASM") {
//     //   console.log("Send mpped WASM to clients");
//     //   sendClientMessage(event.data);
//     // }
//
//     if (data["origin"] == "WASM") {
//       console.log("Send non-ampped WASM to clients");
//       sendClientMessage(data);
//     }
//
//     if (data["message_type"] == "upload") {
//       console.log("Send upload", wasms);
//       sendWASMMessage("get_elem_count");
//     }
//
//     if (data["message_type"] == "l1_norm") {
//       console.log("Send upload", wasms);
//       sendWASMMessage("l1_norm");
//     }
//
//     // console.log("CLIENTS", clients);
//
//     // By using methods of a struct as reaction to messages passed to the
//     // worker, we can preserve our state between messages.
//     // var worker_result = num_eval.is_even(event.data);
//
//     // Send response back to be handled by callback in main thread.
//     // port.postMessage(e.data);
//     // port.postMessage({
//     //   origin: "shared_worker",
//     //   message_type: "ack",
//     //   payload: "im the worker",
//     // });
//     // };
//     //
//   };
// };
//
// onerror = (event) => {
//   console.error("WORKEER ERRRE", event);
// };

init_wasm_in_worker();
