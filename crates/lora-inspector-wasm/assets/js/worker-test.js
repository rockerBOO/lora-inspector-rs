importScripts("/pkg/lora_inspector_wasm.js");

const { LoraWorker, LoRAFile, BufferedLoRAWeight } = wasm_bindgen;

function init_wasm_in_worker() {
	// Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
	wasm_bindgen("/pkg/lora_inspector_wasm_bg.wasm")
		.then(() => {
			onerror = (event) => {
				console.log("There is an error inside your worker!", event);
			};

			onmessage = async (e) => {
				if (e.data.messageType === "file_upload") {
					fileUploadHandler(e);
				} else {
					console.log("got message ---", e.data);
				}
			};
		})
		.catch((e) => {
			console.log("error", e);
		});
}

async function fileUploadHandler(e) {
	console.time("file_upload");
	const file = e.data.file;
	// files.set(file.name, file);
	loadWorker(file);
}

async function readFile(file) {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = (e) => {
			const buffer = new Uint8Array(e.target.result);
			resolve(buffer);
		};
		reader.readAsArrayBuffer(file);
	});
}

async function loadWorker(file) {
	const buffer = await readFile(file);

	try {
		const loraWorker = new LoraWorker(buffer, file.name);

		console.timeEnd("file_upload");
		self.postMessage({
			messageType: "metadata",
			filename: file.name,
			metadata: loraWorker.metadata(),
		});

		if (!self.crossOriginIsolated) {
			console.log(
				"performance.measureUserAgentSpecificMemory() is only available in cross-origin-isolated pages",
			);
			console.log("See https://web.dev/coop-coep/ to learn more");
		} else {
			console.log("is cross origin isolated");
		}
		const memorySample =
			await self.performance.measureUserAgentSpecificMemory();
		console.log("1--", memorySample);
		loraWorker.dims();

		const memorySample2 =
			await self.performance.measureUserAgentSpecificMemory();
		console.log("2--", memorySample2);
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

console.log("worker...");

init_wasm_in_worker();
