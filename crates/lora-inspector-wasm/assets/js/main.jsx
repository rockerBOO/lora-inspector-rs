import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import init from "/pkg";
import { Metadata, Support } from "./components/index.js";
import {
	addWorker,
	clearWorkers,
	terminatePreviousProcessing,
} from "./workers";

function attachDragEvents() {
	const dropbox = document.querySelector("#dropbox");

	for (const eventName of ["dragover", "dragenter"]) {
		dropbox.addEventListener(eventName, () => {
			dropbox.classList.add("is-dragover");
		});
	}

	for (const eventName of ["dragleave", "dragend", "drop"]) {
		dropbox.addEventListener(eventName, () => {
			dropbox.classList.remove("is-dragover");
		});
	}

	const isAdvancedUpload = (() => {
		const div = document.createElement("div");
		return (
			("draggable" in div || ("ondragstart" in div && "ondrop" in div)) &&
			"FormData" in window &&
			"FileReader" in window
		);
	})();

	if (isAdvancedUpload) {
		document.querySelector("#dropbox").classList.add("has-advanced-upload");
	}
}

attachDragEvents();

const files = new Map();
let mainFilename;

init().then(() => {
	const dropbox = document.querySelector("#dropbox");
	dropbox.addEventListener("drop", async (e) => {
		e.preventDefault();
		if (e.dataTransfer.items) {
			// Use DataTransferItemList interface to access the file(s)
			[...e.dataTransfer.items].forEach((item, i) => {
				if (item.type !== "") {
					addErrorMessage("Invalid filetype. Try a .safetensors file.");
					return;
				}

				// If dropped items aren't files, reject them
				if (item.kind === "file") {
					const file = item.getAsFile();
					console.log(`data transfer items… file[${i}].name = ${file.name}`);

					processFile(file);
				}
			});
		} else {
			// Use DataTransfer interface to access the file(s)
			[...e.dataTransfer.files].forEach((file, i) => {
				processFile(file.item(i));
				console.log(`… file[${i}].name = ${file.name}`);
			});
		}
	});

	document.querySelector("#file").addEventListener("change", async (e) => {
		e.preventDefault();
		e.stopPropagation();

		const files = e.target.files;

		for (let i = 0; i < files.length; i++) {
			if (files.item(i).type !== "") {
				addErrorMessage("Invalid filetype. Try a .safetensors file.");
				continue;
			}

			processFile(files.item(i));
		}
	});
});

const domNode = document.getElementById("results");
const root = createRoot(domNode);

async function handleMetadata(metadata, filename, worker) {
	dropbox.classList.remove("box__open");
	dropbox.classList.add("box__closed");
	document.querySelector(".support").classList.remove("hidden");
	document.querySelector(".home")?.classList.remove("home");
	document.querySelector(".box").classList.remove("box__open");
	document.querySelector(".box__intro").classList.add("hidden");
	document.querySelector(".note").classList.add("hidden");
	root.render(
		<StrictMode>
			<Metadata metadata={metadata} filename={filename} worker={worker} />
		</StrictMode>,
	);
}

// Test the new JSX Support component
(() => {
	const root = createRoot(document.querySelector(".support"));
	root.render(
		<StrictMode>
			<Support />
		</StrictMode>,
	);
})();

let uploadTimeoutHandler;
let processingMetadata = false;

async function processFile(file) {
	clearWorkers();
	const worker = await addWorker(file.name);

	processingMetadata = terminatePreviousProcessing(
		file.name,
		processingMetadata,
	);

	mainFilename = undefined;

	worker.postMessage({ messageType: "file_upload", file: file });
	processingMetadata = true;
	const cancel = loading(file.name);

	uploadTimeoutHandler = setTimeout(() => {
		cancel();
		addErrorMessage("Timeout loading LoRA. Try again.");
	}, 12000);

	window.addEventListener("keyup", (e) => {
		if (e.key === "Escape") {
			cancelLoading(file, processingMetadata, uploadTimeoutHandler);
		}
	});

	function messageHandler(e) {
		clearTimeout(uploadTimeoutHandler);
		if (e.data.messageType === "metadata") {
			processingMetadata = false;

			// Setup some access points to the file
			// (we shouldn't hold on to the file handlers but just he metadata)
			files.set(file.name, file);
			mainFilename = e.data.filename;

			handleMetadata(e.data.metadata, file.name, worker).then(() => {
				worker.postMessage({
					messageType: "network_module",
					name: mainFilename,
				});
				worker.postMessage({ messageType: "network_args", name: mainFilename });
				worker.postMessage({ messageType: "network_type", name: mainFilename });
				worker.postMessage({ messageType: "weight_keys", name: mainFilename });
				worker.postMessage({ messageType: "alpha_keys", name: mainFilename });
				worker.postMessage({ messageType: "base_names", name: mainFilename });
				worker.postMessage({ messageType: "weight_norms", name: mainFilename });
			});
			finishLoading();
		} else {
			// console.log("UNHANDLED MESSAGE", e.data);
		}
	}

	worker.addEventListener("message", messageHandler);
}

function loading(file) {
	const loadingEle = document.createElement("div");
	const loadingOverlayEle = document.createElement("div");

	loadingEle.classList.add("loading-file");
	loadingOverlayEle.classList.add("loading-overlay");
	loadingOverlayEle.id = "loading-overlay";

	loadingEle.textContent = "loading...";
	loadingOverlayEle.appendChild(loadingEle);
	document.body.appendChild(loadingOverlayEle);

	let clicks = 0;
	loadingOverlayEle.addEventListener("click", () => {
		clicks += 1;

		// The user is getting fustrated or we are about to make them made. Either way.
		if (clicks > 1) {
			cancelLoading(file, processingMetadata, uploadTimeoutHandler);
		}
	});

	return function cancel() {
		cancelLoading(file, processingMetadata, uploadTimeoutHandler);
	};
}

function finishLoading() {
	const overlay = document.querySelector("#loading-overlay");

	if (overlay) {
		overlay.remove();
	}
}

function addErrorMessage(errorMessage) {
	const errorEle = document.createElement("div");
	const errorOverlayEle = document.createElement("div");
	const errorBlockEle = document.createElement("div");

	errorEle.classList.add("error");
	errorBlockEle.classList.add("error-block");
	errorOverlayEle.classList.add("error-overlay");
	errorOverlayEle.id = "error-overlay";

	errorEle.textContent = errorMessage;

	const button = document.createElement("button");
	button.textContent = "Close";
	button.addEventListener("click", closeErrorMessage);

	errorBlockEle.append(errorEle, button);

	errorBlockEle.addEventListener("click", (e) => {
		e.preventDefault();
		e.stopPropagation();
	});

	errorOverlayEle.appendChild(errorBlockEle);

	errorOverlayEle.addEventListener("click", (e) => {
		e.preventDefault();
		closeErrorMessage();
	});

	document.body.appendChild(errorOverlayEle);
}

function closeErrorMessage() {
	const overlay = document.querySelector("#error-overlay");

	if (overlay) {
		overlay.remove();
	}
}

function cancelLoading(file, _processingMetadata, uploadTimeoutHandler) {
	clearTimeout(uploadTimeoutHandler);
	// processingMetadata = false;
	console.log("Cancel loading", file.name);
}
