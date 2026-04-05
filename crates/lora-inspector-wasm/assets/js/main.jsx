import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import init from "/pkg";
import { CompareMetadata, Metadata, Support } from "./components/index.js";
import {
	addWorker,
	clearWorkers,
	terminatePreviousProcessing,
} from "./workers";

function attachDragEvents() {
	const dropbox = document.querySelector("#dropbox");

	for (const eventName of ["dragover", "dragenter"]) {
		dropbox.addEventListener(eventName, (e) => {
			e.preventDefault();
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
let fileA = { metadata: null, filename: null, worker: null };
let fileB = { metadata: null, filename: null, worker: null };

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

function renderView(view) {
	if (view === "compare") {
		root.render(
			<StrictMode>
				<CompareMetadata
					metadataA={fileA.metadata}
					filenameA={fileA.filename}
					metadataB={fileB.metadata}
					filenameB={fileB.filename}
					onViewA={() => renderView("a")}
					onViewB={() => renderView("b")}
				/>
			</StrictMode>,
		);
	} else if (view === "a") {
		root.render(
			<StrictMode>
				<Metadata
					metadata={fileA.metadata}
					filename={fileA.filename}
					worker={fileA.worker}
					nav={
						<nav>
							<ul>
								<li>
									<button type="button" onClick={() => renderView("compare")}>
										Back to comparison
									</button>
								</li>
								<li>
									<button type="button" onClick={() => renderView("b")}>
										View {fileB.filename}
									</button>
								</li>
							</ul>
						</nav>
					}
				/>
			</StrictMode>,
		);
	} else if (view === "b") {
		root.render(
			<StrictMode>
				<Metadata
					metadata={fileB.metadata}
					filename={fileB.filename}
					worker={fileB.worker}
					nav={
						<nav>
							<ul>
								<li>
									<button type="button" onClick={() => renderView("compare")}>
										Back to comparison
									</button>
								</li>
								<li>
									<button type="button" onClick={() => renderView("a")}>
										View {fileA.filename}
									</button>
								</li>
							</ul>
						</nav>
					}
				/>
			</StrictMode>,
		);
	} else {
		root.render(
			<StrictMode>
				<Metadata
					metadata={fileA.metadata}
					filename={fileA.filename}
					worker={fileA.worker}
				/>
			</StrictMode>,
		);
	}
}

async function handleMetadata(metadata, filename, worker) {
	dropbox.classList.remove("box__open");
	dropbox.classList.add("box__closed");
	document.querySelector(".support").classList.remove("hidden");
	document.querySelector(".home")?.classList.remove("home");
	document.querySelector(".box").classList.remove("box__open");
	document.querySelector(".box__intro").classList.add("hidden");
	document.querySelector(".note").classList.add("hidden");

	if (fileA.metadata !== null && filename !== fileA.filename) {
		fileB = { metadata, filename, worker };
		renderView("compare");
	} else {
		fileA = { metadata, filename, worker };
		fileB = { metadata: null, filename: null, worker: null };
		renderView("single");
		showCompareButton();
	}
}

function showCompareButton() {
	const existing = document.querySelector("#compare-dropbox");
	if (existing) return;

	const input = document.createElement("input");
	input.type = "file";
	input.accept = ".safetensors";
	input.id = "compare-file";
	input.className = "box__file";
	input.addEventListener("change", async (e) => {
		const file = e.target.files[0];
		if (file) processFile(file, true);
	});

	const label = document.createElement("label");
	label.htmlFor = "compare-file";
	label.innerHTML =
		'<strong>Compare LoRA</strong><span class="box__dragndrop"> or drag it here</span>.';

	const boxInput = document.createElement("div");
	boxInput.className = "box__input";
	boxInput.appendChild(input);
	boxInput.appendChild(label);

	const form = document.createElement("form");
	form.id = "compare-dropbox";
	form.className = "box has-advanced-upload";
	form.enctype = "multipart/form-data";
	form.appendChild(boxInput);

	for (const eventName of ["dragover", "dragenter"]) {
		form.addEventListener(eventName, (e) => {
			e.preventDefault();
			form.classList.add("is-dragover");
		});
	}
	for (const eventName of ["dragleave", "dragend", "drop"]) {
		form.addEventListener(eventName, () => {
			form.classList.remove("is-dragover");
		});
	}
	form.addEventListener("drop", (e) => {
		e.preventDefault();
		const file = e.dataTransfer.items
			? [...e.dataTransfer.items].find((i) => i.kind === "file")?.getAsFile()
			: e.dataTransfer.files[0];
		if (file) processFile(file, true);
	});

	document.querySelector(".support").insertAdjacentElement("beforebegin", form);
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

async function processFile(file, isComparison = false) {
	if (!isComparison) {
		fileA = { metadata: null, filename: null, worker: null };
		fileB = { metadata: null, filename: null, worker: null };
		const existing = document.querySelector("#compare-dropbox");
		if (existing) existing.remove();
	}

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
		} else if (
			e.data.messageType === "metadata_error" ||
			e.data.messageType === "worker_error"
		) {
			processingMetadata = false;
			finishLoading();
			// Terminate the broken worker so future uploads start fresh
			clearWorkers();
			const msg = e.data.message || "Unknown error";
			if (e.data.isPanic) {
				addPanicMessage(msg);
			} else {
				addErrorMessage(`Could not load LoRA: ${msg}`);
			}
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

// Frames that are pure panic/hook infrastructure — not useful for finding
// the source of a bug. Everything else (user code, stdlib ops) is kept.
const PANIC_NOISE = [
	"__rust_abort",
	"__rust_start_panic",
	"::rust_panic",
	"rust_panic_with_hook",
	"begin_panic_handler",
	"__rust_end_short_backtrace",
	"rust_begin_unwind",
	"core::panicking::panic_fmt",
	"core::panicking::panic::",
	"console_panic_hook::",
	"core::ops::function::Fn::call",
	"__wbg_",
	"logError@",
	"LoraWorker@",
	"loadWorker@",
	"fileUploadHandler@",
	"init_wasm_in_worker",
	"EventHandlerNonNull",
	"wasm-bindgen-test",
];

function parsePanic(raw) {
	// Split hook output from call stack
	const callStackSep = "\nCall stack:\n\n";
	const callStackIdx = raw.indexOf(callStackSep);
	const hookSection = callStackIdx >= 0 ? raw.slice(0, callStackIdx) : raw;
	const callStackSection =
		callStackIdx >= 0 ? raw.slice(callStackIdx + callStackSep.length) : "";

	// Extract panic location + message from hook output.
	// New format: "panicked at FILE:LINE:COL:\nMESSAGE"
	// Old format: "panicked at 'MESSAGE', FILE:LINE:COL"
	let location = "";
	let message = "";
	const newFmt = hookSection.match(
		/^panicked at ([^\n]+):\n([\s\S]+?)(?:\n\nStack:|$)/,
	);
	const oldFmt = hookSection.match(/^panicked at '([^']+)',\s*(.+)/m);
	if (newFmt) {
		location = newFmt[1].trim();
		message = newFmt[2].trim();
	} else if (oldFmt) {
		message = oldFmt[1].trim();
		location = oldFmt[2].trim();
	} else {
		message = hookSection.trim();
	}

	// Clean a raw stack frame line down to just the function name.
	function cleanFrame(line) {
		// Strip leading "at " (Chrome/Node format)
		let t = line.trim().replace(/^at\s+/, "");
		// Cut everything from "@" onward (URL + wasm address)
		const atIdx = t.indexOf("@");
		if (atIdx >= 0) t = t.slice(0, atIdx);
		return (
			t
				.replace(/^lora_inspector_wasm\.wasm\./, "")
				// Strip __rustc[HASH]:: prefixes
				.replace(/__rustc\[[0-9a-f]+\]::/g, "")
				// Strip trailing ::h<hex> hash suffixes
				.replace(/::h[0-9a-f]{15,}$/g, "")
				.trim()
		);
	}

	// Filter and clean call stack frames.
	const relevantFrames = callStackSection
		.split("\n")
		.filter((line) => {
			const t = line.trim();
			if (!t || t === "Error" || t === "RuntimeError: unreachable")
				return false;
			return !PANIC_NOISE.some((noise) => t.includes(noise));
		})
		.map(cleanFrame)
		.filter(Boolean)
		.join("\n");

	return { location, message, relevantFrames };
}

function addPanicMessage(errorMessage) {
	const { location, message, relevantFrames } = parsePanic(errorMessage);

	const errorOverlayEle = document.createElement("div");
	const errorBlockEle = document.createElement("div");
	errorOverlayEle.classList.add("error-overlay");
	errorOverlayEle.id = "error-overlay";
	errorBlockEle.classList.add("error-block", "panic-block");

	// Close button — top right corner
	const closeBtn = document.createElement("button");
	closeBtn.classList.add("panic-close");
	closeBtn.setAttribute("aria-label", "Close");
	closeBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;
	closeBtn.addEventListener("click", closeErrorMessage);

	const header = document.createElement("div");
	header.classList.add("panic-header");

	const title = document.createElement("div");
	title.classList.add("error");
	title.textContent = "LoRA Inspector crashed";

	header.append(title, closeBtn);

	const msgEl = document.createElement("div");
	msgEl.classList.add("panic-message");
	msgEl.textContent = message || errorMessage;

	const locEl = document.createElement("pre");
	locEl.classList.add("panic-location");
	locEl.textContent = location;

	const stackLabel = document.createElement("p");
	stackLabel.classList.add("panic-stack-label");
	stackLabel.textContent = "Call stack:";

	const stackEl = document.createElement("pre");
	stackEl.classList.add("panic-stack");
	stackEl.textContent =
		relevantFrames ||
		"(no demangled frames — use a debug build for full trace)";

	// Footer: copy icon + report link
	const footer = document.createElement("div");
	footer.classList.add("panic-footer");

	const copyBtn = document.createElement("button");
	copyBtn.classList.add("panic-copy");
	copyBtn.setAttribute("aria-label", "Copy full report");
	copyBtn.title = "Copy full report";
	copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
	copyBtn.addEventListener("click", (e) => {
		e.stopPropagation();
		navigator.clipboard.writeText(errorMessage).then(() => {
			copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>`;
			setTimeout(() => {
				copyBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>`;
			}, 2000);
		});
	});

	const reportLink = document.createElement("a");
	reportLink.classList.add("panic-report-link");
	reportLink.href = "https://github.com/rockerBOO/lora-inspector-rs/issues/new";
	reportLink.target = "_blank";
	reportLink.rel = "noopener noreferrer";
	reportLink.textContent = "Report on GitHub →";
	reportLink.addEventListener("click", (e) => e.stopPropagation());

	footer.append(copyBtn, reportLink);

	const children = [header, msgEl];
	if (location) children.push(locEl);
	if (relevantFrames) children.push(stackLabel, stackEl);
	children.push(footer);

	errorBlockEle.append(...children);

	// Only stop propagation (not default) so links inside still work
	errorBlockEle.addEventListener("click", (e) => e.stopPropagation());
	errorOverlayEle.appendChild(errorBlockEle);
	errorOverlayEle.addEventListener("click", closeErrorMessage);
	document.body.appendChild(errorOverlayEle);
}

function cancelLoading(file, _processingMetadata, uploadTimeoutHandler) {
	clearTimeout(uploadTimeoutHandler);
	processingMetadata = false;
	finishLoading();
	clearWorkers();
	console.log("Cancel loading", file?.name ?? file);
}
