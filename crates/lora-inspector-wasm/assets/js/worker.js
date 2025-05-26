import { parseSDKey } from "./moduleBlocks";
import { simd } from "wasm-feature-detect";

const files = new Map();

// loraWorkers are specific objects that manage the LoRA file, buffer, and inspection.
const loraWorkers = new Map();

const loraWorkersRegistry = new FinalizationRegistry((key) => {
	if (!loraWorkers.get(key)?.deref()) {
		loraWorkers.delete(key);
	}
});

function addWorker(name, worker) {
	console.log("Adding worker ", name);
	loraWorkers.set(name, worker);

	loraWorkersRegistry.register(worker, name);
	return getWorker(name);
}

function removeWorker(workerName) {
	const loraWorker = getWorker(workerName);

	loraWorker.unload();
	loraWorker.free();

	loraWorkers.set(workerName, undefined);
	loraWorkers.delete(workerName);
}

function getWorker(workerName) {
	const loraWorker = loraWorkers.get(workerName);

	if (!loraWorker) {
		throw new Error(`Could not find worker ${workerName}`);
	}

	return loraWorker;
}

async function init_wasm_in_worker() {
	let worker;
	// if (await simd()) {
	// 	const { initSync, LoraWorker } = await import("/pkg/lora-inspector-simd");
	// 	worker = LoraWorker;
	// 	const resolvedUrl = (await import("/pkg/lora-inspector-simd_bg.wasm?url")).default;
	// 	await fetch(resolvedUrl)
	// 		.then((response) => response.arrayBuffer())
	// 		.then((bytes) => {
	// 			return initSync(bytes);
	// 		});
	// } else {
	const { initSync, LoraWorker } = await import("/pkg/lora-inspector");
	worker = LoraWorker;
	const resolvedUrl = (await import("/pkg/lora-inspector-simd_bg.wasm?url"))
		.default;
	await fetch(resolvedUrl)
		.then((response) => response.arrayBuffer())
		.then((bytes) => {
			return initSync(bytes);
		});
	// }
	// Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
	self.onerror = (error) => {
		console.log("There is an error inside your worker!", error);
	};

	self.onmessage = async (e) => {
		if (e.data.messageType === "file_upload") {
			fileUploadHandler(e, worker);
		} else if (e.data.messageType === "unload") {
			unloadWorker(e).then(() => {
				if (e.data.reply) {
					self.postMessage({
						messageType: "is_available",
					});
				}
			});
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
			getL2Norms(e).then((norms) => {
				if (e.data.reply) {
					self.postMessage({
						messageType: "l2_norm",
						norms,
					});
				}
			});
		} else if (e.data.messageType === "norms") {
			getNorms(e).then(([norms, error]) => {
				if (e.data.reply) {
					if (error) {
						self.postMessage({
							messageType: "norms",
							error,
							norms,
							baseName: e.data.baseName,
						});
					} else {
						self.postMessage({
							messageType: "norms",
							norms,
							baseName: e.data.baseName,
						});
					}
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
		} else if (e.data.messageType === "weight_decomposition") {
			getWeightDecomposition(e).then((weightDecomposition) => {
				if (e.data.reply) {
					self.postMessage({
						messageType: "weight_decomposition",
						weightDecomposition,
					});
				}
			});
		} else if (e.data.messageType === "rank_stabilized") {
			getRankStabilized(e).then((rankStabilized) => {
				if (e.data.reply) {
					self.postMessage({
						messageType: "rank_stabilized",
						rankStabilized,
					});
				}
			});
		} else if (e.data.messageType === "dora_scales") {
			getDoraScales(e).then((doraScales) => {
				if (e.data.reply) {
					self.postMessage({
						messageType: "dora_scales",
						doraScales,
					});
				}
			});
		}
	};
}

init_wasm_in_worker();

// FUNCTIONS
// ============================

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

async function loadWorker(file, worker) {
	const buffer = await readFile(file);

	try {
		const loraWorker = addWorker(file.name, new worker(buffer, file.name));

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

async function fileUploadHandler(e, worker) {
	console.time("file_upload");
	const file = e.data.file;
	files.set(file.name, file);
	loadWorker(file, worker);
}

async function unloadWorker(e) {
	removeWorker(e.data.name);
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
	const name = e.data.name;
	const loraWorker = getWorker(name);
	const baseName = e.data.baseName;

	try {
		const scaled = loraWorker.norms(baseName, [
			"l1_norm",
			"l2_norm",
			"matrix_norm",
			"max",
			"min",
			"std_dev",
			"median",
		]);

		return [scaled, undefined];
	} catch (e) {
		console.error(e);
		return [undefined, e];
	}
}

async function getL2Norms(e) {
	const loraWorker = getWorker(e.data.name);

	console.time("Calculating norms");

	const baseNames = loraWorker.base_names();
	const totalCount = baseNames.length;

	let currentCount = 0;

	const l2Norms = baseNames
		.map((base_name) => {
			currentCount += 1;

			self.postMessage({
				messageType: "l2_norms_progress",
				currentCount,
				totalCount,
				baseName: base_name,
			});

			try {
				const norm = loraWorker.l2_norm(base_name);
				return [base_name, norm];
			} catch (e) {
				console.error(base_name, e);
				return [base_name, undefined];
			}
		})
		.reduce(
			(acc, [base_name, norm]) => {
				if (norm === undefined) {
					return acc;
				}

				const parts = parseSDKey(base_name);

				const blockName = parts.name;

				acc.block.set(blockName, (acc.block.get(blockName) ?? 0) + norm);
				acc.block_count.set(
					blockName,
					(acc.block_count.get(blockName) ?? 0) + 1,
				);
				acc.block_mean.set(
					blockName,
					acc.block.get(blockName) / acc.block_count.get(blockName),
				);

				acc.metadata.set(blockName, parts);

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

	const norms = Array.from(l2Norms.block_mean).sort(([k, _], [k2, _v]) => {
		return k > k2;
	});

	console.timeEnd("Calculating norms");

	// Split between TE and UNet
	const splitNorms = norms.reduce(
		(acc, item) => {
			const [k, v] = item;

			if (k === undefined) {
				console.log(item);
				console.error("Undefined key for norm reduction");
				return acc;
			}

			if (k.includes("TE")) {
				acc.te.set(k, { mean: v, metadata: l2Norms.metadata.get(k) });
			} else if (k.includes("DB")) {
				acc.db.set(k, { mean: v, metadata: l2Norms.metadata.get(k) });
			} else if (k.includes("SB")) {
				acc.sb.set(k, { mean: v, metadata: l2Norms.metadata.get(k) });
			} else {
				acc.unet.set(k, { mean: v, metadata: l2Norms.metadata.get(k) });
			}
			return acc;
		},
		{ te: new Map(), unet: new Map(), db: new Map(), sb: new Map() },
	);

	return splitNorms;
}

async function getAlphaKeys(e) {
	const name = e.data.name;
	const loraWorker = getWorker(name);

	return loraWorker.alpha_keys();
}

async function getAlphas(e) {
	const loraWorker = getWorker(e.data.name);

	return Array.from(loraWorker.alphas()).sort((a, b) => a > b);
}

async function getWeightDecomposition(e) {
	const loraWorker = getWorker(e.data.name);

	return loraWorker.weight_decomposition();
}

async function getRankStabilized(e) {
	const loraWorker = getWorker(e.data.name);

	return loraWorker.rank_stabilized();
}

async function getDoraScales(e) {
	const loraWorker = getWorker(e.data.name);

	const doraScales = Array.from(loraWorker.doraScales()).sort((a, b) => a > b);

	return doraScales;
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

