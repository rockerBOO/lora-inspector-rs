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
			console.log("loraWorkers", Array.from(loraWorkers.keys()));
			console.log(e.data);
			if (e.data.messageType === "file_upload") {
				// unload old workers for now...
				console.log("Clearing workers!!!!!");
				loraWorkers.clear();
				fileUploadHandler(e);
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
				getKeys(e);
			} else if (e.data.messageType === "base_names") {
				getBaseNames(e);
			} else if (e.data.messageType === "l2_norm") {
				getL2Norms(e).then((norms) => {
					if (e.data.reply) {
						self.postMessage({
							messageType: "l2_norm",
							norms,
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
		reader.onload = function(e) {
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
		console.log("!!!!! file.name", file.name);
		loraWorkers.set(file.name, new LoraWorker(buffer, file.name));

		const loraWorker = loraWorkers.get(file.name);

		console.log("!!!!!Getting metadata...");

		console.timeEnd("file_upload");
		self.postMessage({
			messageType: "metadata",
			filename: file.name,
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

	if (!loraWorker) {
		console.log("!!!!", Array.from(loraWorkers.keys()));
		throw new Error("No LoRA for this name " + name);
	}

	return loraWorker.network_module();
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

	// loraWorker.parse_key(key);
	// console.log("base names", baseNames.map(parseSDKey));
}

async function getL2Norms(e) {
	const name = e.data.name;

	console.assert(name, "name is invalid");
	const loraWorker = loraWorkers.get(name);
	console.log("loraWorkers", loraWorkers);

	console.time("Calculating norms");
	console.log("Calculating l2 norms...");

	const baseNames = loraWorker.base_names();
	const totalCount = baseNames.length;

	let currentCount = 0;

	// const progressInterval = setInterval(
	//   (currentCount, totalCount) => {
	//     console.log("l2_norms_progress", currentCount, totalCount);
	//   },
	//   1000,
	//   currentCount,
	//   totalCount,
	// );

	// console.log("progress interval", progressInterval);

	let l2Norms = baseNames
		.map((base_name) => {
			currentCount += 1;

			self.postMessage({
				messageType: "l2_norms_progress",
				currentCount,
				totalCount,
				baseName: base_name,
			});
			return [base_name, loraWorker.l2_norm(base_name)];
		})
		.reduce(
			(acc, [base_name, norm]) => {

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
		messageType: "l2_norm_progress_finished",
	});

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
	const norms = Array.from(l2Norms["block_mean"]).sort(([k, _], [k2, _v]) => {
		return k > k2;
	});

	console.log("weight_norms mean", norms);
	console.timeEnd("Calculating norms");

	console.log("metadata", l2Norms['metadata'])

	// Split between TE and UNet
	const splitNorms = norms.reduce(
		(acc, [k, v]) => {
			if (k.includes("TE")) {
				acc.te.set(k, { mean: v, metadata: l2Norms['metadata'].get(k) });
			} else {
				acc.unet.set(k, { mean: v, metadata: l2Norms['metadata'].get(k) });
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

	console.log("alpha keys", loraWorker.alpha_keys());
}

async function getAlphas(e) {
	const name = e.data.name;
	const loraWorker = loraWorkers.get(name);

	// console.log("alphas", loraWorker.alphas());

	return Array.from(loraWorker.alphas()).sort((a, b) => a > b);
	// self.postMessage({
	// 	messageType: "alphas",
	// 	alphas: ,
	// 	reply: true
	// });
}

async function getDims(e) {
	const name = e.data.name;
	const loraWorker = loraWorkers.get(name);

	return Array.from(loraWorker.dims()).sort((a, b) => a > b);
	// self.postMessage({
	// 	messageType: "dims",
	// 	dims: ,
	// 	reply: true
	// });
}

async function getNetworkArgs(e) {
	const name = e.data.name;
	const loraWorker = loraWorkers.get(name);

	console.log('metadata', loraWorker.metadata());

	console.log("network args", loraWorker.network_args());

	return loraWorker.network_args();
}

async function getNetworkType(e) {
	const name = e.data.name;
	const loraWorker = loraWorkers.get(name);

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
	let isProjection = false
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
			type = "encoder"
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
