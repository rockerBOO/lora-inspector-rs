import test from "ava";

// Tests for worker message patterns used by components
// Focus on the core logic patterns rather than actual worker communication

test("worker message creation patterns", (t) => {
	// Test message structure used by components
	const createMessage = (type, filename, extras = {}) => ({
		messageType: type,
		name: filename,
		...extras,
	});

	const msg = createMessage("network_type", "test.safetensors");
	t.is(msg.messageType, "network_type");
	t.is(msg.name, "test.safetensors");

	const msgWithExtras = createMessage("dims", "test.safetensors", {
		limit: 10,
	});
	t.is(msgWithExtras.limit, 10);
});

test("network type selection logic", (t) => {
	// Test the logic components use to select network sub-components
	const selectNetworkComponent = (networkType) => {
		if (networkType === "DiagOFT") return "DiagOFTNetwork";
		if (networkType === "BOFT") return "BOFTNetwork";
		if (networkType === "LoKr") return "LoKrNetwork";
		return "LoRANetwork";
	};

	t.is(selectNetworkComponent("DiagOFT"), "DiagOFTNetwork");
	t.is(selectNetworkComponent("BOFT"), "BOFTNetwork");
	t.is(selectNetworkComponent("LoKr"), "LoKrNetwork");
	t.is(selectNetworkComponent("LoRA"), "LoRANetwork");
	t.is(selectNetworkComponent("Unknown"), "LoRANetwork");
});

test("component state update logic", (t) => {
	// Test how components update their state from worker responses
	const updateState = (currentState, response, messageType) => {
		const newState = { ...currentState };

		if (messageType === "network_type" && response.networkType) {
			newState.networkType = response.networkType;
		}
		if (messageType === "dims" && Array.isArray(response.dims)) {
			newState.dims = response.dims;
		}
		if (messageType === "precision" && response.precision) {
			newState.precision = response.precision;
		}

		return newState;
	};

	const initialState = { networkType: "", dims: [], precision: "" };

	const state1 = updateState(
		initialState,
		{ networkType: "LoRA" },
		"network_type",
	);
	t.is(state1.networkType, "LoRA");

	const state2 = updateState(state1, { dims: [4, 8] }, "dims");
	t.deepEqual(state2.dims, [4, 8]);

	const state3 = updateState(state2, { precision: "fp16" }, "precision");
	t.is(state3.precision, "fp16");
});

test("error handling patterns", (t) => {
	// Test error handling in worker communication
	const safeWorkerCall = (response, fallback) => {
		try {
			if (!response) return fallback;
			return response;
		} catch (error) {
			return fallback;
		}
	};

	t.deepEqual(safeWorkerCall({ data: "success" }, { data: "fallback" }), {
		data: "success",
	});
	t.deepEqual(safeWorkerCall(null, { data: "fallback" }), { data: "fallback" });
	t.deepEqual(safeWorkerCall(undefined, { data: "fallback" }), {
		data: "fallback",
	});
});

// Mirrors the idle timer + ensureLoaded logic in worker.js
function makeWorkerMemory() {
	const files = new Map();
	const loraWorkers = new Map();
	const idleTimers = new Map();

	function setIdleTimer(name, onExpire) {
		if (idleTimers.has(name)) clearTimeout(idleTimers.get(name));
		idleTimers.set(
			name,
			setTimeout(() => {
				// idle: just unload weights, keep the worker object (metadata stays)
				const w = loraWorkers.get(name);
				if (w) w.unload();
				idleTimers.delete(name);
				onExpire?.(name);
			}, 100),
		);
	}

	function clearIdleTimer(name) {
		if (idleTimers.has(name)) {
			clearTimeout(idleTimers.get(name));
			idleTimers.delete(name);
		}
	}

	function addWorker(name, worker) {
		loraWorkers.set(name, worker);
		return loraWorkers.get(name);
	}

	function removeWorker(name) {
		loraWorkers.delete(name);
		clearIdleTimer(name);
	}

	async function ensureLoaded(name, onExpire) {
		const loraWorker = loraWorkers.get(name);
		const file = files.get(name);

		if (!loraWorker) {
			// Worker was fully freed — reconstruct from scratch
			if (!file)
				throw new Error(`Cannot reload worker ${name}: file not found`);
			addWorker(name, {
				name,
				tensorsLoaded: true,
				metadata: file.metadata,
				data: file.data,
				unload() {
					this.tensorsLoaded = false;
				},
				reload_from_buffer(buf) {
					this.tensorsLoaded = true;
					this.data = buf;
				},
				is_tensors_loaded() {
					return this.tensorsLoaded;
				},
			});
		} else if (!loraWorker.is_tensors_loaded()) {
			// Weights were idle-unloaded — reload weights only, metadata stays
			if (!file)
				throw new Error(`Cannot reload weights for ${name}: file not found`);
			loraWorker.reload_from_buffer(file.data);
		}

		setIdleTimer(name, onExpire);
	}

	async function withWorker(name, fn, onExpire) {
		await ensureLoaded(name, onExpire);
		return fn(loraWorkers.get(name));
	}

	return {
		files,
		loraWorkers,
		idleTimers,
		addWorker,
		removeWorker,
		ensureLoaded,
		withWorker,
	};
}

test("idle timer unloads weights but keeps worker in map", async (t) => {
	const mem = makeWorkerMemory();
	mem.files.set("model.safetensors", {
		data: "buffer",
		metadata: { name: "test" },
	});
	mem.addWorker("model.safetensors", {
		tensorsLoaded: true,
		unload() {
			this.tensorsLoaded = false;
		},
		is_tensors_loaded() {
			return this.tensorsLoaded;
		},
		reload_from_buffer(buf) {
			this.tensorsLoaded = true;
			this.data = buf;
		},
	});

	await new Promise((resolve) => {
		mem.ensureLoaded("model.safetensors", resolve);
	});

	// Worker still in map (metadata preserved), but weights unloaded
	t.true(mem.loraWorkers.has("model.safetensors"));
	t.false(mem.loraWorkers.get("model.safetensors").is_tensors_loaded());
});

test("idle timer resets on activity", async (t) => {
	const mem = makeWorkerMemory();
	mem.files.set("model.safetensors", { data: "buffer", metadata: {} });
	mem.addWorker("model.safetensors", {
		tensorsLoaded: true,
		unload() {
			this.tensorsLoaded = false;
		},
		is_tensors_loaded() {
			return this.tensorsLoaded;
		},
		reload_from_buffer(buf) {
			this.tensorsLoaded = true;
			this.data = buf;
		},
	});

	let expiredCount = 0;
	const onExpire = () => {
		expiredCount++;
	};

	// Call ensureLoaded twice quickly — timer should reset, fire only once
	await mem.ensureLoaded("model.safetensors", onExpire);
	await mem.ensureLoaded("model.safetensors", onExpire);

	await new Promise((resolve) => setTimeout(resolve, 200));
	t.is(expiredCount, 1);
});

test("ensureLoaded reloads weights after idle unload, metadata unchanged", async (t) => {
	const mem = makeWorkerMemory();
	mem.files.set("model.safetensors", {
		data: "reloaded-buffer",
		metadata: { name: "kept" },
	});
	const worker = {
		tensorsLoaded: false,
		metadata: { name: "kept" },
		unload() {
			this.tensorsLoaded = false;
		},
		is_tensors_loaded() {
			return this.tensorsLoaded;
		},
		reload_from_buffer(buf) {
			this.tensorsLoaded = true;
			this.data = buf;
		},
	};
	mem.addWorker("model.safetensors", worker);

	// weights already unloaded (simulating idle expiry)
	t.false(worker.is_tensors_loaded());
	t.is(worker.metadata.name, "kept");

	await mem.ensureLoaded("model.safetensors");

	t.true(worker.is_tensors_loaded());
	t.is(worker.data, "reloaded-buffer");
	t.is(worker.metadata.name, "kept"); // metadata untouched
});

test("withWorker reloads and calls fn", async (t) => {
	const mem = makeWorkerMemory();
	mem.files.set("model.safetensors", { data: "buffer" });

	// Worker not loaded yet
	t.false(mem.loraWorkers.has("model.safetensors"));

	const result = await mem.withWorker("model.safetensors", (w) => w.data);
	t.is(result, "buffer");
});

test("ensureLoaded throws when file is missing", async (t) => {
	const mem = makeWorkerMemory();
	// No file registered, no worker loaded
	await t.throwsAsync(() => mem.ensureLoaded("missing.safetensors"), {
		message: /Cannot reload worker missing.safetensors/,
	});
});

test("dependency tracking for useEffect", (t) => {
	// Test dependency comparison logic used in useEffect hooks
	const shouldUpdate = (prevDeps, newDeps) => {
		return (
			prevDeps.filename !== newDeps.filename ||
			prevDeps.worker !== newDeps.worker
		);
	};

	t.true(
		shouldUpdate(
			{ filename: "old.safetensors", worker: "w1" },
			{ filename: "new.safetensors", worker: "w1" },
		),
	);

	t.true(
		shouldUpdate(
			{ filename: "test.safetensors", worker: "w1" },
			{ filename: "test.safetensors", worker: "w2" },
		),
	);

	t.false(
		shouldUpdate(
			{ filename: "test.safetensors", worker: "w1" },
			{ filename: "test.safetensors", worker: "w1" },
		),
	);
});
