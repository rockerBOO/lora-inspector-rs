const workers = new Map();

export async function addWorker(file) {
	const worker = new Worker(new URL("./worker.js", import.meta.url), {
		type: "module",
	});

	workers.set(file, worker);

	return new Promise((resolve, reject) => {
		const timeouts = [];
		const worker = workers.get(file);

		worker.onmessage = (event) => {
			timeouts.map((timeout) => clearTimeout(timeout));
			worker.onmessage = undefined;
			resolve(worker);
		};

		function checkIfAvailable() {
			worker.postMessage({ messageType: "is_available", reply: true });
			const timeout = setTimeout(() => {
				checkIfAvailable();
			}, 200);

			timeouts.push(timeout);
		}

		checkIfAvailable();
	});
}

export function getWorker(file) {
	return workers.get(file);
}

export function removeWorker(file) {
	const worker = workers.get(file);
	worker.terminate();
	return workers.delete(file);
}

export function clearWorkers() {
	for (const key of Array.from(workers.keys())) {
		removeWorker(key);
	}
}

// if we are processing the uploaded file
// we want to be able to terminate the worker if we are still working on a previous file
// in the current implementation
export function terminatePreviousProcessing(file, processingMetadata) {
	const worker = getWorker(file);
	if (processingMetadata) {
		// restart the worker
		worker.terminate();
		// make a new worker
		removeWorker(worker);
		addWorker(file);
	}

	return false;
}

export function cancelLoading(file, processingMetadata, uploadTimeoutHandler) {
	terminatePreviousProcessing(file, processingMetadata);
	finishLoading();
	clearTimeout(uploadTimeoutHandler);
}
