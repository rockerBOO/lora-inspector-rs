import { getWorker } from "./workers";

export async function trySyncMessage(message, worker, matches = []) {
	if (!worker) {
		throw new Error("Invalid worker");
	}

	return new Promise((resolve) => {
		worker.postMessage({ ...message, reply: true });

		const workerHandler = (e) => {
			if (matches.length > 0) {
				const hasMatches =
					matches.filter((match) => e.data[match] === message[match]).length ===
					matches.length;
				if (hasMatches) {
					worker.removeEventListener("message", workerHandler);
					resolve(e.data);
				}
			} else if (e.data.messageType === message.messageType) {
				worker.removeEventListener("message", workerHandler);
				resolve(e.data);
			}
		};

		worker.addEventListener("message", workerHandler);
	});
}

export async function listenProgress(messageType, file) {
	const worker = getWorker(file);
	let isFinished = false;
	function finishedWorkerHandler(e) {
		if (e.data.messageType === `${messageType}_finished`) {
			worker.removeEventListener("message", finishedWorkerHandler);
			isFinished = true;
		}
	}

	worker.addEventListener("message", finishedWorkerHandler);

	return async function* listen() {
		if (isFinished) {
			return;
		}

		yield await new Promise((resolve) => {
			function workerHandler(e) {
				if (e.data.messageType === messageType) {
					worker.removeEventListener("message", workerHandler);
					resolve(e.data);
				}
			}

			worker.addEventListener("message", workerHandler);
		});
	};
}
