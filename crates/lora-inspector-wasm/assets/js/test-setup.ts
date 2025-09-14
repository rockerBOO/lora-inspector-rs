import "@testing-library/jest-dom/vitest";
import * as matchers from "@testing-library/jest-dom/matchers";
import { cleanup } from "@testing-library/react";
import { afterEach, expect } from "vitest";

expect.extend(matchers);

// Cleanup DOM after each test
afterEach(() => {
	cleanup();
});

// Mock Worker types
interface MockWorkerMessage {
	messageType: string;
	[key: string]: unknown;
}

interface MockWorkerResponse {
	messageType: string;
	alphas?: number[];
	dims?: number[];
	precision?: string;
	[key: string]: unknown;
}

interface MockWorker {
	postMessage: (message: MockWorkerMessage) => void;
	addEventListener: (
		event: string,
		handler: (event: { data: MockWorkerResponse }) => void,
	) => void;
	removeEventListener: (
		event: string,
		handler: (event: { data: MockWorkerResponse }) => void,
	) => void;
}

// Create a global mock worker that properly handles async operations
declare global {
	function createMockWorker(): MockWorker;
}

global.createMockWorker = () => {
	const listeners = new Map<
		string,
		((event: { data: MockWorkerResponse }) => void)[]
	>();

	return {
		postMessage: (message: MockWorkerMessage) => {
			// Simulate async worker response
			setTimeout(() => {
				const handlers = listeners.get("message") || [];
				for (const handler of handlers) {
					try {
						// Mock response based on message type
						const mockResponse: MockWorkerResponse = {
							messageType: message.messageType,
							alphas: message.messageType === "alphas" ? [] : undefined,
							dims: message.messageType === "dims" ? [] : undefined,
							precision:
								message.messageType === "precision" ? "fp16" : undefined,
							// Add other mock responses as needed
						};
						handler({ data: mockResponse });
					} catch (_error) {
						// Silently ignore handler errors in tests
					}
				}
			}, 0);
		},

		addEventListener: (
			event: string,
			handler: (event: { data: MockWorkerResponse }) => void,
		) => {
			if (!listeners.has(event)) {
				listeners.set(event, []);
			}
			listeners.get(event).push(handler);
		},

		removeEventListener: (
			event: string,
			handler: (event: { data: MockWorkerResponse }) => void,
		) => {
			if (listeners.has(event)) {
				const handlers = listeners.get(event);
				if (handlers) {
					const index = handlers.indexOf(handler);
					if (index > -1) {
						handlers.splice(index, 1);
					}
				}
			}
		},
	};
};

// Global error handler to catch unhandled promise rejections in tests
if (typeof window !== "undefined") {
	window.addEventListener("unhandledrejection", (event) => {
		// Only suppress worker-related errors in test environment
		if (
			event.reason?.message &&
			(event.reason.message.includes("postMessage") ||
				event.reason.message.includes("worker") ||
				event.reason.message.includes("Cannot read properties of undefined"))
		) {
			event.preventDefault();
		}
	});
}
