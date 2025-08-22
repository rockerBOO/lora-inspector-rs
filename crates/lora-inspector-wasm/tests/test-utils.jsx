import { render } from "@testing-library/react";

// Wrapper for rendering components that might have worker errors
export function renderWithErrorBoundary(component) {
	// Temporarily suppress console errors during render
	const originalError = console.error;
	console.error = (...args) => {
		// Only suppress worker-related errors
		if (
			args.some(
				(arg) =>
					typeof arg === "string" &&
					(arg.includes("postMessage") ||
						arg.includes("Cannot read properties of undefined") ||
						arg.includes("worker")),
			)
		) {
			return;
		}
		originalError(...args);
	};

	let result;
	try {
		result = render(component);
	} finally {
		// Restore console.error
		console.error = originalError;
	}

	return result;
}

// Enhanced mock worker that handles all message types
export function createComprehensiveMockWorker() {
	const listeners = new Map();
	const messageQueue = [];

	const worker = {
		postMessage: (message) => {
			// Store the message for potential processing
			messageQueue.push(message);

			// Simulate async response with more realistic delay
			setTimeout(() => {
				const handlers = listeners.get("message") || [];
				for (const handler of handlers) {
					try {
						// Create more comprehensive mock responses
						const mockResponse = {
							messageType: message.messageType,
							// Network-related responses
							alphas: message.messageType === "alphas" ? [1.0] : undefined,
							dims: message.messageType === "dims" ? [32] : undefined,
							// Analysis-related responses
							precision:
								message.messageType === "precision" ? "fp16" : undefined,
							// Add any other message types as needed
							reply: true,
						};

						// Only call handler if it's still valid
						if (typeof handler === "function") {
							handler({ data: mockResponse });
						}
					} catch (error) {
						// Silently ignore any handler errors
					}
				}
			}, 1); // Very small delay to simulate async behavior
		},

		addEventListener: (event, handler) => {
			if (typeof handler !== "function") return;

			if (!listeners.has(event)) {
				listeners.set(event, []);
			}
			listeners.get(event).push(handler);
		},

		removeEventListener: (event, handler) => {
			if (listeners.has(event)) {
				const handlers = listeners.get(event);
				const index = handlers.indexOf(handler);
				if (index > -1) {
					handlers.splice(index, 1);
				}
			}
		},

		// Additional worker properties for completeness
		terminate: () => {
			listeners.clear();
			messageQueue.length = 0;
		},
	};

	return worker;
}

// Mock metadata helper
export function createMockMetadata(overrides = {}) {
	const defaultMetadata = new Map([
		["ss_network_module", "networks.lora"],
		["ss_network_rank", "32"],
		["ss_network_alpha", "32.0"],
		["ss_optimizer", "AdamW8bit"],
		["ss_seed", "12345"],
		["ss_loss_type", "l2"],
		["modelspec.title", "Test LoRA Model"],
	]);

	// Apply overrides
	for (const [key, value] of Object.entries(overrides)) {
		defaultMetadata.set(key, value);
	}

	return defaultMetadata;
}
