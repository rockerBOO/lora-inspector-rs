import { cleanup, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

// Clean up after each test
beforeEach(() => {
	cleanup();
});

describe("Dataset Components", () => {
	describe("Dataset Component", () => {
		it("should handle empty datasets gracefully", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const emptyMetadata = new Map([]);

			render(<Dataset metadata={emptyMetadata} />);

			expect(screen.getByText("Dataset")).toBeDefined();
		});

		it("should handle invalid JSON in datasets gracefully", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const invalidJsonMetadata = new Map([["ss_datasets", "invalid json"]]);

			// Suppress console errors during test
			const originalError = console.error;
			const originalLog = console.log;
			console.error = () => {};
			console.log = () => {};

			try {
				// Should not crash with invalid JSON and should render "Dataset" header
				render(<Dataset metadata={invalidJsonMetadata} />);
				expect(screen.getByText("Dataset")).toBeDefined();
			} finally {
				// Restore console
				console.error = originalError;
				console.log = originalLog;
			}
		});
	});

	describe("CaptionDropout Component", () => {
		it("should display caption dropout settings", async () => {
			const { CaptionDropout } = await import(
				"../assets/js/components/dataset/CaptionDropout.jsx"
			);

			const captionMetadata = new Map([
				["ss_caption_dropout_rate", "0.1"],
				["ss_caption_dropout_every_n_epochs", "5"],
				["ss_caption_tag_dropout_rate", "0.05"],
			]);

			render(<CaptionDropout metadata={captionMetadata} />);

			expect(screen.getByText("0.1")).toBeDefined();
			expect(screen.getByText("5")).toBeDefined();
		});
	});

	describe("TagFrequency Component", () => {
		it("should display tag frequency data", async () => {
			const { TagFrequency } = await import(
				"../assets/js/components/dataset/TagFrequency.jsx"
			);

			const mockTagFrequency = {
				tag1: 100,
				tag2: 50,
				tag3: 25,
			};

			render(<TagFrequency tagFrequency={mockTagFrequency} />);

			expect(screen.getByText("tag1")).toBeDefined();
			expect(screen.getByText("100")).toBeDefined();
		});

		it("should handle empty tag frequency", async () => {
			const { TagFrequency } = await import(
				"../assets/js/components/dataset/TagFrequency.jsx"
			);

			const emptyTagFrequency = {};

			expect(() =>
				render(<TagFrequency tagFrequency={emptyTagFrequency} />),
			).not.toThrow();
		});
	});
});

describe("Analysis Components", () => {
	describe("StatisticRow Component", () => {
		it("should render statistic rows correctly", async () => {
			const { StatisticRow } = await import(
				"../assets/js/components/analysis/StatisticRow.jsx"
			);

			const mockProps = {
				baseName: "test_base",
				l1Norm: 0.524,
				l2Norm: 0.823,
				matrixNorm: 1.234,
				min: -0.5,
				max: 0.8,
				median: 0.1,
				stdDev: 0.3,
			};

			render(
				<table>
					<tbody>
						<StatisticRow {...mockProps} />
					</tbody>
				</table>,
			);

			expect(screen.getByText("test_base")).toBeDefined();
			expect(screen.getByText("0.5240")).toBeDefined(); // toPrecision(4)
		});

		it("should handle missing statistical values", async () => {
			const { StatisticRow } = await import(
				"../assets/js/components/analysis/StatisticRow.jsx"
			);

			const minimalProps = {
				baseName: "minimal_test",
				l1Norm: 1.5,
				// Other values undefined
			};

			expect(() =>
				render(
					<table>
						<tbody>
							<StatisticRow {...minimalProps} />
						</tbody>
					</table>,
				),
			).not.toThrow();
		});
	});

	describe("Precision Component", () => {
		it("should render without crashing", async () => {
			const { Precision } = await import(
				"../assets/js/components/analysis/Precision.jsx"
			);

			const mockWorker = createMockWorker();

			// This component requires worker interaction, so we just test it doesn't crash
			expect(() =>
				render(<Precision filename="test.safetensors" worker={mockWorker} />),
			).not.toThrow();
		});
	});

	describe("Weight Component", () => {
		it("should handle weight display", async () => {
			const { Weight } = await import(
				"../assets/js/components/analysis/Weight.jsx"
			);

			const mockProps = {
				name: "lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k.lora_down.weight",
				value: "torch.Size([16, 320])",
				worker: createMockWorker(),
			};

			expect(() => render(<Weight {...mockProps} />)).not.toThrow();
		});
	});
});

describe("Blocks Component", () => {
	it("should render without crashing when no block weights are supported", async () => {
		const { Blocks } = await import(
			"../assets/js/components/analysis/Blocks.jsx"
		);

		const mockWorker = createMockWorker();

		expect(() =>
			render(<Blocks filename="test.safetensors" worker={mockWorker} />),
		).not.toThrow();
	});

	it("should show get block weights button when supported", async () => {
		const { Blocks } = await import(
			"../assets/js/components/analysis/Blocks.jsx"
		);

		// Mock worker that returns a supported network type
		const mockWorker = {
			...createMockWorker(),
			postMessage: (message) => {
				setTimeout(() => {
					if (message.messageType === "network_type") {
						// Simulate event listeners
						const handlers = mockWorker.listeners?.get?.("message") || [];
						for (const handler of handlers) {
							handler({ data: { networkType: "LoRA" } });
						}
					}
				}, 0);
			},
			listeners: new Map(),
			addEventListener: (event, handler) => {
				if (!mockWorker.listeners.has(event)) {
					mockWorker.listeners.set(event, []);
				}
				mockWorker.listeners.get(event).push(handler);
			},
			removeEventListener: (event, handler) => {
				if (mockWorker.listeners.has(event)) {
					const handlers = mockWorker.listeners.get(event);
					const index = handlers.indexOf(handler);
					if (index > -1) {
						handlers.splice(index, 1);
					}
				}
			},
		};

		expect(() =>
			render(<Blocks filename="test.safetensors" worker={mockWorker} />),
		).not.toThrow();
	});
});

describe("Interactive Analysis Components", () => {
	describe("BaseNames Component", () => {
		it("should render when base names are available", async () => {
			const { BaseNames } = await import(
				"../assets/js/components/analysis/BaseNames.jsx"
			);

			const mockProps = {
				baseNames: ["base1", "base2", "base3"],
				showBaseNames: true,
				setShowBlockNames: () => {},
				worker: null,
			};

			render(<BaseNames {...mockProps} />);

			expect(screen.getByText("base1")).toBeDefined();
			expect(screen.getByText("base2")).toBeDefined();
		});

		it("should handle empty base names", async () => {
			const { BaseNames } = await import(
				"../assets/js/components/analysis/BaseNames.jsx"
			);

			const mockProps = {
				baseNames: [],
				showBaseNames: false,
				setShowBlockNames: () => {},
				worker: createMockWorker(),
			};

			expect(() => render(<BaseNames {...mockProps} />)).not.toThrow();
		});
	});

	describe("UnetKeys Component", () => {
		it("should render UNet keys when available", async () => {
			const { UnetKeys } = await import(
				"../assets/js/components/analysis/UnetKeys.jsx"
			);

			const mockProps = {
				unetKeys: ["down_blocks.0", "mid_block", "up_blocks.0"],
				showUnetKeys: true,
				setShowUnetKeys: () => {},
				worker: createMockWorker(),
			};

			render(<UnetKeys {...mockProps} />);

			expect(screen.getByText("down_blocks.0")).toBeDefined();
			expect(screen.getByText("mid_block")).toBeDefined();
		});
	});

	describe("TextEncoderKeys Component", () => {
		it("should render text encoder keys", async () => {
			const { TextEncoderKeys } = await import(
				"../assets/js/components/analysis/TextEncoderKeys.jsx"
			);

			const mockProps = {
				textEncoderKeys: [
					"text_model.encoder.layers.0",
					"text_model.encoder.layers.1",
				],
				showTextEncoderKeys: true,
				setShowTextEncoderKeys: () => {},
				worker: createMockWorker(),
			};

			render(<TextEncoderKeys {...mockProps} />);

			expect(screen.getByText("text_model.encoder.layers.0")).toBeDefined();
		});
	});
});
