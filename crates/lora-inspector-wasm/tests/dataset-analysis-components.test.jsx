import { cleanup, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

// Clean up after each test
beforeEach(() => {
	cleanup();
});

describe("Dataset Components", () => {
	describe("Dataset Component", () => {
		it("should render without crashing for empty datasets", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const emptyMetadata = new Map([]);

			expect(() => render(<Dataset metadata={emptyMetadata} />)).not.toThrow();
		});

		it("shows joined Directories when all datasets share settings", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const datasets = [
				{
					image_directory: "photos",
					enable_bucket: true,
					resolution: [1024, 1024],
					num_repeats: 1,
				},
				{
					image_directory: "sketches",
					enable_bucket: true,
					resolution: [1024, 1024],
					num_repeats: 1,
				},
			];
			const metadata = new Map([["ss_datasets", JSON.stringify(datasets)]]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("Directories")).toBeDefined();
			expect(screen.getByText("photos, sketches")).toBeDefined();
			// Shared settings shown once
			expect(screen.getAllByText("True").length).toBe(1);
		});

		it("renders each dataset separately when settings differ", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const datasets = [
				{
					image_directory: "hires",
					enable_bucket: true,
					resolution: [1024, 1024],
					num_repeats: 2,
				},
				{
					image_directory: "lores",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
				},
			];
			const metadata = new Map([["ss_datasets", JSON.stringify(datasets)]]);

			render(<Dataset metadata={metadata} />);

			// Falls back to per-dataset rendering — both directories shown as Image directory
			expect(screen.getByText("hires")).toBeDefined();
			expect(screen.getByText("lores")).toBeDefined();
			// Should NOT show the single "Directories" label
			expect(screen.queryByText("Directories")).toBeNull();
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
				expect(() =>
					render(<Dataset metadata={invalidJsonMetadata} />),
				).not.toThrow();
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

	describe("CaptionSettings Component", () => {
		it("imports from CaptionSettings.jsx and displays caption settings", async () => {
			const { CaptionSettings } = await import(
				"../assets/js/components/dataset/CaptionSettings.jsx"
			);

			const captionMetadata = new Map([
				["ss_caption_dropout_rate", "0.2"],
				["ss_caption_dropout_every_n_epochs", "3"],
				["ss_caption_tag_dropout_rate", "0.1"],
			]);

			render(<CaptionSettings metadata={captionMetadata} />);

			expect(screen.getByText("0.2")).toBeDefined();
			expect(screen.getByText("3")).toBeDefined();
		});
	});

	describe("Max token length in ss_datasets path", () => {
		it("renders Max token length when ss_max_token_length present in shared-settings branch", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const datasets = [
				{
					image_directory: "photos",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
				},
				{
					image_directory: "sketches",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
				},
			];
			const metadata = new Map([
				["ss_datasets", JSON.stringify(datasets)],
				["ss_max_token_length", "225"],
			]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("Max token length")).toBeDefined();
			expect(screen.getByText("225")).toBeDefined();
		});

		it("renders Max token length when ss_max_token_length present in per-dataset branch", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const datasets = [
				{
					image_directory: "hires",
					enable_bucket: true,
					resolution: [1024, 1024],
					num_repeats: 2,
				},
				{
					image_directory: "lores",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
				},
			];
			const metadata = new Map([
				["ss_datasets", JSON.stringify(datasets)],
				["ss_max_token_length", "150"],
			]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("Max token length")).toBeDefined();
			expect(screen.getByText("150")).toBeDefined();
		});

		it("does not render Max token length when ss_max_token_length absent", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const datasets = [
				{
					image_directory: "photos",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
				},
			];
			const metadata = new Map([["ss_datasets", JSON.stringify(datasets)]]);

			render(<Dataset metadata={metadata} />);

			expect(screen.queryByText("Max token length")).toBeNull();
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

	describe("ss_tag_frequency top-level metadata key", () => {
		it("renders tag frequencies from ss_tag_frequency (aleriia format)", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			// aleriia_metadata.safetensors format: ss_tag_frequency at top level,
			// no ss_datasets, paired with ss_dataset_dirs/ss_resolution
			const tagFrequency = {
				"2_aleriia v": { "1girl": 60, "breasts": 56, "black hair": 13 },
			};
			const metadata = new Map([
				["ss_resolution", "(512, 512)"],
				["ss_enable_bucket", "True"],
				["ss_tag_frequency", JSON.stringify(tagFrequency)],
			]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("Tag frequencies")).toBeDefined();
			expect(screen.getByText("2_aleriia v")).toBeDefined();
			expect(screen.getByText("1girl")).toBeDefined();
			expect(screen.getByText("60")).toBeDefined();
		});

		it("renders tag frequencies from ss_tag_frequency with multiple dirs (boo format)", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			// boo.safetensors format: multiple directories in ss_tag_frequency
			const tagFrequency = {
				"150_cp-ede": { "1girl": 120, "smile": 80 },
				"8_anime": { "anime style": 8 },
			};
			const metadata = new Map([
				["ss_resolution", "(768, 768)"],
				["ss_enable_bucket", "True"],
				["ss_tag_frequency", JSON.stringify(tagFrequency)],
			]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("150_cp-ede")).toBeDefined();
			expect(screen.getByText("8_anime")).toBeDefined();
			expect(screen.getByText("1girl")).toBeDefined();
			expect(screen.getByText("anime style")).toBeDefined();
		});

		it("still renders tag_frequency from ss_datasets datasets", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			// Original ss_datasets format with tag_frequency inside dataset objects
			const datasets = [
				{
					image_directory: "photos",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
					tag_frequency: {
						photos: { portrait: 40, outdoor: 20 },
					},
				},
			];
			const metadata = new Map([["ss_datasets", JSON.stringify(datasets)]]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("Tag frequencies")).toBeDefined();
			expect(screen.getByText("portrait")).toBeDefined();
		});

		it("renders both ss_datasets tag_frequency and ss_tag_frequency when both present", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const datasets = [
				{
					image_directory: "photos",
					enable_bucket: true,
					resolution: [512, 512],
					num_repeats: 1,
					tag_frequency: { photos: { portrait: 40 } },
				},
			];
			const topLevelTagFreq = {
				"extra_dir": { "extra_tag": 10 },
			};
			const metadata = new Map([
				["ss_datasets", JSON.stringify(datasets)],
				["ss_tag_frequency", JSON.stringify(topLevelTagFreq)],
			]);

			render(<Dataset metadata={metadata} />);

			expect(screen.getByText("portrait")).toBeDefined();
			expect(screen.getByText("extra_tag")).toBeDefined();
		});

		it("does not render tag frequencies section when ss_tag_frequency is absent", async () => {
			const { Dataset } = await import(
				"../assets/js/components/dataset/Dataset.jsx"
			);

			const metadata = new Map([
				["ss_resolution", "(512, 512)"],
				["ss_enable_bucket", "True"],
			]);

			render(<Dataset metadata={metadata} />);

			expect(screen.queryByText("Tag frequencies")).toBeNull();
		});
	});
});

describe("Buckets Component", () => {
	const emptyMetadata = new Map();

	it("renders video directory when video_directory present", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = { video_directory: "/data/videos", enable_bucket: true };
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Video directory")).toBeDefined();
		expect(screen.getByText("/data/videos")).toBeDefined();
	});

	it("renders video fields when target_frames present", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			video_directory: "/data/videos",
			target_frames: [1, 25, 45],
			frame_extraction: "chunk",
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Target frames")).toBeDefined();
		expect(screen.getByText("1, 25, 45")).toBeDefined();
		expect(screen.getByText("Frame extraction")).toBeDefined();
		expect(screen.getByText("chunk")).toBeDefined();
	});

	it("formats target_frames array as comma-separated string", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = { target_frames: [1, 25, 45], enable_bucket: false };
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("1, 25, 45")).toBeDefined();
	});

	it("shows image_jsonl_file source label", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_jsonl_file: "/data/images.jsonl",
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Image JSONL")).toBeDefined();
		expect(screen.getByText("/data/images.jsonl")).toBeDefined();
	});

	it("shows video_jsonl_file source label", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			video_jsonl_file: "/data/videos.jsonl",
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Video JSONL")).toBeDefined();
		expect(screen.getByText("/data/videos.jsonl")).toBeDefined();
	});

	it("renders control fields when control_directory present", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_directory: "/data/images",
			control_directory: "/data/control",
			control_resolution: [1024, 1024],
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Control directory")).toBeDefined();
		expect(screen.getByText("/data/control")).toBeDefined();
		expect(screen.getByText("Control resolution")).toBeDefined();
		expect(screen.getByText("1024, 1024")).toBeDefined();
	});

	it("shows has_control when true and no control directory present", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_directory: "/data/images",
			has_control: true,
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Has control")).toBeDefined();
	});

	it("hides control row when has_control is false and no control fields", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_directory: "/data/images",
			has_control: false,
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.queryByText("Has control")).toBeNull();
	});

	it("renders FramePack group when fp_latent_window_size present", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_directory: "/data/images",
			fp_latent_window_size: 9,
			fp_1f_clean_indices: [0, 1],
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Latent window size")).toBeDefined();
		expect(screen.getByText("9")).toBeDefined();
		expect(screen.getByText("1F clean indices")).toBeDefined();
		expect(screen.getByText("0, 1")).toBeDefined();
	});

	it("does not render video fields for image-only dataset", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_directory: "/data/images",
			enable_bucket: true,
			resolution: [512, 512],
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.queryByText("Target frames")).toBeNull();
		expect(screen.queryByText("Frame extraction")).toBeNull();
	});

	it("shows cache_directory when present", async () => {
		const { Buckets } = await import(
			"../assets/js/components/dataset/Buckets.jsx"
		);

		const dataset = {
			image_directory: "/data/images",
			cache_directory: "/cache/latents",
			enable_bucket: true,
		};
		render(<Buckets dataset={dataset} metadata={emptyMetadata} />);

		expect(screen.getByText("Cache directory")).toBeDefined();
		expect(screen.getByText("/cache/latents")).toBeDefined();
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

describe("Subset Component", () => {
	it("renders image_dir as a heading", async () => {
		const { Subset } = await import(
			"../assets/js/components/dataset/Subset.jsx"
		);

		const subset = { image_dir: "/data/my_images" };
		render(<Subset subset={subset} />);

		expect(screen.getByRole("heading", { level: 4 })).toBeDefined();
		expect(screen.getByText("/data/my_images")).toBeDefined();
	});

	it("renders class_tokens when present", async () => {
		const { Subset } = await import(
			"../assets/js/components/dataset/Subset.jsx"
		);

		const subset = { image_dir: "/data/images", class_tokens: "my_character" };
		render(<Subset subset={subset} />);

		expect(screen.getByText("Class tokens")).toBeDefined();
		expect(screen.getByText("my_character")).toBeDefined();
	});

	it("renders num_repeats, keep_tokens, and is_reg when present", async () => {
		const { Subset } = await import(
			"../assets/js/components/dataset/Subset.jsx"
		);

		const subset = {
			image_dir: "/data/images",
			num_repeats: 10,
			keep_tokens: 2,
			is_reg: true,
		};
		render(<Subset subset={subset} />);

		expect(screen.getByText("Repeats")).toBeDefined();
		expect(screen.getByText("10")).toBeDefined();
		expect(screen.getByText("Keep tokens")).toBeDefined();
		expect(screen.getByText("2")).toBeDefined();
		expect(screen.getByText("Regularization")).toBeDefined();
		expect(screen.getByText("True")).toBeDefined();
	});

	it("does not render rows for absent fields", async () => {
		const { Subset } = await import(
			"../assets/js/components/dataset/Subset.jsx"
		);

		const subset = { image_dir: "/data/images" };
		render(<Subset subset={subset} />);

		expect(screen.queryByText("Class tokens")).toBeNull();
		expect(screen.queryByText("Repeats")).toBeNull();
		expect(screen.queryByText("Keep tokens")).toBeNull();
		expect(screen.queryByText("Regularization")).toBeNull();
		expect(screen.queryByText("Flip aug")).toBeNull();
		expect(screen.queryByText("Color aug")).toBeNull();
	});

	it("renders boolean fields as True/False strings", async () => {
		const { Subset } = await import(
			"../assets/js/components/dataset/Subset.jsx"
		);

		const subset = {
			image_dir: "/data/images",
			flip_aug: false,
			color_aug: true,
			shuffle_caption: false,
			random_crop: true,
			enable_wildcard: false,
		};
		render(<Subset subset={subset} />);

		expect(screen.getByText("Flip aug")).toBeDefined();
		expect(screen.getByText("Color aug")).toBeDefined();
		const trues = screen.getAllByText("True");
		const falses = screen.getAllByText("False");
		expect(trues.length).toBe(2);
		expect(falses.length).toBe(3);
	});

	it("renders face_crop_aug_range as min, max string", async () => {
		const { Subset } = await import(
			"../assets/js/components/dataset/Subset.jsx"
		);

		const subset = {
			image_dir: "/data/images",
			face_crop_aug_range: [1.0, 3.0],
		};
		render(<Subset subset={subset} />);

		expect(screen.getByText("Face crop aug range")).toBeDefined();
		expect(screen.getByText("1, 3")).toBeDefined();
	});
});
