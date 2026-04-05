import { cleanup, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

// Clean up after each test
beforeEach(() => {
	cleanup();
});

describe("Training Components", () => {
	describe("Optimizer Component", () => {
		it("should display optimizer configuration", async () => {
			const { Optimizer } = await import(
				"../assets/js/components/training/Optimizer.jsx"
			);

			const optimizerMetadata = new Map([["ss_optimizer", "AdamW8bit"]]);

			render(<Optimizer metadata={optimizerMetadata} />);

			expect(screen.getByText("AdamW8bit")).toBeDefined();
		});

		it("should handle missing optimizer data", async () => {
			const { Optimizer } = await import(
				"../assets/js/components/training/Optimizer.jsx"
			);

			const minimalMetadata = new Map([["ss_optimizer", "AdamW"]]);

			render(<Optimizer metadata={minimalMetadata} />);

			expect(screen.getByText("AdamW")).toBeDefined();
		});
	});

	describe("LRScheduler Component", () => {
		it("should display learning rate scheduler information", async () => {
			const { LRScheduler } = await import(
				"../assets/js/components/training/LRScheduler.jsx"
			);

			const schedulerMetadata = new Map([
				["ss_lr_scheduler", "cosine_with_restarts"],
				["ss_lr_warmup_steps", "100"],
				["ss_num_cycles", "1"],
			]);

			render(<LRScheduler metadata={schedulerMetadata} />);

			expect(screen.getByText("cosine_with_restarts")).toBeDefined();
			expect(screen.getByText("100")).toBeDefined();
		});

		it("shows global learning rate when both specific LRs absent", async () => {
			const { LRScheduler } = await import(
				"../assets/js/components/training/LRScheduler.jsx"
			);

			const metadata = new Map([
				["ss_lr_scheduler", "constant"],
				["ss_learning_rate", "0.0001"],
			]);

			render(<LRScheduler metadata={metadata} />);

			expect(screen.getByText("0.0001")).toBeDefined();
		});

		it("hides global learning rate when unet_lr is set", async () => {
			const { LRScheduler } = await import(
				"../assets/js/components/training/LRScheduler.jsx"
			);

			const metadata = new Map([
				["ss_lr_scheduler", "constant"],
				["ss_learning_rate", "0.0001"],
				["ss_unet_lr", "1e-4"],
				["ss_text_encoder_lr", "5e-5"],
			]);

			render(<LRScheduler metadata={metadata} />);

			expect(screen.queryByText("0.0001")).toBeNull();
		});

		it("shows global learning rate when both specific LRs are None", async () => {
			const { LRScheduler } = await import(
				"../assets/js/components/training/LRScheduler.jsx"
			);

			const metadata = new Map([
				["ss_lr_scheduler", "constant"],
				["ss_learning_rate", "1.0"],
				["ss_unet_lr", "None"],
				["ss_text_encoder_lr", "None"],
			]);

			render(<LRScheduler metadata={metadata} />);

			expect(screen.getByText("1.0")).toBeDefined();
		});

		it("hides global learning rate when only one specific LR is set", async () => {
			const { LRScheduler } = await import(
				"../assets/js/components/training/LRScheduler.jsx"
			);

			const metadata = new Map([
				["ss_lr_scheduler", "constant"],
				["ss_learning_rate", "0.0001"],
				["ss_unet_lr", "1e-4"],
				["ss_text_encoder_lr", "None"],
			]);

			render(<LRScheduler metadata={metadata} />);

			expect(screen.queryByText("0.0001")).toBeNull();
		});
	});

	describe("Batch Component", () => {
		it("should display batch configuration", async () => {
			const { Batch } = await import(
				"../assets/js/components/training/Batch.jsx"
			);

			const batchMetadata = new Map([
				["ss_batch_size_per_device", "4"],
				["ss_gradient_accumulation_steps", "2"],
				["ss_total_batch_size", "8"],
			]);

			render(<Batch metadata={batchMetadata} />);

			expect(screen.getByText("4")).toBeDefined();
			expect(screen.getByText("2")).toBeDefined();
		});

		it("should calculate batch size from datasets when direct value missing", async () => {
			const { Batch } = await import(
				"../assets/js/components/training/Batch.jsx"
			);

			const datasetBatchMetadata = new Map([
				["ss_datasets", '[{"batch_size_per_device": 6, "name": "dataset1"}]'],
				["ss_gradient_accumulation_steps", "1"],
			]);

			expect(() =>
				render(<Batch metadata={datasetBatchMetadata} />),
			).not.toThrow();
		});
	});

	describe("EpochStep Component", () => {
		it("should display epoch and step information", async () => {
			const { EpochStep } = await import(
				"../assets/js/components/training/EpochStep.jsx"
			);

			const epochMetadata = new Map([
				["ss_epoch", "10"],
				["ss_max_train_steps", "1000"],
				["ss_num_train_epochs", "10"],
			]);

			render(<EpochStep metadata={epochMetadata} />);

			expect(screen.getByText("10")).toBeDefined();
			expect(screen.getByText("1000")).toBeDefined();
		});

		it("should display seed", async () => {
			const { EpochStep } = await import(
				"../assets/js/components/training/EpochStep.jsx"
			);

			const metadata = new Map([
				["ss_epoch", "1"],
				["ss_seed", "42"],
			]);

			render(<EpochStep metadata={metadata} />);

			expect(screen.getByText("42")).toBeDefined();
		});

		it("should display gradient checkpointing when present", async () => {
			const { EpochStep } = await import(
				"../assets/js/components/training/EpochStep.jsx"
			);

			const metadata = new Map([
				["ss_epoch", "1"],
				["ss_gradient_checkpointing", "True"],
			]);

			render(<EpochStep metadata={metadata} />);

			expect(screen.getByText("True")).toBeDefined();
		});

		it("should not show gradient checkpointing when absent", async () => {
			const { EpochStep } = await import(
				"../assets/js/components/training/EpochStep.jsx"
			);

			const metadata = new Map([["ss_epoch", "1"]]);

			render(<EpochStep metadata={metadata} />);

			expect(screen.queryByText("Gradient Checkpointing")).toBeNull();
		});
	});

	describe("Noise Component", () => {
		it("should display noise configuration", async () => {
			const { Noise } = await import(
				"../assets/js/components/training/Noise.jsx"
			);

			const noiseMetadata = new Map([
				["ss_noise_offset", "0.1"],
				["ss_adaptive_noise_scale", "0.00357"],
				["ss_noise_offset_random_strength", "0.02"],
			]);

			render(<Noise metadata={noiseMetadata} />);

			expect(screen.getByText("0.1")).toBeDefined();
			expect(screen.getByText("0.00357")).toBeDefined();
		});

		it("should handle missing noise settings", async () => {
			const { Noise } = await import(
				"../assets/js/components/training/Noise.jsx"
			);

			const noNoiseMetadata = new Map([["ss_learning_rate", "0.0001"]]);

			expect(() => render(<Noise metadata={noNoiseMetadata} />)).not.toThrow();
		});
	});

	describe("Loss Component", () => {
		it("should display loss function information", async () => {
			const { Loss } = await import(
				"../assets/js/components/training/Loss.jsx"
			);

			const lossMetadata = new Map([["ss_loss_type", "l2"]]);

			render(<Loss metadata={lossMetadata} />);

			expect(screen.getByText("l2")).toBeDefined();
		});

		it("should display huber loss configuration when type is huber", async () => {
			const { Loss } = await import(
				"../assets/js/components/training/Loss.jsx"
			);

			const huberLossMetadata = new Map([
				["ss_loss_type", "huber"],
				["ss_huber_schedule", "snr"],
				["ss_huber_c", "0.1"],
			]);

			render(<Loss metadata={huberLossMetadata} />);

			expect(screen.getByText("huber")).toBeDefined();
			expect(screen.getByText("snr")).toBeDefined();
			expect(screen.getByText("0.1")).toBeDefined();
		});
	});

	describe("Experimental Component", () => {
		it("returns null when no experimental keys present", async () => {
			const { Experimental } = await import(
				"../assets/js/components/training/Experimental.jsx"
			);

			const metadata = new Map([["ss_loss_type", "l2"]]);
			const { container } = render(<Experimental metadata={metadata} />);

			expect(container.firstChild).toBeNull();
		});

		it("shows noise offset group when ss_noise_offset present", async () => {
			const { Experimental } = await import(
				"../assets/js/components/training/Experimental.jsx"
			);

			const metadata = new Map([["ss_noise_offset", "0.1"]]);
			render(<Experimental metadata={metadata} />);

			expect(screen.getByText("Noise offset")).toBeDefined();
			expect(screen.getByText("0.1")).toBeDefined();
		});

		it("shows multires noise group when ss_multires_noise_iterations present", async () => {
			const { Experimental } = await import(
				"../assets/js/components/training/Experimental.jsx"
			);

			const metadata = new Map([["ss_multires_noise_iterations", "6"]]);
			render(<Experimental metadata={metadata} />);

			expect(screen.getByText("MultiRes noise")).toBeDefined();
			expect(screen.getByText("6")).toBeDefined();
		});

		it("shows SNR group when ss_min_snr_gamma present", async () => {
			const { Experimental } = await import(
				"../assets/js/components/training/Experimental.jsx"
			);

			const metadata = new Map([["ss_min_snr_gamma", "5"]]);
			render(<Experimental metadata={metadata} />);

			expect(screen.getByText("SNR")).toBeDefined();
			expect(screen.getByText("5")).toBeDefined();
		});
	});

	describe("WaveletLoss Component", () => {
		it("should display wavelet loss configuration when enabled", async () => {
			const { WaveletLoss } = await import(
				"../assets/js/components/training/WaveletLoss.jsx"
			);

			const waveletMetadata = new Map([
				["ss_wavelet_loss", "True"],
				["ss_wavelet_loss_alpha", "0.1"],
			]);

			render(<WaveletLoss metadata={waveletMetadata} />);

			expect(screen.getByText("0.1")).toBeDefined();
		});

		it("should handle disabled wavelet configuration", async () => {
			const { WaveletLoss } = await import(
				"../assets/js/components/training/WaveletLoss.jsx"
			);

			const noWaveletMetadata = new Map([
				["ss_wavelet_loss", "False"],
				["ss_loss_type", "l2"],
			]);

			expect(() =>
				render(<WaveletLoss metadata={noWaveletMetadata} />),
			).not.toThrow();
		});

		it("should handle missing wavelet configuration", async () => {
			const { WaveletLoss } = await import(
				"../assets/js/components/training/WaveletLoss.jsx"
			);

			const noWaveletMetadata = new Map([["ss_loss_type", "l2"]]);

			expect(() =>
				render(<WaveletLoss metadata={noWaveletMetadata} />),
			).not.toThrow();
		});
	});
});
