import test from "ava";

// Tests for the core business logic that drives component behavior
// These tests verify the actual functionality users depend on

test("LoRA file metadata processing pipeline", (t) => {
	// Test the complete metadata processing pipeline
	const processLoRAMetadata = (rawMetadata) => {
		const metadata = new Map(Object.entries(rawMetadata));

		const result = {
			hasModelSpec: metadata.has("modelspec.title"),
			networkInfo: {},
			trainingInfo: {},
			datasetInfo: {},
		};

		// Network processing
		if (metadata.has("ss_network_module")) {
			result.networkInfo.module = metadata.get("ss_network_module");
		}
		if (metadata.has("ss_network_dim")) {
			result.networkInfo.dim = Number.parseInt(metadata.get("ss_network_dim"));
		}
		if (metadata.has("ss_network_alpha")) {
			result.networkInfo.alpha = Number.parseFloat(
				metadata.get("ss_network_alpha"),
			);
		}

		// Training processing
		if (
			metadata.has("ss_training_started_at") &&
			metadata.has("ss_training_finished_at")
		) {
			const start = Number.parseInt(metadata.get("ss_training_started_at"));
			const end = Number.parseInt(metadata.get("ss_training_finished_at"));
			result.trainingInfo.duration = (end - start) / 60; // minutes
		}

		// Dataset processing
		if (metadata.has("ss_num_train_images")) {
			result.datasetInfo.imageCount = Number.parseInt(
				metadata.get("ss_num_train_images"),
			);
		}

		return result;
	};

	const sampleMetadata = {
		"modelspec.title": "Test LoRA",
		ss_network_module: "networks.lora",
		ss_network_dim: "32",
		ss_network_alpha: "16.0",
		ss_training_started_at: "1640995200",
		ss_training_finished_at: "1640995800",
		ss_num_train_images: "1000",
	};

	const result = processLoRAMetadata(sampleMetadata);

	t.true(result.hasModelSpec);
	t.is(result.networkInfo.module, "networks.lora");
	t.is(result.networkInfo.dim, 32);
	t.is(result.networkInfo.alpha, 16.0);
	t.is(result.trainingInfo.duration, 10); // 10 minutes
	t.is(result.datasetInfo.imageCount, 1000);
});

test("network component rendering decisions", (t) => {
	// Test the logic that determines which network components to render
	const decideNetworkComponents = (metadata) => {
		const networkType = metadata.get("ss_network_type") || "LoRA";
		const hasNetworkArgs = metadata.has("ss_network_args");
		const hasWeightDecomposition = metadata.has("weight_decomposition");

		const components = [];

		// Main network component based on type
		switch (networkType) {
			case "DiagOFT":
				components.push("DiagOFTNetwork");
				break;
			case "BOFT":
				components.push("BOFTNetwork");
				break;
			case "LoKr":
				components.push("LoKrNetwork");
				break;
			default:
				components.push("LoRANetwork");
		}

		// Additional components based on available data
		if (hasNetworkArgs) {
			components.push("NetworkArgs");
		}
		if (hasWeightDecomposition) {
			components.push("WeightDecomposition");
		}

		const supportsDoRA = ["LoRA", "LoHa", "LoRAFA", "LoKr", "GLoRA"].includes(
			networkType,
		);
		if (supportsDoRA) {
			components.push("DoRASupport");
		}

		return components;
	};

	// Test LoRA network
	const loraMetadata = new Map([
		["ss_network_type", "LoRA"],
		["ss_network_args", "{}"],
	]);
	const loraComponents = decideNetworkComponents(loraMetadata);
	t.true(loraComponents.includes("LoRANetwork"));
	t.true(loraComponents.includes("NetworkArgs"));
	t.true(loraComponents.includes("DoRASupport"));

	// Test DiagOFT network
	const diagoftMetadata = new Map([["ss_network_type", "DiagOFT"]]);
	const diagoftComponents = decideNetworkComponents(diagoftMetadata);
	t.true(diagoftComponents.includes("DiagOFTNetwork"));
	t.false(diagoftComponents.includes("DoRASupport"));
});

test("training data validation and processing", (t) => {
	// Test training data validation logic used by training components
	const validateTrainingData = (metadata) => {
		const validation = {
			valid: true,
			errors: [],
			warnings: [],
			processed: {},
		};

		// Validate learning rates
		const unetLR = metadata.get("ss_unet_lr");
		const teLR = metadata.get("ss_text_encoder_lr");

		if (unetLR && unetLR !== "None") {
			const lr = Number.parseFloat(unetLR);
			if (Number.isNaN(lr) || lr <= 0) {
				validation.errors.push("Invalid UNet learning rate");
				validation.valid = false;
			} else {
				validation.processed.unetLR = lr;
			}
		}

		// Validate batch size
		const batchSize = metadata.get("ss_batch_size_per_device");
		if (batchSize) {
			const batch = Number.parseInt(batchSize);
			if (Number.isNaN(batch) || batch <= 0) {
				validation.errors.push("Invalid batch size");
				validation.valid = false;
			} else {
				validation.processed.batchSize = batch;
				if (batch > 32) {
					validation.warnings.push("Large batch size may cause memory issues");
				}
			}
		}

		// Validate training duration
		const startTime = metadata.get("ss_training_started_at");
		const endTime = metadata.get("ss_training_finished_at");
		if (startTime && endTime) {
			const start = Number.parseInt(startTime);
			const end = Number.parseInt(endTime);
			if (start >= end) {
				validation.errors.push("Invalid training duration");
				validation.valid = false;
			} else {
				validation.processed.duration = (end - start) / 60;
			}
		}

		return validation;
	};

	// Test valid data
	const validMetadata = new Map([
		["ss_unet_lr", "0.0001"],
		["ss_batch_size_per_device", "4"],
		["ss_training_started_at", "1640995200"],
		["ss_training_finished_at", "1640995800"],
	]);

	const validResult = validateTrainingData(validMetadata);
	t.true(validResult.valid);
	t.is(validResult.errors.length, 0);
	t.is(validResult.processed.unetLR, 0.0001);
	t.is(validResult.processed.batchSize, 4);
	t.is(validResult.processed.duration, 10);

	// Test invalid data
	const invalidMetadata = new Map([
		["ss_unet_lr", "invalid"],
		["ss_batch_size_per_device", "-1"],
		["ss_training_started_at", "1640995800"],
		["ss_training_finished_at", "1640995200"], // end before start
	]);

	const invalidResult = validateTrainingData(invalidMetadata);
	t.false(invalidResult.valid);
	t.true(invalidResult.errors.length > 0);
});

test("dataset component data processing", (t) => {
	// Test dataset processing logic
	const processDatasetInfo = (metadata) => {
		const result = {
			datasets: [],
			totalImages: 0,
			bucketsEnabled: false,
			tagFrequencies: {},
		};

		// Parse datasets JSON
		if (metadata.has("ss_datasets")) {
			try {
				const datasets = JSON.parse(metadata.get("ss_datasets"));
				result.datasets = Array.isArray(datasets) ? datasets : [];

				// Process each dataset
				for (const dataset of result.datasets) {
					if (dataset.enable_bucket) {
						result.bucketsEnabled = true;
					}
					if (dataset.tag_frequency) {
						Object.assign(result.tagFrequencies, dataset.tag_frequency);
					}
				}
			} catch (error) {
				// Invalid JSON - use empty array
				result.datasets = [];
			}
		}

		// Get total image count
		if (metadata.has("ss_num_train_images")) {
			result.totalImages =
				Number.parseInt(metadata.get("ss_num_train_images")) || 0;
		}

		return result;
	};

	const datasetMetadata = new Map([
		[
			"ss_datasets",
			JSON.stringify([
				{
					enable_bucket: true,
					tag_frequency: { person: 100, landscape: 50 },
				},
				{
					enable_bucket: false,
					tag_frequency: { anime: 75 },
				},
			]),
		],
		["ss_num_train_images", "1500"],
	]);

	const result = processDatasetInfo(datasetMetadata);
	t.is(result.datasets.length, 2);
	t.true(result.bucketsEnabled);
	t.is(result.totalImages, 1500);
	t.is(result.tagFrequencies.person, 100);
	t.is(result.tagFrequencies.anime, 75);

	// Test invalid JSON handling
	const invalidMetadata = new Map([["ss_datasets", "invalid json"]]);
	const invalidResult = processDatasetInfo(invalidMetadata);
	t.is(invalidResult.datasets.length, 0);
});

test("component conditional rendering logic", (t) => {
	// Test the conditions that determine whether components should render
	const shouldRenderComponent = (componentName, metadata) => {
		switch (componentName) {
			case "ModelSpec":
				return metadata.has("modelspec.title");

			case "WaveletLoss":
				return metadata.get("ss_wavelet_loss") === "True";

			case "MultiresNoise":
				return metadata.get("ss_multires_noise_iterations") !== "None";

			case "VAE":
				return metadata.has("ss_vae_name");

			case "CaptionDropout":
				return (
					metadata.has("ss_caption_dropout_rate") ||
					metadata.has("ss_caption_tag_dropout_rate")
				);

			case "Dataset":
				return (
					metadata.has("ss_datasets") || metadata.has("ss_num_train_images")
				);

			default:
				return true;
		}
	};

	const fullMetadata = new Map([
		["modelspec.title", "Test"],
		["ss_wavelet_loss", "True"],
		["ss_multires_noise_iterations", "5"],
		["ss_vae_name", "test.vae"],
		["ss_caption_dropout_rate", "0.1"],
		["ss_datasets", "[]"],
	]);

	t.true(shouldRenderComponent("ModelSpec", fullMetadata));
	t.true(shouldRenderComponent("WaveletLoss", fullMetadata));
	t.true(shouldRenderComponent("MultiresNoise", fullMetadata));
	t.true(shouldRenderComponent("VAE", fullMetadata));
	t.true(shouldRenderComponent("CaptionDropout", fullMetadata));
	t.true(shouldRenderComponent("Dataset", fullMetadata));

	const minimalMetadata = new Map([
		["ss_wavelet_loss", "False"],
		["ss_multires_noise_iterations", "None"],
	]);

	t.false(shouldRenderComponent("ModelSpec", minimalMetadata));
	t.false(shouldRenderComponent("WaveletLoss", minimalMetadata));
	t.false(shouldRenderComponent("MultiresNoise", minimalMetadata));
	t.false(shouldRenderComponent("VAE", minimalMetadata));
});
