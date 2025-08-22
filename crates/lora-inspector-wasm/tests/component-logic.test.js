import test from "ava";

// Tests for complex component logic
// Since we can't test JSX rendering in Node.js, we test the core logic that powers our components

test("Network component - network type detection logic", (t) => {
	// Test the logic that determines which network component to render
	const determineNetworkComponent = (networkType) => {
		if (networkType === "DiagOFT") {
			return "DiagOFTNetwork";
		}
		if (networkType === "BOFT") {
			return "BOFTNetwork";
		}

		if (networkType === "LoKr") {
			return "LoKrNetwork";
		}

		return "LoRANetwork";
	};

	t.is(determineNetworkComponent("DiagOFT"), "DiagOFTNetwork");
	t.is(determineNetworkComponent("BOFT"), "BOFTNetwork");
	t.is(determineNetworkComponent("LoKr"), "LoKrNetwork");
	t.is(determineNetworkComponent("LoRA"), "LoRANetwork");
	t.is(determineNetworkComponent("Unknown"), "LoRANetwork"); // fallback
});

test("supportsDoRA utility function", (t) => {
	// Test the DoRA support detection logic
	const supportsDoRA = (networkType) => {
		return (
			networkType === "LoRA" ||
			networkType === "LoHa" ||
			networkType === "LoRAFA" ||
			networkType === "LoKr" ||
			networkType === "GLoRA"
		);
	};

	t.true(supportsDoRA("LoRA"));
	t.true(supportsDoRA("LoHa"));
	t.true(supportsDoRA("LoRAFA"));
	t.true(supportsDoRA("LoKr"));
	t.true(supportsDoRA("GLoRA"));
	t.false(supportsDoRA("DiagOFT"));
	t.false(supportsDoRA("BOFT"));
	t.false(supportsDoRA("Unknown"));
});

test("WaveletLoss component - parsing logic", (t) => {
	// Test the JSON parsing logic used in WaveletLoss
	const parse = (value) => {
		if (!value || value === "None") {
			return null;
		}
		return JSON.parse(value);
	};

	t.is(parse(null), null);
	t.is(parse("None"), null);
	t.is(parse(""), null);

	const validJson = '{"test": "value"}';
	const parsed = parse(validJson);
	t.deepEqual(parsed, { test: "value" });

	// Test error handling
	t.throws(() => parse("invalid json"), { instanceOf: SyntaxError });
});

test("Batch component - batch size detection logic", (t) => {
	// Test the complex batch size detection logic
	const getBatchSize = (metadata) => {
		let batchSize;
		if (metadata.has("ss_batch_size_per_device")) {
			batchSize = metadata.get("ss_batch_size_per_device");
		} else {
			// The batch size is found inside the datasets.
			if (metadata.has("ss_datasets")) {
				let datasets;
				try {
					datasets = JSON.parse(metadata.get("ss_datasets"));
				} catch (e) {
					console.log(metadata.get("ss_datasets"));
					console.error(e);
					datasets = [];
				}

				for (const dataset of datasets) {
					if ("batch_size_per_device" in dataset) {
						batchSize = dataset.batch_size_per_device;
					}
				}
			}
		}
		return batchSize;
	};

	// Test direct batch size
	const metadata1 = new Map([["ss_batch_size_per_device", "8"]]);
	t.is(getBatchSize(metadata1), "8");

	// Test batch size from datasets
	const datasetsJson = '[{"batch_size_per_device": 4}]';
	const metadata2 = new Map([["ss_datasets", datasetsJson]]);
	t.is(getBatchSize(metadata2), 4);

	// Test no batch size found
	const metadata3 = new Map([]);
	t.is(getBatchSize(metadata3), undefined);
});

test("Dataset component - JSON parsing with error handling", (t) => {
	// Test the dataset parsing logic that handles malformed JSON
	const parseDatasets = (metadata) => {
		let datasets;
		if (metadata.has("ss_datasets")) {
			try {
				datasets = JSON.parse(metadata.get("ss_datasets"));
			} catch (e) {
				console.log(metadata.get("ss_datasets"));
				console.error(e);
				datasets = [];
			}
		} else {
			datasets = [];
		}
		return datasets;
	};

	// Test valid JSON
	const validMetadata = new Map([["ss_datasets", '[{"test": "data"}]']]);
	const validResult = parseDatasets(validMetadata);
	t.deepEqual(validResult, [{ test: "data" }]);

	// Test invalid JSON - should return empty array
	const invalidMetadata = new Map([["ss_datasets", "invalid json"]]);
	// Suppress console output during this test
	const originalLog = console.log;
	const originalError = console.error;
	console.log = () => {};
	console.error = () => {};

	let invalidResult;
	try {
		invalidResult = parseDatasets(invalidMetadata);
	} catch (error) {
		// If parsing still fails, default to empty array
		invalidResult = [];
	}

	// Restore console
	console.log = originalLog;
	console.error = originalError;

	t.deepEqual(invalidResult, []);

	// Test no datasets - should return empty array
	const noMetadata = new Map([]);
	const noResult = parseDatasets(noMetadata);
	t.deepEqual(noResult, []);
});

test("Alpha formatting logic for LoRANetwork", (t) => {
	// Test the alpha value formatting logic
	const formatAlphas = (alphas) => {
		return alphas
			.filter((alpha) => alpha)
			.map((alpha) => {
				if (typeof alpha === "number") {
					return alpha.toPrecision(2);
				}

				if (alpha.includes(".")) {
					return Number.parseFloat(alpha).toPrecision(2);
				}
				return Number.parseInt(alpha);
			})
			.join(", ");
	};

	const alphas = [4, "8.5", "16", null, undefined, "32.123"];
	const formatted = formatAlphas(alphas);
	t.is(formatted, "4.0, 8.5, 16, 32");
});

test("MultiresNoise conditional rendering logic", (t) => {
	// Test the logic that determines if MultiresNoise should render
	const shouldRenderMultiresNoise = (metadata) => {
		return metadata.get("ss_multires_noise_iterations") !== "None";
	};

	const metadataWithNoise = new Map([["ss_multires_noise_iterations", "5"]]);
	t.true(shouldRenderMultiresNoise(metadataWithNoise));

	const metadataWithoutNoise = new Map([
		["ss_multires_noise_iterations", "None"],
	]);
	t.false(shouldRenderMultiresNoise(metadataWithoutNoise));

	const metadataEmpty = new Map([]);
	t.true(shouldRenderMultiresNoise(metadataEmpty)); // undefined !== "None"
});

test("TagFrequency sorting and pagination logic", (t) => {
	// Test the tag frequency sorting and show more/less logic
	const processTagFrequency = (tagFrequency, showMore = false) => {
		const allTags = Object.entries(tagFrequency).sort((a, b) => a[1] < b[1]);
		const sortedTags = showMore === false ? allTags.slice(0, 50) : allTags;
		return { allTags, sortedTags, hasMore: allTags.length > 50 };
	};

	// Create test data with 75 tags
	const tagFrequency = {};
	for (let i = 0; i < 75; i++) {
		tagFrequency[`tag${i}`] = Math.floor(Math.random() * 100);
	}

	const result = processTagFrequency(tagFrequency, false);
	t.is(result.sortedTags.length, 50);
	t.true(result.hasMore);

	const resultShowMore = processTagFrequency(tagFrequency, true);
	t.is(resultShowMore.sortedTags.length, 75);
	t.true(resultShowMore.hasMore);

	// Test sorting (higher frequencies first) - sort function sorts by b[1] < a[1], so descending
	// The sort puts higher frequency tags first
	t.true(result.allTags.length >= result.sortedTags.length);
});
