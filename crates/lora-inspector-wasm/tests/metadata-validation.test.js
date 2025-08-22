import test from "ava";

// Tests for metadata validation and edge cases that components handle

test("metadata Map validation and safety", (t) => {
	// Test safe metadata access patterns used by components
	const safeMetadataGet = (metadata, key, defaultValue = undefined) => {
		if (!metadata || typeof metadata.get !== "function") {
			return defaultValue;
		}
		return metadata.get(key) ?? defaultValue;
	};

	const safeMetadataHas = (metadata, key) => {
		if (!metadata || typeof metadata.has !== "function") {
			return false;
		}
		return metadata.has(key);
	};

	// Test with valid metadata
	const validMetadata = new Map([["ss_network_rank", "4"]]);
	t.is(safeMetadataGet(validMetadata, "ss_network_rank"), "4");
	t.is(safeMetadataGet(validMetadata, "missing_key", "default"), "default");
	t.true(safeMetadataHas(validMetadata, "ss_network_rank"));
	t.false(safeMetadataHas(validMetadata, "missing_key"));

	// Test with null/undefined metadata
	t.is(safeMetadataGet(null, "key", "fallback"), "fallback");
	t.is(safeMetadataGet(undefined, "key", "fallback"), "fallback");
	t.false(safeMetadataHas(null, "key"));
	t.false(safeMetadataHas(undefined, "key"));

	// Test with invalid metadata object
	t.is(safeMetadataGet({}, "key", "fallback"), "fallback");
	t.false(safeMetadataHas({}, "key"));
});

test("numeric metadata validation", (t) => {
	// Test validation for numeric metadata fields
	const validateNumericField = (
		value,
		min = Number.NEGATIVE_INFINITY,
		max = Number.POSITIVE_INFINITY,
	) => {
		if (value === null || value === undefined || value === "None") {
			return null;
		}

		const num = typeof value === "string" ? Number.parseFloat(value) : value;
		if (Number.isNaN(num)) {
			return null;
		}

		return Math.max(min, Math.min(max, num));
	};

	// Test valid numbers
	t.is(validateNumericField("4.5"), 4.5);
	t.is(validateNumericField(8), 8);
	t.is(validateNumericField("16"), 16);

	// Test invalid values
	t.is(validateNumericField("invalid"), null);
	t.is(validateNumericField("None"), null);
	t.is(validateNumericField(null), null);
	t.is(validateNumericField(undefined), null);

	// Test range validation
	t.is(validateNumericField("10", 0, 5), 5); // clamped to max
	t.is(validateNumericField("-5", 0, 10), 0); // clamped to min
});

test("boolean metadata validation", (t) => {
	// Test validation for boolean metadata fields
	const validateBooleanField = (value, defaultValue = false) => {
		if (value === null || value === undefined || value === "None") {
			return defaultValue;
		}

		if (typeof value === "boolean") {
			return value;
		}

		if (typeof value === "string") {
			const lower = value.toLowerCase();
			return lower === "true" || lower === "1" || lower === "yes";
		}

		return Boolean(value);
	};

	// Test boolean values
	t.true(validateBooleanField(true));
	t.false(validateBooleanField(false));

	// Test string values
	t.true(validateBooleanField("true"));
	t.true(validateBooleanField("True"));
	t.true(validateBooleanField("1"));
	t.true(validateBooleanField("yes"));
	t.false(validateBooleanField("false"));
	t.false(validateBooleanField("False"));
	t.false(validateBooleanField("0"));
	t.false(validateBooleanField("no"));

	// Test edge cases
	t.false(validateBooleanField(null));
	t.false(validateBooleanField(undefined));
	t.false(validateBooleanField("None"));
	t.true(validateBooleanField("None", true)); // with default
});

test("date metadata validation", (t) => {
	// Test date validation and formatting
	const validateDateField = (value) => {
		if (!value || value === "None") {
			return null;
		}

		// Handle Unix timestamp
		if (typeof value === "number" || /^\d+$/.test(value)) {
			const timestamp =
				typeof value === "number" ? value : Number.parseInt(value);
			return new Date(timestamp * 1000);
		}

		// Handle ISO string
		const date = new Date(value);
		return Number.isNaN(date.getTime()) ? null : date;
	};

	// Test Unix timestamp
	const unixTime = 1640995200; // Jan 1, 2022
	const dateFromUnix = validateDateField(unixTime);
	t.true(dateFromUnix instanceof Date);
	t.false(Number.isNaN(dateFromUnix.getTime()));

	// Test string timestamp
	const dateFromString = validateDateField("1640995200");
	t.true(dateFromString instanceof Date);

	// Test ISO string
	const isoString = "2022-01-01T00:00:00Z";
	const dateFromISO = validateDateField(isoString);
	t.true(dateFromISO instanceof Date);

	// Test invalid values
	t.is(validateDateField("invalid"), null);
	t.is(validateDateField("None"), null);
	t.is(validateDateField(null), null);
});

test("JSON metadata validation", (t) => {
	// Test JSON field validation with error handling
	const validateJSONField = (value, defaultValue = null) => {
		if (!value || value === "None") {
			return defaultValue;
		}

		try {
			return JSON.parse(value);
		} catch (error) {
			console.warn("Invalid JSON in metadata field:", error.message);
			return defaultValue;
		}
	};

	// Test valid JSON
	const validJson = '{"rank": 4, "alpha": 4.0}';
	const parsed = validateJSONField(validJson);
	t.deepEqual(parsed, { rank: 4, alpha: 4.0 });

	// Test invalid JSON
	const invalidJson = '{"rank": 4, "alpha":}';
	const invalid = validateJSONField(invalidJson, {});
	t.deepEqual(invalid, {});

	// Test edge cases
	t.is(validateJSONField("None"), null);
	t.is(validateJSONField(null), null);
	t.deepEqual(validateJSONField("", []), []);
});

test("ModelSpec component data validation", (t) => {
	// Test the specific validation logic used by ModelSpec
	const validateModelSpec = (metadata) => {
		const hasModelSpec = metadata.has("modelspec.title");

		if (!hasModelSpec) {
			return { hasModelSpec: false, trainingData: {} };
		}

		const trainingData = {};

		// Training times
		if (
			metadata.has("ss_training_started_at") &&
			metadata.has("ss_training_finished_at")
		) {
			const start = Number.parseInt(metadata.get("ss_training_started_at"));
			const end = Number.parseInt(metadata.get("ss_training_finished_at"));

			if (!Number.isNaN(start) && !Number.isNaN(end) && end > start) {
				trainingData.startTime = new Date(start * 1000);
				trainingData.endTime = new Date(end * 1000);
				trainingData.elapsedMinutes = (end - start) / 60;
			}
		}

		return { hasModelSpec: true, trainingData };
	};

	// Test with full ModelSpec
	const fullMetadata = new Map([
		["modelspec.title", "Test Model"],
		["modelspec.date", "2022-01-01T00:00:00Z"],
		["ss_training_started_at", "1640995200"],
		["ss_training_finished_at", "1640995800"],
	]);

	const fullResult = validateModelSpec(fullMetadata);
	t.true(fullResult.hasModelSpec);
	t.is(fullResult.trainingData.elapsedMinutes, 10);

	// Test without ModelSpec
	const noModelSpecMetadata = new Map([
		["ss_training_started_at", "1640995200"],
	]);

	const noModelSpecResult = validateModelSpec(noModelSpecMetadata);
	t.false(noModelSpecResult.hasModelSpec);
});

test("Network args parsing edge cases", (t) => {
	// Test network args parsing used by Network component
	const parseNetworkArgs = (metadata) => {
		if (!metadata.has("ss_network_args")) {
			return null;
		}

		try {
			const args = JSON.parse(metadata.get("ss_network_args"));

			// Validate expected structure
			if (typeof args !== "object" || args === null) {
				return null;
			}

			return args;
		} catch (error) {
			return null;
		}
	};

	// Test valid network args
	const validArgs = '{"module_dropout": 0.1, "rank_dropout": 0.0}';
	const validMetadata = new Map([["ss_network_args", validArgs]]);
	const parsed = parseNetworkArgs(validMetadata);
	t.deepEqual(parsed, { module_dropout: 0.1, rank_dropout: 0.0 });

	// Test invalid JSON
	const invalidArgs = '{"module_dropout": 0.1,}';
	const invalidMetadata = new Map([["ss_network_args", invalidArgs]]);
	t.is(parseNetworkArgs(invalidMetadata), null);

	// Test missing field
	const emptyMetadata = new Map([]);
	t.is(parseNetworkArgs(emptyMetadata), null);
});
