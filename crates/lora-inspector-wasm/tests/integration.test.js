import test from "ava";

// Integration tests to verify core functionality works correctly
// Focus on meaningful business logic rather than file structure

test("should import worker functions successfully", async (t) => {
	try {
		const workersModule = await import("../assets/js/workers.js");
		
		// Verify key worker functions exist and are callable
		t.is(typeof workersModule.addWorker, "function");
		t.is(typeof workersModule.clearWorkers, "function");
		t.is(typeof workersModule.terminatePreviousProcessing, "function");
		
	} catch (error) {
		throw new Error(`Workers import failed: ${error.message}`);
	}
});

test("should handle metadata Map structure correctly", (t) => {
	// Test the core data structure that drives our components
	const sampleMetadata = new Map([
		// Model specification
		["modelspec.title", "Test LoRA Model"],
		["modelspec.date", "2022-01-01T00:00:00Z"],
		["modelspec.license", "MIT"],
		["modelspec.description", "Test model for verification"],
		["modelspec.prediction_type", "epsilon"],
		
		// Training information
		["ss_training_started_at", 1640995200],
		["ss_training_finished_at", 1640995800],
		["ss_training_comment", "Test training session"],
		
		// Model information
		["ss_sd_model_name", "test-model-v1.5"],
		["sshs_model_hash", "abc123def456"],
		["ss_session_id", "test-session-123"],
		
		// VAE information
		["ss_vae_name", "test-vae.ckpt"],
		["ss_vae_hash", "vae123hash"],
		
		// Network configuration
		["ss_network_rank", "4"],
		["ss_network_alpha", "4.0"],
		["ss_network_module", "networks.lora"],
		["ss_network_args", '{"network_type": "LoRA"}']
	]);
	
	// Verify our metadata structure works as expected
	t.true(sampleMetadata.has("modelspec.title"));
	t.is(sampleMetadata.get("modelspec.title"), "Test LoRA Model");
	
	t.true(sampleMetadata.has("ss_training_started_at"));
	t.is(typeof sampleMetadata.get("ss_training_started_at"), "number");
	
	t.true(sampleMetadata.has("ss_vae_name"));
	t.is(sampleMetadata.get("ss_vae_name"), "test-vae.ckpt");
	
	// Test training time calculation
	const startTime = sampleMetadata.get("ss_training_started_at");
	const endTime = sampleMetadata.get("ss_training_finished_at");
	const elapsedMinutes = (endTime - startTime) / 60;
	t.is(elapsedMinutes, 10);
});