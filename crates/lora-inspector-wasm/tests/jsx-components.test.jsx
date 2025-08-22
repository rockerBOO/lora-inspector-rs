import { describe, it, expect, beforeEach } from "vitest";
import { render, screen, cleanup } from "@testing-library/react";

// Clean up after each test
beforeEach(() => {
	cleanup();
});

describe("Core UI Components", () => {
	describe("MetaAttribute Component", () => {
		it("should render name and value correctly", async () => {
			const { MetaAttribute } = await import("../assets/js/components/ui/MetaAttribute.jsx");
			
			render(<MetaAttribute name="Model Name" value="test-model-v1.5" />);
			
			expect(screen.getByText("Model Name")).toBeDefined();
			expect(screen.getByText("test-model-v1.5")).toBeDefined();
		});

		it("should handle empty values gracefully", async () => {
			const { MetaAttribute } = await import("../assets/js/components/ui/MetaAttribute.jsx");
			
			render(<MetaAttribute name="Empty Field" value="" />);
			
			expect(screen.getByText("Empty Field")).toBeDefined();
			// Component should render without crashing with empty value
			const container = document.querySelector('.meta-attribute-value');
			expect(container).toBeDefined();
		});

		it("should handle long values properly", async () => {
			const { MetaAttribute } = await import("../assets/js/components/ui/MetaAttribute.jsx");
			
			const longValue = "This is a very long value that might wrap or need special handling in the UI component";
			render(<MetaAttribute name="Long Value" value={longValue} />);
			
			expect(screen.getByText("Long Value")).toBeDefined();
			expect(screen.getByText(longValue)).toBeDefined();
		});
	});

	describe("Header Component", () => {
		it("should render with complete metadata", async () => {
			const { Header } = await import("../assets/js/components/layout/Header.jsx");
			
			const completeMetadata = new Map([
				["modelspec.title", "Advanced LoRA Model"],
				["modelspec.description", "A comprehensive test model"],
				["ss_training_started_at", "1640995200"],
				["ss_training_finished_at", "1640995800"],
				["ss_session_id", "session-12345"]
			]);
			
			render(<Header metadata={completeMetadata} />);
			
			// Should render the title
			expect(screen.getByText("Advanced LoRA Model")).toBeDefined();
		});

		it("should handle minimal metadata gracefully", async () => {
			const { Header } = await import("../assets/js/components/layout/Header.jsx");
			
			const minimalMetadata = new Map([
				["modelspec.title", "Basic Model"]
			]);
			
			render(<Header metadata={minimalMetadata} />);
			
			expect(screen.getByText("Basic Model")).toBeDefined();
			// Should not crash with minimal data
		});

		it("should handle empty metadata", async () => {
			const { Header } = await import("../assets/js/components/layout/Header.jsx");
			
			const emptyMetadata = new Map();
			
			// Should not crash with empty metadata
			expect(() => render(<Header metadata={emptyMetadata} />)).not.toThrow();
		});
	});
});

describe("Metadata Display Components", () => {
	describe("ModelSpec Component", () => {
		it("should display model specification details", async () => {
			const { ModelSpec } = await import("../assets/js/components/metadata/ModelSpec.jsx");
			
			const modelMetadata = new Map([
				["modelspec.title", "Test Model"],
				["modelspec.description", "A test model for validation"],
				["modelspec.license", "MIT"],
				["modelspec.prediction_type", "epsilon"]
			]);
			
			render(<ModelSpec metadata={modelMetadata} />);
			
			expect(screen.getByText("Test Model")).toBeDefined();
			expect(screen.getByText("A test model for validation")).toBeDefined();
		});

		it("should handle missing modelspec fields", async () => {
			const { ModelSpec } = await import("../assets/js/components/metadata/ModelSpec.jsx");
			
			const partialMetadata = new Map([
				["modelspec.title", "Partial Model"]
				// Missing other fields
			]);
			
			render(<ModelSpec metadata={partialMetadata} />);
			
			expect(screen.getByText("Partial Model")).toBeDefined();
			// Should render without crashing
		});
	});

	describe("VAE Component", () => {
		it("should display VAE information when available", async () => {
			const { VAE } = await import("../assets/js/components/metadata/VAE.jsx");
			
			const vaeMetadata = new Map([
				["ss_vae_name", "vae-ft-mse-840000-ema-pruned.ckpt"],
				["ss_vae_hash", "abc123def456"]
			]);
			
			render(<VAE metadata={vaeMetadata} />);
			
			expect(screen.getByText("vae-ft-mse-840000-ema-pruned.ckpt")).toBeDefined();
		});

		it("should handle missing VAE data", async () => {
			const { VAE } = await import("../assets/js/components/metadata/VAE.jsx");
			
			const noVaeMetadata = new Map([
				["modelspec.title", "Model without VAE"]
			]);
			
			// Should not crash when VAE data is missing
			expect(() => render(<VAE metadata={noVaeMetadata} />)).not.toThrow();
		});
	});
});