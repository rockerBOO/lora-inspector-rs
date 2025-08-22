import { cleanup, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

// Clean up after each test
beforeEach(() => {
	cleanup();
});

describe("Network Components", () => {
	describe("LoRANetwork Component", () => {
		it("should render network structure correctly", async () => {
			const { LoRANetwork } = await import(
				"../assets/js/components/network/LoRANetwork.jsx"
			);

			const loraMetadata = new Map([
				["ss_network_module", "networks.lora"],
				["ss_network_dim", "32"],
				["ss_network_alpha", "32.0"],
				["ss_network_args", '{"network_type": "LoRA", "conv_rank": 16}'],
			]);

			const mockWorker = createMockWorker();

			render(
				<LoRANetwork
					metadata={loraMetadata}
					filename="test.safetensors"
					worker={mockWorker}
				/>,
			);

			// Should render the component structure
			expect(screen.getByText("Network Rank/Dimension")).toBeDefined();
			expect(screen.getByText("Network Alpha")).toBeDefined();
		});

		it("should handle missing network rank gracefully", async () => {
			const { LoRANetwork } = await import(
				"../assets/js/components/network/LoRANetwork.jsx"
			);

			const incompleteMetadata = new Map([
				["ss_network_module", "networks.lora"],
				["ss_network_alpha", "16.0"],
			]);

			const mockWorker = createMockWorker();

			expect(() =>
				render(
					<LoRANetwork
						metadata={incompleteMetadata}
						filename="test.safetensors"
						worker={mockWorker}
					/>,
				),
			).not.toThrow();
		});
	});

	describe("LoKrNetwork Component", () => {
		it("should render LoKr network structure", async () => {
			const { LoKrNetwork } = await import(
				"../assets/js/components/network/LoKrNetwork.jsx"
			);

			const lokrMetadata = new Map([
				["ss_network_module", "networks.lokr"],
				["ss_network_rank", "16"],
				["ss_network_alpha", "1.0"],
				["ss_network_args", '{"decompose_both": true, "factor": 8}'],
			]);

			const mockWorker = createMockWorker();

			render(
				<LoKrNetwork
					metadata={lokrMetadata}
					filename="test.safetensors"
					worker={mockWorker}
				/>,
			);

			// Should render the component structure
			expect(screen.getByText("Network Rank/Dimension")).toBeDefined();
			expect(screen.getByText("Network Alpha")).toBeDefined();
		});
	});

	describe("DiagOFTNetwork Component", () => {
		it("should display DiagOFT network configuration", async () => {
			const { DiagOFTNetwork } = await import(
				"../assets/js/components/network/DiagOFTNetwork.jsx"
			);

			const diagoftMetadata = new Map([
				["ss_network_module", "lycoris.kohya"],
				["ss_network_args", '{"algo": "diag-oft", "block_size": 4}'],
			]);

			const mockWorker = createMockWorker();

			expect(() =>
				render(
					<DiagOFTNetwork
						metadata={diagoftMetadata}
						filename="test.safetensors"
						worker={mockWorker}
					/>,
				),
			).not.toThrow();
		});
	});

	describe("BOFTNetwork Component", () => {
		it("should display BOFT network information", async () => {
			const { BOFTNetwork } = await import(
				"../assets/js/components/network/BOFTNetwork.jsx"
			);

			const boftMetadata = new Map([
				["ss_network_module", "lycoris.kohya"],
				["ss_network_args", '{"algo": "boft", "block_size": 8}'],
			]);

			const mockWorker = createMockWorker();

			expect(() =>
				render(
					<BOFTNetwork
						metadata={boftMetadata}
						filename="test.safetensors"
						worker={mockWorker}
					/>,
				),
			).not.toThrow();
		});
	});

	describe("Network Component (Main)", () => {
		it("should render LoRA network type correctly", async () => {
			const { Network } = await import(
				"../assets/js/components/network/Network.jsx"
			);

			const loraMetadata = new Map([
				["ss_network_module", "networks.lora"],
				["ss_network_rank", "64"],
				["ss_network_alpha", "64.0"],
			]);

			const mockWorker = createMockWorker();

			render(
				<Network
					metadata={loraMetadata}
					filename="test.safetensors"
					worker={mockWorker}
				/>,
			);

			// Should render the network structure
			expect(screen.getByText("Network Rank/Dimension")).toBeDefined();
		});

		it("should handle different network types", async () => {
			const { Network } = await import(
				"../assets/js/components/network/Network.jsx"
			);

			const lokrMetadata = new Map([
				["ss_network_module", "networks.lokr"],
				["ss_network_rank", "32"],
				["ss_network_alpha", "1.0"],
			]);

			const mockWorker = createMockWorker();

			expect(() =>
				render(
					<Network
						metadata={lokrMetadata}
						filename="test.safetensors"
						worker={mockWorker}
					/>,
				),
			).not.toThrow();
		});

		it("should handle unknown network types gracefully", async () => {
			const { Network } = await import(
				"../assets/js/components/network/Network.jsx"
			);

			const unknownMetadata = new Map([
				["ss_network_module", "networks.unknown"],
				["ss_network_rank", "16"],
			]);

			const mockWorker = createMockWorker();

			expect(() =>
				render(
					<Network
						metadata={unknownMetadata}
						filename="test.safetensors"
						worker={mockWorker}
					/>,
				),
			).not.toThrow();
		});
	});
});

describe("Network Utilities", () => {
	describe("DoRA Support Detection", () => {
		it("should correctly identify DoRA-compatible networks", async () => {
			const { supportsDoRA } = await import(
				"../assets/js/components/network/NetworkUtils.js"
			);

			expect(supportsDoRA("LoRA")).toBe(true);
			expect(supportsDoRA("LoHa")).toBe(true);
			expect(supportsDoRA("LoRAFA")).toBe(true);
			expect(supportsDoRA("LoKr")).toBe(true);
			expect(supportsDoRA("GLoRA")).toBe(true);

			expect(supportsDoRA("DiagOFT")).toBe(false);
			expect(supportsDoRA("BOFT")).toBe(false);
			expect(supportsDoRA("Unknown")).toBe(false);
		});
	});
});
