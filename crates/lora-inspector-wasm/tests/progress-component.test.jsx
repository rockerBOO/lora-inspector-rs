import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
	cleanup();
	vi.restoreAllMocks();
});

async function renderProgress(props) {
	const { Progress } = await import(
		"../assets/js/components/analysis/Progress.jsx"
	);
	return render(<Progress {...props} />);
}

describe("Progress component", () => {
	describe("math correctness", () => {
		it("computes remaining time and throughput correctly", async () => {
			// startTime = 1000ms, now = 3000ms → 2000ms elapsed for 50/100 items
			// msPerItem = 2000/50 = 40ms
			// remainingMs = 40 * 50 = 2000ms → 2.0s
			// perSecond = 50 / 2 = 25.00 it/s
			vi.spyOn(performance, "now").mockReturnValue(3000);

			await renderProgress({
				startTime: 1000,
				currentCount: 50,
				totalCount: 100,
				statisticProgress: 0.5,
				currentItemName: "layer_50",
			});

			expect(screen.getByText(/50\.0%/)).toBeDefined();
			expect(screen.getByText(/50\/100/)).toBeDefined();
			expect(screen.getByText(/25\.00it\/s/)).toBeDefined();
			expect(screen.getByText(/2\.0s remaining/)).toBeDefined();
		});

		it("shows 0% and no remaining time at the start", async () => {
			vi.spyOn(performance, "now").mockReturnValue(1000);

			await renderProgress({
				startTime: 1000,
				currentCount: 0,
				totalCount: 100,
				statisticProgress: 0,
				currentItemName: "",
			});

			expect(screen.getByText(/0\.0%/)).toBeDefined();
			expect(screen.getByText(/0\/100/)).toBeDefined();
			// No divide-by-zero: perSecond and remainingMs should be 0
			expect(screen.getByText(/0\.00it\/s/)).toBeDefined();
			expect(screen.getByText(/0\.0s remaining/)).toBeDefined();
		});

		it("shows 100% with 0s remaining when done", async () => {
			vi.spyOn(performance, "now").mockReturnValue(5000);

			await renderProgress({
				startTime: 1000,
				currentCount: 100,
				totalCount: 100,
				statisticProgress: 1.0,
				currentItemName: "last_layer",
			});

			expect(screen.getByText(/100\.0%/)).toBeDefined();
			expect(screen.getByText(/100\/100/)).toBeDefined();
			expect(screen.getByText(/0\.0s remaining/)).toBeDefined();
		});

		it("handles null startTime without crashing", async () => {
			vi.spyOn(performance, "now").mockReturnValue(5000);

			expect(() =>
				renderProgress({
					startTime: null,
					currentCount: 10,
					totalCount: 50,
					statisticProgress: 0.2,
					currentItemName: "test_layer",
				}),
			).not.toThrow();
		});

		it("remaining time scales linearly with work left", async () => {
			// 1000ms elapsed for 25/100 items → 3000ms remaining
			vi.spyOn(performance, "now").mockReturnValue(2000);

			await renderProgress({
				startTime: 1000,
				currentCount: 25,
				totalCount: 100,
				statisticProgress: 0.25,
				currentItemName: "layer_25",
			});

			// msPerItem = 1000/25 = 40ms, remaining = 40 * 75 = 3000ms = 3.0s
			expect(screen.getByText(/3\.0s remaining/)).toBeDefined();
		});
	});

	describe("rendering", () => {
		it("displays the current item name", async () => {
			vi.spyOn(performance, "now").mockReturnValue(2000);

			await renderProgress({
				startTime: 1000,
				currentCount: 10,
				totalCount: 50,
				statisticProgress: 0.2,
				currentItemName: "lora_unet_double_blocks_5",
			});

			expect(screen.getByText(/lora_unet_double_blocks_5/)).toBeDefined();
		});

		it("renders without crashing with minimal props", async () => {
			vi.spyOn(performance, "now").mockReturnValue(1000);

			expect(() =>
				renderProgress({
					startTime: 1000,
					currentCount: 1,
					totalCount: 10,
					statisticProgress: 0.1,
					currentItemName: "",
				}),
			).not.toThrow();
		});
	});
});
