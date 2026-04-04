import { cleanup, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

beforeEach(() => {
	cleanup();
});

const FILTERED_KEYS = [
	"ss_training_finished_at",
	"ss_training_started_at",
	"ss_session_id",
	"ss_output_name",
	"modelspec.date",
];

async function renderCompare(metadataA, metadataB, props = {}) {
	const { CompareMetadata } = await import(
		"../assets/js/components/layout/CompareMetadata.jsx"
	);
	render(
		<CompareMetadata
			metadataA={new Map(Object.entries(metadataA))}
			filenameA="file-a.safetensors"
			metadataB={new Map(Object.entries(metadataB))}
			filenameB="file-b.safetensors"
			onViewA={() => {}}
			onViewB={() => {}}
			{...props}
		/>,
	);
}

describe("CompareMetadata", () => {
	describe("filtered keys", () => {
		for (const key of FILTERED_KEYS) {
			it(`does not show ${key} even when values differ`, async () => {
				await renderCompare({ [key]: "value-a" }, { [key]: "value-b" });
				expect(screen.queryByText(key)).toBeNull();
			});
		}
	});

	describe("diff logic", () => {
		it("shows keys that differ between files", async () => {
			await renderCompare({ ss_epoch: "1" }, { ss_epoch: "3" });
			expect(screen.getByText("ss_epoch")).toBeDefined();
			expect(screen.getByText("1")).toBeDefined();
			expect(screen.getByText("3")).toBeDefined();
		});

		it("does not show keys with identical values", async () => {
			await renderCompare(
				{ ss_network_dim: "32" },
				{ ss_network_dim: "32" },
			);
			expect(screen.queryByText("ss_network_dim")).toBeNull();
		});

		it("shows missing key as em dash when only one file has it", async () => {
			await renderCompare({ ss_steps: "1503" }, {});
			expect(screen.getByText("ss_steps")).toBeDefined();
			expect(screen.getByText("—")).toBeDefined();
		});

		it("shows no differences message when all visible keys match", async () => {
			await renderCompare(
				{ ss_network_dim: "32", ss_session_id: "aaa" },
				{ ss_network_dim: "32", ss_session_id: "bbb" },
			);
			expect(screen.getByText("No differences found.")).toBeDefined();
		});
	});

	describe("ss_datasets collapsed", () => {
		it("renders ss_datasets inside a details element", async () => {
			const datasets = JSON.stringify([{ image_directory: "a" }]);
			const datasetsB = JSON.stringify([{ image_directory: "b" }]);
			await renderCompare(
				{ ss_datasets: datasets },
				{ ss_datasets: datasetsB },
			);
			expect(screen.getByText("ss_datasets")).toBeDefined();
			const summaries = document.querySelectorAll("details summary");
			expect(summaries.length).toBeGreaterThan(0);
		});

		it("renders ss_datasets value inside a details element, not directly in the table cell", async () => {
			const datasets = JSON.stringify([{ image_directory: "my-dataset" }]);
			const datasetsB = JSON.stringify([{ image_directory: "other" }]);
			await renderCompare(
				{ ss_datasets: datasets },
				{ ss_datasets: datasetsB },
			);
			// Value exists in DOM but is wrapped in <details>, not a bare table cell
			const detailsElements = document.querySelectorAll("details");
			const valueInDetails = [...detailsElements].some((el) =>
				el.textContent.includes("my-dataset"),
			);
			expect(valueInDetails).toBe(true);
		});
	});

	describe("navigation", () => {
		it("renders view buttons for both filenames", async () => {
			await renderCompare({}, {});
			expect(screen.getByText("View file-a.safetensors")).toBeDefined();
			expect(screen.getByText("View file-b.safetensors")).toBeDefined();
		});

		it("calls onViewA when file A button is clicked", async () => {
			const { CompareMetadata } = await import(
				"../assets/js/components/layout/CompareMetadata.jsx"
			);
			let called = false;
			render(
				<CompareMetadata
					metadataA={new Map()}
					filenameA="file-a.safetensors"
					metadataB={new Map()}
					filenameB="file-b.safetensors"
					onViewA={() => {
						called = true;
					}}
					onViewB={() => {}}
				/>,
			);
			screen.getByText("View file-a.safetensors").click();
			expect(called).toBe(true);
		});
	});
});
