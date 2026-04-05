import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

const { SectionNav } = await import(
	"../assets/js/components/layout/SectionNav.jsx"
);

describe("SectionNav", () => {
	it("renders all 6 section links", () => {
		render(<SectionNav filename="boo.safetensors" />);

		expect(screen.getByRole("link", { name: "Metadata" })).toBeDefined();
		expect(screen.getByRole("link", { name: "Network" })).toBeDefined();
		expect(screen.getByRole("link", { name: "Training" })).toBeDefined();
		expect(screen.getByRole("link", { name: "Optimizer" })).toBeDefined();
		expect(screen.getByRole("link", { name: "Dataset" })).toBeDefined();
		expect(screen.getByRole("link", { name: "Advanced" })).toBeDefined();
	});

	it("links point to correct section anchors", () => {
		render(<SectionNav filename="boo.safetensors" />);

		expect(screen.getByRole("link", { name: "Metadata" }).href).toContain(
			"#metadata",
		);
		expect(screen.getByRole("link", { name: "Network" }).href).toContain(
			"#network",
		);
		expect(screen.getByRole("link", { name: "Training" }).href).toContain(
			"#training",
		);
		expect(screen.getByRole("link", { name: "Optimizer" }).href).toContain(
			"#optimizer",
		);
		expect(screen.getByRole("link", { name: "Dataset" }).href).toContain(
			"#dataset",
		);
		expect(screen.getByRole("link", { name: "Advanced" }).href).toContain(
			"#advanced",
		);
	});

	it("has navigation landmark with label", () => {
		render(<SectionNav filename="boo.safetensors" />);
		expect(
			screen.getByRole("navigation", { name: "Page sections" }),
		).toBeDefined();
	});

	it("shows filename in file bar", () => {
		render(<SectionNav filename="boo.safetensors" />);
		expect(screen.getByText("boo.safetensors")).toBeDefined();
	});

	it("marks first section active by default", () => {
		render(<SectionNav filename="boo.safetensors" />);
		const metadataLink = screen.getByRole("link", { name: "Metadata" });
		expect(metadataLink.getAttribute("aria-current")).toBe("true");
	});
});
