import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

const { MetaAttribute } = await import(
	"../assets/js/components/ui/MetaAttribute.jsx"
);

describe("MetaAttribute", () => {
	it("renders label and value", () => {
		render(<MetaAttribute name="Optimizer" value="AdamW" />);
		expect(screen.getByText("Optimizer")).toBeDefined();
		expect(screen.getByText("AdamW")).toBeDefined();
	});

	it("shows em-dash when value is undefined", () => {
		render(<MetaAttribute name="Loss type" value={undefined} />);
		expect(screen.getByText("—")).toBeDefined();
	});

	it("shows em-dash when value is null", () => {
		render(<MetaAttribute name="fp8 base" value={null} />);
		expect(screen.getByText("—")).toBeDefined();
	});

	it('shows "None" as-is when value is the string None', () => {
		render(<MetaAttribute name="Network args" value="None" />);
		expect(screen.getByText("None")).toBeDefined();
	});

	it('shows "null" as-is when value is the string null', () => {
		render(<MetaAttribute name="Network args" value="null" />);
		expect(screen.getByText("null")).toBeDefined();
	});

	it("shows False as-is when value is the string False", () => {
		render(<MetaAttribute name="Full fp16" value="False" />);
		expect(screen.getByText("False")).toBeDefined();
	});

	it("renders secondary value when provided", () => {
		render(
			<MetaAttribute
				name="Rank"
				value="4"
				secondaryName="Alpha"
				secondary="4.0"
			/>,
		);
		expect(screen.getByText("4")).toBeDefined();
		expect(screen.getByText("Alpha")).toBeDefined();
		expect(screen.getByText("4.0")).toBeDefined();
	});
});
