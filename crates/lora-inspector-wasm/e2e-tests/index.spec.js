import { expect, test } from "@playwright/test";

test("has title", async ({ page }) => {
	await page.goto("/");

	// Expect a title "to contain" a substring.
	await expect(page).toHaveTitle(/LoRA Inspector/);
});

async function uploadBooSafetensors(page) {
	await page.evaluate(() => {
		const hiddenInput = document.getElementById("file");
		if (hiddenInput) {
			hiddenInput.style.display = "block";
		}
	});

	page.on("filechooser", (fileChooser) => {
		fileChooser.setFiles(["boo.safetensors"]);
	});

	await page.getByText("Choose a safetensors file").click();
}

test("file upload shows network data", async ({ page }) => {
	await page.goto("/");
	await uploadBooSafetensors(page);

	await expect(page.locator("#network-rank")).toContainText("4", {
		timeout: 10000,
	});
	await expect(page.locator("#network-alpha")).toContainText("4.0");
	await expect(page.locator("#network-module")).toContainText("kohya-ss/lora");
	await expect(page.locator("#network-type")).toContainText("LoRA");
});

test("section nav renders all 6 sections after file upload", async ({
	page,
}) => {
	await page.goto("/");
	await uploadBooSafetensors(page);

	await expect(page.locator("#network-rank")).toContainText("4", {
		timeout: 10000,
	});

	const nav = page.getByRole("navigation", { name: "Page sections" });
	await expect(nav).toBeVisible();

	for (const section of [
		"Metadata",
		"Network",
		"Training",
		"Optimizer",
		"Dataset",
		"Advanced",
	]) {
		await expect(nav.getByRole("link", { name: section })).toBeVisible();
	}
});

test("sections exist on page after file upload", async ({ page }) => {
	await page.goto("/");
	await uploadBooSafetensors(page);

	await expect(page.locator("#network-rank")).toContainText("4", {
		timeout: 10000,
	});

	for (const id of [
		"metadata",
		"network",
		"training",
		"optimizer",
		"dataset",
		"advanced",
	]) {
		await expect(page.locator(`#${id}`)).toBeAttached();
	}
});

test("metadata section shows SD model name", async ({ page }) => {
	await page.goto("/");
	await uploadBooSafetensors(page);

	await expect(page.locator("#network-rank")).toContainText("4", {
		timeout: 10000,
	});

	await expect(page.locator("#metadata")).toContainText(
		"v1-5-pruned-emaonly.ckpt",
	);
});

test("drag and drop file upload", async ({ page }) => {
	await page.goto("/");

	// Simulate drag and drop from within the page context
	await page.evaluate(async () => {
		// Fetch the file
		const response = await fetch(
			"https://lora-inspector-test-files.us-east-1.linodeobjects.com/boo.safetensors",
		);
		const arrayBuffer = await response.arrayBuffer();
		const blob = new Blob([new Uint8Array(arrayBuffer)], { type: "" });
		const file = new File([blob], "boo.safetensors", { type: "" });

		// Get the dropbox element
		const dropbox = document.querySelector("#dropbox");

		// Create DataTransfer with the file
		const dataTransfer = new DataTransfer();
		dataTransfer.items.add(file);

		// Simulate dragenter event
		const dragenterEvent = new DragEvent("dragenter", {
			bubbles: true,
			cancelable: true,
			dataTransfer: dataTransfer,
		});
		dropbox.dispatchEvent(dragenterEvent);

		// Simulate dragover event
		const dragoverEvent = new DragEvent("dragover", {
			bubbles: true,
			cancelable: true,
			dataTransfer: dataTransfer,
		});
		dropbox.dispatchEvent(dragoverEvent);

		// Simulate drop event
		const dropEvent = new DragEvent("drop", {
			bubbles: true,
			cancelable: true,
			dataTransfer: dataTransfer,
		});
		dropbox.dispatchEvent(dropEvent);
	});

	// Verify that the file was processed correctly
	await expect(page.locator("#network-rank")).toContainText("4", {
		timeout: 10000,
	});
	await expect(page.locator("#network-alpha")).toContainText("4.0");
	await expect(page.locator("#network-module")).toContainText("kohya-ss/lora");
	await expect(page.locator("#network-type")).toContainText("LoRA");

	// Verify we didn't navigate away from the page
	await expect(page).toHaveURL("/");
});

// test('get started link', async ({ page }) => {
//   await page.goto('/');
//
//   // Click the get started link.
//   await page.getByRole('link', { name: 'Get started' }).click();
//
//   // Expects page to have a heading with the name of Installation.
//   await expect(page.getByRole('heading', { name: 'Installation' })).toBeVisible();
// });
