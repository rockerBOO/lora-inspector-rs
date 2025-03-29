import { test, expect } from "@playwright/test";

test("has title", async ({ page }) => {
	await page.goto("/");

	// Expect a title "to contain" a substring.
	await expect(page).toHaveTitle(/LoRA Inspector/);
});

test("test", async ({ page }) => {
	await page.goto("/");
	// await page.getByText("Choose a file").click();

	// const file = await fetch(
	//   "https://lora-inspector-test-files.us-east-1.linodeobjects.com/boo.safetensors",
	// ).then((resp) => {
	//   return resp.arrayBuffer();
	// });
	// console.log(file.suggestedFilename());

	// Click on <strong> "Choose a file"
	// await page.click('text=Choose a file');
	//
	await page.evaluate(() => {
		const hiddenInput = document.getElementById("file");
		if (hiddenInput) {
			// Modify CSS properties to make the element visible
			hiddenInput.style.display = "block";
		}
	});

	// const [fileChooser] = await Promise.all([
	//    // It is important to call waitForEvent before click to set up waiting.
	//    page.waitForEvent('filechooser'),
	//    // Opens the file chooser.
	//    ,
	//  ])

	page.on("filechooser", (fileChooser) => {
		fileChooser.setFiles(["boo.safetensors"]);
	});

	await page.getByText("Choose a safetensors file").click();

	// await expect(page.locator(".box__input input")).toBeVisible();

	// await page.locator("#file").setInputFiles("boo.safetensors");
	// await fileChooser.setFiles(["boo.safetensors"]);

	await expect(page.locator("#network-rank")).toContainText("4", {
		timeout: 10000,
	});
	await expect(page.locator("#network-alpha")).toContainText("4.0");
	await expect(page.locator("#network-module")).toContainText("kohya-ss/lora");
	await expect(page.locator("#network-type")).toContainText("LoRA");
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
