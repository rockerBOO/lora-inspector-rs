import { test, expect } from "@playwright/test";

test("has title", async ({ page }) => {
  await page.goto("/");

  // Expect a title "to contain" a substring.
  await expect(page).toHaveTitle(/LoRA Inspector/);
});

test("test", async ({ page }) => {
  page.on("pageerror", (error) => {
    console.error("Page error:", error.message);
  });
  await page.goto("/");
  // await page.getByText("Choose a file").click();

  // const file = await fetch(
  //   "https://lora-inspector-test-files.us-east-1.linodeobjects.com/boo.safetensors",
  // ).then((resp) => {
  //   return resp.arrayBuffer();
  // });
  // console.log(file.suggestedFilename());

  // Click on <strong> "Choose a file"
  // await page.click('text=Choose a safetensors file');

  // await page.evaluate(() => {
  //   const hiddenInput = document.getElementById("file");
  //   if (hiddenInput) {
  //     // Modify CSS properties to make the element visible
  //     hiddenInput.style.display = "block";
  //   }
  // });
  // Check if the page navigates or if any JavaScript errors occur

  // await page.getByText("Choose a safetensors file").click();

  // Wait for the file chooser to appear
  // const [fileChooser] = await Promise.all([
  //   page.waitForEvent('filechooser'),
  //   page.getByText("Choose a safetensors file").click(), // This might need to be checked
  // ]);
  //
  // await page.waitForTimeout(500);
  //
  //  // Wait for the file input to be visible
  //  await expect(page.locator("#file")).toBeVisible();
  //
  //  // Wait for the file input to be ready
  //  const [fileChooser] = await Promise.all([
  //    page.waitForEvent('filechooser'), // Wait for the file chooser to appear
  //    page.getByRole("#file").click() // Click to open it
  //  ]);
  //
  //  // Set the file directly using the file chooser
  //  await fileChooser.setFiles("boo.safetensors"); // Make sure the path is correct
  await page.getByText("Choose a safetensors file").click();

  await page.locator("#file").setInputFiles("boo.safetensors");

  // Check the selected file
  const inputValue = await page.locator("#file").inputValue();
  console.log('Selected file:', inputValue);

  // Optionally, add a short wait to allow processing time
  await page.waitForTimeout(1000);

  // // Set the file

  await expect(page.locator("#network-rank")).toContainText("4", {});
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
