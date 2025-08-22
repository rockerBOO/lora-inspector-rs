import { resolve } from "node:path";
import process from "node:process";
import { defineConfig, devices } from "@playwright/test";

/**
 * Read environment variables from file.
 * https://github.com/motdotla/dotenv
 */
// require('dotenv').config();

/**
 * @see https://playwright.dev/docs/test-configuration
 */
export default defineConfig({
	testDir: "./e2e-tests",
	/* Run tests in files in parallel */
	fullyParallel: true,
	/* Fail the build on CI if you accidentally left test.only in the source code. */
	forbidOnly: !!process.env.CI,
	/* Retry on CI only */
	retries: process.env.CI ? 2 : 0,
	/* Opt out of parallel tests on CI. */
	workers: process.env.CI ? 1 : undefined,
	/* Reporter to use. See https://playwright.dev/docs/test-reporters */
	reporter: "html",
	/* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
	use: {
		/* Base URL to use in actions like `await page.goto('/')`. */
		baseURL: process.env.BASE_URL ?? "http://localhost:4173",

		/* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
		trace: "on-first-retry",
	},

	/* Configure projects for major browsers */
	projects: [
		// {
		//   name: "setup",
		//   testMatch: /global-setup\.mjs/,
		//   // teardown: "cleanup fs"
		// },
		// {
		//   name: "cleanup fs",
		//   testMatch: /global-teardown\.mjs/,
		// },
		{
			name: "chromium",
			use: { ...devices["Desktop Chrome"] },
			// dependencies: ["setup"],
		},

		{
			name: "firefox",
			use: { ...devices["Desktop Firefox"] },
			// dependencies: ["setup"],
		},

		{
			name: "webkit",
			use: { ...devices["Desktop Safari"] },
			// dependencies: ["setup"],
		},

		/* Test against mobile viewports. */
		// {
		//   name: 'Mobile Chrome',
		//   use: { ...devices['Pixel 5'] },
		// },
		// {
		//   name: 'Mobile Safari',
		//   use: { ...devices['iPhone 12'] },
		// },

		/* Test against branded browsers. */
		// {
		//   name: 'Microsoft Edge',
		//   use: { ...devices['Desktop Edge'], channel: 'msedge' },
		// },
		// {
		//   name: 'Google Chrome',
		//   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
		// },
	],

	/* Run your local dev server before starting the tests */
	webServer: {
		command: "yarn build && yarn preview --host 0.0.0.0",
		url: "http://localhost:4173",
		reuseExistingServer: !process.env.CI,
	},

	globalSetup: "./global-setup",
	// globalTeardown: './global-teardown',
});
