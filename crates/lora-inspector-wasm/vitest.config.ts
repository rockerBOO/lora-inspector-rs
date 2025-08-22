import { fileURLToPath } from "node:url";
import { configDefaults, defineConfig, mergeConfig } from "vitest/config";
import viteConfig from "./vite.config";

export default mergeConfig(
	viteConfig,
	defineConfig({
		test: {
			environment: "jsdom",
			include: ["tests/**/*.test.jsx"],
			exclude: [...configDefaults.exclude, "e2e-test/**", "e2e-tests/**", "tests/**/*.test.js"],
			root: fileURLToPath(new URL("./", import.meta.url)),
			setupFiles: ["./assets/js/test-setup.ts"],
		},
	}),
);
