import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";

export default defineConfig({
	// worker: {
	//   // Enable classic web workers
	//   type: "classic",
	// },
	plugins: [wasm()],
	build: {
		// main: {
		//   entry: resolve(__dirname, "index.html"),
		// },
		// lib: {
		//   // Could also be a dictionary or array of multiple entry points
		//   entry: resolve(__dirname, "assets/js/lib.js"),
		//   name: "Inspector",
		//   // the proper extensions will be added
		//   fileName: "lora-inspector-lib",
		// },
		// rollupOptions: {
		//   // make sure to externalize deps that shouldn't be bundled
		//   // into your library
		//   external: ["vue"],
		//   output: {
		//     // Provide global variables to use in the UMD build
		//     // for externalized deps
		//     globals: {
		//       vue: "Vue",
		//     },
		//   },
		// },
	},
});
