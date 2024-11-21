import { defineConfig } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

export default defineConfig({
  plugins: [
    viteStaticCopy({
      targets: [
        // Chartist isn't supported in vite, and we use a very old version so we will just move it
        {
          src: "./assets/js/*.min.js",
          dest: "./assets/js",
        },
      ],
      verbose: true,
    }),
    wasm(),
    topLevelAwait(),
  ],
  assetsInclude: ["**/*.wasm"],
  build: {
    commonjsOptions: { transformMixedEsModules: true },
  },
  worker: {
    format: "es",
  },
});
