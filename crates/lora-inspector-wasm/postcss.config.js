import cssnano from "cssnano";
import postcssPresetEnv from "postcss-preset-env";

export default {
	plugins: [
		postcssPresetEnv({
			features: {},
		}),
		cssnano({
			preset: "default",
		}),
	],
};
