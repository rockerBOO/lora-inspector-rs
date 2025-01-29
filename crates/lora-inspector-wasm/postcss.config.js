import postcssPresetEnv from "postcss-preset-env";
import cssnano from "cssnano";

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
