export default {
	files: ["tests/**/*"],
	nodeArguments: ["--loader=@babel/register"],
	extensions: ["js", "jsx"],
	babel: {
		compileEnhancements: false,
		presets: [
			["@babel/preset-env", { targets: { node: "current" } }],
			["@babel/preset-react", { runtime: "automatic" }],
		],
	},
};
