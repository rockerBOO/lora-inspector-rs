/** @type {import('stylelint').Config} */
export default {
	extends: ["stylelint-config-standard"],
	ignoreFiles: ["assets/css/chartist.min.css", "assets/css/reset.css"],
	rules: {
		"custom-property-empty-line-before": null,
		"no-descending-specificity": null,
		"no-duplicate-selectors": null,
		"selector-class-pattern": [
			"^([a-z][a-z0-9-]*)(__[a-z0-9-]+)?(--[a-z0-9-]+)?$",
			{ "message": "Expected class selector to be BEM or kebab-case" },
		],
		"declaration-property-value-no-unknown": null,
		"declaration-property-value-keyword-no-deprecated": null,
	},
};
