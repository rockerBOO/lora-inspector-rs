import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function MultiresNoise({ metadata }) {
	if (metadata.get("ss_multires_noise_iterations") === "None") {
		return [];
	}

	return [
		<MetaAttribute
			name="MultiRes Noise Iterations"
			valueClassName="number"
			value={metadata.get("ss_multires_noise_iterations")}
			key="multires noise iterations"
		/>,
		<MetaAttribute
			name="MultiRes Noise Discount"
			valueClassName="number"
			value={metadata.get("ss_multires_noise_discount")}
			key="multires noise discount"
		/>,
	];
}
