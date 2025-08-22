import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { MultiresNoise } from "./MultiresNoise.jsx";

export function Noise({ metadata }) {
	return (
		<div className="row space-apart">
			<MetaAttribute
				name="IP noise gamma"
				valueClassName="number"
				value={metadata.get("ss_ip_noise_gamma")}
				{...(metadata.get("ss_ip_noise_gamma_random_strength") !==
					undefined && {
					secondaryName: "Random strength:",
					secondary: metadata.get("ss_ip_noise_gamma_random_strength")
						? "True"
						: "False",
				})}
			/>
			<MetaAttribute
				name="Noise offset"
				valueClassName="number"
				value={metadata.get("ss_noise_offset")}
				{...(metadata.get("ss_ip_noise_gamma_random_strength") !==
					undefined && {
					secondaryName: "Random strength:",
					secondary: metadata.get("ss_noise_offset_random_strength")
						? "True"
						: "False",
				})}
			/>
			<MetaAttribute
				name="Adaptive noise scale"
				valueClassName="number"
				value={metadata.get("ss_adaptive_noise_scale")}
			/>
			<MultiresNoise metadata={metadata} />
		</div>
	);
}
