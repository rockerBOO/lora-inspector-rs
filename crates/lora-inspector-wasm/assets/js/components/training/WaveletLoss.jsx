import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { WaveletLossWeights } from "./WaveletLossWeights.jsx";

export function WaveletLoss({ metadata }) {
	if (metadata.get("ss_wavelet_loss") !== "True") {
		return [];
	}

	const parse = (key) => {
		const value = metadata.get(key);
		if (!value || value === "None") {
			return null;
		}
		return JSON.parse(value);
	};

	const bandWeights = parse("ss_wavelet_loss_band_weights");
	const bandLevelWeights = parse("ss_wavelet_loss_band_level_weights");
	const quaternionComponentWeights = parse(
		"ss_wavelet_loss_quaternion_component_weights",
	);

	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Wavelet loss alpha"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_alpha")}
			/>
			<MetaAttribute
				name="Band weights"
				valueClassName="attribute"
				value={
					bandWeights ? <WaveletLossWeights weights={bandWeights} /> : "None"
				}
			/>
			<MetaAttribute
				name="Band level weights"
				valueClassName="attribute"
				value={
					bandLevelWeights ? (
						<WaveletLossWeights weights={bandLevelWeights} />
					) : (
						"None"
					)
				}
			/>
			<MetaAttribute
				name="Energy ratio"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_energy_ratio")}
			/>
			<MetaAttribute
				name="Energy scale factor"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_energy_scale_factor")}
			/>
			<MetaAttribute
				name="Wavelet level"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_level")}
			/>
			<MetaAttribute
				name="LL level threshold"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_ll_level_threshold")}
			/>
			<MetaAttribute
				name="Wavelet loss primary"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_primary")}
			/>
			<MetaAttribute
				name="Quaternion component weights"
				valueClassName="attribute"
				value={
					quaternionComponentWeights ? (
						<WaveletLossWeights weights={quaternionComponentWeights} />
					) : (
						"None"
					)
				}
			/>
			<MetaAttribute
				name="Transform"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_transform")}
			/>
			<MetaAttribute
				name="Wavelet loss function"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_type")}
			/>
			<MetaAttribute
				name="Wavelet"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_wavelet")}
			/>
			<MetaAttribute
				name="Normalize bands"
				valueClassName="attribute"
				value={metadata.get("ss_wavelet_loss_normalize_bands")}
			/>
		</div>
	);
}
