import { MetaAttribute } from "../ui/MetaAttribute.jsx";

const FLUX_KEYS = [
	"ss_weighting_scheme",
	"ss_timestep_sampling",
	"ss_discrete_flow_shift",
	"ss_guidance_scale",
	"ss_logit_mean",
	"ss_logit_std",
	"ss_mode_scale",
	"ss_sigmoid_scale",
	"ss_model_prediction_type",
	"ss_bypass_flux_guidance",
	"ss_blocks_to_swap",
];

export function FluxTraining({ metadata }) {
	if (!FLUX_KEYS.some((k) => metadata.has(k))) return null;

	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Weighting scheme"
				value={metadata.get("ss_weighting_scheme")}
			/>
			<MetaAttribute
				name="Timestep sampling"
				value={metadata.get("ss_timestep_sampling")}
			/>
			<MetaAttribute
				name="Discrete flow shift"
				value={metadata.get("ss_discrete_flow_shift")}
			/>
			<MetaAttribute
				name="Guidance scale"
				valueClassName="number"
				value={metadata.get("ss_guidance_scale")}
			/>
			<MetaAttribute name="Logit mean" value={metadata.get("ss_logit_mean")} />
			<MetaAttribute name="Logit std" value={metadata.get("ss_logit_std")} />
			<MetaAttribute name="Mode scale" value={metadata.get("ss_mode_scale")} />
			<MetaAttribute
				name="Sigmoid scale"
				value={metadata.get("ss_sigmoid_scale")}
			/>
			{metadata.has("ss_model_prediction_type") && (
				<MetaAttribute
					name="Model prediction type"
					value={metadata.get("ss_model_prediction_type")}
				/>
			)}
			{metadata.has("ss_bypass_flux_guidance") && (
				<MetaAttribute
					name="Bypass flux guidance"
					valueClassName="boolean"
					value={metadata.get("ss_bypass_flux_guidance")}
				/>
			)}
			{metadata.has("ss_blocks_to_swap") && (
				<MetaAttribute
					name="Blocks to swap"
					valueClassName="number"
					value={metadata.get("ss_blocks_to_swap")}
				/>
			)}
		</div>
	);
}
