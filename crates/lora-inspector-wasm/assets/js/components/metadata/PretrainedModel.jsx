import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { VAE } from "./VAE.jsx";

export function PretrainedModel({ metadata }) {
	return (
		<div className="pretrained-model row space-apart">
			<MetaAttribute
				name="SD model name"
				value={metadata.get("ss_sd_model_name")}
			/>
			{metadata.has("ss_model_type") && (
				<MetaAttribute
					name="Model type"
					value={metadata.get("ss_model_type")}
				/>
			)}
			{metadata.has("ss_base_model_version") && (
				<MetaAttribute
					name="Base model version"
					value={metadata.get("ss_base_model_version")}
				/>
			)}
			{metadata.has("ss_output_name") && (
				<MetaAttribute
					name="Output name"
					value={metadata.get("ss_output_name")}
				/>
			)}
			<div className="span-2">
				<MetaAttribute
					name="Model hash"
					value={metadata.get("sshs_model_hash")}
					valueClassName="hash"
					metadata={metadata}
				/>
				<MetaAttribute
					name="Legacy model hash"
					value={metadata.get("sshs_legacy_hash")}
					metadata={metadata}
				/>
			</div>
			<div>
				<MetaAttribute
					name="Session ID"
					value={metadata.get("ss_session_id")}
				/>
				{metadata.has("ss_sd_scripts_commit_hash") && (
					<MetaAttribute
						name="sd-scripts commit hash"
						value={metadata.get("ss_sd_scripts_commit_hash")}
						valueClassName="hash"
					/>
				)}
			</div>
			<div>
				<VAE metadata={metadata} />
			</div>
		</div>
	);
}
