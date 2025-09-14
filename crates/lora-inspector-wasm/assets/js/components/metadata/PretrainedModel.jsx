import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { VAE } from "./VAE.jsx";

export function PretrainedModel({ metadata }) {
	return (
		<div className="pretrained-model row space-apart">
			<MetaAttribute
				name="SD model name"
				value={metadata.get("ss_sd_model_name")}
			/>
			<div>
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
				<MetaAttribute
					name="sd-scripts commit hash"
					value={metadata.get("ss_sd_scripts_commit_hash")}
					valueClassName="hash"
				/>
			</div>
			<div>
				<VAE metadata={metadata} />
			</div>
		</div>
	);
}
