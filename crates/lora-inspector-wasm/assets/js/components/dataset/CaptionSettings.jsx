import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function CaptionSettings({ metadata }) {
	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Caption dropout rate"
				valueClassName="number"
				value={metadata.get("ss_caption_dropout_rate")}
			/>
			<MetaAttribute
				name="Caption dropout every n epochs"
				valueClassName="number"
				value={metadata.get("ss_caption_dropout_every_n_epochs")}
			/>

			<MetaAttribute
				name="Caption tag dropout rate"
				valueClassName="number"
				value={metadata.get("ss_caption_tag_dropout_rate")}
			/>
		</div>
	);
}
