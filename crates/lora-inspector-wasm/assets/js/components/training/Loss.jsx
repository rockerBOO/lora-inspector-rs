import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function Loss({ metadata }) {
	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Loss type"
				valueClassName="attribute"
				value={metadata.get("ss_loss_type")}
			/>
			{metadata.get("ss_loss_type") === "huber" && (
				<MetaAttribute
					name="Huber schedule"
					valueClassName="attribute"
					value={metadata.get("ss_huber_schedule")}
				/>
			)}
			{metadata.get("ss_loss_type") === "huber" && (
				<MetaAttribute
					name="Huber c"
					valueClassName="number"
					value={metadata.get("ss_huber_c")}
				/>
			)}
			{metadata.get("ss_loss_type") === "huber" && (
				<MetaAttribute
					name="Huber scale"
					valueClassName="number"
					value={metadata.get("ss_huber_scale") ?? 1.0}
				/>
			)}
			{metadata.has("ss_masked_loss") && (
				<MetaAttribute
					name="Masked Loss"
					value={metadata.get("ss_masked_loss")}
				/>
			)}
		</div>
	);
}
