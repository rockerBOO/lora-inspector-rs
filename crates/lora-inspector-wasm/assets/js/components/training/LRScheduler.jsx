import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function LRScheduler({ metadata }) {
	const lrScheduler = metadata.has("ss_lr_scheduler_type")
		? metadata.get("ss_lr_scheduler_type")
		: metadata.get("ss_lr_scheduler");

	return (
		<div className="row space-apart">
			<MetaAttribute
				name="LR Scheduler"
				containerProps={{ style: { gridColumn: "1 / span 3" } }}
				value={lrScheduler}
			/>

			{metadata.has("ss_lr_warmup_steps") && (
				<MetaAttribute
					name="Warmup steps"
					valueClassName="lr number"
					value={metadata.get("ss_lr_warmup_steps")}
				/>
			)}
			{(metadata.get("ss_unet_lr") === "None" ||
				metadata.get("ss_text_encoder_lr") === "None") && (
				<MetaAttribute
					name="Learning rate"
					valueClassName="lr number"
					value={metadata.get("ss_learning_rate")}
				/>
			)}
			<MetaAttribute
				name="UNet learning rate"
				valueClassName="lr number"
				value={metadata.get("ss_unet_lr")}
			/>
			<MetaAttribute
				name="Text encoder learning rate"
				valueClassName="lr number"
				value={metadata.get("ss_text_encoder_lr")}
			/>
		</div>
	);
}
