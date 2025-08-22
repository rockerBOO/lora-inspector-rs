import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function EpochStep({ metadata }) {
	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Epoch"
				valueClassName="number"
				value={metadata.get("ss_epoch")}
			/>
			<MetaAttribute
				name="Steps"
				valueClassName="number"
				value={metadata.get("ss_steps")}
			/>
			{metadata.get("ss_max_train_steps") === undefined && (
				<MetaAttribute
					name="Max Train Epochs"
					valueClassName="number"
					value={metadata.get("ss_max_train_epochs")}
				/>
			)}
			{metadata.get("ss_max_train_epochs") === undefined && (
				<MetaAttribute
					name="Max Train Steps"
					valueClassName="number"
					value={metadata.get("ss_max_train_steps")}
				/>
			)}
		</div>
	);
}
