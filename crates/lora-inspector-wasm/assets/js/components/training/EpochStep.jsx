import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function EpochStep({ metadata }) {
	// ss_epoch = current epoch at time of save
	// ss_num_epochs = total planned epochs (from --max_train_epochs)
	// ss_max_train_epochs = alias used by some trainers
	const totalEpochs =
		metadata.get("ss_num_epochs") ?? metadata.get("ss_max_train_epochs");

	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Epoch"
				valueClassName="number"
				value={metadata.get("ss_epoch")}
			/>
			{totalEpochs !== undefined && (
				<MetaAttribute
					name="Total epochs"
					valueClassName="number"
					value={totalEpochs}
				/>
			)}
			<MetaAttribute
				name="Steps"
				valueClassName="number"
				value={metadata.get("ss_steps")}
			/>
			<MetaAttribute
				name="Max Train Steps"
				valueClassName="number"
				value={metadata.get("ss_max_train_steps")}
			/>
			<MetaAttribute
				name="Seed"
				valueClassName="number"
				value={metadata.get("ss_seed")}
			/>
			{metadata.has("ss_gradient_checkpointing") && (
				<MetaAttribute
					name="Gradient Checkpointing"
					valueClassName="boolean"
					value={metadata.get("ss_gradient_checkpointing")}
				/>
			)}
		</div>
	);
}
