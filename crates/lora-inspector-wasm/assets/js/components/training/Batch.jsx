import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function Batch({ metadata }) {
	let batchSize;
	if (metadata.has("ss_batch_size_per_device")) {
		batchSize = metadata.get("ss_batch_size_per_device");
	} else {
		// The batch size is found inside the datasets.
		if (metadata.has("ss_datasets")) {
			let datasets;
			try {
				datasets = JSON.parse(metadata.get("ss_datasets"));
			} catch (e) {
				console.log(metadata.get("ss_datasets"));
				console.error(e);
				datasets = [];
			}

			for (const dataset of datasets) {
				if ("batch_size_per_device" in dataset) {
					batchSize = dataset.batch_size_per_device;
				}
			}
		}
	}

	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Num train images"
				valueClassName="number"
				value={metadata.get("ss_num_train_images")}
			/>
			<MetaAttribute
				name="Num batches per epoch"
				valueClassName="number"
				value={metadata.get("ss_num_batches_per_epoch")}
			/>
			<MetaAttribute
				name="Batch size"
				valueClassName="number"
				value={batchSize}
			/>
			<MetaAttribute
				name="Gradient Accumulation Steps"
				valueClassName="number"
				value={metadata.get("ss_gradient_accumulation_steps")}
			/>
		</div>
	);
}
