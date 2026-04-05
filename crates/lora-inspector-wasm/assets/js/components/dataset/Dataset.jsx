import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { Buckets } from "./Buckets.jsx";

function DatasetDir({ dir, info, label }) {
	return (
		<div className="row space-apart">
			<MetaAttribute name={label} value={dir} />
			<MetaAttribute
				name="Repeats"
				valueClassName="number"
				value={info.n_repeats}
			/>
			<MetaAttribute
				name="Images"
				valueClassName="number"
				value={info.img_count}
			/>
		</div>
	);
}

function KohyaDataset({ metadata }) {
	let datasetDirs = null;
	let regDatasetDirs = null;

	try {
		if (metadata.has("ss_dataset_dirs")) {
			datasetDirs = JSON.parse(metadata.get("ss_dataset_dirs"));
		}
		if (metadata.has("ss_reg_dataset_dirs")) {
			regDatasetDirs = JSON.parse(metadata.get("ss_reg_dataset_dirs"));
		}
	} catch (e) {
		console.error(e);
	}

	const resolution = metadata.get("ss_resolution");
	const enableBucket = metadata.get("ss_enable_bucket");

	return (
		<div>
			<div className="row space-apart">
				<MetaAttribute name="Resolution" value={resolution} />
				<MetaAttribute name="Enable bucket" value={enableBucket} />
				{enableBucket !== "False" && (
					<>
						<MetaAttribute
							name="Min bucket resolution"
							valueClassName="number"
							value={metadata.get("ss_min_bucket_reso")}
						/>
						<MetaAttribute
							name="Max bucket resolution"
							valueClassName="number"
							value={metadata.get("ss_max_bucket_reso")}
						/>
					</>
				)}
				<MetaAttribute
					name="Shuffle caption"
					value={metadata.get("ss_shuffle_caption")}
				/>
				{metadata.has("ss_num_reg_images") && (
					<MetaAttribute
						name="Reg images"
						valueClassName="number"
						value={metadata.get("ss_num_reg_images")}
					/>
				)}
			</div>

			{datasetDirs && Object.keys(datasetDirs).length > 0 && (
				<div>
					<h3>Training datasets</h3>
					{Object.entries(datasetDirs).map(([dir, info]) => (
						<DatasetDir key={dir} dir={dir} info={info} label="Directory" />
					))}
				</div>
			)}

			{regDatasetDirs && Object.keys(regDatasetDirs).length > 0 && (
				<div>
					<h3>Regularization datasets</h3>
					{Object.entries(regDatasetDirs).map(([dir, info]) => (
						<DatasetDir key={dir} dir={dir} info={info} label="Directory" />
					))}
				</div>
			)}
		</div>
	);
}

export function Dataset({ metadata }) {
	if (metadata.has("ss_datasets")) {
		let datasets;
		try {
			datasets = JSON.parse(metadata.get("ss_datasets"));
		} catch (e) {
			console.error(e);
			datasets = [];
		}
		return (
			<div>
				{datasets.map((dataset, i) => (
					<Buckets
						key={`dataset-${dataset.name ?? dataset.image_directory ?? i}`}
						dataset={dataset}
						metadata={metadata}
					/>
				))}
			</div>
		);
	}

	if (metadata.has("ss_dataset_dirs") || metadata.has("ss_resolution")) {
		return <KohyaDataset metadata={metadata} />;
	}

	return null;
}
