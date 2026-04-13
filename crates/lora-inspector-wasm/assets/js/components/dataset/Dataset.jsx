import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { Buckets } from "./Buckets.jsx";
import { CaptionSettings } from "./CaptionSettings.jsx";
import { TagFrequency } from "./TagFrequency.jsx";

const SOURCE_KEYS = [
	"image_directory",
	"video_directory",
	"image_jsonl_file",
	"video_jsonl_file",
];

function getSource(dataset) {
	for (const k of SOURCE_KEYS) {
		if (k in dataset) return dataset[k];
	}
	return null;
}

function withoutSource(dataset) {
	const result = { ...dataset };
	for (const k of SOURCE_KEYS) delete result[k];
	return result;
}

function datasetsShareSettings(datasets) {
	if (datasets.length <= 1) return true;
	const first = JSON.stringify(withoutSource(datasets[0]));
	return datasets.every((d) => JSON.stringify(withoutSource(d)) === first);
}

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

function TopLevelTagFrequency({ metadata }) {
	if (!metadata.has("ss_tag_frequency")) return null;

	let tagFrequency;
	try {
		tagFrequency = JSON.parse(metadata.get("ss_tag_frequency"));
	} catch (e) {
		console.error(e);
		return null;
	}

	const dirs = Object.entries(tagFrequency);
	if (dirs.length === 0) return null;

	return (
		<div>
			<h3>Tag frequencies</h3>
			<div className="tag-frequencies row space-apart">
				{dirs.map(([dir, frequency]) => (
					<div key={dir}>
						<h4>{dir}</h4>
						<TagFrequency tagFrequency={frequency} />
					</div>
				))}
			</div>
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

		if (datasetsShareSettings(datasets)) {
			const shared = withoutSource(datasets[0]);
			const sources = datasets.map(getSource).filter(Boolean);
			return (
				<div>
					<MetaAttribute name="Directories" value={sources.join(", ")} />
					<Buckets dataset={shared} metadata={metadata} />
					{metadata.has("ss_max_token_length") && (
						<MetaAttribute
							name="Max token length"
							valueClassName="number"
							value={metadata.get("ss_max_token_length")}
						/>
					)}
					<CaptionSettings metadata={metadata} />
					<TopLevelTagFrequency metadata={metadata} />
				</div>
			);
		}

		return (
			<div>
				{datasets.map((dataset, i) => (
					<Buckets
						key={`dataset-${dataset.image_directory ?? i}`}
						dataset={dataset}
						metadata={metadata}
					/>
				))}
				{metadata.has("ss_max_token_length") && (
					<MetaAttribute
						name="Max token length"
						valueClassName="number"
						value={metadata.get("ss_max_token_length")}
					/>
				)}
				<CaptionSettings metadata={metadata} />
				<TopLevelTagFrequency metadata={metadata} />
			</div>
		);
	}

	if (metadata.has("ss_dataset_dirs") || metadata.has("ss_resolution")) {
		return (
			<div>
				<KohyaDataset metadata={metadata} />
				<MetaAttribute
					name="Max token length"
					valueClassName="number"
					value={metadata.get("ss_max_token_length")}
				/>
				<CaptionSettings metadata={metadata} />
				<TopLevelTagFrequency metadata={metadata} />
			</div>
		);
	}

	return null;
}
