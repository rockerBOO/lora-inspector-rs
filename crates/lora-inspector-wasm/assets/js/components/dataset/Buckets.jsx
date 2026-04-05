import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { BucketInfo } from "./BucketInfo.jsx";
import { Subset } from "./Subset.jsx";
import { TagFrequency } from "./TagFrequency.jsx";

function formatResolution(resolution) {
	if (Array.isArray(resolution)) return `${resolution[0]}x${resolution[1]}`;
	return resolution;
}

export function Buckets({ dataset, metadata }) {
	return [
		<div key="buckets" className="row space-apart">
			{"image_directory" in dataset && (
				<MetaAttribute name="Directory" value={dataset.image_directory} />
			)}
			{"num_repeats" in dataset && (
				<MetaAttribute
					name="Repeats"
					valueClassName="number"
					value={dataset.num_repeats}
				/>
			)}
			<MetaAttribute
				name="Buckets"
				value={dataset.enable_bucket ? "True" : "False"}
			/>
			{"bucket_no_upscale" in dataset && (
				<MetaAttribute
					name="No upscale"
					value={dataset.bucket_no_upscale ? "True" : "False"}
				/>
			)}
			{"min_bucket_reso" in dataset && (
				<MetaAttribute
					name="Min bucket resolution"
					valueClassName="number"
					value={dataset.min_bucket_reso}
				/>
			)}
			{"max_bucket_reso" in dataset && (
				<MetaAttribute
					name="Max bucket resolution"
					valueClassName="number"
					value={dataset.max_bucket_reso}
				/>
			)}
			<MetaAttribute
				name="Resolution"
				valueClassName="number"
				value={formatResolution(dataset.resolution)}
			/>
			{"caption_extension" in dataset && (
				<MetaAttribute
					name="Caption extension"
					value={dataset.caption_extension}
				/>
			)}
			{"has_control" in dataset && (
				<MetaAttribute
					name="Has control"
					value={dataset.has_control ? "True" : "False"}
				/>
			)}
		</div>,

		<div key="bucket-info">
			<BucketInfo metadata={metadata} dataset={dataset} />
		</div>,

		...("subsets" in dataset
			? [
					<h3 key="subsets-header" className="row space-apart">
						Subsets:
					</h3>,
					<div key="subsets" className="subsets">
						{dataset.subsets.map((subset, i) => (
							<Subset
								key={`subset-${subset.image_dir}-${i}`}
								metadata={metadata}
								subset={subset}
							/>
						))}
					</div>,
				]
			: []),

		...(Object.keys(dataset?.tag_frequency ?? {}).length > 0
			? [
					<h3 key="header-tag-frequencies">Tag frequencies</h3>,
					<div
						key="tag-frequencies"
						className="tag-frequencies row space-apart"
					>
						{Object.entries(dataset.tag_frequency).map(([dir, frequency]) => (
							<div key={dir}>
								<h3>{dir}</h3>
								<TagFrequency
									key="tag-frequency"
									tagFrequency={frequency}
									metadata={metadata}
								/>
							</div>
						))}
					</div>,
				]
			: []),
	];
}
