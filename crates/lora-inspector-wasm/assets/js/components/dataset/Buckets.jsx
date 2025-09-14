import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { BucketInfo } from "./BucketInfo.jsx";
import { Subset } from "./Subset.jsx";
import { TagFrequency } from "./TagFrequency.jsx";

export function Buckets({ dataset, metadata }) {
	return [
		<div key="buckets" className="row space-apart">
			<MetaAttribute
				name="Buckets"
				value={dataset.enable_bucket ? "True" : "False"}
			/>
			<MetaAttribute
				name="Min bucket resolution"
				valueClassName="number"
				value={dataset.min_bucket_reso}
			/>
			<MetaAttribute
				name="Max bucket resolution"
				valueClassName="number"
				value={dataset.max_bucket_reso}
			/>
			<MetaAttribute
				name="Resolution"
				valueClassName="number"
				value={`${dataset.resolution[0]}x${dataset.resolution[0]}`}
			/>
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

		<h3 key="header-tag-frequencies">Tag frequencies</h3>,

		<div key="tag-frequencies" className="tag-frequencies row space-apart">
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
	];
}
