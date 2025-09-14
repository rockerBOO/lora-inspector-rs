import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function BucketInfo({ dataset }) {
	// No bucket info
	if (!dataset.bucket_info) {
		return;
	}

	// No buckets data
	if (!dataset.bucket_info.buckets) {
		return;
	}

	return (
		<div className="bucket-infos">
			{Object.entries(dataset.bucket_info.buckets).map(([key, bucket]) => {
				return (
					<div key={key} className="bucket">
						<MetaAttribute
							name={`Bucket ${key}`}
							value={`${bucket.resolution[0]}x${bucket.resolution[1]}: ${
								bucket.count
							} image${bucket.count > 1 ? "s" : ""}`}
						/>
					</div>
				);
			})}
		</div>
	);
}
