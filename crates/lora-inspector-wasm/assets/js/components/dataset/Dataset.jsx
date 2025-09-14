import { Buckets } from "./Buckets.jsx";

export function Dataset({ metadata }) {
	let datasets;
	if (metadata.has("ss_datasets")) {
		try {
			datasets = JSON.parse(metadata.get("ss_datasets"));
		} catch (e) {
			console.log(metadata.get("ss_datasets"));
			console.error(e);
			datasets = [];
		}
	} else {
		datasets = [];
	}
	return (
		<div>
			<h2>Dataset</h2>

			{datasets.map((dataset, i) => {
				return (
					<Buckets
						key={`dataset-${dataset.name || dataset.id || i}`}
						dataset={dataset}
						metadata={metadata}
					/>
				);
			})}
		</div>
	);
}
