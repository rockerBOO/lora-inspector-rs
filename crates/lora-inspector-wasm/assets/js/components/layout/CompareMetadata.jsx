const FILTERED_KEYS = new Set([
	"ss_training_finished_at",
	"ss_training_started_at",
	"ss_session_id",
	"ss_output_name",
	"modelspec.date",
]);

const COLLAPSED_KEYS = new Set(["ss_datasets"]);

function CollapsedValue({ value }) {
	return (
		<details>
			<summary>Show</summary>
			{value}
		</details>
	);
}

export function CompareMetadata({
	metadataA,
	filenameA,
	metadataB,
	filenameB,
	onViewA,
	onViewB,
}) {
	const entriesA = metadataA ? Object.fromEntries(metadataA) : {};
	const entriesB = metadataB ? Object.fromEntries(metadataB) : {};

	const allKeys = Array.from(
		new Set([...Object.keys(entriesA), ...Object.keys(entriesB)]),
	).sort();

	const diffKeys = allKeys.filter(
		(key) => !FILTERED_KEYS.has(key) && entriesA[key] !== entriesB[key],
	);

	return (
		<div className="compare-metadata">
			<nav>
				<ul>
					<li>
						<button type="button" onClick={onViewA}>
							View {filenameA}
						</button>
					</li>
					<li>
						<button type="button" onClick={onViewB}>
							View {filenameB}
						</button>
					</li>
				</ul>
			</nav>
			<h2>Metadata Comparison</h2>
			<table>
				<thead>
					<tr>
						<th>Key</th>
						<th title={filenameA}>{filenameA}</th>
						<th title={filenameB}>{filenameB}</th>
					</tr>
				</thead>
				<tbody>
					{diffKeys.map((key) => {
						const collapsed = COLLAPSED_KEYS.has(key);
						const valA = entriesA[key];
						const valB = entriesB[key];
						return (
							<tr key={key}>
								<td className="compare-key">{key}</td>
								<td
									className={
										valA === undefined ? "compare-missing" : "compare-value"
									}
								>
									{valA === undefined ? (
										"—"
									) : collapsed ? (
										<CollapsedValue value={valA} />
									) : (
										valA
									)}
								</td>
								<td
									className={
										valB === undefined ? "compare-missing" : "compare-value"
									}
								>
									{valB === undefined ? (
										"—"
									) : collapsed ? (
										<CollapsedValue value={valB} />
									) : (
										valB
									)}
								</td>
							</tr>
						);
					})}
					{diffKeys.length === 0 && (
						<tr>
							<td colSpan={3}>No differences found.</td>
						</tr>
					)}
				</tbody>
			</table>
		</div>
	);
}
