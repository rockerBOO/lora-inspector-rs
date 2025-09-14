import { useState } from "react";

export function Raw({ metadata, filename }) {
	const [showRaw, setShowRaw] = useState(undefined);
	const [wrapText, setWrapText] = useState(false);

	if (showRaw) {
		const entries = Object.fromEntries(metadata);

		const sortedEntries = Object.keys(entries)
			.sort()
			.reduce((obj, key) => {
				obj[key] = entries[key];
				return obj;
			}, {});

		return (
			<div className="full-overlay">
				<pre className={wrapText ? "wrap" : ""}>
					{JSON.stringify(sortedEntries, null, 2)}
				</pre>

				<div className="action-overlay">
					<button
						type="button"
						className="download"
						onClick={() => {
							const data = `text/json;charset=utf-8,${encodeURIComponent(JSON.stringify(sortedEntries, null, 2))}`;
							const a = document.createElement("a");
							a.href = `data:${data}`;
							a.download = `${filename.replace(".safetensors", "")}-metadata.json`;

							const container = document.body;
							container.appendChild(a);
							a.click();
							a.remove();
						}}
					>
						Download
					</button>
					<button
						type="button"
						className="close"
						onClick={() => {
							setWrapText(!wrapText);
						}}
					>
						Wrap
					</button>
					<button
						type="button"
						className="close"
						onClick={() => {
							setShowRaw(false);
						}}
					>
						Close
					</button>
				</div>
			</div>
		);
	}

	return (
		<div
			style={{
				display: "grid",
				justifyItems: "end",
			}}
		>
			<button
				type="button"
				className="raw"
				onClick={() => {
					setShowRaw(true);
				}}
			>
				Show raw
			</button>
		</div>
	);
}
