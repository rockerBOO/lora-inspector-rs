import React from "react";
import { Raw } from "./Raw.jsx";

export function Headline({ metadata, filename }) {
	let raw;
	if (metadata) {
		raw = <Raw key="raw" metadata={metadata} filename={filename} />;
	}

	return (
		<div className="headline">
			<div key="headline">
				<div key="lora file">LoRA file</div>
				<h1 key="filename">{filename}</h1>
			</div>
			{raw}
		</div>
	);
}
