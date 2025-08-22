import React from "react";
import { Headline } from "./Headline.jsx";
import { Weight } from "../analysis/Weight.jsx";

export function NoMetadata({ metadata, filename, worker }) {
	return (
		<main>
			<Headline filename={filename} />
			<Weight metadata={metadata} filename={filename} worker={worker} />
		</main>
	);
}
