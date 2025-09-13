import React from "react";
import { Weight } from "../analysis/Weight.jsx";
import { Headline } from "./Headline.jsx";

export function NoMetadata({ metadata, filename, worker }) {
	return (
		<main>
			<Headline filename={filename} />
			<Weight metadata={metadata} filename={filename} worker={worker} />
		</main>
	);
}
