import React from "react";
import { ModelSpec } from "../metadata/ModelSpec.jsx";

export function Header({ metadata }) {
	return (
		<header>
			<ModelSpec metadata={metadata} />
		</header>
	);
}
