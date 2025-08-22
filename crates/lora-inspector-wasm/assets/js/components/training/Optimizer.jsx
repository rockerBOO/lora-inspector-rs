import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function Optimizer({ metadata }) {
	return (
		<div className="row space-apart">
			<MetaAttribute
				name="Optimizer"
				containerProps={{ className: "span-3" }}
				value={metadata.get("ss_optimizer")}
			/>
			<MetaAttribute
				name="Seed"
				valueClassName="number"
				value={metadata.get("ss_seed")}
			/>
		</div>
	);
}
