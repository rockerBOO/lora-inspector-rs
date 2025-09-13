import React from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function Attention({ k, q, v, out }) {
	// k + q => softmax (*v) -> proj
	return (
		<div>
			<div>
				<MetaAttribute name="Key" value={k} />
				<MetaAttribute name="Query" value={q} />
			</div>
			<div>
				<MetaAttribute name="Value" value={v} />
			</div>
			<div>
				<MetaAttribute name="Out Proj" value={out} />
			</div>
		</div>
	);
}
