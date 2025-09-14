import { useEffect, useState } from "react";
import { trySyncMessage } from "../../message.js";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function LoRANetwork({ metadata, filename, worker }) {
	const [alphas, setAlphas] = useState([
		metadata?.get("ss_network_alpha") ?? undefined,
	]);
	const [dims, setDims] = useState([
		metadata?.get("ss_network_dim") ?? undefined,
	]);

	useEffect(() => {
		trySyncMessage({ messageType: "alphas", name: filename }, worker).then(
			(resp) => {
				setAlphas(resp.alphas);
			},
		);
		trySyncMessage({ messageType: "dims", name: filename }, worker).then(
			(resp) => {
				setDims(resp.dims);
			},
		);
	}, [filename, worker]);

	return [
		<MetaAttribute
			name="Network Rank/Dimension"
			containerProps={{ id: "network-rank" }}
			valueClassName="rank"
			value={dims.join(", ")}
			key="network-rank"
		/>,
		<MetaAttribute
			name="Network Alpha"
			containerProps={{ id: "network-alpha" }}
			valueClassName="alpha"
			value={alphas
				.filter((alpha) => alpha)
				.map((alpha) => {
					if (typeof alpha === "number") {
						return alpha.toPrecision(2);
					}

					if (alpha.includes(".")) {
						return Number.parseFloat(alpha).toPrecision(2);
					}
					return Number.parseInt(alpha);
				})
				.join(", ")}
			key="network-alpha"
		/>,
	];
}
