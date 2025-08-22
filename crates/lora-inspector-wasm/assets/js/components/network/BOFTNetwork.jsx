import React, { useState, useEffect } from "react";
import { trySyncMessage } from "../../message.js";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function BOFTNetwork({ metadata, filename, worker }) {
	const [dims, setDims] = useState([metadata.get("ss_network_dim")]);

	useEffect(() => {
		trySyncMessage({ messageType: "dims", name: filename }, worker).then(
			(resp) => {
				setDims(resp.dims);
			},
		);
	}, [filename, worker]);

	return (
		<MetaAttribute
			name="Network factor"
			valueClassName="rank"
			value={dims.join(", ")}
		/>
	);
}
