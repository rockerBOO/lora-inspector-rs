import React, { useState, useEffect } from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { trySyncMessage } from "../../message.js";

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
