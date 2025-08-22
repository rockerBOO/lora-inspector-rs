import React, { useState, useEffect } from "react";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { trySyncMessage } from "../../message.js";

export function Precision({ filename, worker }) {
	const [precision, setPrecision] = useState("");

	useEffect(() => {
		trySyncMessage({ messageType: "precision", name: filename }, worker).then(
			(resp) => {
				setPrecision(resp.precision);
			},
		);
	}, [filename, worker]);

	return (
		<MetaAttribute name="Precision" valueClassName="number" value={precision} />
	);
}
