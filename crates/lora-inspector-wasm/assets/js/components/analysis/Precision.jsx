import { useEffect, useState } from "react";
import { trySyncMessage } from "../../message.js";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";

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
