import { MetaAttribute } from "../ui/MetaAttribute.jsx";

export function LoKrNetwork({ metadata }) {
	return [
		<MetaAttribute
			name="Network Rank/Dimension"
			containerProps={{ id: "network-rank" }}
			valueClassName="rank"
			value={metadata.get("ss_network_dim")}
			key="network-rank"
		/>,
		<MetaAttribute
			name="Network Alpha"
			containerProps={{ id: "network-alpha" }}
			valueClassName="alpha"
			value={metadata.get("ss_network_alpha")}
			key="network-alpha"
		/>,
	];
}
