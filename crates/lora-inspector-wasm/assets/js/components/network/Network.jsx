import React, { useState, useEffect } from "react";
import { trySyncMessage } from "../../message.js";
import { MetaAttribute } from "../ui/MetaAttribute.jsx";
import { BOFTNetwork } from "./BOFTNetwork.jsx";
import { DiagOFTNetwork } from "./DiagOFTNetwork.jsx";
import { LoKrNetwork } from "./LoKrNetwork.jsx";
import { LoRANetwork } from "./LoRANetwork.jsx";
import { supportsDoRA } from "./NetworkUtils.js";

export function Network({ metadata, filename, worker }) {
	const [networkModule, setNetworkModule] = useState(
		metadata.get("ss_network_module"),
	);
	const [networkType, setNetworkType] = useState("");
	const [networkArgs, setNetworkArgs] = useState(
		metadata.has("ss_network_args")
			? JSON.parse(metadata.get("ss_network_args"))
			: null,
	);
	const [weightDecomposition, setWeightDecomposition] = useState(null);
	const [rankStabilized, setRankStabilized] = useState(false);

	useEffect(() => {
		trySyncMessage(
			{ messageType: "network_args", name: filename },
			worker,
		).then((resp) => {
			setNetworkArgs(resp.networkArgs);
		});
		trySyncMessage(
			{ messageType: "network_module", name: filename },
			worker,
		).then((resp) => {
			setNetworkModule(resp.networkModule);
		});
		trySyncMessage(
			{ messageType: "network_type", name: filename },
			worker,
		).then((resp) => {
			setNetworkType(resp.networkType);
		});
		trySyncMessage(
			{ messageType: "weight_decomposition", name: filename },
			worker,
		).then((resp) => {
			setWeightDecomposition(resp.weightDecomposition);
		});
		trySyncMessage(
			{ messageType: "rank_stabilized", name: filename },
			worker,
		).then((resp) => {
			setRankStabilized(resp.rankStabilized);
		});
	}, [filename, worker]);

	let networkOptions;

	if (networkType === "DiagOFT") {
		networkOptions = (
			<DiagOFTNetwork metadata={metadata} filename={filename} worker={worker} />
		);
	} else if (networkType === "BOFT") {
		networkOptions = (
			<BOFTNetwork metadata={metadata} filename={filename} worker={worker} />
		);
	} else if (networkType === "LoKr") {
		networkOptions = <LoKrNetwork metadata={metadata} />;
	} else {
		networkOptions = (
			<LoRANetwork metadata={metadata} filename={filename} worker={worker} />
		);
	}

	return [
		<div key="network" className="row space-apart">
			<MetaAttribute
				containerProps={{ id: "network-module" }}
				name="Network module"
				valueClassName="attribute"
				value={networkModule}
			/>
			<MetaAttribute
				containerProps={{ id: "network-type" }}
				name="Network type"
				valueClassName="attribute"
				value={networkType}
			/>
			{networkOptions}
			<MetaAttribute
				name="Network dropout"
				valueClassName="number"
				value={metadata.get("ss_network_dropout")}
			/>
			<MetaAttribute
				name="Module dropout"
				valueClassName="number"
				value={
					networkArgs && "module_dropout" in networkArgs
						? networkArgs.module_dropout
						: "None"
				}
			/>
			<MetaAttribute
				name="Rank dropout"
				valueClassName="number"
				value={
					networkArgs && "rank_dropout" in networkArgs
						? networkArgs.rank_dropout
						: "None"
				}
			/>
		</div>,
		<div key="network-rank" className="row space-apart">
			{supportsDoRA(networkType) && (
				<MetaAttribute
					name="Weight decomposition (DoRA)"
					valueClassName="number"
					value={weightDecomposition ?? "False"}
				/>
			)}
			<MetaAttribute
				name="Rank-stabilized"
				valueClassName="number"
				value={rankStabilized ? "True" : "False"}
			/>
			<MetaAttribute
				name="Gradient Checkpointing"
				valueClassName="number"
				value={metadata.get("ss_gradient_checkpointing")}
			/>
		</div>,
		<div key="network-args" className="row space-apart">
			<MetaAttribute
				name="Network args"
				containerProps={{
					id: "network-args",
					style: { gridColumn: "1 / span 6" },
				}}
				valueClassName="args"
				value={networkArgs ? JSON.stringify(networkArgs) : "None"}
			/>
		</div>,
	];
}
