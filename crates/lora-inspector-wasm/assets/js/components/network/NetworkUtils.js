export function supportsDoRA(networkType) {
	return (
		networkType === "LoRA" ||
		networkType === "LoHa" ||
		networkType === "LoRAFA" ||
		networkType === "LoKr" ||
		networkType === "GLoRA"
	);
}
