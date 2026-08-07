import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

// MiniMax H3 (diffusers-native PEFT LoRA) uses bare "transformer_blocks.N.*"
// keys (no "transformer." prefix, unlike the Flux PEFT convention) plus a
// "token_refiner.refiner_blocks.N.*" block group, with lora_A/lora_B tensors
// carrying a ".default." adapter-name suffix.

test("parseSDKey maps bare transformer_blocks keys to TB blocks", (t) => {
	const result = parseSDKey(
		"transformer_blocks.5.attn.to_q.lora_A.default.weight",
	);

	t.is(result.name, "TB05");
	t.is(result.blockId, 5);
	t.is(result.idx, 5);
	t.true(result.isAttention);
});

test("parseSDKey maps token_refiner.refiner_blocks keys to TR blocks", (t) => {
	const result = parseSDKey(
		"token_refiner.refiner_blocks.1.attn.to_q.lora_A.default.weight",
	);

	t.is(result.name, "TR01");
	t.is(result.blockId, 1);
	t.is(result.idx, 1);
	t.true(result.isAttention);
});

test("parseSDKey distinguishes ff blocks from attn blocks for token_refiner", (t) => {
	const result = parseSDKey(
		"token_refiner.refiner_blocks.0.ff.net.0.proj.lora_A.default.weight",
	);

	t.is(result.name, "TR00");
	t.false(result.isAttention);
});
