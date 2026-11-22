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

// MiniMax H3 (ai-toolkit trained) uses "diffusion_model.blocks.N.*" and
// "diffusion_model.token_refiner.blocks.N.*" keys with attn.qkv_proj,
// attn.out_proj, mlp.fc1, mlp.fc2 sub-blocks and lora_A/lora_B tensors.

test("parseSDKey maps diffusion_model.blocks keys to TB blocks", (t) => {
	const result = parseSDKey("diffusion_model.blocks.3.mlp.fc1.lora_A.weight");

	t.is(result.name, "TB03");
	t.is(result.blockId, 3);
	t.is(result.idx, 3);
	t.false(result.isAttention);
});

test("parseSDKey maps diffusion_model.blocks attn.qkv_proj keys as attention", (t) => {
	const result = parseSDKey(
		"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
	);

	t.is(result.name, "TB00");
	t.true(result.isAttention);
});

test("parseSDKey maps diffusion_model.token_refiner.blocks keys to TR blocks", (t) => {
	const result = parseSDKey(
		"diffusion_model.token_refiner.blocks.1.attn.out_proj.lora_B.weight",
	);

	t.is(result.name, "TR01");
	t.is(result.blockId, 1);
	t.true(result.isAttention);
});
