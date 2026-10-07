import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

// Krea 2 (musubi-tuner networks.lora_krea2) uses a flat DiT block naming:
// lora_unet_blocks_{0-27}_(attn_*|mlp_*), plus txtfusion layerwise/refiner
// blocks and a handful of single-tensor modules (first, last_linear, etc).

test("parseSDKey maps Krea2 main block keys to TB blocks", (t) => {
	const result = parseSDKey("lora_unet_blocks_5_attn_wq");

	t.is(result.name, "TB05");
	t.is(result.blockId, "5");
	t.is(result.idx, 5);
	t.true(result.isAttention);
});

test("parseSDKey maps Krea2 mlp block keys without isAttention", (t) => {
	const result = parseSDKey("lora_unet_blocks_12_mlp_gate");

	t.is(result.name, "TB12");
	t.false(result.isAttention);
});

test("parseSDKey maps Krea2 txtfusion layerwise/refiner blocks separately from main blocks", (t) => {
	const layerwise = parseSDKey(
		"lora_unet_txtfusion_layerwise_blocks_0_attn_wk",
	);
	const refiner = parseSDKey("lora_unet_txtfusion_refiner_blocks_1_mlp_up");

	t.is(layerwise.name, "TFL00");
	t.is(refiner.name, "TFR01");
});

test("parseSDKey handles Krea2 single-tensor modules", (t) => {
	const keys = [
		"lora_unet_first",
		"lora_unet_last_linear",
		"lora_unet_tmlp_0",
		"lora_unet_tproj_1",
		"lora_unet_txtfusion_projector",
		"lora_unet_txtmlp_1",
	];

	for (const key of keys) {
		const result = parseSDKey(key);
		t.is(result.type, "embedder", `Type should be embedder for ${key}`);
	}
});

// musubi-tuner also saves Krea2 LoRAs with dot-separated "diffusion_model."
// prefixed keys (e.g. rank_64 checkpoints trained directly against the DiT
// module names) rather than the "lora_unet_"-prefixed, underscore-joined
// keys above.
test("parseSDKey maps dot-separated diffusion_model txtfusion blocks", (t) => {
	const layerwise = parseSDKey(
		"diffusion_model.txtfusion.layerwise_blocks.0.attn.wk",
	);
	const refiner = parseSDKey(
		"diffusion_model.txtfusion.refiner_blocks.1.mlp.down",
	);

	t.is(layerwise.name, "TFL00");
	t.true(layerwise.isAttention);

	t.is(refiner.name, "TFR01");
	t.false(refiner.isAttention);
});

test("parseSDKey handles dot-separated diffusion_model Krea2 single-tensor modules", (t) => {
	const keys = [
		"diffusion_model.first",
		"diffusion_model.last_linear",
		"diffusion_model.last.linear",
		"diffusion_model.tmlp.0",
		"diffusion_model.tproj.1",
		"diffusion_model.txtfusion.projector",
		"diffusion_model.txtmlp.1",
	];

	for (const key of keys) {
		const result = parseSDKey(key);
		t.is(result.type, "embedder", `Type should be embedder for ${key}`);
	}
});
