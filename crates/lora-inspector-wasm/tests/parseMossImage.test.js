import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

// MOSS-Image (sd-scripts networks.lora_generic) names its DiT blocks
// "layers.N", with attention projections proj_q/k/v/o and a SwiGLU FFN
// exposed as proj_ffn_up_gate/proj_ffn_down. Keys are underscore-joined
// under the usual lora_unet_ prefix.

const MOSS_ATTENTION_MODULES = ["proj_q", "proj_k", "proj_v", "proj_o"];
const MOSS_MLP_MODULES = ["proj_ffn_up_gate", "proj_ffn_down"];

test("parseSDKey maps MOSS-Image attention projections to TB blocks", (t) => {
	for (const module of MOSS_ATTENTION_MODULES) {
		const key = `lora_unet_layers_5_${module}`;
		const result = parseSDKey(key);

		t.is(result.name, "TB05", `name for ${key}`);
		t.is(result.blockId, 5, `blockId for ${key}`);
		t.is(result.idx, 5, `idx for ${key}`);
		t.is(result.blockType, "layers", `blockType for ${key}`);
		t.is(result.type, "attentions", `type for ${key}`);
		t.true(result.isAttention, `isAttention for ${key}`);
		t.is(result.key, key);
	}
});

test("parseSDKey maps MOSS-Image FFN projections to TB blocks without isAttention", (t) => {
	for (const module of MOSS_MLP_MODULES) {
		const key = `lora_unet_layers_19_${module}`;
		const result = parseSDKey(key);

		t.is(result.name, "TB19", `name for ${key}`);
		t.is(result.blockId, 19, `blockId for ${key}`);
		t.is(result.idx, 19, `idx for ${key}`);
		t.is(result.type, "mlp", `type for ${key}`);
		t.false(result.isAttention, `isAttention for ${key}`);
	}
});

test("parseSDKey pads MOSS-Image block names and keeps them sortable", (t) => {
	const names = [0, 8, 9, 10, 41].map(
		(layer) => parseSDKey(`lora_unet_layers_${layer}_proj_q`).name,
	);

	t.deepEqual(names, ["TB00", "TB08", "TB09", "TB10", "TB41"]);
	// localeCompare ordering must keep layer 10 after layer 9.
	t.deepEqual(
		[...names].sort((a, b) => a.localeCompare(b)),
		names,
	);
});

test("parseSDKey handles every base name of a rank-8 MOSS-Image LoRA", (t) => {
	// A rank-8 42-layer MOSS-Image LoRA trains layers.N.proj_{q,k,v,o} and
	// layers.N.proj_{ffn_up_gate,ffn_down}, which sd-scripts joins into
	// lora_unet_layers_N_proj_* base names.
	const NUM_LAYERS = 42;
	const modules = [...MOSS_ATTENTION_MODULES, ...MOSS_MLP_MODULES];
	const baseNames = [];

	for (let layer = 0; layer < NUM_LAYERS; layer++) {
		for (const module of modules) {
			baseNames.push(`lora_unet_layers_${layer}_${module}`);
		}
	}

	for (const baseName of baseNames) {
		const result = parseSDKey(baseName);
		t.is(result.name, `TB${String(result.idx).padStart(2, "0")}`);
	}

	t.is(baseNames.length, NUM_LAYERS * modules.length);
});
