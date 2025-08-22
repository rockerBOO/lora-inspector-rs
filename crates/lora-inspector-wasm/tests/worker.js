import test from "ava";

import { SDRE, parseSDKey } from "../assets/js/moduleBlocks.js";

// Text Encoder Tests
// ------------------

test("parses text encoder self attention block", (t) => {
	const result = parseSDKey("lora_te_text_model_1_self_attn");
	t.is(result.type, "encoder");
	t.is(result.idx, 1);
	t.is(result.blockId, "1");
	t.is(result.blockType, "self_attn");
	t.is(result.name, "TE01");
	t.true(result.isAttention);
});

test("parses second text encoder self attention block", (t) => {
	const result = parseSDKey(
		"lora_te2_text_model_encoder_layers_5_self_attn_q_proj",
	);
	t.is(result.type, "encoder");
	t.is(result.idx, 5);
	t.is(result.blockId, "5");
	t.is(result.blockType, "self_attn");
	t.is(result.name, "TE05");
	t.true(result.isAttention);
});

test("parses text encoder mlp block", (t) => {
	const result = parseSDKey("lora_te_text_model_2_mlp");
	t.is(result.type, "encoder");
	t.is(result.idx, 2);
	t.is(result.blockId, "2");
	t.is(result.blockType, "mlp");
	t.is(result.name, "TE02");
	t.false(result.isAttention);
});

// UNet Down Block Tests
// ---------------------

test("parses down block resnet", (t) => {
	const result = parseSDKey("down_blocks_0_resnets_1");
	t.is(result.type, "resnets");
	t.is(result.idx, 1);
	t.is(result.blockType, "down");
	t.is(result.blockId, 0);
	t.is(result.subBlockId, 1);
	t.is(result.name, "IN01");
	t.true(result.isConv);
	t.false(result.isAttention);
	t.false(result.isSampler);
});

test("parses down block attention", (t) => {
	const result = parseSDKey("lora_down_blocks_1_attentions_0");
	t.is(result.type, "attentions");
	t.is(result.blockType, "down");
	t.is(result.blockId, 1);
	t.is(result.subBlockId, 0);
	t.is(result.name, "IN03");
	t.false(result.isConv);
	t.true(result.isAttention);
	t.false(result.isSampler);
});

test("parses down block downsampler", (t) => {
	const result = parseSDKey("down_blocks_0_downsamplers_0");
	t.is(result.type, "downsamplers");
	t.is(result.idx, 2);
	t.is(result.blockType, "down");
	t.is(result.blockId, 0);
	t.is(result.subBlockId, 0);
	t.is(result.name, "IN02");
	t.false(result.isConv);
	t.false(result.isAttention);
	t.true(result.isSampler);
});

// UNet Up Block Tests
test("parses up block resnet", (t) => {
	const result = parseSDKey("lora_up_blocks_1_resnets_0");
	t.is(result.type, "resnets");
	t.is(result.idx, 3);
	t.is(result.blockType, "up");
	t.is(result.blockId, 1);
	t.is(result.subBlockId, 0);
	t.is(result.name, "OUT03");
	t.true(result.isConv);
	t.false(result.isAttention);
	t.false(result.isSampler);
});

test("parses up block upsampler", (t) => {
	const result = parseSDKey("up_blocks_0_upsamplers_0");
	t.is(result.type, "upsamplers");
	t.is(result.idx, 2);
	t.is(result.blockType, "up");
	t.is(result.blockId, 0);
	t.is(result.subBlockId, 0);
	t.is(result.name, "OUT02");
	t.false(result.isConv);
	t.false(result.isAttention);
	t.true(result.isSampler);
});

// Mid Block Tests
test("parses mid block resnet", (t) => {
	const result = parseSDKey("lora_mid_block_resnets_0_0");
	t.is(result.type, "resnets");
	t.is(result.idx, 0);
	t.is(result.blockType, "mid");
	t.is(result.blockId, "0");
	t.is(result.name, "MID00");
	t.true(result.isConv);
	t.false(result.isAttention);
});

test("parses mid block attention", (t) => {
	const result = parseSDKey("lora_mid_block_attentions_0_0");
	t.is(result.type, "attentions");
	t.is(result.idx, 0);
	t.is(result.blockType, "mid");
	t.is(result.blockId, "0");
	t.is(result.name, "MID00");
	t.false(result.isConv);
	t.true(result.isAttention);
});

// SDXL Tests
// ----------

// Where does output/input blocks come from?
// test("parses SDXL unet transformer blocks", (t) => {
//   const result = parseSDKey(
//     "lora_unet_output_blocks_2_1_transformer_blocks_0_ff_net_2",
//   );
//   t.is(result.type, "encoder");
//   t.is(result.blockId, 1);
//   t.is(result.blockType, "self_attn");
//   t.is(result.name, "TE01");
//   t.true(result.isAttention);
// });
//
// test("should parse different component types", (t) => {
//   const key = "lora_unet_output_blocks_2_1_transformer_blocks_0_attn_1";
//   const matches = key.match(SDXL_UNET_SDRE);
//
//   t.truthy(matches);
//   t.is(matches.groups.component_type, "attn");
//   t.is(matches.groups.component_id, "1");
// });

test("should handle various numeric values", (t) => {
	const key = "lora_unet_down_blocks_100_attentions_200_proj_in";
	const matches = key.match(SDRE);

	t.truthy(matches);
	t.is(matches.groups.block_id, "100");
	t.is(matches.groups.subblock_id, "200");
});

test("should not match invalid formats", (t) => {
	const invalidKeys = [
		"lora_unet_invalid_blocks_2_1_transformer_blocks_0_ff_net_2",
		"lora_unet_output_blocks_2_transformer_blocks_0_ff_net_2", // missing subblock_id
		"lora_unet_output_blocks_2_1_transformer_0_ff_net_2", // incorrect transformer format
	];

	for (const key in invalidKeys) {
		const matches = key.match(SDRE);
		t.falsy(matches, `Should not match invalid key: ${key}`);
	}
});

// test("should match different block types", (t) => {
//   const keys = [
//     "lora_unet_down_blocks_1_1_transformer_blocks_0_ff_net_2",
//     "lora_unet_up_blocks_2_1_transformer_blocks_0_ff_net_2",
//   ];
//
//   for (const key in keys) {
//     const matches = keys[key].match(SDRE);
//     t.truthy(matches, `Should match key: ${keys[key]} | ${SDRE}`);
//   }
// });

// Invalid Input Test
test("handles invalid input", (t) => {
	t.throws(() => parseSDKey("invalid_key"));
});
