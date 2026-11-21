import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

// SDXL LoRAs from kohya sd-scripts use SGM (Stability) U-Net naming:
// input_blocks_{0-8}, middle_block_{0-2}, output_blocks_{0-8}.
// Block weights follow the IN00-IN08 / MID / OUT00-OUT08 convention where
// the block name comes from the first index only.

test("parseSDKey maps SDXL input_blocks attention keys to IN blocks", (t) => {
	const result = parseSDKey(
		"lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_v",
	);

	t.is(result.name, "IN04");
	t.is(result.blockId, "4");
	t.is(result.subBlockId, "1");
	t.is(result.idx, 4);
	t.is(result.blockIdx, 5);
	t.is(result.blockType, "input");
	t.is(result.type, "attentions");
	t.true(result.isAttention);
	t.false(result.isConv);
});

test("parseSDKey does not leak SDXL transformer block index into the block name", (t) => {
	// SDXL has up to 10 transformer blocks per attention; the transformer
	// index must not shift the block name.
	const keys = [
		"lora_unet_input_blocks_7_1_transformer_blocks_0_attn1_to_out_0",
		"lora_unet_input_blocks_7_1_transformer_blocks_3_attn1_to_out_0",
		"lora_unet_input_blocks_7_1_transformer_blocks_9_attn2_to_q",
	];

	for (const key of keys) {
		const result = parseSDKey(key);
		t.is(result.name, "IN07", `Name should be IN07 for ${key}`);
		t.true(result.isAttention);
	}
});

test("parseSDKey handles SDXL feed-forward keys as attention", (t) => {
	const result = parseSDKey(
		"lora_unet_input_blocks_5_1_transformer_blocks_1_ff_net_0_proj",
	);

	t.is(result.name, "IN05");
	t.is(result.type, "attentions");
	t.true(result.isAttention);
});

test("parseSDKey handles SDXL proj_in/proj_out keys", (t) => {
	const projIn = parseSDKey("lora_unet_input_blocks_4_1_proj_in");
	t.is(projIn.name, "IN04");
	t.is(projIn.type, "attentions");
	t.true(projIn.isAttention);

	const projOut = parseSDKey("lora_unet_output_blocks_0_1_proj_out");
	t.is(projOut.name, "OUT00");
	t.is(projOut.type, "attentions");
	t.true(projOut.isAttention);
});

test("parseSDKey handles SDXL resnet conv keys", (t) => {
	const keys = [
		"lora_unet_input_blocks_1_0_in_layers_2",
		"lora_unet_input_blocks_1_0_emb_layers_1",
		"lora_unet_input_blocks_1_0_out_layers_3",
		"lora_unet_input_blocks_4_0_skip_connection",
	];

	for (const key of keys) {
		const result = parseSDKey(key);
		t.is(result.type, "resnets", `Type should be resnets for ${key}`);
		t.true(result.isConv, `isConv should be true for ${key}`);
		t.false(result.isAttention);
	}

	t.is(parseSDKey("lora_unet_input_blocks_1_0_in_layers_2").name, "IN01");
	t.is(parseSDKey("lora_unet_input_blocks_4_0_skip_connection").name, "IN04");
});

test("parseSDKey handles SDXL downsampler op keys", (t) => {
	const result = parseSDKey("lora_unet_input_blocks_3_0_op");

	t.is(result.name, "IN03");
	t.is(result.type, "downsamplers");
	t.true(result.isSampler);
});

test("parseSDKey handles SDXL upsampler conv keys", (t) => {
	const result = parseSDKey("lora_unet_output_blocks_2_2_conv");

	t.is(result.name, "OUT02");
	t.is(result.type, "upsamplers");
	t.true(result.isSampler);
});

test("parseSDKey maps SDXL middle_block keys to MID", (t) => {
	const attn = parseSDKey(
		"lora_unet_middle_block_1_transformer_blocks_2_attn1_to_out_0",
	);
	t.is(attn.name, "MID00");
	t.is(attn.blockIdx, 10);
	t.is(attn.type, "attentions");
	t.true(attn.isAttention);

	const resnet = parseSDKey("lora_unet_middle_block_0_in_layers_2");
	t.is(resnet.name, "MID00");
	t.is(resnet.blockIdx, 10);
	t.is(resnet.type, "resnets");
	t.true(resnet.isConv);

	const resnet2 = parseSDKey("lora_unet_middle_block_2_out_layers_3");
	t.is(resnet2.name, "MID00");
	t.true(resnet2.isConv);
});

test("parseSDKey maps SDXL output_blocks keys to OUT blocks", (t) => {
	const result = parseSDKey(
		"lora_unet_output_blocks_8_1_transformer_blocks_1_attn2_to_q",
	);

	t.is(result.name, "OUT08");
	t.is(result.blockId, "8");
	t.is(result.idx, 8);
	t.is(result.blockIdx, 19);
	t.is(result.blockType, "output");
	t.true(result.isAttention);

	const resnet = parseSDKey("lora_unet_output_blocks_6_0_emb_layers_1");
	t.is(resnet.name, "OUT06");
	t.true(resnet.isConv);
});

test("parseSDKey parses every SDXL LoCon key pattern without throwing", (t) => {
	// Representative patterns from a real full-preset LoCon SDXL file
	const keys = [
		"lora_unet_input_blocks_1_0_emb_layers_1",
		"lora_unet_input_blocks_1_0_in_layers_2",
		"lora_unet_input_blocks_3_0_op",
		"lora_unet_input_blocks_2_0_out_layers_3",
		"lora_unet_input_blocks_4_0_skip_connection",
		"lora_unet_input_blocks_8_1_transformer_blocks_4_attn2_to_q",
		"lora_unet_middle_block_0_emb_layers_1",
		"lora_unet_middle_block_2_in_layers_2",
		"lora_unet_middle_block_0_out_layers_3",
		"lora_unet_middle_block_1_transformer_blocks_6_attn1_to_out_0",
		"lora_unet_output_blocks_5_2_conv",
		"lora_unet_output_blocks_3_0_emb_layers_1",
		"lora_unet_output_blocks_3_0_in_layers_2",
		"lora_unet_output_blocks_3_0_out_layers_3",
		"lora_unet_output_blocks_3_0_skip_connection",
		"lora_unet_output_blocks_1_1_transformer_blocks_8_attn2_to_q",
	];

	for (const key of keys) {
		t.notThrows(() => parseSDKey(key), `Should parse key: ${key}`);
		const result = parseSDKey(key);
		t.regex(
			result.name,
			/^(IN0[0-8]|MID00|OUT0[0-8])$/,
			`Block name "${result.name}" should be a valid SDXL block for ${key}`,
		);
	}
});

test("parseSDKey splits SDXL dual text encoders into TE1/TE2 blocks", (t) => {
	const te1 = parseSDKey(
		"lora_te1_text_model_encoder_layers_4_self_attn_q_proj",
	);
	t.is(te1.name, "TE1-04");
	t.is(te1.type, "encoder");
	t.true(te1.isAttention);

	const te2 = parseSDKey("lora_te2_text_model_encoder_layers_17_mlp_fc1");
	t.is(te2.name, "TE2-17");
	t.is(te2.type, "encoder");
	t.false(te2.isAttention);
});

test("parseSDKey keeps single text encoder naming for SD1.5", (t) => {
	const result = parseSDKey(
		"lora_te_text_model_encoder_layers_5_self_attn_q_proj",
	);

	t.is(result.name, "TE05");
	t.is(result.type, "encoder");
	t.true(result.isAttention);
});
