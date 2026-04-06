import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

// Test suite for Flux LoRA key parsing
// These keys are from actual Flux LoRA files that should parse correctly

test("parseSDKey handles Flux double_blocks with txt modality", (t) => {
	const key = "lora_unet_double_blocks_0_txt_mlp_2";
	const result = parseSDKey(key);

	t.is(result.blockType, "double_blocks");
	t.is(result.blockId, "0");
	t.is(result.idx, 0);
	t.is(result.name, "DB00");
	t.is(result.type, "transformer");
	t.true(result.isAttention);
	t.is(result.key, key);
});

test("parseSDKey handles Flux double_blocks with img modality", (t) => {
	const key = "lora_unet_double_blocks_2_img_attn_qkv";
	const result = parseSDKey(key);

	t.is(result.blockType, "double_blocks");
	t.is(result.blockId, "2");
	t.is(result.idx, 2);
	t.is(result.name, "DB02");
	t.is(result.type, "transformer");
	t.true(result.isAttention);
	t.is(result.key, key);
});

test("parseSDKey handles Flux double_blocks with different subblock types", (t) => {
	const keys = [
		"lora_unet_double_blocks_4_txt_attn_proj",
		"lora_unet_double_blocks_4_txt_attn_qkv",
		"lora_unet_double_blocks_3_txt_mlp_0",
		"lora_unet_double_blocks_7_img_mlp_2",
	];

	for (const key of keys) {
		const result = parseSDKey(key);
		t.is(result.blockType, "double_blocks");
		t.is(result.type, "transformer");
		t.true(result.isAttention);
		t.is(result.key, key);
	}
});

test("parseSDKey handles Flux single_blocks with linear1", (t) => {
	const key = "lora_unet_single_blocks_13_linear1";
	const result = parseSDKey(key);

	t.is(result.blockType, "single_blocks");
	t.is(result.blockId, "13");
	t.is(result.name, "SB13");
	t.is(result.type, "transformer");
	t.true(result.isAttention);
	t.is(result.subBlockId, "0");
	t.is(result.key, key);
});

test("parseSDKey handles Flux single_blocks with linear2", (t) => {
	const key = "lora_unet_single_blocks_11_linear2";
	const result = parseSDKey(key);

	t.is(result.blockType, "single_blocks");
	t.is(result.blockId, "11");
	t.is(result.name, "SB11");
	t.is(result.type, "transformer");
	t.true(result.isAttention);
	t.is(result.key, key);
});

test("parseSDKey handles all Flux single_blocks from real file", (t) => {
	const keys = [
		"lora_unet_single_blocks_13_linear1",
		"lora_unet_single_blocks_11_linear1",
		"lora_unet_single_blocks_7_linear1",
		"lora_unet_single_blocks_11_linear2",
		"lora_unet_single_blocks_4_linear1",
	];

	for (const key of keys) {
		t.notThrows(() => parseSDKey(key), `Should parse key: ${key}`);
		const result = parseSDKey(key);
		t.is(result.blockType, "single_blocks");
		t.is(result.type, "transformer");
		t.true(result.isAttention);
	}
});

test("parseSDKey handles all Flux double_blocks from real file", (t) => {
	const keys = [
		"lora_unet_double_blocks_2_img_attn_qkv",
		"lora_unet_double_blocks_7_img_mlp_2",
		"lora_unet_double_blocks_1_img_mlp_2",
		"lora_unet_double_blocks_0_txt_mlp_2",
		"lora_unet_double_blocks_4_img_attn_qkv",
		"lora_unet_double_blocks_3_txt_mlp_0",
		"lora_unet_double_blocks_2_txt_mlp_2",
		"lora_unet_double_blocks_4_txt_attn_proj",
		"lora_unet_double_blocks_0_txt_attn_proj",
		"lora_unet_double_blocks_6_txt_attn_proj",
		"lora_unet_double_blocks_4_txt_attn_qkv",
		"lora_unet_double_blocks_2_txt_attn_proj",
		"lora_unet_double_blocks_6_txt_mlp_2",
		"lora_unet_double_blocks_0_img_attn_proj",
		"lora_unet_double_blocks_1_txt_mlp_0",
	];

	for (const key of keys) {
		t.notThrows(() => parseSDKey(key), `Should parse key: ${key}`);
		const result = parseSDKey(key);
		t.is(result.blockType, "double_blocks");
		t.is(result.type, "transformer");
		t.true(result.isAttention);
	}
});

test("parseSDKey handles Flux input embedders", (t) => {
	const embedders = [
		{ key: "lora_unet_txt_in", name: "TXT_IN", type: "embedder" },
		{ key: "lora_unet_img_in", name: "IMG_IN", type: "embedder" },
		{ key: "lora_unet_time_in", name: "TIME_IN", type: "embedder" },
		{ key: "lora_unet_vector_in", name: "VEC_IN", type: "in" },
		{ key: "lora_unet_guidance_in", name: "GUI_IN", type: "embedder" },
	];

	for (const { key, name, type } of embedders) {
		const result = parseSDKey(key);
		t.is(result.name, name, `Name should be ${name} for ${key}`);
		t.is(result.type, type, `Type should be ${type} for ${key}`);
		t.is(result.blockIdx, 0);
		t.is(result.idx, 0);
	}
});

test("parseSDKey maintains backwards compatibility with SD1.5/SDXL keys", (t) => {
	// Ensure we didn't break existing functionality
	const sd15Keys = [
		"lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_k",
		"lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn1_to_k",
		"lora_te_text_model_encoder_layers_5_self_attn_q_proj",
	];

	for (const key of sd15Keys) {
		t.notThrows(() => parseSDKey(key), `Should parse SD1.5/SDXL key: ${key}`);
	}
});

test("parseSDKey throws error for invalid/unknown key patterns", (t) => {
	const invalidKeys = [
		"invalid_key_pattern",
		"lora_unknown_structure",
		"random_text",
	];

	for (const key of invalidKeys) {
		t.throws(
			() => parseSDKey(key),
			{ instanceOf: Error },
			`Should throw error for invalid key: ${key}`,
		);
	}
});

test("parseSDKey handles Flux double_blocks block ID extraction correctly", (t) => {
	// Test various block IDs to ensure correct parsing
	const testCases = [
		{
			key: "lora_unet_double_blocks_0_txt_mlp_2",
			expectedId: "0",
			expectedIdx: 0,
			expectedName: "DB00",
		},
		{
			key: "lora_unet_double_blocks_5_img_attn_qkv",
			expectedId: "5",
			expectedIdx: 5,
			expectedName: "DB05",
		},
		{
			key: "lora_unet_double_blocks_15_txt_attn_proj",
			expectedId: "15",
			expectedIdx: 15,
			expectedName: "DB15",
		},
	];

	for (const { key, expectedId, expectedIdx, expectedName } of testCases) {
		const result = parseSDKey(key);
		t.is(
			result.blockId,
			expectedId,
			`Block ID should be "${expectedId}" for ${key}`,
		);
		t.is(result.idx, expectedIdx, `Index should be ${expectedIdx} for ${key}`);
		t.is(
			result.name,
			expectedName,
			`Name should be ${expectedName} for ${key}`,
		);
	}
});

test("parseSDKey handles Flux single_blocks block ID extraction correctly", (t) => {
	// Test various block IDs to ensure correct parsing
	const testCases = [
		{
			key: "lora_unet_single_blocks_0_linear1",
			expectedId: "0",
			expectedName: "SB00",
		},
		{
			key: "lora_unet_single_blocks_7_linear2",
			expectedId: "7",
			expectedName: "SB07",
		},
		{
			key: "lora_unet_single_blocks_20_linear1",
			expectedId: "20",
			expectedName: "SB20",
		},
	];

	for (const { key, expectedId, expectedName } of testCases) {
		const result = parseSDKey(key);
		t.is(
			result.blockId,
			expectedId,
			`Block ID should be ${expectedId} for ${key}`,
		);
		t.is(
			result.name,
			expectedName,
			`Name should be ${expectedName} for ${key}`,
		);
	}
});
