import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

const rawKeys = `lora_unet_txt_in.lora_down.weight
lora_unet_txt_in.lora_up.weight
lora_unet_txt_in.alpha
lora_unet_img_in.lora_down.weight
lora_unet_img_in.lora_up.weight
lora_unet_img_in.alpha
lora_unet_guidance_in.lora_down.weight
lora_unet_guidance_in.lora_up.weight
lora_unet_guidance_in.alpha
lora_unet_vector_in.lora_down.weight
lora_unet_vector_in.lora_up.weight
lora_unet_vector_in.alpha
lora_unet_time_in.lora_down.weight
lora_unet_time_in.lora_up.weight
lora_unet_time_in.alpha
lora_unet_single_blocks_0_linear1.alpha
lora_unet_single_blocks_0_linear1.lora_down.weight
lora_unet_single_blocks_0_linear1.lora_up.weight
lora_unet_single_blocks_0_linear2.alpha
lora_unet_single_blocks_0_linear2.lora_down.weight
lora_unet_single_blocks_0_linear2.lora_up.weight
lora_unet_single_blocks_0_modulation_lin.alpha
lora_unet_single_blocks_0_modulation_lin.lora_down.weight
lora_unet_single_blocks_0_modulation_lin.lora_up.weight
lora_unet_single_blocks_10_linear1.alpha
lora_unet_single_blocks_10_linear1.lora_down.weight
lora_unet_single_blocks_10_linear1.lora_up.weight
lora_unet_single_blocks_10_linear2.alpha
lora_unet_single_blocks_10_linear2.lora_down.weight
lora_unet_single_blocks_10_linear2.lora_up.weight
lora_unet_single_blocks_10_modulation_lin.alpha
lora_unet_single_blocks_10_modulation_lin.lora_down.weight
lora_unet_single_blocks_10_modulation_lin.lora_up.weight
lora_unet_single_blocks_11_linear1.alpha
lora_unet_single_blocks_11_linear1.lora_down.weight
lora_unet_single_blocks_11_linear1.lora_up.weight
lora_unet_single_blocks_11_linear2.alpha
lora_unet_single_blocks_11_linear2.lora_down.weight
lora_unet_single_blocks_11_linear2.lora_up.weight
lora_unet_single_blocks_11_modulation_lin.alpha
lora_unet_single_blocks_11_modulation_lin.lora_down.weight
lora_unet_single_blocks_11_modulation_lin.lora_up.weight
lora_unet_single_blocks_12_linear1.alpha
lora_unet_single_blocks_12_linear1.lora_down.weight
lora_unet_single_blocks_12_linear1.lora_up.weight
lora_unet_single_blocks_12_linear2.alpha
lora_unet_single_blocks_12_linear2.lora_down.weight
lora_unet_single_blocks_12_linear2.lora_up.weight
lora_unet_single_blocks_12_modulation_lin.alpha
lora_unet_single_blocks_12_modulation_lin.lora_down.weight
lora_unet_single_blocks_12_modulation_lin.lora_up.weight
lora_unet_single_blocks_13_linear1.alpha
lora_unet_single_blocks_13_linear1.lora_down.weight
lora_unet_single_blocks_13_linear1.lora_up.weight
lora_unet_single_blocks_13_linear2.alpha
lora_unet_single_blocks_13_linear2.lora_down.weight
lora_unet_single_blocks_13_linear2.lora_up.weight
lora_unet_single_blocks_13_modulation_lin.alpha
lora_unet_single_blocks_13_modulation_lin.lora_down.weight
lora_unet_single_blocks_13_modulation_lin.lora_up.weight
lora_unet_single_blocks_14_linear1.alpha
lora_unet_single_blocks_14_linear1.lora_down.weight
lora_unet_single_blocks_14_linear1.lora_up.weight
lora_unet_single_blocks_14_linear2.alpha
lora_unet_single_blocks_14_linear2.lora_down.weight
lora_unet_single_blocks_14_linear2.lora_up.weight
lora_unet_single_blocks_14_modulation_lin.alpha
lora_unet_single_blocks_14_modulation_lin.lora_down.weight
lora_unet_single_blocks_14_modulation_lin.lora_up.weight
lora_unet_single_blocks_15_linear1.alpha
lora_unet_single_blocks_15_linear1.lora_down.weight
lora_unet_single_blocks_15_linear1.lora_up.weight
lora_unet_single_blocks_15_linear2.alpha
lora_unet_single_blocks_15_linear2.lora_down.weight
lora_unet_single_blocks_15_linear2.lora_up.weight
lora_unet_single_blocks_15_modulation_lin.alpha
lora_unet_single_blocks_15_modulation_lin.lora_down.weight
lora_unet_single_blocks_15_modulation_lin.lora_up.weight
lora_unet_single_blocks_16_linear1.alpha
lora_unet_single_blocks_16_linear1.lora_down.weight
lora_unet_single_blocks_16_linear1.lora_up.weight
lora_unet_single_blocks_16_linear2.alpha
lora_unet_single_blocks_16_linear2.lora_down.weight
lora_unet_single_blocks_16_linear2.lora_up.weight
lora_unet_single_blocks_16_modulation_lin.alpha
lora_unet_single_blocks_16_modulation_lin.lora_down.weight
lora_unet_single_blocks_16_modulation_lin.lora_up.weight
lora_unet_single_blocks_17_linear1.alpha
lora_unet_single_blocks_17_linear1.lora_down.weight
lora_unet_single_blocks_17_linear1.lora_up.weight
lora_unet_single_blocks_17_linear2.alpha
lora_unet_single_blocks_17_linear2.lora_down.weight
lora_unet_single_blocks_17_linear2.lora_up.weight
lora_unet_single_blocks_17_modulation_lin.alpha
lora_unet_single_blocks_17_modulation_lin.lora_down.weight
lora_unet_single_blocks_17_modulation_lin.lora_up.weight
lora_unet_single_blocks_18_linear1.alpha
lora_unet_single_blocks_18_linear1.lora_down.weight
lora_unet_single_blocks_18_linear1.lora_up.weight
lora_unet_single_blocks_18_linear2.alpha
lora_unet_single_blocks_18_linear2.lora_down.weight
lora_unet_single_blocks_18_linear2.lora_up.weight
lora_unet_single_blocks_18_modulation_lin.alpha
lora_unet_single_blocks_18_modulation_lin.lora_down.weight
lora_unet_single_blocks_18_modulation_lin.lora_up.weight
lora_unet_single_blocks_19_linear1.alpha
lora_unet_single_blocks_19_linear1.lora_down.weight
lora_unet_single_blocks_19_linear1.lora_up.weight
lora_unet_single_blocks_19_linear2.alpha
lora_unet_single_blocks_19_linear2.lora_down.weight
lora_unet_single_blocks_19_linear2.lora_up.weight
lora_unet_single_blocks_19_modulation_lin.alpha
lora_unet_single_blocks_19_modulation_lin.lora_down.weight
lora_unet_single_blocks_19_modulation_lin.lora_up.weight
lora_unet_single_blocks_1_linear1.alpha
lora_unet_single_blocks_1_linear1.lora_down.weight
lora_unet_single_blocks_1_linear1.lora_up.weight
lora_unet_single_blocks_1_linear2.alpha
lora_unet_single_blocks_1_linear2.lora_down.weight
lora_unet_single_blocks_1_linear2.lora_up.weight
lora_unet_single_blocks_1_modulation_lin.alpha
lora_unet_single_blocks_1_modulation_lin.lora_down.weight
lora_unet_single_blocks_1_modulation_lin.lora_up.weight
lora_unet_single_blocks_20_linear1.alpha
lora_unet_single_blocks_20_linear1.lora_down.weight
lora_unet_single_blocks_20_linear1.lora_up.weight
lora_unet_single_blocks_20_linear2.alpha
lora_unet_single_blocks_20_linear2.lora_down.weight
lora_unet_single_blocks_20_linear2.lora_up.weight
lora_unet_single_blocks_20_modulation_lin.alpha
lora_unet_single_blocks_20_modulation_lin.lora_down.weight
lora_unet_single_blocks_20_modulation_lin.lora_up.weight
lora_unet_single_blocks_21_linear1.alpha
lora_unet_single_blocks_21_linear1.lora_down.weight
lora_unet_single_blocks_21_linear1.lora_up.weight
lora_unet_single_blocks_21_linear2.alpha
lora_unet_single_blocks_21_linear2.lora_down.weight
lora_unet_single_blocks_21_linear2.lora_up.weight
lora_unet_single_blocks_21_modulation_lin.alpha
lora_unet_single_blocks_21_modulation_lin.lora_down.weight
lora_unet_single_blocks_21_modulation_lin.lora_up.weight
lora_unet_single_blocks_22_linear1.alpha
lora_unet_single_blocks_22_linear1.lora_down.weight
lora_unet_single_blocks_22_linear1.lora_up.weight
lora_unet_single_blocks_22_linear2.alpha
lora_unet_single_blocks_22_linear2.lora_down.weight
lora_unet_single_blocks_22_linear2.lora_up.weight
lora_unet_single_blocks_22_modulation_lin.alpha
lora_unet_single_blocks_22_modulation_lin.lora_down.weight
lora_unet_single_blocks_22_modulation_lin.lora_up.weight
lora_unet_single_blocks_23_linear1.alpha
lora_unet_single_blocks_23_linear1.lora_down.weight
lora_unet_single_blocks_23_linear1.lora_up.weight
lora_unet_single_blocks_23_linear2.alpha
lora_unet_single_blocks_23_linear2.lora_down.weight
lora_unet_single_blocks_23_linear2.lora_up.weight
lora_unet_single_blocks_23_modulation_lin.alpha
lora_unet_single_blocks_23_modulation_lin.lora_down.weight
lora_unet_single_blocks_23_modulation_lin.lora_up.weight
lora_unet_single_blocks_24_linear1.alpha
lora_unet_single_blocks_24_linear1.lora_down.weight
lora_unet_single_blocks_24_linear1.lora_up.weight
lora_unet_single_blocks_24_linear2.alpha
lora_unet_single_blocks_24_linear2.lora_down.weight
lora_unet_single_blocks_24_linear2.lora_up.weight
lora_unet_single_blocks_24_modulation_lin.alpha
lora_unet_single_blocks_24_modulation_lin.lora_down.weight
lora_unet_single_blocks_24_modulation_lin.lora_up.weight
lora_unet_single_blocks_25_linear1.alpha
lora_unet_single_blocks_25_linear1.lora_down.weight
lora_unet_single_blocks_25_linear1.lora_up.weight
lora_unet_single_blocks_25_linear2.alpha
lora_unet_single_blocks_25_linear2.lora_down.weight
lora_unet_single_blocks_25_linear2.lora_up.weight
lora_unet_single_blocks_25_modulation_lin.alpha
lora_unet_single_blocks_25_modulation_lin.lora_down.weight
lora_unet_single_blocks_25_modulation_lin.lora_up.weight
lora_unet_single_blocks_26_linear1.alpha
lora_unet_single_blocks_26_linear1.lora_down.weight
lora_unet_single_blocks_26_linear1.lora_up.weight
lora_unet_single_blocks_26_linear2.alpha
lora_unet_single_blocks_26_linear2.lora_down.weight
lora_unet_single_blocks_26_linear2.lora_up.weight
lora_unet_single_blocks_26_modulation_lin.alpha
lora_unet_single_blocks_26_modulation_lin.lora_down.weight
lora_unet_single_blocks_26_modulation_lin.lora_up.weight
lora_unet_single_blocks_27_linear1.alpha
lora_unet_single_blocks_27_linear1.lora_down.weight
lora_unet_single_blocks_27_linear1.lora_up.weight
lora_unet_single_blocks_27_linear2.alpha
lora_unet_single_blocks_27_linear2.lora_down.weight
lora_unet_single_blocks_27_linear2.lora_up.weight
lora_unet_single_blocks_27_modulation_lin.alpha
lora_unet_single_blocks_27_modulation_lin.lora_down.weight
lora_unet_single_blocks_27_modulation_lin.lora_up.weight
lora_unet_single_blocks_28_linear1.alpha
lora_unet_single_blocks_28_linear1.lora_down.weight
lora_unet_single_blocks_28_linear1.lora_up.weight
lora_unet_single_blocks_28_linear2.alpha
lora_unet_single_blocks_28_linear2.lora_down.weight
lora_unet_single_blocks_28_linear2.lora_up.weight
lora_unet_single_blocks_28_modulation_lin.alpha
lora_unet_single_blocks_28_modulation_lin.lora_down.weight
lora_unet_single_blocks_28_modulation_lin.lora_up.weight
lora_unet_single_blocks_29_linear1.alpha
lora_unet_single_blocks_29_linear1.lora_down.weight
lora_unet_single_blocks_29_linear1.lora_up.weight
lora_unet_single_blocks_29_linear2.alpha
lora_unet_single_blocks_29_linear2.lora_down.weight
lora_unet_single_blocks_29_linear2.lora_up.weight
lora_unet_single_blocks_29_modulation_lin.alpha
lora_unet_single_blocks_29_modulation_lin.lora_down.weight
lora_unet_single_blocks_29_modulation_lin.lora_up.weight
lora_unet_single_blocks_2_linear1.alpha
lora_unet_single_blocks_2_linear1.lora_down.weight
lora_unet_single_blocks_2_linear1.lora_up.weight
lora_unet_single_blocks_2_linear2.alpha
lora_unet_single_blocks_2_linear2.lora_down.weight
lora_unet_single_blocks_2_linear2.lora_up.weight
lora_unet_single_blocks_2_modulation_lin.alpha
lora_unet_single_blocks_2_modulation_lin.lora_down.weight
lora_unet_single_blocks_2_modulation_lin.lora_up.weight
lora_unet_single_blocks_30_linear1.alpha
lora_unet_single_blocks_30_linear1.lora_down.weight
lora_unet_single_blocks_30_linear1.lora_up.weight
lora_unet_single_blocks_30_linear2.alpha
lora_unet_single_blocks_30_linear2.lora_down.weight
lora_unet_single_blocks_30_linear2.lora_up.weight
lora_unet_single_blocks_30_modulation_lin.alpha
lora_unet_single_blocks_30_modulation_lin.lora_down.weight
lora_unet_single_blocks_30_modulation_lin.lora_up.weight
lora_unet_single_blocks_31_linear1.alpha
lora_unet_single_blocks_31_linear1.lora_down.weight
lora_unet_single_blocks_31_linear1.lora_up.weight
lora_unet_single_blocks_31_linear2.alpha
lora_unet_single_blocks_31_linear2.lora_down.weight
lora_unet_single_blocks_31_linear2.lora_up.weight
lora_unet_single_blocks_31_modulation_lin.alpha
lora_unet_single_blocks_31_modulation_lin.lora_down.weight
lora_unet_single_blocks_31_modulation_lin.lora_up.weight
lora_unet_single_blocks_32_linear1.alpha
lora_unet_single_blocks_32_linear1.lora_down.weight
lora_unet_single_blocks_32_linear1.lora_up.weight
lora_unet_single_blocks_32_linear2.alpha
lora_unet_single_blocks_32_linear2.lora_down.weight
lora_unet_single_blocks_32_linear2.lora_up.weight
lora_unet_single_blocks_32_modulation_lin.alpha
lora_unet_single_blocks_32_modulation_lin.lora_down.weight
lora_unet_single_blocks_32_modulation_lin.lora_up.weight
lora_unet_single_blocks_33_linear1.alpha
lora_unet_single_blocks_33_linear1.lora_down.weight
lora_unet_single_blocks_33_linear1.lora_up.weight
lora_unet_single_blocks_33_linear2.alpha
lora_unet_single_blocks_33_linear2.lora_down.weight
lora_unet_single_blocks_33_linear2.lora_up.weight
lora_unet_single_blocks_33_modulation_lin.alpha
lora_unet_single_blocks_33_modulation_lin.lora_down.weight
lora_unet_single_blocks_33_modulation_lin.lora_up.weight
lora_unet_single_blocks_34_linear1.alpha
lora_unet_single_blocks_34_linear1.lora_down.weight
lora_unet_single_blocks_34_linear1.lora_up.weight
lora_unet_single_blocks_34_linear2.alpha
lora_unet_single_blocks_34_linear2.lora_down.weight
lora_unet_single_blocks_34_linear2.lora_up.weight
lora_unet_single_blocks_34_modulation_lin.alpha
lora_unet_single_blocks_34_modulation_lin.lora_down.weight
lora_unet_single_blocks_34_modulation_lin.lora_up.weight
lora_unet_single_blocks_35_linear1.alpha
lora_unet_single_blocks_35_linear1.lora_down.weight
lora_unet_single_blocks_35_linear1.lora_up.weight
lora_unet_single_blocks_35_linear2.alpha
lora_unet_single_blocks_35_linear2.lora_down.weight
lora_unet_single_blocks_35_linear2.lora_up.weight
lora_unet_single_blocks_35_modulation_lin.alpha
lora_unet_single_blocks_35_modulation_lin.lora_down.weight
lora_unet_single_blocks_35_modulation_lin.lora_up.weight
lora_unet_single_blocks_36_linear1.alpha
lora_unet_single_blocks_36_linear1.lora_down.weight
lora_unet_single_blocks_36_linear1.lora_up.weight
lora_unet_single_blocks_36_linear2.alpha
lora_unet_single_blocks_36_linear2.lora_down.weight
lora_unet_single_blocks_36_linear2.lora_up.weight
lora_unet_single_blocks_36_modulation_lin.alpha
lora_unet_single_blocks_36_modulation_lin.lora_down.weight
lora_unet_single_blocks_36_modulation_lin.lora_up.weight
lora_unet_single_blocks_37_linear1.alpha
lora_unet_single_blocks_37_linear1.lora_down.weight
lora_unet_single_blocks_37_linear1.lora_up.weight
lora_unet_single_blocks_37_linear2.alpha
lora_unet_single_blocks_37_linear2.lora_down.weight
lora_unet_single_blocks_37_linear2.lora_up.weight
lora_unet_single_blocks_37_modulation_lin.alpha
lora_unet_single_blocks_37_modulation_lin.lora_down.weight
lora_unet_single_blocks_37_modulation_lin.lora_up.weight
lora_unet_single_blocks_3_linear1.alpha
lora_unet_single_blocks_3_linear1.lora_down.weight
lora_unet_single_blocks_3_linear1.lora_up.weight
lora_unet_single_blocks_3_linear2.alpha
lora_unet_single_blocks_3_linear2.lora_down.weight
lora_unet_single_blocks_3_linear2.lora_up.weight
lora_unet_single_blocks_3_modulation_lin.alpha
lora_unet_single_blocks_3_modulation_lin.lora_down.weight
lora_unet_single_blocks_3_modulation_lin.lora_up.weight
lora_unet_single_blocks_4_linear1.alpha
lora_unet_single_blocks_4_linear1.lora_down.weight
lora_unet_single_blocks_4_linear1.lora_up.weight
lora_unet_single_blocks_4_linear2.alpha
lora_unet_single_blocks_4_linear2.lora_down.weight
lora_unet_single_blocks_4_linear2.lora_up.weight
lora_unet_single_blocks_4_modulation_lin.alpha
lora_unet_single_blocks_4_modulation_lin.lora_down.weight
lora_unet_single_blocks_4_modulation_lin.lora_up.weight
lora_unet_single_blocks_5_linear1.alpha
lora_unet_single_blocks_5_linear1.lora_down.weight
lora_unet_single_blocks_5_linear1.lora_up.weight
lora_unet_single_blocks_5_linear2.alpha
lora_unet_single_blocks_5_linear2.lora_down.weight
lora_unet_single_blocks_5_linear2.lora_up.weight
lora_unet_single_blocks_5_modulation_lin.alpha
lora_unet_single_blocks_5_modulation_lin.lora_down.weight
lora_unet_single_blocks_5_modulation_lin.lora_up.weight
lora_unet_single_blocks_6_linear1.alpha
lora_unet_single_blocks_6_linear1.lora_down.weight
lora_unet_single_blocks_6_linear1.lora_up.weight
lora_unet_single_blocks_6_linear2.alpha
lora_unet_single_blocks_6_linear2.lora_down.weight
lora_unet_single_blocks_6_linear2.lora_up.weight
lora_unet_single_blocks_6_modulation_lin.alpha
lora_unet_single_blocks_6_modulation_lin.lora_down.weight
lora_unet_single_blocks_6_modulation_lin.lora_up.weight
lora_unet_single_blocks_7_linear1.alpha
lora_unet_single_blocks_7_linear1.lora_down.weight
lora_unet_single_blocks_7_linear1.lora_up.weight
lora_unet_single_blocks_7_linear2.alpha
lora_unet_single_blocks_7_linear2.lora_down.weight
lora_unet_single_blocks_7_linear2.lora_up.weight
lora_unet_single_blocks_7_modulation_lin.alpha
lora_unet_single_blocks_7_modulation_lin.lora_down.weight
lora_unet_single_blocks_7_modulation_lin.lora_up.weight
lora_unet_single_blocks_8_linear1.alpha
lora_unet_single_blocks_8_linear1.lora_down.weight
lora_unet_single_blocks_8_linear1.lora_up.weight
lora_unet_single_blocks_8_linear2.alpha
lora_unet_single_blocks_8_linear2.lora_down.weight
lora_unet_single_blocks_8_linear2.lora_up.weight
lora_unet_single_blocks_8_modulation_lin.alpha
lora_unet_single_blocks_8_modulation_lin.lora_down.weight
lora_unet_single_blocks_8_modulation_lin.lora_up.weight
lora_unet_single_blocks_9_linear1.alpha
lora_unet_single_blocks_9_linear1.lora_down.weight
lora_unet_single_blocks_9_linear1.lora_up.weight
lora_unet_single_blocks_9_linear2.alpha
lora_unet_single_blocks_9_linear2.lora_down.weight
lora_unet_single_blocks_9_linear2.lora_up.weight
lora_unet_single_blocks_9_modulation_lin.alpha
lora_unet_single_blocks_9_modulation_lin.lora_down.weight
lora_unet_single_blocks_9_modulation_lin.lora_up.weight`;

test("test keys", async (t) => {
	const keys = rawKeys.split("\n");
	for (const key of keys) {
		if (!key.trim()) continue; // Skip empty lines

		const parsed = parseSDKey(key);

		// Test that all required properties exist and have the correct types
		t.truthy(parsed, `Failed to parse key: ${key}`);
		t.is(
			typeof parsed.name,
			"string",
			`'name' should be a string for key: ${key}`,
		);
		t.is(
			typeof parsed.blockIdx,
			"number",
			`'blockIdx' should be a number for key: ${key}`,
		);
		t.is(
			typeof parsed.idx,
			"number",
			`'idx' should be a number for key: ${key}`,
		);
		t.is(
			typeof parsed.blockId,
			"string",
			`'blockId' should be a string for key: ${key}`,
		);
		t.is(
			typeof parsed.subBlockId,
			"string",
			`'subBlockId' should be a string for key: ${key}`,
		);
		t.is(
			typeof parsed.type,
			"string",
			`'type' should be a string for key: ${key}`,
		);
		t.is(
			typeof parsed.blockType,
			"string",
			`'blockType' should be a string for key: ${key}`,
		);
		t.is(
			typeof parsed.isConv,
			"boolean",
			`'isConv' should be a boolean for key: ${key}`,
		);
		t.is(
			typeof parsed.isAttention,
			"boolean",
			`'isAttention' should be a boolean for key: ${key}`,
		);
		t.is(
			typeof parsed.isSampler,
			"boolean",
			`'isSampler' should be a boolean for key: ${key}`,
		);
		t.is(
			parsed.key,
			key,
			`'key' should contain the original key string for key: ${key}`,
		);

		// Test that the values are consistent with expectations
		if (parsed.isConv) {
			t.is(
				parsed.type,
				"resnet",
				`When isConv is true, type should be 'resnet' for key: ${key}`,
			);
		}

		if (parsed.isSampler) {
			t.true(
				["upscaler", "downscaler"].includes(parsed.type),
				`When isSampler is true, type should be 'upscaler' or 'downscaler' for key: ${key}`,
			);
		}

		// Test that blockIdx is within the expected range
		t.true(
			parsed.blockIdx >= -1 && parsed.blockIdx <= 48,
			`'blockIdx' (${parsed.blockIdx}) should be between 0 and 48 for key: ${key}`,
		);

		//    if (parsed.isAttention) {
		// 		const firstTry = await t.try((tt, p) => {
		//      tt.is(p.type, 'attentions', `When isAttention is true, type should be 'attentions' for key: ${key}`);
		// 		}, parsed);
		//
		// 			if (firstTry.passed) {
		// 	firstTry.commit();
		// }
		//
		// 			const secondTry = await t.try((tt, p) => {
		//      tt.is(p.type, 'transformer', `When isAttention is true, type should be 'transformer' for key: ${key}`);
		// }, parsed);
		// secondTry.commit();
		//    }
	}
});
