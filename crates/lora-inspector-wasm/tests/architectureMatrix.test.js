import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_ROOT = path.join(__dirname, "../../inspector/tests/fixtures");

const FIXTURE_CELLS = [
	"sd15/kohya.json",
	"sdxl/kohya.json",
	"sdxl/lycoris.json",
	"flux1/kohya.json",
	"flux1/diffusers.json",
	"flux2_klein/kohya.json",
	"flux2_klein/diffusers.json",
	"wan/kohya.json",
	"wan/diffusers.json",
	"krea2/diffusers.json",
	"krea2/lycoris.json",
	"h3/diffusers.json",
	"qwen_image/kohya.json",
	"ltx23/diffusers.json",
];

function loadFixture(relativePath) {
	const raw = fs.readFileSync(path.join(FIXTURES_ROOT, relativePath), "utf8");
	return JSON.parse(raw);
}

// A handful of key suffixes (alpha, dora_scale, etc.) aren't block-structural
// and parseSDKey isn't expected to classify them -- these fixtures cover
// parseSDKey's own supported key shapes (double_blocks/single_blocks/
// transformer_blocks/blocks/te), not every tensor a safetensors file can hold.
const SKIP_SUFFIXES = [".alpha", ".dora_scale"];

for (const cell of FIXTURE_CELLS) {
	test(`parseSDKey handles every key in ${cell}`, (t) => {
		const fixture = loadFixture(cell);
		const keys = Object.keys(fixture.keys).filter(
			(k) => !SKIP_SUFFIXES.some((suffix) => k.endsWith(suffix)),
		);

		t.true(keys.length > 0, `${cell} should have parseable keys`);

		for (const key of keys) {
			t.notThrows(() => parseSDKey(key), `Should parse key: ${key}`);
		}
	});
}

// Golden-value assertions for representative keys from each of the new
// branches added alongside this matrix (Wan, LTX 2.3, Qwen-Image, and the
// SDXL LyCORIS resnet/downsampler fallback). t.notThrows above only proves
// parseSDKey doesn't throw -- it says nothing about whether the returned
// name/blockIdx/type are actually correct, which is what let the SDXL
// resnet indexing bug (input_blocks_3_0_op mis-numbered as IN09 instead of
// IN03) slip through with a fully green suite.
test("parseSDKey golden values: SDXL LyCORIS resnet/downsampler naming", (t) => {
	// input_blocks_3 is SGM's third flat U-Net block (IN03), a downsampler
	// ("op"). SDXL_RESNET_RE's block_id is already the flat SGM index, unlike
	// the transformer_blocks path where block_id is a supergroup index that
	// gets multiplied out.
	t.like(parseSDKey("lora_unet_input_blocks_3_0_op.hada_w1_a"), {
		name: "IN03",
		blockIdx: 4,
		idx: 3,
		type: "downsamplers",
		blockType: "input",
	});

	t.like(parseSDKey("lora_unet_input_blocks_1_0_in_layers_2.hada_w1_a"), {
		name: "IN01",
		blockIdx: 2,
		idx: 1,
		type: "resnets",
		blockType: "input",
	});

	t.like(parseSDKey("lora_unet_output_blocks_2_2_conv.hada_w1_a"), {
		name: "OUT02",
		idx: 2,
		type: "upsamplers",
		blockType: "output",
	});
});

test("parseSDKey golden values: Wan 2.x kohya blocks", (t) => {
	t.like(parseSDKey("lora_unet_blocks_0_cross_attn_k.lora_down.weight"), {
		name: "TB00",
		idx: 0,
		blockIdx: 0,
		type: "attentions",
		isAttention: true,
	});
});

test("parseSDKey golden values: LTX 2.3 diffusers transformer blocks", (t) => {
	t.like(
		parseSDKey("diffusion_model.transformer_blocks.0.attn1.to_k.lora_A.weight"),
		{
			name: "TB00",
			idx: 0,
			type: "attentions",
			isAttention: true,
		},
	);
});

test("parseSDKey golden values: Qwen-Image bare transformer blocks", (t) => {
	t.like(parseSDKey("transformer_blocks.0.attn.add_k_proj.lora_down.weight"), {
		name: "TB00",
		idx: 0,
		type: "attentions",
		isAttention: true,
	});
});
