import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import test from "ava";
import { parseSDKey } from "../assets/js/moduleBlocks.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES_ROOT = path.join(
	__dirname,
	"../../inspector/tests/fixtures",
);

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
