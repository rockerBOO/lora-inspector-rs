# LoRA architecture/format test matrix

## Problem

The parsing logic (`crates/inspector/src/key.pest`, `weight.rs`, and
`crates/lora-inspector-wasm/assets/js/moduleBlocks.js`) supports many model
architectures (SD1.5, SDXL, Flux.1, Flux.2/Klein, Wan/Wan2.x, Krea2, MiniMax
H3, Qwen-Image, LTX2.3) and two key-naming conventions per architecture
(Kohya-style `lora_unet_..._lora_down/up.weight` and diffusers/PEFT-style
`...lora_A/B.weight`). Coverage is uneven and ad hoc: most existing tests are
hand-picked keys or `#[ignore]`d regression tests tied to one developer
machine's file paths (e.g. `load_minimax_h3_file`, `load_women_flux2_file`).
There is no systematic guarantee that every supported architecture parses
correctly in both naming conventions, and regressions (like the
`transformer.single_transformer_blocks...` Flux bug already logged in
`BUGS.md`) can reappear silently.

Real LoRA files for most architecture/format combinations exist locally
under `/mnt/900/lora`, but they range from a few MB to several GB and can't
reasonably live in the git repo.

## Goals

- One committed, lightweight fixture per (architecture, format) cell that
  can be checked out anywhere, without needing the original multi-GB
  safetensors files.
- Rust tests that exercise the full `LoRAFile` pipeline (format detection,
  `NetworkModule`, `base_names()`, `dims()`, `unet_keys()`/
  `text_encoder_keys()`, `scale_weight()`) for every matrix cell.
- JS tests that exercise `parseSDKey` against the same key sets, so the UI's
  block-naming logic is covered too.
- Fix the bugs the matrix surfaces (the known Flux `transformer.` prefix bug,
  and the `diffusion_model.`-prefixed `unet_keys()` gap for Wan/Krea2/LTX2.3)
  so the full matrix passes green, rather than leaving them `#[ignore]`d.

## Non-goals

- LLM LoRAs (Gemma3) — out of scope, this tool targets SD-family diffusion
  models.
- Fabricating fixtures for architecture/format combinations that don't
  exist on disk (see Gaps below) — those cells get a single-format test
  instead, with a comment noting the gap.
- Testing numerical correctness of SVD/scale outputs — the matrix uses
  zero-filled synthetic tensors, so it only proves the parsing/shape-handling
  path doesn't panic or misclassify. Numerical correctness is covered by
  existing `svd.rs`/`weight.rs` unit tests.

## Coverage matrix

| Architecture | Kohya-style source | Diffusers/PEFT-style source |
|---|---|---|
| SD1.5 | `add_detail.safetensors` | `spo-sd-v1-5_4k-p_10ep_lora_diffusers.safetensors` (dot-separated `lora.down.weight`) |
| SDXL | `sdxl/anyaXL.safetensors` | *(gap — none found)* |
| SDXL LyCORIS | `sdxl/gwei.safetensors` (LoHA/LoKr) | — |
| Flux.1 | `flux/DracoFelis.safetensors` | `flux/FLUX-daubrez-DB4RZ.safetensors` (`transformer.single_transformer_blocks...` — the known BUGS.md case) |
| Flux.2/Klein | `flux2-klein-4b/nipplediffusion-f2-klein-4b.safetensors` | `flux2-klein-9b/dx8152-Klein-Migration.safetensors` (`diffusion_model.double_blocks...lora_A/B`) |
| Wan/Wan2.x | `wan2.2/SmoothXXXAnimation_High.safetensors` (`diffusion_model.blocks...lora_down/up`) | `wan2.2/wan2.2_5b_c0wg1rl_72_000002500.safetensors` (`...lora_A/B`) |
| Krea2 | *(gap — none found)* | `krea2/bloomgirls-ultrarealism-krea2_4k.safetensors` |
| Krea2 LyCORIS | — | `krea2/realism_engine_krea2_v2-svd.safetensors` |
| MiniMax H3 | — | `h3/minimax_h3_fl2v_turbo_4step_v0.1.safetensors` (already has an `#[ignore]`d regression test; matrix supersedes it) |
| Qwen-Image | `qwen-image/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors` (bare `transformer_blocks.N...lora_down/up.alpha`, no `lora_unet` prefix) | *(gap — none found)* |
| LTX2.3 | *(gap — none found)* | `ltx2.3/DR34ML4Y_LT3X_V3.safetensors` (`diffusion_model.transformer_blocks...lora_A/B`) |

Gap cells get a single test using the one format that exists, with a comment
explaining why the other cell is absent. No fixture is fabricated.

## Fixture format and extraction

New `extract-fixture` subcommand on the `lora-inspector` CLI
(`crates/lora-inspector/src/main.rs`):

```
lora-inspector extract-fixture --file <path> --out <path.json>
```

It reads only the safetensors header (8-byte length prefix + JSON header —
no tensor data is loaded, so this works even on multi-GB files) plus
`Metadata::new_from_buffer`, and writes:

```json
{
  "source": "flux/FLUX-daubrez-DB4RZ.safetensors",
  "keys": {
    "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight": { "shape": [16, 3072], "dtype": "F16" }
  },
  "metadata": { "...": "raw __metadata__ block, if present" }
}
```

Fixtures are committed under
`crates/inspector/tests/fixtures/<arch>/<format>.json` (e.g.
`fixtures/flux1/kohya.json`, `fixtures/flux1/diffusers.json`,
`fixtures/wan/kohya.json`, `fixtures/wan/diffusers.json`). `source` is kept
only as provenance/documentation, not read by tests.

## Rust test layer

New integration test file `crates/inspector/tests/architecture_matrix.rs`.

A shared helper synthesizes an in-memory safetensors buffer from a fixture:
for each `(key, {shape, dtype})`, allocate a zero-filled tensor of that shape
and dtype and write a valid safetensors buffer (header + zero data). This
buffer is fed to `LoRAFile::new_from_buffer`, exercising the real load path
end-to-end without needing real weights — zero data is sufficient because
these tests assert on *parsing/shape/classification* behavior, not numeric
output.

One test function per matrix cell (e.g. `flux1_kohya`, `flux1_diffusers`,
`wan_kohya`, `wan_diffusers`, `sdxl_kohya`, `sdxl_lycoris`, `krea2_diffusers`,
`krea2_lycoris`, `qwen_image`, `ltx23_diffusers`, `sd15_kohya`,
`sd15_diffusers`, `h3_diffusers`). Each asserts:

- `is_tensors_loaded()` is true
- `format()` matches the expected `LoRAFormat` (Kohya/Peft/Lycoris where
  applicable)
- `base_names()` is non-empty
- `dims()` is non-empty
- `unet_keys()` (and `text_encoder_keys()` where the fixture has text
  encoder keys) is non-empty — this is the assertion that currently fails
  for `diffusion_model.`-prefixed architectures, since `unet_keys()` filters
  on the literal substring `"lora_unet"`
- `scale_weight()` succeeds (doesn't panic/error) for at least one base name

The existing `#[ignore]`d machine-specific tests in `file.rs`
(`load_women_flux2_file`, `reproduce_flux2_panic`, `load_minimax_h3_file`)
are left as-is; they're developer-machine regression aids, not part of the
committed matrix.

## JS test layer

New file `crates/lora-inspector-wasm/tests/architectureMatrix.test.js`,
following the existing AVA pattern in `moduleBlocks.test.js`. It imports the
same fixture JSON files (via a relative path into
`crates/inspector/tests/fixtures/`), and for each fixture:

```js
for (const key of Object.keys(fixture.keys)) {
  t.notThrows(() => parseSDKey(key), `Should parse key: ${key}`);
}
```

with per-architecture assertions on `blockType`/`type` for a sample of keys,
mirroring the existing Flux tests in `moduleBlocks.test.js`.

## Bug fixes required for the matrix to pass

1. **Flux diffusers-native prefix** (`BUGS.md`): keys like
   `transformer.single_transformer_blocks.9.proj_out` fail to match. Fix the
   relevant regex/pest rule in `key.pest` and/or the JS regex in
   `moduleBlocks.js` (both currently anchor on `lora_unet`/`diffusion_model`
   prefixes and don't handle a bare `transformer.` prefix).
2. **`unet_keys()`/`text_encoder_keys()` prefix gap**: these filter on the
   literal substrings `"lora_unet"` / `"lora_te"`
   (`crates/inspector/src/weight.rs`), so `diffusion_model.`-prefixed
   architectures (Wan, Krea2, LTX2.3, Flux.2/Klein diffusers files) return
   empty regardless of whether the keys are otherwise valid LoRA weights.
   Fix by broadening the match (e.g. also matching `diffusion_model.` as a
   unet-equivalent prefix), scoped narrowly to avoid misclassifying
   text-encoder-only keys.

Both fixes are scoped narrowly to the key-classification logic; no change to
`base_names()`/`scale_weight()` math is expected.

## Testing

- `cargo test -p inspector architecture_matrix`
- `cd crates/lora-inspector-wasm && yarn ava tests/architectureMatrix.test.js`
- `make test` for full suite regression
