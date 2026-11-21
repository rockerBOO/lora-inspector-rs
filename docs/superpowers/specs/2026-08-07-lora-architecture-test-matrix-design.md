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
correctly in both naming conventions, and regressions can reappear silently
without a matrix pinning each case down. (The specific case logged in
`BUGS.md` — `transformer.single_transformer_blocks.9.proj_out` failing to
match — turned out to already be fixed in the current `FLUX_PEFT` regex in
`moduleBlocks.js`; `BUGS.md` is stale. The matrix still covers this key
shape so a regression there is caught.)

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
- Fix the one real bug the matrix surfaces (the `diffusion_model.`/bare-prefix
  `unet_keys()`/`text_encoder_keys()` gap affecting Wan, Krea2, LTX2.3, and
  Qwen-Image) so the full matrix passes green, rather than leaving it
  `#[ignore]`d.

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
| SD1.5 | `add_detail.safetensors` | *(gap — the only candidate, `spo-sd-v1-5_4k-p_10ep_lora_diffusers.safetensors`, uses a third, dot-separated `lora.down.weight`/`lora.up.weight` convention that `get_base_name` doesn't recognize; fixing that is a separate, unscoped change, so this cell is kohya-only for now)* |
| SDXL | `sdxl/anyaXL.safetensors` | *(gap — none found)* |
| SDXL LyCORIS | `sdxl/gwei.safetensors` (LoHA, `ss_network_module=lycoris.kohya`, `algo=loha` in metadata) | — |
| Flux.1 | `flux/DracoFelis.safetensors` | `flux/FLUX-daubrez-DB4RZ.safetensors` (`transformer.single_transformer_blocks...`) |
| Flux.2/Klein | `flux2-klein-4b/nipplediffusion-f2-klein-4b.safetensors` | `flux2-klein-9b/dx8152-Klein-Migration.safetensors` (`diffusion_model.double_blocks...lora_A/B`) |
| Wan/Wan2.x | `wan2.2/wan2.1-i2v-480p-pov-insertion-v1.0.safetensors` (`lora_unet_blocks...lora_down/up`) | `wan2.2/wan2.2_5b_c0wg1rl_72_000002500.safetensors` (`diffusion_model.blocks...lora_A/B`) |
| Krea2 | *(gap — none found)* | `krea2/bloomgirls-ultrarealism-krea2_4k.safetensors` |
| Krea2 LyCORIS | — | `krea2/realism_engine_krea2_v2.safetensors` (LoKr, **no `__metadata__` block at all** — see caveat below) |
| MiniMax H3 | — | `h3/minimax_h3_fl2v_turbo_4step_v0.1.safetensors` (already has an `#[ignore]`d regression test; matrix supersedes it) |
| Qwen-Image | `qwen-image/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors` (bare `transformer_blocks.N...lora_down/up.alpha`, no `lora_unet` prefix) | *(gap — none found)* |
| LTX2.3 | *(gap — none found)* | `ltx2.3/DR34ML4Y_LT3X_V3.safetensors` (`diffusion_model.transformer_blocks...lora_A/B`) |

Gap cells get a single test using the one format that exists, with a comment
explaining why the other cell is absent. No fixture is fabricated.

**Krea2 LyCORIS caveat:** `LoRAFile::scale_weight` dispatches to the correct
decomposition (`scale_lokr_weight`, `scale_hada_weight`, ...) by reading
`ss_network_args`/`ss_network_module` from the file's training metadata; with
no metadata it falls back to plain `scale_lora_weight`, which doesn't
recognize LoKr-only keys and errors. The SDXL LyCORIS fixture has real kohya
training metadata so scale dispatch works correctly and is asserted; the
Krea2 LyCORIS fixture has none (musubi-tuner didn't write it), so for that
one cell the matrix test stops at `base_names()`/`dims()`/`unet_keys()` and
does not assert `scale_weight()` succeeds. Making network-type detection
metadata-independent is a separate, unscoped feature.

## Fixture format and extraction

New `extract-fixture` subcommand on the `lora-inspector` CLI
(`crates/lora-inspector/src/main.rs`):

```
lora-inspector extract-fixture --file <path> --out <path.json>
```

It reads only the safetensors header (8-byte little-endian length prefix +
that many bytes of JSON header — no tensor data is loaded, so this works
even on multi-GB files, mirroring the existing `metadata_from_file` helper
already in `main.rs`) and writes:

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

`LoRAFormat` currently isn't reachable from outside the crate (`weight` is a
private module even though `LoRAFile::format()` returns `weight::LoRAFormat`
publicly), so it needs `pub use crate::weight::LoRAFormat;` added to `lib.rs`
before an external `tests/` file can name it.

One test function per matrix cell: `sd15_kohya`, `sdxl_kohya`,
`sdxl_lycoris`, `flux1_kohya`, `flux1_diffusers`, `flux2_klein_kohya`,
`flux2_klein_diffusers`, `wan_kohya`, `wan_diffusers`, `krea2_diffusers`,
`krea2_lycoris`, `h3_diffusers`, `qwen_image_kohya`, `ltx23_diffusers` (14
cells — SD1.5/Krea2/Qwen-Image/LTX2.3 kohya-or-diffusers gaps noted above are
simply absent, not stubbed). Each asserts:

- `is_tensors_loaded()` is true
- `format()` matches the expected `LoRAFormat` (`LoRAFormat::Lycoris` is
  dead code — never constructed anywhere in `weight.rs` — so LyCORIS cells
  still expect `Kohya`; only the naming-convention axis, Kohya vs Peft, maps
  to `format()`)
- `base_names()` is non-empty
- `dims()` is non-empty
- `unet_keys()` is non-empty — this is the assertion that currently fails
  for `diffusion_model.`-prefixed and bare-prefixed architectures (see fix
  below)
- `scale_weight()` succeeds for `base_names()[0]`, **except** for
  `krea2_lycoris` (see caveat above)

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

## Bug fix required for the matrix to pass

**`unet_keys()`/`text_encoder_keys()` prefix gap**: both
`BufferedLoRAWeight` and `LoRAWeight` in `crates/inspector/src/weight.rs`
implement these by filtering all keys on the literal substrings
`"lora_unet"` / `"lora_te"`. Every architecture that doesn't use the kohya
`lora_unet_`/`lora_te` prefix — Wan, Krea2, LTX2.3, Qwen-Image,
Flux.2/Klein's diffusers files — returns an empty `unet_keys()` regardless
of whether the keys are otherwise valid LoRA weights. This is user-visible:
`unet_keys()`/`text_encoder_keys()` back the "Advanced" panel's key listing
in the UI (`assets/js/components/analysis/Advanced.jsx` via
`worker.js`/`worker.rs`).

Fix: invert the logic. Add a private `is_text_encoder_key(key: &str) -> bool`
matching `"lora_te"`, `"text_encoder"`, or `"text_model"`, then define
`text_encoder_keys()` as `self.keys()` filtered to matches and `unet_keys()`
as `self.keys()` filtered to non-matches — i.e. "not a text-encoder key"
rather than "matches a specific unet prefix". This is architecture-agnostic
so it doesn't need a prefix allowlist, and preserves current behavior for
kohya SD1.5/SDXL files (every key there contains either `lora_unet` or
`lora_te`, so the counts don't change — verified against the existing
`unet_keys`/`text_encoder_keys` tests in `worker.rs` which expect 576/216
for `boo.safetensors`).

This fix is scoped narrowly to key-classification; no change to
`base_names()`/`scale_weight()` math.

## Testing

- `cargo test -p inspector architecture_matrix`
- `cd crates/lora-inspector-wasm && yarn ava tests/architectureMatrix.test.js`
- `make test` for full suite regression
