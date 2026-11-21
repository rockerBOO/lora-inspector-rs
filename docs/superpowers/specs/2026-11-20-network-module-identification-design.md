# Network module identification

Date: 2026-11-20
Status: awaiting approval

## Intent

A MOSS-Image LoRA (`mj-base-test-moss-1`) shows `no_module_found` where the
network module should be named. The value is wrong twice over: the file _does_
declare its module (`ss_network_module = networks.lora_generic`), and the
whitelist that drops it is the same one that has had to be extended by hand for
Lumina, SD3, Krea2, Flux2, `lora_fa`, `dylora` and `oft`.

The deeper problem is that `NetworkModule` is one flat list that mixes toolkits
(`kohya-ss/...`, `musubi-tuner/...`, `lycoris`) with modules, so "what made this
file" and "which trainer module produced it" cannot both be answered.

Goal: for any LoRA, state which toolkit produced it and which of that toolkit's
network modules, from evidence in the file — and stop reporting
`no_module_found` for files that carry the evidence.

## Goals

- `networks.lora_generic` displays as kohya's `lora_generic`, with
  `NetworkType::LoRA` so block weights are offered.
- Every other kohya `networks.<module>` that has no table entry still displays
  its module name (family `kohya`), rather than `no_module_found`.
- When `ss_network_module` is absent, derive the family from
  `modelspec.implementation` / `modelspec.implementation_version`, then from key
  shape (`weight.rs::is_peft`), then report unknown.
- Files with no usable metadata (e.g. `Aether_Ghost_v1.1_LoRA`,
  `amateur_style_v1_pony`: zero metadata entries) report family `unknown`
  instead of a misleading module name.

## Non-goals

- No new architecture key parsers, and no change to weight/norm math.
- No inventing a module name that is not in the file. A family without a module
  shows the family only.
- No inference of the _model architecture_ (SDXL vs MOSS vs Flux) — only the
  toolkit that wrote the LoRA.

## Anti-goals

- Echoing an arbitrary metadata string as if it were a recognised module.
- Typing unknown modules (e.g. `networks.loha`) as LoRA just because they live
  under `networks.`; they keep `NetworkType` unset until someone maps them.

## Evidence

| file                      | metadata signal                                                                                                   | correct answer             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `mj-base-test-moss-1`     | `ss_network_module=networks.lora_generic`, `ss_sd_scripts_commit_hash`, `implementation_version=sd-scripts/<sha>` | kohya · `lora_generic`     |
| `pony-style-hardstuck-v1` | no `ss_network_module`; `modelspec.implementation=.../diffusers`                                                  | diffusers · unknown module |
| `JN_Kurashiki_Reika`      | `ss_network_module=networks.lora`                                                                                 | kohya · `lora`             |
| `BoobsgamesLycoXL`        | `ss_network_module=lycoris.kohya`                                                                                 | lycoris · `kohya`          |
| `Aether_Ghost_v1.1_LoRA`  | none                                                                                                              | unknown                    |
| `MiniMax H3` PEFT keys    | none; `lora_A`/`lora_B`                                                                                           | peft                       |

`networks/lora_generic.py` was confirmed present in `kohya-ss/sd-scripts`
alongside 15+ sibling modules.

## Approaches considered

1. **Grow the flat enum.** Add `KohyaSSLoRAGeneric` + missing siblings. Small,
   but repeats the whack-a-mole and still cannot describe PEFT files.
2. **Family and module, no inference.** Right shape, but files without
   `ss_network_module` stay unknown.
3. **Family and module with the detection chain (chosen).** Metadata first,
   key shape last.

## Design

`NetworkFamily`: `Kohya`, `MusubiTuner`, `Lycoris`, `Diffusers`, `Peft`,
`Unknown`.

Detection, first match wins, in `metadata.rs`:

1. `ss_network_module`:
   - `networks.<module>` → family from a small tool table
     (`lora_flux_2` → `MusubiTunerLoRAFlux2`, `lora_krea2` →
     `MusubiTunerLoRAKrea2`, everything else → `Kohya`), module = `<module>`.
   - `lycoris.kohya` → `Lycoris`, module `kohya`.
2. `modelspec.implementation_version` prefix, then `modelspec.implementation`
   URL: `sd-scripts/` → `Kohya`, `.../diffusers` → `Diffusers`,
   `OpenMOSS/MOSS-Image` → `Kohya` (it trains through sd-scripts).
3. Key shape: `weight.rs::is_peft` → `Peft`.
4. Otherwise `Unknown`.

Known modules keep their existing friendly strings (`kohya-ss/lora`,
`musubi-tuner/lora_krea2`, …) so nothing regresses. Unknown-but-attributed
modules render as `<family> · <module>`; a family with no module renders
`<family>` alone; nothing at all renders `unknown`.

`worker.rs` gains `network_family()` and `network_module_name()` alongside the
existing `network_module()`, so the wire format for `Network.jsx` stays as it is
today and gains two fields. `network_type()` maps `lora_generic` to `LoRA`.

## Calls left to me

- `NetworkFamily` as a separate serde enum returned over the wire, rather than
  folding the family into the existing display string.
- Treating MOSS-Image as kohya because it trains through sd-scripts.
- Leaving `SS_network_module`-less diffusers files at "diffusers · unknown
  module" instead of guessing `lora_diffusers`.

## Tests

- Rust: table for `networks.lora_generic`, an unmapped `networks.lora_anima`,
  `lycoris.kohya`, diffusers-only metadata, empty metadata + PEFT keys, empty
  metadata + kohya keys, and the two existing sd-scripts strings.
- Rust: `network_type()` returns `LoRA` for `lora_generic`.
- Existing `metadata.rs` assertions for `network_module()` updated to the new
  return shape; snapshot tests re-blessed where they encode the module string.
- JS: `Network.jsx` renders family and module rows for the three shapes (both,
  family only, unknown).
