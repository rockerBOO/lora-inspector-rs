# LoRA Inspector

![Screenshot 2023-11-23 at 02-56-54 LoRA Inspector](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/64eb186a-414c-49b7-ad4b-0eb1567c0801)

[https://lora-inspector.rocker.boo/](https://lora-inspector.rocker.boo/)

Rust version of [LoRA inspector](https://github.com/rockerBOO/lora-inspector)

- WASM version
- Binary (coming soon)

## WASM

## Setup

### Build

```bash
wasm-pack build --target web
```

### Deploy

```bash
fly deploy
```

## Contributions

Welcome

## Bugs

- Loading a new file doesn't unload the previous LoRA properly (be aware of loading multiple LoRAS in the future)
- Loading some LoRAs fail to be loaded into the buffer
- Loading some LoRAs fail to load their block weights (undefined error)
- Average TE/UNet blocks are now invalid (generally)
