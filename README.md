# LoRA Inspector

![Screenshot 2023-11-23 at 02-56-54 LoRA Inspector](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/64eb186a-414c-49b7-ad4b-0eb1567c0801)

[https://lora-inspector.rocker.boo/](https://lora-inspector.rocker.boo/)

Rust version of [LoRA inspector](https://github.com/rockerBOO/lora-inspector)

Web based tool for inspecting your LoRAs. All done in the browser, no servers. Private. No dependencies like torch or python.

![Screenshot 2023-12-05 at 03-50-31 LoRA Inspector](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/344cea55-9b1e-4321-93c8-3e3ddc70a9d2)

See the different blocks of the different networks. Ideally uses the more common format but still good to see how the weights are effectively. We use Frobenius norm for the magnitude and a vector norm for the strength.

![Screenshot 2023-12-05 at 03-33-38 LoRA Inspector](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/e128240f-ef1a-4c6c-a019-4548e57892e9)

See the different settings for the LoRA file. What model it was trained on. Any VAE. Network Dim/Rank and Alpha. Learning rates. Optimizer settings, learning rate schedulers.

![8efa06b5-4b86-4ffb-bf19-8842eaed5503](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/0df99556-73bf-4e2b-a576-2e2f0693377e)

Dataset with buckets. Bucket resolutions.

![aef217c2-59a2-40c1-b2a3-bef3df800895](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/26c8daec-353c-45da-a6a2-1e6a791dbf42)

Subsets showing the different subset datasets, image augments, captions.

![c2182030-8b1f-4eff-8007-3feafa60b577](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/cff02f1b-5f75-48ba-a126-3f781bcb1005)

Tags used (tags are phrases separated by , )

- WASM version
- Binary (coming soon)

## WASM

## Setup

### Build

```bash
wasm-pack build --target no-modules --out-dir pkg crates/lora-inspector-wasm --release
```

### Usage

Any http-server works (like python or nginx)

```bash
cd crates/lora-inspector-wasm
npx http-server # or any http server, static 
```

Then we can view it in the browser.

```
http://localhost:8080
```

### Deploy

```bash
fly deploy
```

## Contributions

Welcome

### Missing metadata options

- LR Warmup
