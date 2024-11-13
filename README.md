# LoRA Inspector

![home](https://github.com/user-attachments/assets/48bebf44-187c-4fd7-91a0-d8611a8092d3)

[https://lora-inspector.rocker.boo/](https://lora-inspector.rocker.boo/)

Rust version of [LoRA inspector](https://github.com/rockerBOO/lora-inspector)

![lora-training-metadata](https://github.com/user-attachments/assets/b87df1b5-57f0-4cce-8dc4-2dc2ebc7780d)

Web based tool for inspecting your LoRAs. All done in the browser, no servers. Private. No dependencies like torch or python.

![block-weight-norms](https://github.com/user-attachments/assets/363239b3-3cba-4b97-83b8-d2fa0557f8c9)

See the different blocks of the different networks. Ideally uses the more common format but still good to see how the weights are effectively. We use Frobenius norm for the magnitude and a vector norm for the strength.

![lora-network-module-metadata](https://github.com/user-attachments/assets/cee5b3cc-6c39-478a-aa68-cb95b2a0aa83)

![learning-rate-metadata](https://github.com/user-attachments/assets/06609fd7-394f-4bd7-9cba-27f9539aabf7)

See the different settings for the LoRA file. What model it was trained on. Any VAE. Network Dim/Rank and Alpha. Learning rates. Optimizer settings, learning rate schedulers.

![epoch-dataset-steps](https://github.com/user-attachments/assets/aa6e3b9a-4ebb-458a-b0dd-ef07d640ac60)

Dataset with buckets. Bucket resolutions.

![dataset-subset-metadata](https://github.com/user-attachments/assets/6d3a135d-51fa-4a6d-b9a5-f1a0322a36c3)

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
