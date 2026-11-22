# LoRA Inspector

![LoRA Inspector Logo](logo.png)

A comprehensive tool for inspecting LoRA (Low-Rank Adaptation) files for machine learning models, particularly Stable Diffusion.
![block-weight-norms](https://github.com/user-attachments/assets/363239b3-3cba-4b97-83b8-d2fa0557f8c9)

See the different blocks of the different networks. Ideally uses the more common format but↵
still good to see how the weights are effectively. We use Frobenius norm for the magnitude↵
and a vector norm for the strength.

![lora-network-module-metadata](https://github.com/user-attachments/assets/cee5b3cc-6c39-478a-aa68-cb95b2a0aa83)

![learning-rate-metadata](https://github.com/user-attachments/assets/06609fd7-394f-4bd7-9cba-27f9539aabf7)

See the different settings for the LoRA file. What model it was trained on. Any VAE. Networ↵
k Dim/Rank and Alpha. Learning rates. Optimizer settings, learning rate schedulers.

![epoch-dataset-steps](https://github.com/user-attachments/assets/aa6e3b9a-4ebb-458a-b0dd-ef07d640ac60)

Dataset with buckets. Bucket resolutions.

![dataset-subset-metadata](https://github.com/user-attachments/assets/6d3a135d-51fa-4a6d-b9a5-f1a0322a36c3)

Subsets showing the different subset datasets, image augments, captions.

![c2182030-8b1f-4eff-8007-3feafa60b577](https://github.com/rockerBOO/lora-inspector-rs/assets/15027/cff02f1b-5f75-48ba-a126-3f781bcb1005)

## Features

- Browser-based LoRA file inspection
- Detailed metadata analysis
- Weight and network characteristic visualization
- WebAssembly-powered performance
- No external dependencies (like torch or python)

## Reading the rank numbers

The `--rank` section reports two kinds of quantity, and only one of them travels between files.

**Absolute** — comparable across any two LoRAs of the same architecture:

- `effective_rank`: `exp(H)` over the singular-value energy distribution `s²` of the update
  (`up @ down`). `5.6` means the energy is spread as if ~5.6 directions were in use; it is a
  concentration measure, not a count of dimensions.
- `top1_energy` (`s₀² / Σs²`), `dominance` (`s₀ / s₁`), `eff_scale` (`‖up@down‖_F · alpha/rank`,
  so alpha is included).
- `summary.rank.mean_effective_rank` / `median_effective_rank`.

**Relative** — meaningful only inside one file, because the denominator is the file itself:

- `balance` = `effective_rank / nominal_rank`, and therefore every health bucket
  (Good ≥ 0.75, Ok ≥ 0.50, Weak ≥ 0.25, Poor below). Raise `dim` from 16 to 32 and an
  unchanged effective rank of 5.6 moves from *Weak* to *Poor* without the LoRA changing. Use
  `effective_rank` to compare across `dim`; use `balance`/health to describe one file.
- `outlier`, and `summary.rank.outliers`: `eff_scale > 1.5 × median(eff_scale of this file)`.
  A uniformly stronger or weaker LoRA moves the median with it, so the count stays put and the
  flag cannot detect a change in overall strength. The threshold and the median it came from are
  both emitted (`summary.rank.outlier_threshold`, `median_eff_scale`) so you can normalise
  yourself — `eff_scale / median_eff_scale` is the scale-invariant version of the flag.
- Block-level `outliers` (in `blocks[]` and `summary.block_weight`) are counted over **blocks**;
  layer-level `outlier` (in `rank[]`/`layers[]`) is counted over **layers**. Same word, different
  denominator.

## Project Structure

- `crates/`: Project sub-modules
  - `inspector/`: Core Rust library
  - `lora-inspector-wasm/`: WebAssembly web frontend
  - `lora-inspector/`: CLI application

## Technologies

- Rust
- WebAssembly
- React
- Vite
- Candle (machine learning library)

## Testing Strategy

### Test Frameworks

- lora-inspector-wasm:
  - AVA (JavaScript unit tests)
  - Playwright (E2E testing)
- inspector: Rust built-in test module
- lora-inspector: Limited test coverage

### Test Coverage

- Unit tests for key parsing
- E2E browser testing
- Metadata validation
- Cross-browser compatibility

## Getting Started

### Prerequisites

- Rust toolchain
- Node.js
- Yarn 4.9.1+
- wasm-pack

### Installation

```bash
# Clone the repository
git clone https://github.com/rockerBOO/lora-inspector-rs.git

# Navigate to the project directory
cd lora-inspector-rs

# Install dependencies for web frontend
cd crates/lora-inspector-wasm
yarn install --immutable
```

### Development

```bash
# Start development server
make dev-wasm

# Build the project
make build

# Run tests
make test
```

## Crate-Specific Documentation

- [LoRA Inspector WASM Frontend](/crates/lora-inspector-wasm/README.md)
- [LoRA Inspector CLI](/crates/lora-inspector/README.md)

## Contributing

Contributions are welcome! Please check individual crate READMEs for specific guidelines.

### Current Roadmap

- [x] WebAssembly frontend
- [ ] Add LR Warmup metadata support
- [ ] Improve weight visualization
- [ ] Expand metadata parsing capabilities
- [ ] Increase test coverage

## License

MIT License - See [LICENSE](LICENSE) file
