# LoRA Inspector Core Library

Core Rust library for parsing and analyzing LoRA (Low-Rank Adaptation) model files.

## Features

- Parse safetensors files
- Extract and analyze network metadata
- Calculate tensor norms and statistics
- Support for multiple LoRA variants:
  - LoRA
  - LoKR
  - LoHA
  - OFT
  - BOFT
  - GLoRA

## Technologies

- Rust
- Candle (machine learning library)
- SafeTensors
- WebAssembly support

## Key Modules

- `file`: LoRA file parsing
- `metadata`: Metadata extraction
- `network`: Network type handling
- `norms`: Tensor norm calculations
- `statistic`: Statistical analysis

## Usage

```rust
use inspector::{file, metadata, norms};

// Load a LoRA file
let lora_file = file::LoRAFile::new_from_buffer(data, filename, &device);

// Extract base names
let base_names = lora_file.base_names();

// Calculate norms
let l2_norm = norms::l2(&scale_weight)?;
```

## WebAssembly Support

Includes WASM bindings for browser-based use.

## Testing

```bash
cargo test
```

## Contributing

- Improve parsing support
- Add more norm/statistic calculations
- Enhance error handling

## License

See the root project LICENSE file
