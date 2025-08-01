# LoRA Inspector CLI

Command-line interface for analyzing LoRA (Low-Rank Adaptation) model files.

## Features

- Inspect block weights of safetensors files
- Compare metadata between different LoRA files
- Detailed weight and norm analysis
- Multiple output formats (JSON, text)

## Installation

```bash
cargo install --path .
```

## Usage

### Block Weights Analysis

```bash
# Analyze block weights (default JSON output)
lora-inspector block-weights --file path/to/model.safetensors

# Specify output format
lora-inspector block-weights --file path/to/model.safetensors --output-format text
```

### Metadata Comparison

```bash
# Compare metadata between two LoRA files
lora-inspector compare-metadata --file1 model1.safetensors --file2 model2.safetensors
```

## Output Formats

- `json`: Detailed JSON output (default)
- `text`: Human-readable text format with statistics and visualizations

## Optional Features

- CUDA support: Enable with `--features cuda`

## Contributing

- Suggest improvements or report issues on GitHub
- Pull requests welcome

## License

See the root project LICENSE file