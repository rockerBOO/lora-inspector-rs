# LoRA Inspector Web Application

A WebAssembly-powered web application for inspecting LoRA (Low-Rank Adaptation) files for machine learning models.

## Overview

This web application provides a browser-based tool for analyzing LoRA training metadata, weights, and network characteristics, powered by Rust and WebAssembly.

## Technologies

- Rust
- WebAssembly
- React (v18.3.1)
- Vite
- Chartist

## Testing Strategy

### Test Frameworks
- AVA (JavaScript unit tests)
- Playwright (E2E testing)

### Test Coverage
- Unit tests for key parsing
- E2E browser testing
- Metadata validation
- Cross-browser compatibility

### Supported Browsers
- Chromium
- Firefox
- WebKit (Safari)

## Prerequisites

- Rust toolchain
- Node.js
- Yarn 4.9.1+
- wasm-pack

## Installation

```bash
# Install dependencies
yarn install --immutable

# Install wasm-pack (if not already installed)
cargo install wasm-pack
```

## Development Workflow

### Start Development Server

```bash
yarn dev
```

### Build for Production

```bash
# Build WebAssembly package
wasm-pack build --target web --out-dir pkg

# Build frontend
yarn build
```

### Testing

```bash
# Run JavaScript tests
yarn test

# Run E2E tests
yarn e2e-test
```

### Code Quality

```bash
# Format code
yarn format

# Lint code
yarn lint
```

## Project Structure

- `src/`: Rust WebAssembly sources
- `assets/js/`: JavaScript modules
  - `main.js`: Entry point
  - `worker.js`: Web worker implementations
  - `components.js`: React components
  - `lib.js`: Utility functions

## Deployment

The application can be deployed as a static website. Build the project and serve the `dist/` directory.

## E2E Testing Details

### Playwright Configuration
- Parallel test execution
- Supports multiple browser targets
- Configurable for CI/CD environments

### Test Scenarios
- Page title verification
- File upload interaction
- Network metadata validation
- Cross-browser compatibility checks

## Contributing

Contributions are welcome! Please ensure:
- Code follows project formatting and linting rules
- All tests pass
- New features are accompanied by appropriate tests

## Roadmap
- [ ] Increase test coverage
- [ ] Expand E2E test scenarios
- [ ] Improve browser compatibility

## License

See the root project LICENSE file