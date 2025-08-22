# Makefile for building and running WASM projects

# General settings
WASM_DIR := crates/lora-inspector-wasm
OUT_DIR := pkg
# TARGET := --target bundler 
TARGET :=  --target web 
RELEASE := --release
RELEASE := 
WEAK_REFS := --weak-refs
WEAK_REFS := 
SIMD := RUSTFLAGS="-C target-feature=+simd128"

# Default target
.PHONY: all
all: test build-wasm build-frontend

# Run tests for the whole workspace
.PHONY: test
test:
	cargo test --workspace && \
		make wasm-bindgen-test && \
		yarn --cwd $(WASM_DIR) test

# Build WASM for production (optimized)
.PHONY: build-wasm
build-wasm:
	wasm-pack build $(TARGET) --out-name lora-inspector --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS)

.PHONY: build-wasm-simd
build-wasm-simd:
	$(SIMD) wasm-pack build $(TARGET) --out-name lora-inspector-simd --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS)

.PHONY: build-frontend
build-frontend:
	yarn --cwd $(WASM_DIR) build

.PHONY: build-frontend
preview:
	yarn --cwd $(WASM_DIR) preview

build:
	 make build-wasm && make build-wasm-simd && make build-frontend

# Start a local HTTP server for serving the WASM package (simple)
.PHONY: dev-wasm
dev-wasm:
	make build-wasm && \
	make build-wasm-simd && \
		cd $(WASM_DIR) && \
		yarn vite

# Start a custom server (e.g., with CORS enabled) for development
.PHONY: dev-wasm-cors
dev-wasm-cors:
	cd $(WASM_DIR) && python simple-cors-server.py

deploy:
	fly deploy

fmt: 
	cargo fmt && (cd $(WASM_DIR) && yarn format)

run:
	cargo run --manifest-path crates/lora-inspector/cargo.toml

.PHONY: 
wasm-bindgen-test: 
	wasm-pack test --headless --firefox crates/lora-inspector-wasm

# Clean build artifacts (optional)
.PHONY: clean
clean:
	rm -rf $(OUT_DIR)/*
