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
		(cd $(WASM_DIR) && yarn test)

# Build WASM for production (optimized)
.PHONY: build-wasm
build-wasm:
	wasm-pack build $(TARGET) --out-name lora-inspector --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS)

.PHONY: build-wasm-simd
build-wasm-simd:
	$(SIMD) wasm-pack build $(TARGET) --out-name lora-inspector-simd --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS)

.PHONY: build-frontend
build-frontend:
	(cd $(WASM_DIR) && yarn build)

.PHONY: build-frontend
preview:
	(cd $(WASM_DIR) && yarn preview)

build:
	 make build-wasm && make build-wasm-simd && make build-frontend

# Start a local dev server with debug WASM builds.
# --dev skips wasm-opt entirely so DWARF debug info from rustc is preserved,
# giving demangled Rust function names in browser DevTools and Node.js stack traces.
.PHONY: dev-wasm
dev-wasm:
	wasm-pack build $(TARGET) --dev --out-name lora-inspector --out-dir $(OUT_DIR) $(WASM_DIR) && \
	$(SIMD) wasm-pack build $(TARGET) --dev --out-name lora-inspector-simd --out-dir $(OUT_DIR) $(WASM_DIR) && \
		(cd $(WASM_DIR) && \
		yarn vite)

# Release build with debug info preserved — optimized but stack traces still show
# demangled Rust names. Useful for debugging panics on staging/production.
# wasm-opt requires at least one pass (-O) for --debuginfo (-g) to take effect.
.PHONY: build-wasm-debuginfo
build-wasm-debuginfo:
	wasm-pack build $(TARGET) --out-name lora-inspector --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS) --no-opt && \
	wasm-opt -O -g $(OUT_DIR)/lora-inspector_bg.wasm -o $(OUT_DIR)/lora-inspector_bg.wasm && \
	$(SIMD) wasm-pack build $(TARGET) --out-name lora-inspector-simd --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS) --no-opt && \
	wasm-opt -O -g $(OUT_DIR)/lora-inspector-simd_bg.wasm -o $(OUT_DIR)/lora-inspector-simd_bg.wasm

# Start a custom server (e.g., with CORS enabled) for development
.PHONY: dev-wasm-cors
dev-wasm-cors:
	cd $(WASM_DIR) && python simple-cors-server.py

deploy:
	fly deploy

fmt: 
	cargo fmt && (cd $(WASM_DIR) && yarn format)

run:
	cargo run --manifest-path crates/lora-inspector/Cargo.toml

lint: 
	cargo clippy && (cd $(WASM_DIR) && yarn lint --fix)

.PHONY: wasm-bindgen-test
wasm-bindgen-test:
	wasm-pack test --headless --firefox crates/lora-inspector-wasm

# Run wasm tests in Node.js — gives demangled Rust function names in stack
# traces, making panics like index-out-of-bounds much easier to locate.
# Debug build (no --release) keeps full DWARF symbols.
.PHONY: wasm-bindgen-test-node
wasm-bindgen-test-node:
	wasm-pack test --node crates/lora-inspector-wasm

.PHONY: test-panic-hook
test-panic-hook:
	wasm-pack test --headless --firefox crates/console-panic-hook

# Same but in Node.js for better stack traces when debugging panics.
.PHONY: test-panic-hook-node
test-panic-hook-node:
	wasm-pack test --node crates/console-panic-hook --features test-node

# Clean build artifacts (optional)
.PHONY: clean
clean:
	rm -rf $(OUT_DIR)/*
