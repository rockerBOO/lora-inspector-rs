# Makefile for building and running WASM projects

# General settings
WASM_DIR := crates/lora-inspector-wasm
OUT_DIR := pkg
# TARGET := --target bundler
TARGET :=  --target web
RELEASE := --release
DEV := --dev
WEAK_REFS := --weak-refs
WEAK_REFS :=
SIMD := RUSTFLAGS="-C target-feature=+simd128"

# Default target
.PHONY: all
all: test build-wasm build-frontend

# Run tests for the whole workspace (debug build)
.PHONY: test
test:
	cargo test --workspace && \
		make wasm-bindgen-test && \
		(cd $(WASM_DIR) && yarn test)

# Run tests for the whole workspace under the release profile (opt-level=3
# for the inspector crate — see Cargo.toml's [profile.release.package.inspector]).
# Use this to verify perf-sensitive changes (svd/candle numeric code) under
# the same optimization level the shipped wasm build uses.
.PHONY: test-release
test-release:
	cargo test --workspace --release

# Build WASM for production (optimized, slow to compile)
.PHONY: build-wasm
build-wasm:
	wasm-pack build $(TARGET) --out-name lora-inspector --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS)

.PHONY: build-wasm-simd
build-wasm-simd:
	$(SIMD) wasm-pack build $(TARGET) --out-name lora-inspector-simd --out-dir $(OUT_DIR) $(WASM_DIR) $(RELEASE) $(WEAK_REFS)

# Fast, unoptimized WASM builds (debug info, no optimization) for local dev iteration.
.PHONY: build-wasm-dev
build-wasm-dev:
	wasm-pack build $(TARGET) --out-name lora-inspector --out-dir $(OUT_DIR) $(WASM_DIR) $(DEV) $(WEAK_REFS)

.PHONY: build-wasm-simd-dev
build-wasm-simd-dev:
	$(SIMD) wasm-pack build $(TARGET) --out-name lora-inspector-simd --out-dir $(OUT_DIR) $(WASM_DIR) $(DEV) $(WEAK_REFS)

.PHONY: build-frontend
build-frontend:
	(cd $(WASM_DIR) && yarn build)

.PHONY: build-frontend
preview:
	(cd $(WASM_DIR) && yarn preview)

build:
	 make build-wasm && make build-wasm-simd && make build-frontend

# Start a local dev server against fast/unoptimized WASM builds (default local loop).
.PHONY: dev-wasm
dev-wasm:
	make build-wasm-dev && \
	make build-wasm-simd-dev && \
		(cd $(WASM_DIR) && \
		yarn vite)

# Start a local dev server against the real optimized/release WASM build, to
# reproduce perf-sensitive behavior (e.g. rank/effective-scale computation)
# as it actually ships. Slower to (re)build than dev-wasm.
.PHONY: dev-wasm-release
dev-wasm-release:
	make build-wasm && \
	make build-wasm-simd && \
		(cd $(WASM_DIR) && \
		yarn vite)

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

.PHONY: 
wasm-bindgen-test: 
	wasm-pack test --headless --firefox crates/lora-inspector-wasm

# Clean build artifacts (optional)
.PHONY: clean
clean:
	rm -rf $(OUT_DIR)/*
