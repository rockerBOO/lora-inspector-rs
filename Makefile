

test:
	cargo test --workspace

build-wasm:
	wasm-pack build --target no-modules  --out-dir pkg crates/lora-inspector-wasm --release --weak-refs

build-dev-wasm:
	wasm-pack build --target no-modules --out-dir pkg crates/lora-inspector-wasm --release --weak-refs

dev-wasm:
	python -m http.server -d crates/lora-inspector-wasm
