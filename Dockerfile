FROM pierrezemb/gostatic
COPY crates/lora-inspector-wasm/dist/index.html /srv/http/index.html
COPY crates/lora-inspector-wasm/pkg /srv/http/pkg
COPY crates/lora-inspector-wasm/dist/assets /srv/http/assets
CMD ["-port","8080","-https-promote", "-enable-logging"]
