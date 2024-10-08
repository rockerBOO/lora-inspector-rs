FROM pierrezemb/gostatic
COPY crates/lora-inspector-wasm/dist /srv/http
COPY crates/lora-inspector-wasm/pkg /srv/http/pkg
CMD ["-port","8080","-https-promote", "-enable-logging"]
