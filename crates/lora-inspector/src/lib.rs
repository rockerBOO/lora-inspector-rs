//! Library target for the `lora-inspector` CLI.
//!
//! The binary (`src/main.rs`) shares these modules; the separate library target
//! exists so the integration tests under `tests/` can link the report data
//! model directly. Add each new `src/<name>.rs` module here when it becomes
//! part of the shared surface.

pub mod report;
