//! Library target for the `lora-inspector` CLI.
//!
//! This is the single shared surface: the binary (`src/main.rs`) and the
//! integration tests under `tests/` both link these modules, so a `Report`
//! built by one is the same type everywhere. Add each new shared `src/<name>.rs`
//! module here.

pub mod render;
pub mod report;
pub mod terminal;
