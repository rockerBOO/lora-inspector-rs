//! End-to-end tests for the `inspect` command, run against the real binary so
//! the CLI surface (`--json` error contract, load rules, progress routing) is
//! exercised rather than just the shared library modules.

// The shared fixture module is compiled once per test binary, so helpers that
// another test file uses (but this one does not) would warn as dead code here.
#![allow(dead_code)]

mod support;

use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_lora-inspector");

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("run lora-inspector")
}

/// Run `inspect <fixture> --json` with optional extra flags, asserting success
/// and returning the parsed document.
fn run_json(fixture: &support::Fixture, extra: &[&str]) -> serde_json::Value {
    let path = fixture.path().to_str().expect("utf-8 temp path");
    let mut args = vec!["inspect", path, "--json"];
    args.extend_from_slice(extra);
    let output = run(&args);
    assert!(
        output.status.success(),
        "inspect --json failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    serde_json::from_str(stdout.trim()).expect("stdout parses as a single JSON document")
}

#[test]
fn json_is_a_single_object_and_independent_of_width() {
    let fixture = support::Fixture::write();
    let narrow = run_json(&fixture, &["--width", "40"]);
    let wide = run_json(&fixture, &["--width", "200"]);

    assert!(narrow.is_object(), "--json must emit a single object");
    assert_eq!(narrow, wide, "--json must not depend on --width");
    assert_eq!(narrow["load"], "full");
    assert!(narrow["ungrouped"].is_object());
    assert!(narrow["skipped"].is_object());
}

#[test]
fn json_rank_is_null_unless_requested() {
    let fixture = support::Fixture::write();

    let without = run_json(&fixture, &[]);
    assert!(without["rank"].is_null());

    let with = run_json(&fixture, &["--rank"]);
    assert!(
        with["rank"].is_array(),
        "rank is a flat array when computed"
    );
}

#[test]
fn no_blocks_skips_the_block_payload() {
    let fixture = support::Fixture::write();
    let value = run_json(&fixture, &["--no-blocks"]);
    assert!(value["blocks"].is_null());
    assert_eq!(value["load"], "header_only");
    assert!(
        value["ungrouped"].is_null(),
        "ungrouped is null when the block pass did not run"
    );
    assert!(
        value["skipped"].is_null(),
        "skipped is null when the block pass did not run"
    );
}

#[test]
fn a_file_with_nothing_groupable_gives_an_empty_array_not_null() {
    let fixture = support::Fixture::write_ungroupable();
    let value = run_json(&fixture, &[]);

    assert_eq!(value["blocks"], serde_json::json!([]));
    assert!(
        value["ungrouped"]["layers"].as_u64().expect("layer count") > 0,
        "the ungroupable layer must be counted"
    );
}

#[test]
fn progress_never_lands_on_stdout() {
    let fixture = support::Fixture::write();
    let path = fixture.path().to_str().expect("utf-8 temp path");
    let output = run(&["inspect", path, "--json", "--rank"]);

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    let value: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("stdout must be JSON only");
    assert!(value.is_object());

    for glyph in ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏', '│', '█']
    {
        assert!(
            !stdout.contains(glyph),
            "stdout must not contain the progress glyph {glyph:?}"
        );
    }
}

#[test]
fn block_weights_alias_still_works() {
    let fixture = support::Fixture::write();
    let path = fixture.path().to_str().expect("utf-8 temp path");
    let output = run(&["block-weights", "--file", path, "--output-format", "text"]);

    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("utf-8 stdout");
    assert!(
        stdout.contains("block weight"),
        "block-weights text output must render the block section: {stdout}"
    );
}

#[test]
fn narrow_width_does_not_panic() {
    let fixture = support::Fixture::write();
    let path = fixture.path().to_str().expect("utf-8 temp path");

    for width in ["1", "0"] {
        let output = run(&["inspect", path, "--width", width]);
        assert!(
            output.status.success(),
            "--width {width} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("panicked"),
            "--width {width} panicked: {stderr}"
        );
    }
}

#[test]
fn metadata_is_null_unless_requested() {
    let fixture = support::Fixture::write_metadata();

    let without = run_json(&fixture, &[]);
    assert!(without["metadata"].is_null());

    let with = run_json(&fixture, &["--metadata"]);
    assert!(with["metadata"].is_object(), "metadata is opt-in");
}

#[test]
fn schema_version_is_one() {
    let fixture = support::Fixture::write();
    let value = run_json(&fixture, &[]);
    assert_eq!(value["schema_version"], 1);
}

#[test]
fn a_missing_file_is_one_json_error_object_with_exit_one() {
    let missing = std::env::temp_dir().join("lora-inspector-missing-file-never-exists.safetensors");
    let output = run(&["inspect", missing.to_str().expect("utf-8 path"), "--json"]);

    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stdout.is_empty(),
        "nothing may reach stdout on failure"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf-8 stderr");
    let value: serde_json::Value =
        serde_json::from_str(stderr.trim()).expect("stderr is one JSON error object");
    assert_eq!(value["error"]["kind"], "io");
    assert!(value["error"]["message"].is_string());
}

#[test]
fn a_bad_file_is_an_error_not_a_partial_report() {
    let fixture = support::Fixture::write();
    let mut truncated = fixture.bytes();
    // Drop the last four payload bytes so the buffer no longer reaches the
    // declared end offset: a truncation, not a missing header.
    truncated.truncate(truncated.len() - 4);
    let path = std::env::temp_dir().join(format!(
        "lora-inspector-truncated-{}.safetensors",
        std::process::id()
    ));
    std::fs::write(&path, truncated).expect("write truncated fixture");

    let output = run(&["inspect", path.to_str().expect("utf-8 path"), "--json"]);
    std::fs::remove_file(&path).ok();

    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stdout.is_empty(),
        "a partial report must never reach stdout"
    );
    let stderr = String::from_utf8(output.stderr).expect("utf-8 stderr");
    let value: serde_json::Value =
        serde_json::from_str(stderr.trim()).expect("stderr is one JSON error object");
    assert_eq!(value["error"]["kind"], "format");
}
