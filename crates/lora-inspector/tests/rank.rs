// The shared fixture module is compiled once per test binary, so helpers that
// another test file uses (but this one does not) would warn as dead code here.
#![allow(dead_code)]

mod support;

use candle_core::Device;
use inspector::blocks::parse_block_key;
use inspector::file::LoRAFile;
use inspector::metadata::Metadata;
use lora_inspector::report::{build_report, Report, ReportOptions};

fn opts(blocks: bool, rank: bool, jobs: usize) -> ReportOptions {
    ReportOptions {
        blocks,
        rank,
        top: 8,
        jobs,
        slow: false,
        metadata: false,
    }
}

fn build(blocks: bool, rank: bool, jobs: usize) -> (Report, support::Fixture) {
    let fixture = support::Fixture::write();
    let bytes = fixture.bytes();
    let file = LoRAFile::new_from_buffer(
        &bytes,
        fixture.path().to_str().expect("utf-8 temp path"),
        &Device::Cpu,
    );
    let meta = Metadata::new_from_buffer(&bytes).expect("fixture metadata");
    let report = build_report(fixture.path(), &file, &meta, &opts(blocks, rank, jobs))
        .expect("build report");
    (report, fixture)
}

#[test]
fn rank_section_is_identical_for_one_and_four_jobs() {
    let (one, _one_fixture) = build(false, true, 1);
    let (four, _four_fixture) = build(false, true, 4);

    assert_eq!(
        serde_json::to_string(&one.rank).expect("serialize 1-job rank"),
        serde_json::to_string(&four.rank).expect("serialize 4-job rank"),
    );

    let one_summary = one.summary.rank.expect("1-job rank summary");
    let four_summary = four.summary.rank.expect("4-job rank summary");
    assert_eq!(one_summary.layers, four_summary.layers);
    assert_eq!(one_summary.by_health, four_summary.by_health);
    assert_eq!(one_summary.median_balance, four_summary.median_balance);
    assert_eq!(one_summary.weak_or_worse, four_summary.weak_or_worse);

    // `elapsed_ms` is measured wall time, so it is the one summary field that
    // is deliberately outside the byte-identical comparison.
}

#[test]
fn rank_layers_are_sorted_by_base_name() {
    let (report, _fixture) = build(false, true, 4);
    let rank = report.rank.expect("rank computed");

    assert!(!rank.is_empty());
    assert!(rank
        .windows(2)
        .all(|pair| pair[0].base_name.as_str() <= pair[1].base_name.as_str()));
}

#[test]
fn health_summary_counts_every_layer() {
    let (report, _fixture) = build(false, true, 1);
    let rank = report.rank.expect("rank computed");
    let summary = report.summary.rank.expect("rank summary");

    assert_eq!(summary.layers, rank.len());
    assert_eq!(
        summary.by_health.values().sum::<usize>(),
        rank.len(),
        "health counts must account for every rank row"
    );
}

#[test]
fn rank_labels_match_the_block_pass() {
    let (report, _fixture) = build(true, true, 1);
    let blocks = report.blocks.expect("blocks computed");
    let rank = report.rank.expect("rank computed");

    for row in &rank {
        let key = parse_block_key(&row.base_name);
        let expected = key.group.tag(key.block_index);
        assert_eq!(row.label, expected, "rank label for {}", row.base_name);

        // Where the block pass also produced a row (groupable, non-skipped),
        // the two labels must agree; `Other` layers are routed to `ungrouped`.
        if let Some(block) = blocks
            .iter()
            .find(|block| block.group == key.group && block.index == key.block_index)
        {
            assert_eq!(row.label, block.label, "label for {}", row.base_name);
        }
    }
}

#[test]
fn health_summary_always_has_all_five_keys() {
    let (report, _fixture) = build(false, true, 1);
    let summary = report.summary.rank.expect("rank summary");

    for word in ["Good", "Ok", "Weak", "Poor", "Collapsed"] {
        assert!(
            summary.by_health.contains_key(word),
            "missing health word {word}"
        );
    }
}

#[test]
fn rank_outlier_uses_the_layer_median_rule() {
    let (report, _fixture) = build(false, true, 1);
    let rank = report.rank.expect("rank computed");

    let proj_out = rank
        .iter()
        .find(|row| row.base_name == "down_blocks.0.attn.proj_out")
        .expect("proj_out rank row");

    assert!(proj_out.outlier);
    assert_eq!(rank.iter().filter(|row| row.outlier).count(), 1);

    // The fixture's layer scales are 0.3, 1.0, 2.0, 2.5, 3.0, 10.0, so the
    // median is 2.25 and only 10.0 exceeds 1.5 × 2.25 = 3.375.
    assert_eq!(proj_out.eff_scale, Some(10.0));
}
