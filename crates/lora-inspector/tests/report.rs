mod support;

use candle_core::Device;
use inspector::blocks::{BlockFamily, BlockGroup};
use inspector::file::LoRAFile;
use inspector::metadata::Metadata;
use lora_inspector::report::{build_report, round_sig, Report, ReportOptions};

fn opts(blocks: bool) -> ReportOptions {
    ReportOptions {
        blocks,
        rank: false,
        top: 8,
        jobs: 1,
        slow: false,
    }
}

fn build(blocks: bool) -> (Report, support::Fixture) {
    let fixture = support::Fixture::write();
    let bytes = fixture.bytes();
    let file = LoRAFile::new_from_buffer(
        &bytes,
        fixture.path().to_str().expect("utf-8 temp path"),
        &Device::Cpu,
    );
    let meta = Metadata::new_from_buffer(&bytes).expect("fixture metadata");
    let report = build_report(fixture.path(), &file, &meta, &opts(blocks)).expect("build report");
    (report, fixture)
}

#[test]
fn block_means_average_the_layers_in_each_block() {
    let (report, _fixture) = build(true);
    let blocks = report.blocks.expect("blocks computed");

    let in00 = blocks.iter().find(|b| b.label == "IN00").expect("IN00 row");
    let te00 = blocks.iter().find(|b| b.label == "TE00").expect("TE00 row");

    assert_eq!(in00.layers, 2);
    assert_eq!(in00.mean, 6.0);
    assert_eq!(in00.max, 10.0);
    assert_eq!(in00.median, 6.0);

    assert_eq!(te00.layers, 2);
    assert_eq!(te00.mean, 2.0);
    assert_eq!(te00.max, 3.0);
    assert_eq!(te00.median, 2.0);
}

#[test]
fn layers_without_low_rank_factors_are_skipped_not_materialized() {
    let (report, fixture) = build(true);
    let file = LoRAFile::new_from_buffer(
        &fixture.bytes(),
        fixture.path().to_str().expect("utf-8 temp path"),
        &Device::Cpu,
    );

    assert_eq!(report.skipped.layers, 1);
    assert!(report.skipped.reason.is_some());

    // The cheap Gram path declines the hada layer rather than materializing it.
    assert_eq!(
        file.low_rank_effective_scale("down_blocks.1.attn.proj_in")
            .expect("cheap path"),
        None
    );

    let blocks = report.blocks.expect("blocks computed");
    assert!(
        blocks
            .iter()
            .all(|b| !(b.group == BlockGroup::Down && b.index == Some(1))),
        "no block row should carry the skipped hada layer's value"
    );
}

#[test]
fn ungrouped_layers_are_counted_with_their_prefix() {
    let (report, _fixture) = build(true);
    assert_eq!(report.ungrouped.layers, 1);
    assert_eq!(
        report.ungrouped.prefixes,
        vec!["txtfusion_projector".to_string()]
    );
}

#[test]
fn multiple_ranks_and_alphas_are_all_reported() {
    let (report, _fixture) = build(true);
    assert_eq!(report.network.rank, vec![4, 8]);
    assert_eq!(report.network.alpha, vec![4.0, 8.0]);
}

#[test]
fn layer_counts_come_from_block_families_not_name_substrings() {
    let (report, fixture) = build(true);
    let file = LoRAFile::new_from_buffer(
        &fixture.bytes(),
        fixture.path().to_str().expect("utf-8 temp path"),
        &Device::Cpu,
    );

    // The substring helpers see none of these keys (there is no `lora_unet` or
    // `lora_te` prefix), so they are empty even though family-based counting is
    // not. The hada layer and the ungroupable key are part of the walk, so the
    // totals are 6/3/2, not the 4/2/2 a plain four-layer fixture would give.
    assert!(file.unet_keys().is_empty());
    assert!(file.text_encoder_keys().is_empty());

    assert_eq!(report.network.layers.total, 6);
    assert_eq!(report.network.layers.unet, 3);
    assert_eq!(report.network.layers.text_encoder, 2);
}

#[test]
fn every_float_is_rounded_to_six_significant_digits() {
    let (report, _fixture) = build(true);
    let json = serde_json::to_string_pretty(&report).expect("serialize report");
    let value: serde_json::Value = serde_json::from_str(&json).expect("parse report");

    let mut floats = Vec::new();
    collect_floats(&value, &mut floats);
    assert!(!floats.is_empty());

    for float in floats {
        assert_eq!(
            round_sig(float, 6),
            float,
            "{float} is not rounded to six significant digits"
        );
    }
}

#[test]
fn rows_repeat_group_and_family_so_no_join_is_needed() {
    let (report, _fixture) = build(true);
    let blocks = report.blocks.expect("blocks computed");
    assert_eq!(blocks.len(), 2);

    for block in &blocks {
        assert!(!block.label.is_empty());
        match block.group {
            BlockGroup::Down => assert_eq!(block.family, BlockFamily::Unet),
            BlockGroup::TextEncoder => assert_eq!(block.family, BlockFamily::TextEncoder),
            other => panic!("unexpected block group {other:?}"),
        }
    }

    // The flat rows carry their own group/family/label; display order is
    // group rank first, so Down (IN00) precedes TextEncoder (TE00).
    assert_eq!(blocks[0].label, "IN00");
    assert_eq!(blocks[1].label, "TE00");
}

#[test]
fn round_sig_removes_float_noise() {
    assert_eq!(round_sig(0.4000000000000001, 6), 0.4);
    assert_eq!(round_sig(1.0 / 3.0, 6), 0.333333);
    assert_eq!(round_sig(0.0, 6), 0.0);
}

fn collect_floats(value: &serde_json::Value, out: &mut Vec<f64>) {
    match value {
        serde_json::Value::Number(number) if number.is_f64() => {
            out.push(number.as_f64().expect("float number"));
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_floats(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                collect_floats(value, out);
            }
        }
        _ => {}
    }
}
