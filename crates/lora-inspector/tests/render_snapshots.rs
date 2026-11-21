// The shared fixture module is compiled once per test binary, so helpers that
// another test file uses (but this one does not) would warn as dead code here.
#![allow(dead_code)]

mod support;

use std::collections::BTreeMap;

use candle_core::Device;
use inspector::blocks::{BlockFamily, BlockGroup};
use inspector::file::LoRAFile;
use inspector::metadata::Metadata;
use inspector::svd::RankHealth;
use lora_inspector::render::{render, render_header};
use lora_inspector::report::{build_report, Peak, RankLayer, RankSummary, Report, ReportOptions};
use lora_inspector::terminal::Terminal;

fn opts() -> ReportOptions {
    ReportOptions {
        blocks: true,
        rank: false,
        top: 8,
        jobs: 1,
        slow: false,
        metadata: false,
    }
}

fn terminal(width: usize, ascii: bool) -> Terminal {
    Terminal {
        width,
        color: false,
        unicode: !ascii,
    }
}

/// Build the synthetic report with a deterministic filename, so the snapshots
/// do not leak the per-process temp-file PID into the header line.
fn build() -> (Report, Metadata, support::Fixture) {
    let fixture = support::Fixture::write();
    let bytes = fixture.bytes();
    let file = LoRAFile::new_from_buffer(
        &bytes,
        fixture.path().to_str().expect("utf-8 temp path"),
        &Device::Cpu,
    );
    let meta = Metadata::new_from_buffer(&bytes).expect("fixture metadata");
    let mut report = build_report(fixture.path(), &file, &meta, &opts()).expect("build report");
    report.file.path = "fixture.safetensors".to_string();
    (report, meta, fixture)
}

macro_rules! snapshot {
    ($name:ident, $width:expr, $ascii:expr) => {
        #[test]
        fn $name() {
            let (report, meta, _fixture) = build();
            insta::assert_snapshot!(render(&report, &meta, &terminal($width, $ascii), &opts()));
        }
    };
}

snapshot!(render_width_40, 40, false);
snapshot!(render_width_60, 60, false);
snapshot!(render_width_80, 80, false);
snapshot!(render_width_80_ascii, 80, true);
snapshot!(render_width_120, 120, false);
snapshot!(render_width_200, 200, false);

#[test]
fn wide_tier_sorts_rows_by_value_not_by_group() {
    let (report, meta, _fixture) = build();
    let out = render(&report, &meta, &terminal(120, false), &opts());

    // Display order is IN00, IN02, TE00; the wide table re-sorts by mean, so
    // the heaviest block (IN00) must come before the TE row, which precedes
    // the lightest (IN02).
    let in00 = out.find("IN00").expect("IN00 row");
    let te00 = out.find("TE00").expect("TE00 row");
    let in02 = out.find("IN02").expect("IN02 row");
    assert!(in00 < te00, "wide table must sort IN00 before TE00");
    assert!(te00 < in02, "wide table must sort TE00 before IN02");
}

#[test]
fn compact_tier_keeps_family_order() {
    let (mut report, meta, _fixture) = build();

    // Make the text-encoder row heavier than every UNet row, so only a
    // family-based ordering keeps UNet first.
    let blocks = report.blocks.as_mut().expect("blocks computed");
    for block in blocks.iter_mut() {
        if block.family == BlockFamily::Unet {
            block.mean = 0.1;
            block.max = 0.1;
        } else {
            block.mean = 9.0;
            block.max = 9.0;
        }
    }
    let summary = report.summary.block_weight.as_mut().expect("block summary");
    summary.peak = Peak {
        label: "TE00".to_string(),
        value: 9.0,
        ratio_to_median: 9.0,
    };

    let out = render(&report, &meta, &terminal(80, false), &opts());
    let unet = out.find("UNet").expect("UNet sparkline row");
    let te = out.find("TextEncoder").expect("TextEncoder sparkline row");
    assert!(unet < te, "compact tier must keep UNet before TextEncoder");
}

#[test]
fn empty_block_section_prints_a_message() {
    let (mut report, meta, _fixture) = build();
    report.blocks = Some(vec![]);

    let out = render(&report, &meta, &terminal(80, false), &opts());
    assert!(out.contains("no groupable layers"));
    assert!(
        !out.contains("block weight"),
        "no table title for empty blocks"
    );
}

#[test]
fn all_widths_have_identical_json() {
    let (report, meta, _fixture) = build();

    // Rendering is a pure function of `&Report`; the terminal width lives in
    // the `Terminal`, never in the report, so JSON must not vary with width.
    let _ = render(&report, &meta, &terminal(40, false), &opts());
    let _ = render(&report, &meta, &terminal(200, false), &opts());

    let narrow = serde_json::to_string_pretty(&report).expect("serialize report");
    let wide = serde_json::to_string_pretty(&report).expect("serialize report");
    assert_eq!(narrow, wide);
    assert!(
        !narrow.contains("\"width\""),
        "JSON must not carry a width field"
    );
}

#[test]
fn header_prints_every_rank_and_alpha_when_there_are_several() {
    let (report, meta, _fixture) = build();
    // The synthetic fixture declares rank 4 and 8, alpha 4.0 and 8.0.
    assert_eq!(report.network.rank, vec![4, 8]);
    assert_eq!(report.network.alpha, vec![4.0, 8.0]);

    let header = render_header(&report, &meta, &terminal(200, false));
    assert!(
        header.contains("rank 4, 8"),
        "header must print the labelled rank list: {header}"
    );
    assert!(
        header.contains("alpha 4.0, 8.0"),
        "header must print the labelled alpha list: {header}"
    );
}

#[test]
fn skipped_layers_are_reported_with_the_slow_hint() {
    let (mut report, meta, _fixture) = build();
    report.skipped.as_mut().expect("skipped computed").layers = 3;

    let out = render(&report, &meta, &terminal(80, false), &opts());
    assert!(out.contains("3 layers have no low-rank factor"));
    assert!(out.contains("--slow"));
}

/// Build a report with a populated rank section: four rows, one per health
/// class, with long base names and noisy float values so the renderer's
/// rounding and width degradation are actually exercised.
fn build_rank_report() -> (Report, Metadata, support::Fixture) {
    let (mut report, meta, fixture) = build();

    report.rank = Some(vec![
        RankLayer {
            base_name: "lora_unet_down_blocks_3_attentions_0_transformer_blocks_0_ff_net_0_proj"
                .to_string(),
            group: BlockGroup::Down,
            label: "IN03".to_string(),
            index: Some(3),
            nominal_rank: 64,
            effective_rank: 51.234_567,
            balance: 0.801_234,
            top1_energy: 0.112_345,
            dominance: Some(2.345_678),
            health: RankHealth::Good,
            eff_scale: Some(1.02),
            outlier: false,
        },
        RankLayer {
            base_name: "lora_te_text_model_encoder_layers_7_self_attn_o_proj".to_string(),
            group: BlockGroup::TextEncoder,
            label: "TE07".to_string(),
            index: Some(7),
            nominal_rank: 16,
            effective_rank: 9.876_543,
            balance: 0.617_283,
            top1_energy: 0.456_789,
            dominance: Some(5.432_109),
            health: RankHealth::Ok,
            eff_scale: Some(0.9),
            outlier: false,
        },
        RankLayer {
            base_name: "lora_unet_up_blocks_1_attentions_1_proj_out".to_string(),
            group: BlockGroup::Up,
            label: "OUT11".to_string(),
            index: Some(1),
            nominal_rank: 8,
            effective_rank: 1.987_654,
            balance: 0.248_765,
            top1_energy: 0.654_321,
            dominance: None,
            health: RankHealth::Weak,
            eff_scale: Some(3.1),
            outlier: true,
        },
        RankLayer {
            base_name: "lora_unet_mid_block_attentions_0_proj_in".to_string(),
            group: BlockGroup::Mid,
            label: "MID".to_string(),
            index: None,
            nominal_rank: 4,
            effective_rank: 1.001_234,
            balance: 0.123_456,
            top1_energy: 0.987_654,
            dominance: None,
            health: RankHealth::Poor,
            eff_scale: Some(0.1),
            outlier: false,
        },
        RankLayer {
            base_name: "lora_unet_double_blocks_2_img_attn_proj".to_string(),
            group: BlockGroup::Double,
            label: "DB02".to_string(),
            index: Some(2),
            nominal_rank: 2,
            effective_rank: 1.0,
            balance: 0.0,
            top1_energy: 1.0,
            dominance: None,
            health: RankHealth::Collapsed,
            eff_scale: Some(0.05),
            outlier: false,
        },
    ]);

    report.summary.rank = Some(RankSummary {
        layers: 5,
        by_health: BTreeMap::from([
            ("Good".to_string(), 1),
            ("Ok".to_string(), 1),
            ("Weak".to_string(), 1),
            ("Poor".to_string(), 1),
            ("Collapsed".to_string(), 1),
        ]),
        median_balance: 0.5,
        weak_or_worse: 2,
        elapsed_ms: 2400,
    });

    (report, meta, fixture)
}

#[test]
fn rank_rows_stay_within_width_and_keep_health_words() {
    let (mut report, meta, _fixture) = build_rank_report();

    let rows = [
        ("IN03", "Good"),
        ("TE07", "Ok"),
        ("OUT11", "Weak"),
        ("MID", "Poor"),
        ("DB02", "Collapsed"),
    ];

    for width in [40, 60, 120] {
        let term = terminal(width, false);
        let out = render(&report, &meta, &term, &opts());

        for line in out.lines() {
            let used = term.text_width(line);
            assert!(used <= width, "width {width}: {used} columns in {line:?}");
        }

        for (label, health) in rows {
            assert!(
                out.lines()
                    .any(|line| line.contains(label) && line.trim_end().ends_with(health)),
                "width {width}: row {label} must end with health word {health}"
            );
        }
    }

    // The 20-column floor: render without the block section so the rank rows
    // are isolated, and assert the widest health word still survives.
    report.blocks = None;
    let term = terminal(20, false);
    let out = render(&report, &meta, &term, &opts());
    for line in out.lines() {
        let used = term.text_width(line);
        assert!(used <= 20, "20-column floor: {used} columns in {line:?}");
    }
    assert!(
        out.lines()
            .any(|line| line.contains("DB02") && line.trim_end().ends_with("Collapsed")),
        "20-column floor must keep the Collapsed row's health word"
    );
}
