//! Width-adaptive terminal rendering of the [`Report`].
//!
//! Three tiers pick the layout from the detected width: wide (>= 100) is one
//! flat table of the top blocks sorted by value, compact (60..=99) shows one
//! sparkline per display family plus a top-3 table, and minimal (< 60) shows
//! one terse line per family with no bars. Every float is already rounded to
//! six significant digits by [`crate::report::round_sig`], so formatting here
//! is plain `{}` and the human output and `--json` show the same value.

use std::path::Path;

use inspector::blocks::BlockFamily;
use inspector::metadata::Metadata;
use inspector::network::NetworkType;
use inspector::svd::RankHealth;

use crate::report::{BlockStat, BlockWeightSummary, RankLayer, RankSummary, Report, ReportOptions};
use crate::terminal::{Style, Terminal};

const WIDE_MIN: usize = 100;
const COMPACT_MIN: usize = 60;

const SKIPPED_FALLBACK_REASON: &str = "no low-rank factor (LoHA/LoKr/OFT/BOFT)";

const WIDE_LABEL_W: usize = 8;
const WIDE_MEAN_W: usize = 8;
const LAYERS_W: usize = 10;
const COMPACT_FAMILY_W: usize = 12;

/// Family display order for the compact and minimal tiers.
const FAMILY_ORDER: [BlockFamily; 3] = [
    BlockFamily::Unet,
    BlockFamily::TextEncoder,
    BlockFamily::Other,
];

/// Render the whole report as a single string with one trailing newline.
///
/// Sections are separated by one blank line. `meta` is a separate parameter so
/// the header can print `modelspec.architecture`/`modelspec.trigger_phrase`
/// even when `--metadata` is off (`Report.metadata` stays `null` then).
pub fn render(report: &Report, meta: &Metadata, term: &Terminal, opts: &ReportOptions) -> String {
    let mut sections: Vec<String> = Vec::new();

    sections.push(render_header(report, meta, term));
    if let Some(block) = render_block_section(report, term, opts) {
        sections.push(block);
    }
    if let Some(rank) = render_rank_section(report, term, opts) {
        sections.push(rank);
    }

    let mut out = sections.join("\n\n");
    out.push('\n');
    out
}

/// Render the header only: filename and size, the network summary line, and
/// the modelspec line when the metadata carries it.
pub fn render_header(report: &Report, meta: &Metadata, term: &Terminal) -> String {
    let mut lines = Vec::new();

    let filename = Path::new(&report.file.path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| report.file.path.clone());
    let size = human_size(report.file.size_bytes);
    let filename_max = term.width.saturating_sub(term.text_width(&size) + 2);
    lines.push(format!(
        "{}  {}",
        term.truncate_middle(&filename, filename_max),
        size
    ));

    lines.push(header_network_line(report, term));

    if let Some(modelspec) = modelspec_line(meta, term) {
        lines.push(modelspec);
    }

    lines.join("\n")
}

/// Line 2 of the header: the network summary. Narrow widths degrade by
/// dropping whole fields (lowest priority first) instead of ellipsizing a
/// composed sentence.
fn header_network_line(report: &Report, term: &Terminal) -> String {
    let sep = sep(term);
    let module = report.network.module.clone();
    let ty = report
        .network
        .network_type
        .as_ref()
        .map(network_type_label)
        .unwrap_or("-")
        .to_string();
    let rank = if report.network.rank.is_empty() {
        "-".to_string()
    } else {
        join_usize(&report.network.rank)
    };
    let alpha = if report.network.alpha.is_empty() {
        "-".to_string()
    } else {
        join_f64(&report.network.alpha)
    };
    let precision = report.file.precision.as_deref().unwrap_or("-").to_string();
    let layers_total = count_noun(report.network.layers.total, "layer", "layers");
    let layers_full = format!(
        "{} ({} unet, {} te)",
        layers_total, report.network.layers.unet, report.network.layers.text_encoder
    );

    // Drop precision, then the layer breakdown, then type, then module.
    let variants = [
        vec![
            module.clone(),
            ty.clone(),
            rank.clone(),
            alpha.clone(),
            precision.clone(),
            layers_full.clone(),
        ],
        vec![
            module.clone(),
            ty.clone(),
            rank.clone(),
            alpha.clone(),
            layers_full.clone(),
        ],
        vec![
            module.clone(),
            ty.clone(),
            rank.clone(),
            alpha.clone(),
            layers_total.clone(),
        ],
        vec![
            module.clone(),
            rank.clone(),
            alpha.clone(),
            layers_total.clone(),
        ],
        vec![rank.clone(), alpha.clone(), layers_total.clone()],
        vec![rank.clone(), layers_total.clone()],
        vec![layers_total.clone()],
    ];

    for fields in variants {
        let line = fields.join(&format!(" {sep} "));
        if term.text_width(&line) <= term.width {
            return line;
        }
    }

    layers_total
}

fn render_block_section(report: &Report, term: &Terminal, opts: &ReportOptions) -> Option<String> {
    let blocks = report.blocks.as_ref()?;
    let summary = report.summary.block_weight.as_ref();

    let mut lines: Vec<String> = Vec::new();

    if blocks.is_empty() {
        lines.push("no groupable layers".to_string());
    } else if term.width >= WIDE_MIN {
        lines.extend(render_wide(blocks, summary, term, opts));
    } else if term.width >= COMPACT_MIN {
        lines.extend(render_compact(blocks, summary, term));
    } else {
        lines.extend(render_minimal(blocks, summary, term));
    }

    lines.extend(footer_lines(report, term));

    Some(lines.join("\n"))
}

fn render_wide(
    blocks: &[BlockStat],
    summary: Option<&BlockWeightSummary>,
    term: &Terminal,
    opts: &ReportOptions,
) -> Vec<String> {
    let mut lines = vec!["block weight".to_string()];

    if let Some(summary) = summary {
        lines.push(summary_line(summary, blocks.len(), true, term));
    }

    // Blank line between the summary and the value table.
    lines.push(String::new());

    let mut rows: Vec<&BlockStat> = blocks.iter().collect();
    rows.sort_by(by_mean_desc);
    let peak_mean = rows.first().map(|row| row.mean).unwrap_or(0.0);
    // Keep the bar from eating the row: it is capped so the numbers stay near
    // their label instead of being pushed to the far edge.
    let bar_width = term
        .width
        .saturating_sub(2 + WIDE_LABEL_W + 2 + WIDE_MEAN_W + 2 + LAYERS_W + 2)
        .min(40);

    for row in rows.into_iter().take(opts.top) {
        let frac = if peak_mean > 0.0 {
            row.mean / peak_mean
        } else {
            0.0
        };
        let label = term.pad_right(&row.label, WIDE_LABEL_W);
        let bar = term.bar(frac, bar_width);
        let mean = pad_left(term, &fmt_value(row.mean), WIDE_MEAN_W);
        let layers = pad_left(term, &count_noun(row.layers, "layer", "layers"), LAYERS_W);
        lines.push(format!("  {label}  {bar}  {mean}  {layers}"));
    }

    lines
}

fn render_compact(
    blocks: &[BlockStat],
    summary: Option<&BlockWeightSummary>,
    term: &Terminal,
) -> Vec<String> {
    let mut lines = Vec::new();

    if let Some(summary) = summary {
        lines.push(summary_line(summary, blocks.len(), false, term));
    }

    // The sparkline column is a fixed width across families so a single-value
    // family's number lines up with the other families' sparklines.
    let spark_width = term
        .width
        .saturating_sub(2 + COMPACT_FAMILY_W + 2 + 2 + LAYERS_W);

    for (family, members) in families_in_order(blocks) {
        let mut ordered: Vec<&BlockStat> = members;
        ordered.sort_by(|a, b| {
            a.group
                .canonical_rank()
                .cmp(&b.group.canonical_rank())
                .then(a.index.cmp(&b.index))
                .then(a.label.cmp(&b.label))
        });
        let values: Vec<f64> = ordered.iter().map(|row| row.mean).collect();
        let spark = if distinct_count(&values) < 2 {
            pad_left(term, &fmt_value(values[0]), spark_width)
        } else {
            term.sparkline(&values, spark_width)
        };
        let block_count = count_noun(ordered.len(), "block", "blocks");
        lines.push(format!(
            "  {}  {}  {}",
            term.pad_right(family_label(family), COMPACT_FAMILY_W),
            spark,
            block_count
        ));
    }

    // Blank line between the sparklines and the top-3 table.
    lines.push(String::new());

    let mut rows: Vec<&BlockStat> = blocks.iter().collect();
    rows.sort_by(by_mean_desc);
    for row in rows.into_iter().take(3) {
        lines.push(compact_table_row(term, row));
    }

    lines
}

fn compact_table_row(term: &Terminal, row: &BlockStat) -> String {
    let label = term.pad_right(&row.label, WIDE_LABEL_W);
    let mean = pad_left(term, &fmt_value(row.mean), WIDE_MEAN_W);
    let layers = pad_left(term, &count_noun(row.layers, "layer", "layers"), LAYERS_W);
    format!("  {label}  {mean}  {layers}")
}

fn render_minimal(
    blocks: &[BlockStat],
    summary: Option<&BlockWeightSummary>,
    term: &Terminal,
) -> Vec<String> {
    let mut lines = Vec::new();

    for (family, members) in families_in_order(blocks) {
        let mut means: Vec<f64> = members.iter().map(|row| row.mean).collect();
        let median = median(&mut means);
        let mut line = format!(
            "{} {}  med {}",
            family_label(family),
            count_noun(members.len(), "block", "blocks"),
            fmt_value(median)
        );
        if let Some(summary) = summary {
            if members.iter().any(|row| row.label == summary.peak.label) {
                for annotation in peak_annotations(summary, term, " median") {
                    let candidate = format!("{line}  {annotation}");
                    if term.text_width(&candidate) <= term.width {
                        line = candidate;
                        break;
                    }
                }
            }
        }
        lines.push(line);
    }

    lines
}

fn render_rank_section(report: &Report, term: &Terminal, opts: &ReportOptions) -> Option<String> {
    let rank = report.rank.as_ref()?;
    let summary = report.summary.rank.as_ref();

    let mut lines = Vec::new();

    let nominal = distinct_nominal_rank(rank);
    let nominal = if nominal.is_empty() {
        "?".to_string()
    } else {
        nominal
    };
    let layers = summary.map(|s| s.layers).unwrap_or(rank.len());
    let elapsed_s = summary.map(|s| s.elapsed_ms).unwrap_or(0) as f64 / 1000.0;
    let threads = if opts.jobs == 0 {
        "auto".to_string()
    } else {
        opts.jobs.to_string()
    };
    lines.push(fit(
        term,
        &format!(
            "effective rank   r={nominal}, {layers} layers, {elapsed_s:.1} s ({threads} threads)"
        ),
    ));

    if let Some(summary) = summary {
        lines.push(fit(term, &rank_summary_line(summary)));
    }

    lines.push(String::new());

    for row in rank {
        lines.push(fit(term, &rank_row(term, row)));
    }

    Some(lines.join("\n"))
}

fn rank_summary_line(summary: &RankSummary) -> String {
    let counts = ["Good", "Ok", "Weak", "Poor", "Collapsed"]
        .iter()
        .map(|word| {
            format!(
                "{word} {}",
                summary.by_health.get(*word).copied().unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join("  ");
    format!(
        "{counts}  median balance {}  {} weak or worse",
        fmt_f64(summary.median_balance),
        summary.weak_or_worse
    )
}

fn rank_row(term: &Terminal, row: &RankLayer) -> String {
    let health = term.paint(health_style(&row.health), health_label(&row.health));
    format!(
        "  {}  r{}  eff {}  bal {}  {}",
        term.pad_right(&row.label, WIDE_LABEL_W),
        row.nominal_rank,
        fmt_f64(row.effective_rank),
        fmt_f64(row.balance),
        health
    )
}

/// Summary line carrying the headline block numbers. The compact tier folds
/// the `block weight` title into this line; the wide tier has the title on its
/// own line above, so `wide` omits the prefix. The peak annotation degrades by
/// dropping the ratio and then the label rather than being ellipsized.
fn summary_line(
    summary: &BlockWeightSummary,
    block_count: usize,
    wide: bool,
    term: &Terminal,
) -> String {
    let median = fmt_value(summary.median);
    let blocks = count_noun(block_count, "block", "blocks");
    let prefix = if wide {
        format!("{blocks}  median {median}")
    } else {
        format!("block weight  median {median}")
    };
    let suffix = if wide { " median" } else { "" };
    for annotation in peak_annotations(summary, term, suffix) {
        let line = format!("{prefix}  {annotation}");
        if term.text_width(&line) <= term.width {
            return line;
        }
    }
    prefix
}

/// Peak annotations from most to least informative: value with ratio, value
/// with label, bare value.
fn peak_annotations(
    summary: &BlockWeightSummary,
    term: &Terminal,
    ratio_suffix: &str,
) -> Vec<String> {
    let peak = &summary.peak;
    let value = fmt_value(peak.value);
    let ratio = fmt_ratio(peak.ratio_to_median);
    let times = times(term);
    vec![
        format!("peak {} {value} ({ratio}{times}{ratio_suffix})", peak.label),
        format!("peak {} {value}", peak.label),
        format!("peak {value}"),
    ]
}

fn footer_lines(report: &Report, term: &Terminal) -> Vec<String> {
    let mut lines = Vec::new();

    if report.ungrouped.layers > 0 {
        let count = count_noun(report.ungrouped.layers, "layer", "layers");
        let short = format!("{count} ungrouped");
        let full = if report.ungrouped.prefixes.is_empty() {
            short.clone()
        } else {
            format!(
                "{count} ungrouped ({})",
                report.ungrouped.prefixes.join(", ")
            )
        };
        lines.push(pick_fit(term, full, short));
    }

    if report.skipped.layers > 0 {
        let count = count_noun(report.skipped.layers, "layer", "layers");
        let reason = report
            .skipped
            .reason
            .as_deref()
            .unwrap_or(SKIPPED_FALLBACK_REASON);
        let dash = if term.unicode { "—" } else { "-" };
        let full = format!("{count} skipped: {reason} {dash} pass --slow to include");
        let short = format!("{count} skipped; --slow includes them");
        lines.push(pick_fit(term, full, short));
    }

    lines
}

/// Prefer the full footer text; fall back to the short form when it would
/// overflow. Prose is never middle-ellipsized.
fn pick_fit(term: &Terminal, full: String, short: String) -> String {
    if term.text_width(&full) <= term.width {
        full
    } else {
        short
    }
}

fn families_in_order(blocks: &[BlockStat]) -> Vec<(BlockFamily, Vec<&BlockStat>)> {
    let mut out = Vec::new();
    for family in FAMILY_ORDER {
        let members: Vec<&BlockStat> = blocks.iter().filter(|row| row.family == family).collect();
        if !members.is_empty() {
            out.push((family, members));
        }
    }
    out
}

fn distinct_nominal_rank(rank: &[RankLayer]) -> String {
    let mut values: Vec<usize> = rank.iter().map(|row| row.nominal_rank).collect();
    values.sort_unstable();
    values.dedup();
    join_usize(&values)
}

fn by_mean_desc(a: &&BlockStat, b: &&BlockStat) -> std::cmp::Ordering {
    b.mean
        .partial_cmp(&a.mean)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then(a.label.cmp(&b.label))
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = values.len();
    if n.is_multiple_of(2) {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn network_type_label(t: &NetworkType) -> &'static str {
    match t {
        NetworkType::LoRA => "LoRA",
        NetworkType::LoRAC3Lier => "LoRAC3Lier",
        NetworkType::LoRAFA => "LoRAFA",
        NetworkType::LoCon => "LoCon",
        NetworkType::LoHA => "LoHA",
        NetworkType::LoKr => "LoKr",
        NetworkType::IA3 => "IA3",
        NetworkType::DyLoRA => "DyLoRA",
        NetworkType::GLoRA => "GLoRA",
        NetworkType::GLoKr => "GLoKr",
        NetworkType::OFT => "OFT",
        NetworkType::DiagOFT => "DiagOFT",
        NetworkType::BOFT => "BOFT",
    }
}

fn family_label(family: BlockFamily) -> &'static str {
    match family {
        BlockFamily::Unet => "UNet",
        BlockFamily::TextEncoder => "TextEncoder",
        BlockFamily::Other => "Other",
    }
}

fn health_label(health: &RankHealth) -> &'static str {
    match health {
        RankHealth::Collapsed => "Collapsed",
        RankHealth::Good => "Good",
        RankHealth::Ok => "Ok",
        RankHealth::Weak => "Weak",
        RankHealth::Poor => "Poor",
    }
}

fn health_style(health: &RankHealth) -> Style {
    match health {
        RankHealth::Good => Style::Good,
        RankHealth::Ok => Style::Ok,
        RankHealth::Weak => Style::Weak,
        RankHealth::Poor => Style::Poor,
        RankHealth::Collapsed => Style::Collapsed,
    }
}

fn modelspec_line(meta: &Metadata, term: &Terminal) -> Option<String> {
    let architecture = meta_value(meta, "modelspec.architecture");
    let trigger = meta_value(meta, "modelspec.trigger_phrase");
    let sep = format!(" {} ", sep(term));

    let mut variants = Vec::new();
    if let (Some(architecture), Some(trigger)) = (architecture, trigger) {
        variants.push(format!("{architecture}{sep}{trigger}"));
    }
    if let Some(architecture) = architecture {
        variants.push(architecture.to_string());
    }
    if let Some(trigger) = trigger {
        variants.push(trigger.to_string());
    }

    variants
        .into_iter()
        .find(|line| term.text_width(line) <= term.width)
}

fn meta_value<'a>(meta: &'a Metadata, key: &str) -> Option<&'a str> {
    meta.metadata
        .as_ref()?
        .get(key)
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
}

/// The `·` field separator, or `|` in `--ascii` mode so the report emits only
/// ASCII glyphs of its own (the user's filename keeps its own characters).
fn sep(term: &Terminal) -> &'static str {
    if term.unicode {
        "·"
    } else {
        "|"
    }
}

/// The `×` multiplier glyph, or `x` in `--ascii` mode.
fn times(term: &Terminal) -> &'static str {
    if term.unicode {
        "×"
    } else {
        "x"
    }
}

/// Middle-truncate a free-text line when it would exceed the terminal width.
fn fit(term: &Terminal, s: &str) -> String {
    if term.text_width(s) > term.width {
        term.truncate_middle(s, term.width)
    } else {
        s.to_string()
    }
}

/// Right-align `s` in `width` columns; the mirror of [`Terminal::pad_right`]
/// for numeric columns (which are always single-column ASCII digits).
fn pad_left(term: &Terminal, s: &str, width: usize) -> String {
    let used = term.text_width(s);
    if used >= width {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len() + width - used);
    for _ in used..width {
        out.push(' ');
    }
    out.push_str(s);
    out
}

fn fmt_f64(value: f64) -> String {
    format!("{value}")
}

/// Format a block-weight value for display: fixed two decimal places so the
/// decimal points align across rows. A value below 0.005 reads `<0.01` rather
/// than rounding to a misleading `0.00`; a true zero stays `0.00`.
fn fmt_value(value: f64) -> String {
    if value != 0.0 && value.abs() < 0.005 {
        "<0.01".to_string()
    } else {
        format!("{value:.2}")
    }
}

/// Format a peak ratio with one decimal place.
fn fmt_ratio(value: f64) -> String {
    format!("{value:.1}")
}

/// `count` followed by the correctly pluralized noun.
fn count_noun(count: usize, singular: &str, plural: &str) -> String {
    format!("{count} {}", if count == 1 { singular } else { plural })
}

/// Number of distinct values, so a family whose blocks all share one value can
/// print the value instead of a blank sparkline.
fn distinct_count(values: &[f64]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut count = 1;
    for pair in sorted.windows(2) {
        if pair[0] != pair[1] {
            count += 1;
        }
    }
    count
}

fn join_usize(values: &[usize]) -> String {
    values
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn join_f64(values: &[f64]) -> String {
    values
        .iter()
        .copied()
        .map(fmt_f64)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Human-readable file size with one decimal place below 100 of a unit.
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];

    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }

    if unit == 0 {
        format!("{bytes} B")
    } else if value >= 100.0 {
        format!("{value:.0} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}
