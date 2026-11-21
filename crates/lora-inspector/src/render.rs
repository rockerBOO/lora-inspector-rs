//! Width-adaptive terminal rendering of the [`Report`].
//!
//! Three tiers pick the layout from the detected width: wide (>= 100) is one
//! flat table of the top blocks sorted by value, compact (60..=99) shows one
//! sparkline per display family plus a top-3 table, and minimal (< 60) shows
//! one terse line per family with no bars. The `--json` output keeps every
//! float at six significant digits via [`crate::report::round_sig`]; the
//! human view re-formats block-weight values to two decimals for alignment
//! (`6.00`, `0.30`, `<0.01`), which is presentation only.

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

/// Fixed width of a rank row's numeric value column (two decimals).
const RANK_VALUE_W: usize = 6;

/// Minimum columns a rank row's base name must be given before the next
/// numeric column is dropped instead.
const MIN_BASE_NAME: usize = 8;

/// Family display order for the compact and minimal tiers. `BlockFamily::Other`
/// is deliberately absent: `build_blocks` routes `BlockGroup::Other` layers into
/// `ungrouped`, never into `blocks`, so no block row can carry that family.
const FAMILY_ORDER: [BlockFamily; 2] = [BlockFamily::Unet, BlockFamily::TextEncoder];

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
        term.pad_right(&term.truncate_middle(&filename, filename_max), filename_max),
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
        let mean = term.pad_left(&fmt_value(row.mean), WIDE_MEAN_W);
        let layers = term.pad_left(&count_noun(row.layers, "layer", "layers"), LAYERS_W);
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
            term.pad_left(&fmt_value(values[0]), spark_width)
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
    let mean = term.pad_left(&fmt_value(row.mean), WIDE_MEAN_W);
    let layers = term.pad_left(&count_noun(row.layers, "layer", "layers"), LAYERS_W);
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
    let layer_count = count_noun(layers, "layer", "layers");
    let elapsed_s = summary.map(|s| s.elapsed_ms).unwrap_or(0) as f64 / 1000.0;
    let threads = if opts.jobs == 0 {
        "auto threads".to_string()
    } else {
        count_noun(opts.jobs, "thread", "threads")
    };

    // Prose degrades by dropping whole fields (elapsed/jobs, then the layer
    // count, then the rank list), never by middle-ellipsizing. The bare title
    // is the width-bounded floor.
    let title_candidates = [
        format!("effective rank   r={nominal}, {layer_count}, {elapsed_s:.1} s ({threads})"),
        format!("effective rank   r={nominal}, {layer_count}"),
        format!("effective rank   r={nominal}"),
        "effective rank".to_string(),
    ];
    lines.push(pick_first_fit(term, &title_candidates));

    if let Some(summary) = summary {
        let health = rank_summary_line(summary);
        if term.text_width(&health) <= term.width {
            lines.push(health);
        }
    }

    lines.push(String::new());

    for row in rank {
        lines.push(rank_row(term, row));
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
        fmt_2dp(summary.median_balance),
        count_noun(summary.weak_or_worse, "layer", "layers")
    )
}

/// One rank row. The health word is never dropped or truncated; numeric
/// columns degrade by dropping whole columns in the order `dominance`, `top1`,
/// `balance`, `effective_rank`, then `nominal_rank`, leaving `label`,
/// `base_name` and the health word.
fn rank_row(term: &Terminal, row: &RankLayer) -> String {
    let label = term.pad_right(&row.label, WIDE_LABEL_W);
    let health = term.paint(health_style(&row.health), health_label(&row.health));

    // Display order; the first element of each pair is the drop order (1 =
    // dropped first, larger = kept longer).
    let numeric: [(usize, String); 5] = [
        (5, format!("r{}", row.nominal_rank)),
        (4, numeric_column(term, "eff", &fmt_2dp(row.effective_rank))),
        (3, numeric_column(term, "bal", &fmt_2dp(row.balance))),
        (2, numeric_column(term, "top1", &fmt_2dp(row.top1_energy))),
        (
            1,
            numeric_column(
                term,
                "dom",
                &row.dominance
                    .map(fmt_2dp)
                    .unwrap_or_else(|| "-".to_string()),
            ),
        ),
    ];

    for level in 0..=5 {
        let kept: Vec<&str> = numeric
            .iter()
            .filter(|(drop_order, _)| *drop_order > level)
            .map(|(_, text)| text.as_str())
            .collect();
        if let Some(line) =
            try_rank_row(term, &label, &health, &kept, &row.base_name, MIN_BASE_NAME)
        {
            return line;
        }
    }

    // Pathological width: keep label + base_name + health even if the base
    // name has to collapse below the usual minimum.
    try_rank_row(term, &label, &health, &[], &row.base_name, 0)
        .unwrap_or_else(|| format!("  {label}  {health}"))
}

/// Build `  label  [numeric]  base_name  health` with the base name truncated
/// and padded to fill the remaining width. Returns `None` when the fixed
/// columns alone exceed the width or the base name would fall below `min_base`.
fn try_rank_row(
    term: &Terminal,
    label: &str,
    health: &str,
    numeric: &[&str],
    base_name: &str,
    min_base: usize,
) -> Option<String> {
    let leading = format!("  {label}  ");
    let numeric_block = if numeric.is_empty() {
        String::new()
    } else {
        format!("{}  ", numeric.join("  "))
    };
    let trailing = format!("  {health}");

    let fixed =
        term.text_width(&leading) + term.text_width(&numeric_block) + term.text_width(&trailing);
    if fixed > term.width {
        return None;
    }
    let base_width = term.width - fixed;
    if base_width < min_base {
        return None;
    }

    let base_cell = term.pad_right(&term.truncate_middle(base_name, base_width), base_width);
    Some(format!("{leading}{numeric_block}{base_cell}{trailing}"))
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

/// Pick the first candidate (longest first) that fits, falling back to the
/// last (shortest) one. Used where prose degrades by dropping whole fields.
fn pick_first_fit(term: &Terminal, candidates: &[String]) -> String {
    candidates
        .iter()
        .find(|candidate| term.text_width(candidate) <= term.width)
        .cloned()
        .unwrap_or_else(|| candidates.last().cloned().unwrap_or_default())
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
        // Unreachable in the compact/minimal tiers: `build_blocks` routes
        // `BlockGroup::Other` into `ungrouped`, never into `blocks`.
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

/// Format a rank-metric value to a fixed two decimal places.
fn fmt_2dp(value: f64) -> String {
    format!("{value:.2}")
}

/// A rank row's numeric column: a short prefix plus a right-aligned value.
fn numeric_column(term: &Terminal, prefix: &str, value: &str) -> String {
    format!("{prefix} {}", term.pad_left(value, RANK_VALUE_W))
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
