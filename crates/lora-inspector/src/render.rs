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

    let module = &report.network.module;
    let ty = report
        .network
        .network_type
        .as_ref()
        .map(network_type_label)
        .unwrap_or("-");
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
    let precision = report.file.precision.as_deref().unwrap_or("-");
    let counts = format!(
        "{} layers ({} unet, {} te)",
        report.network.layers.total, report.network.layers.unet, report.network.layers.text_encoder
    );
    let sep = sep(term);
    lines.push(fit(
        term,
        &format!("{module} {sep} {ty} {sep} {rank} {sep} {alpha} {sep} {precision} {sep} {counts}"),
    ));

    if let Some(modelspec) = modelspec_line(meta, term) {
        lines.push(fit(term, &modelspec));
    }

    lines.join("\n")
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
        lines.push(fit(term, &summary_line(summary, blocks.len(), true, term)));
    }

    // Blank line between the summary and the value table.
    lines.push(String::new());

    let mut rows: Vec<&BlockStat> = blocks.iter().collect();
    rows.sort_by(by_mean_desc);
    let peak_mean = rows.first().map(|row| row.mean).unwrap_or(0.0);
    let bar_width = term
        .width
        .saturating_sub(2 + WIDE_LABEL_W + 2 + WIDE_MEAN_W + 2 + LAYERS_W + 2);

    for row in rows.into_iter().take(opts.top) {
        let frac = if peak_mean > 0.0 {
            row.mean / peak_mean
        } else {
            0.0
        };
        let label = term.pad_right(&row.label, WIDE_LABEL_W);
        let bar = term.bar(frac, bar_width);
        let mean = pad_left(term, &fmt_f64(row.mean), WIDE_MEAN_W);
        let layers = pad_left(term, &format!("{} layers", row.layers), LAYERS_W);
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
        lines.push(fit(term, &summary_line(summary, blocks.len(), false, term)));
    }

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
        let block_count = format!("{} blocks", ordered.len());
        let spark_width = term
            .width
            .saturating_sub(2 + COMPACT_FAMILY_W + 2 + 2 + term.text_width(&block_count));
        let spark = term.sparkline(&values, spark_width);
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
    let mean = pad_left(term, &fmt_f64(row.mean), WIDE_MEAN_W);
    let layers = pad_left(term, &format!("{} layers", row.layers), LAYERS_W);
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
            "{} {} blocks  med {}",
            family_label(family),
            members.len(),
            fmt_f64(median)
        );
        if let Some(summary) = summary {
            if members.iter().any(|row| row.label == summary.peak.label) {
                line.push_str(&format!(
                    "  peak {}{}{}",
                    fmt_f64(summary.peak.value),
                    times(term),
                    fmt_f64(summary.peak.ratio_to_median)
                ));
            }
        }
        lines.push(fit(term, &line));
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
/// own line above, so `wide` omits the prefix.
fn summary_line(
    summary: &BlockWeightSummary,
    block_count: usize,
    wide: bool,
    term: &Terminal,
) -> String {
    let peak = &summary.peak;
    let median = fmt_f64(summary.median);
    let value = fmt_f64(peak.value);
    let ratio = fmt_f64(peak.ratio_to_median);
    let times = times(term);
    if wide {
        format!(
            "{block_count} blocks  median {median}  peak {} {value} ({ratio}{times} median)",
            peak.label
        )
    } else {
        format!(
            "block weight  median {median}  peak {} {value} ({ratio}{times})",
            peak.label
        )
    }
}

fn footer_lines(report: &Report, term: &Terminal) -> Vec<String> {
    let mut lines = Vec::new();

    if report.ungrouped.layers > 0 {
        let mut text = format!("{} layers ungrouped", report.ungrouped.layers);
        if !report.ungrouped.prefixes.is_empty() {
            text.push_str(&format!(" ({})", report.ungrouped.prefixes.join(", ")));
        }
        lines.push(fit(term, &text));
    }

    if report.skipped.layers > 0 {
        let reason = report
            .skipped
            .reason
            .as_deref()
            .unwrap_or(SKIPPED_FALLBACK_REASON);
        let dash = if term.unicode { "—" } else { "-" };
        lines.push(fit(
            term,
            &format!(
                "{} layers skipped: {reason} {dash} pass --slow to include",
                report.skipped.layers
            ),
        ));
    }

    lines
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
    let mut parts: Vec<&str> = Vec::new();
    if let Some(architecture) = meta_value(meta, "modelspec.architecture") {
        parts.push(architecture);
    }
    if let Some(trigger) = meta_value(meta, "modelspec.trigger_phrase") {
        parts.push(trigger);
    }
    if parts.is_empty() {
        return None;
    }
    Some(parts.join(&format!(" {} ", sep(term))))
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
