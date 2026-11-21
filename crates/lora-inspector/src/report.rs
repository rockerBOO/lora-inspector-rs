//! Serde-shaped data model for the `inspect` terminal report.
//!
//! The struct field names and ordering are the `schema_version: 1` JSON
//! contract: flat arrays whose rows repeat their own group/family/label, stable
//! keys, `null` for "not computed" and `[]` for "computed, nothing found". Every
//! float passes through [`round_sig`] so the human rendering and `--json`
//! output show the same canonical value.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use inspector::blocks::{parse_block_key, BlockFamily, BlockGroup};
use inspector::file::LoRAFile;
use inspector::metadata::Metadata;
use inspector::network::NetworkType;
use inspector::svd::{RankHealth, RankMetrics};
use rayon::prelude::*;
use serde::Serialize;

pub const SCHEMA_VERSION: u32 = 1;

const SKIPPED_REASON: &str = "no 2D low-rank factor";
const UNGROUPED_PREFIX_CAP: usize = 10;
/// Near-zero floor for outlier thresholds and the peak ratio denominator.
const OUTLIER_FLOOR: f64 = 1e-10;

/// Options controlling which sections of the report are computed.
#[derive(Debug, Clone, Serialize)]
pub struct ReportOptions {
    pub blocks: bool,
    pub rank: bool,
    pub top: usize,
    pub jobs: usize,
    pub slow: bool,
    pub metadata: bool,
}

/// Which load path produced this report. The header-only path reads just
/// `8 + header_len` bytes, so payload-derived fields are empty or `null` there;
/// the marker is what lets a consumer tell "not loaded" from "not present".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Load {
    Full,
    HeaderOnly,
}

/// The top-level JSON document emitted by `--json`.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub load: Load,
    pub file: FileInfo,
    pub network: NetworkInfo,
    pub metadata: Option<HashMap<String, String>>,
    pub blocks: Option<Vec<BlockStat>>,
    pub ungrouped: Option<Ungrouped>,
    pub skipped: Option<Skipped>,
    pub rank: Option<Vec<RankLayer>>,
    pub summary: Summary,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileInfo {
    pub path: String,
    pub size_bytes: u64,
    pub precision: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NetworkInfo {
    pub module: String,
    #[serde(rename = "type")]
    pub network_type: Option<NetworkType>,
    pub family: String,
    pub rank: Vec<usize>,
    pub alpha: Vec<f64>,
    pub layers: LayerCounts,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerCounts {
    pub total: usize,
    pub unet: usize,
    pub text_encoder: usize,
}

/// One flat row per `(group, index)`; the group and family repeat on every row.
#[derive(Debug, Clone, Serialize)]
pub struct BlockStat {
    pub group: BlockGroup,
    pub family: BlockFamily,
    pub label: String,
    pub index: Option<usize>,
    pub layers: usize,
    pub mean: f64,
    pub max: f64,
    pub median: f64,
    pub outlier: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct Ungrouped {
    pub layers: usize,
    pub prefixes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Skipped {
    pub layers: usize,
    pub reason: Option<String>,
}

/// One row per layer in the rank pass. Declared here because it is part of the
/// `schema_version: 1` contract; the rank pass fills it in.
#[derive(Debug, Clone, Serialize)]
pub struct RankLayer {
    pub base_name: String,
    pub group: BlockGroup,
    pub label: String,
    pub index: Option<usize>,
    pub nominal_rank: usize,
    pub effective_rank: f64,
    pub balance: f64,
    pub top1_energy: f64,
    pub dominance: Option<f64>,
    pub health: RankHealth,
    pub eff_scale: Option<f64>,
    pub outlier: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct RankSummary {
    pub layers: usize,
    pub by_health: BTreeMap<String, usize>,
    pub median_balance: f64,
    pub weak_or_worse: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub block_weight: Option<BlockWeightSummary>,
    pub rank: Option<RankSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BlockWeightSummary {
    pub median: f64,
    pub peak: Peak,
    pub outliers: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Peak {
    pub label: String,
    pub value: f64,
    pub ratio_to_median: f64,
}

/// Round `x` to `digits` significant decimal digits.
///
/// This is the single helper every float in the report passes through, so the
/// human rendering and the `--json` output show the same canonical value and no
/// value carries `0.4000000000000001`-style float noise.
pub fn round_sig(x: f64, digits: u32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }

    let magnitude = x.abs().log10().floor() as i32;
    let factor = 10f64.powi(digits as i32 - 1 - magnitude);
    (x * factor).round() / factor
}

/// One block's raw per-layer scales before aggregation.
struct RawBlock {
    group: BlockGroup,
    index: Option<usize>,
    family: BlockFamily,
    scales: Vec<f64>,
}

/// One block's aggregated raw values before outlier flagging and rounding.
struct BlockAgg {
    group: BlockGroup,
    index: Option<usize>,
    family: BlockFamily,
    layers: usize,
    mean: f64,
    max: f64,
    median: f64,
}

/// Aggregate per-layer effective scales into one flat row per `(group, index)`.
///
/// This is a single pass over `base_names()`. The cheap Gram path
/// (`low_rank_effective_scale`) is used unless `opts.slow` is set; a layer the
/// path declines (`None`) is skipped and counted, never materialized.
pub fn build_blocks(
    file: &LoRAFile,
    opts: &ReportOptions,
) -> inspector::Result<(Vec<BlockStat>, Ungrouped, Skipped, BlockWeightSummary)> {
    let mut base_names = file.base_names();
    base_names.sort();

    let mut by_block: HashMap<(u8, Option<usize>), RawBlock> = HashMap::new();
    let mut ungrouped_prefixes: Vec<String> = Vec::new();
    let mut ungrouped_layers = 0usize;
    let mut skipped_layers = 0usize;

    for base_name in &base_names {
        let key = parse_block_key(base_name);
        let scale = if opts.slow {
            file.effective_scale(base_name)?
        } else {
            file.low_rank_effective_scale(base_name)?
        };

        let Some(scale) = scale else {
            skipped_layers += 1;
            continue;
        };

        if key.group == BlockGroup::Other {
            ungrouped_layers += 1;
            if let Some(prefix) = key.sublayer.split('.').next().filter(|p| !p.is_empty()) {
                ungrouped_prefixes.push(prefix.to_string());
            }
            continue;
        }

        let (group, index, family) = (key.group, key.block_index, key.family());
        // `canonical_rank()` is unique per `BlockGroup`, so it stands in for the
        // (unhashable) enum as the map key; the real group lives on the value.
        by_block
            .entry((group.canonical_rank(), index))
            .or_insert_with(|| RawBlock {
                group,
                index,
                family,
                scales: Vec::new(),
            })
            .scales
            .push(scale);
    }

    let mut aggs: Vec<BlockAgg> = by_block
        .into_values()
        .map(|raw| {
            let layers = raw.scales.len();
            let mean = raw.scales.iter().sum::<f64>() / layers as f64;
            let max = raw.scales.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            BlockAgg {
                group: raw.group,
                index: raw.index,
                family: raw.family,
                layers,
                mean,
                max,
                median: median(&raw.scales),
            }
        })
        .collect();

    // Display order is part of the contract: group rank, then index, then label.
    aggs.sort_by(|a, b| {
        a.group
            .canonical_rank()
            .cmp(&b.group.canonical_rank())
            .then(a.index.cmp(&b.index))
            .then(a.group.tag(a.index).cmp(&b.group.tag(b.index)))
    });

    // The block outlier rule compares block means to the median block mean,
    // never to the median of the individual layer scales. Averaging a block's
    // layers pulls every block toward the layer median, which would flag almost
    // nothing. The near-zero floor keeps the threshold meaningful when the
    // median is exactly zero.
    let block_means: Vec<f64> = aggs.iter().map(|agg| agg.mean).collect();
    let median_block_mean = median(&block_means);
    let threshold = outlier_threshold(median_block_mean);

    let mut blocks = Vec::with_capacity(aggs.len());
    let mut peak = Peak {
        label: String::new(),
        value: 0.0,
        ratio_to_median: 0.0,
    };
    let mut peak_mean = f64::NEG_INFINITY;

    for agg in &aggs {
        // The first block in display order wins ties, so the peak is stable.
        if agg.mean > peak_mean {
            peak_mean = agg.mean;
            peak = Peak {
                label: agg.group.tag(agg.index),
                value: round_sig(agg.mean, 6),
                // Same floored denominator as the outlier threshold, so a
                // flagged block never reads `ratio_to_median: 0.0`.
                ratio_to_median: round_sig(agg.mean / median_block_mean.max(OUTLIER_FLOOR), 6),
            };
        }

        blocks.push(BlockStat {
            group: agg.group,
            family: agg.family,
            label: agg.group.tag(agg.index),
            index: agg.index,
            layers: agg.layers,
            mean: round_sig(agg.mean, 6),
            max: round_sig(agg.max, 6),
            median: round_sig(agg.median, 6),
            outlier: agg.mean > threshold,
        });
    }

    ungrouped_prefixes.sort();
    ungrouped_prefixes.dedup();
    ungrouped_prefixes.truncate(UNGROUPED_PREFIX_CAP);

    let ungrouped = Ungrouped {
        layers: ungrouped_layers,
        prefixes: ungrouped_prefixes,
    };
    let skipped = Skipped {
        layers: skipped_layers,
        reason: (skipped_layers > 0).then(|| SKIPPED_REASON.to_string()),
    };
    let summary = BlockWeightSummary {
        median: round_sig(median_block_mean, 6),
        peak,
        outliers: blocks.iter().filter(|block| block.outlier).count(),
    };

    Ok((blocks, ungrouped, skipped, summary))
}

/// One layer's raw rank-pass values before outlier flagging and rounding.
struct RankRaw {
    base_name: String,
    group: BlockGroup,
    index: Option<usize>,
    metrics: RankMetrics,
    eff_scale: Option<f64>,
}

/// Parallel effective-rank pass over every base name.
///
/// Each layer's metrics are computed independently, so no cross-layer float
/// depends on scheduling. The collected rows are sorted by `base_name` before
/// the summary is derived, which is what makes the output byte-identical for
/// any thread count.
pub fn build_rank(
    file: &LoRAFile,
    jobs: usize,
) -> inspector::Result<(Vec<RankLayer>, RankSummary)> {
    let start = std::time::Instant::now();

    let mut base_names = file.base_names();
    base_names.sort();

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(rank_threads(jobs))
        .build()
        .map_err(|err| inspector::InspectorError::Msg(err.to_string()))?;

    let mut raw: Vec<RankRaw> = pool
        .install(|| {
            base_names
                .par_iter()
                .map(|base_name| -> inspector::Result<Option<RankRaw>> {
                    let Some(metrics) = file.rank_metrics(base_name)? else {
                        return Ok(None);
                    };
                    let key = parse_block_key(base_name);
                    let eff_scale = file.low_rank_effective_scale(base_name)?;
                    Ok(Some(RankRaw {
                        base_name: base_name.clone(),
                        group: key.group,
                        index: key.block_index,
                        metrics,
                        eff_scale,
                    }))
                })
                .collect::<inspector::Result<Vec<Option<RankRaw>>>>()
        })?
        .into_iter()
        .flatten()
        .collect();

    raw.sort_by(|a, b| a.base_name.cmp(&b.base_name));

    // The layer outlier rule compares each layer's scale to the median layer
    // scale (not the block median), matching `LoRAFile::effective_scales_all`.
    let scales: Vec<f64> = raw.iter().filter_map(|row| row.eff_scale).collect();
    let median_scale = median(&scales);
    let threshold = outlier_threshold(median_scale);

    let mut by_health = BTreeMap::from([
        ("Collapsed".to_string(), 0usize),
        ("Good".to_string(), 0usize),
        ("Ok".to_string(), 0usize),
        ("Weak".to_string(), 0usize),
        ("Poor".to_string(), 0usize),
    ]);
    let mut weak_or_worse = 0usize;
    let balances: Vec<f64> = raw.iter().map(|row| row.metrics.balance).collect();

    let mut layers = Vec::with_capacity(raw.len());
    for row in &raw {
        let word = rank_health_word(&row.metrics.health);
        *by_health.get_mut(word).expect("health word present") += 1;
        if matches!(
            row.metrics.health,
            RankHealth::Weak | RankHealth::Poor | RankHealth::Collapsed
        ) {
            weak_or_worse += 1;
        }

        layers.push(RankLayer {
            base_name: row.base_name.clone(),
            group: row.group,
            label: row.group.tag(row.index),
            index: row.index,
            nominal_rank: row.metrics.nominal_rank,
            effective_rank: round_sig(row.metrics.effective_rank, 6),
            balance: round_sig(row.metrics.balance, 6),
            top1_energy: round_sig(row.metrics.top1_energy, 6),
            dominance: row.metrics.dominance.map(|d| round_sig(d, 6)),
            health: row.metrics.health.clone(),
            eff_scale: row.eff_scale.map(|scale| round_sig(scale, 6)),
            outlier: row.eff_scale.is_some_and(|scale| scale > threshold),
        });
    }

    let summary = RankSummary {
        layers: layers.len(),
        by_health,
        median_balance: round_sig(median(&balances), 6),
        weak_or_worse,
        elapsed_ms: start.elapsed().as_millis(),
    };

    Ok((layers, summary))
}

/// Resolve `--jobs`: `0` means the available core count, falling back to one
/// thread when the platform cannot report parallelism.
fn rank_threads(jobs: usize) -> usize {
    if jobs == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        jobs
    }
}

/// The JSON health word for a [`RankHealth`], matching the enum's `Serialize`
/// spelling (`Good`, `Ok`, `Weak`, `Poor`, `Collapsed`).
fn rank_health_word(health: &RankHealth) -> &'static str {
    match health {
        RankHealth::Collapsed => "Collapsed",
        RankHealth::Good => "Good",
        RankHealth::Ok => "Ok",
        RankHealth::Weak => "Weak",
        RankHealth::Poor => "Poor",
    }
}

pub fn build_network_info(
    file: &LoRAFile,
    meta: &Metadata,
    total_layers: usize,
    unet: usize,
    text_encoder: usize,
) -> NetworkInfo {
    let mut rank: Vec<usize> = file.dims().into_iter().collect();
    rank.sort_unstable();

    let mut alpha: Vec<f64> = file
        .alphas()
        .into_iter()
        .map(|alpha| round_sig(f64::from(alpha.0), 6))
        .collect();
    alpha.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Any UNet layer wins over TextEncoder, because a mixed UNet+TE file is, to
    // a user, a UNet file (matching the example in the JSON contract).
    let family = if unet > 0 {
        "UNet"
    } else if text_encoder > 0 {
        "TextEncoder"
    } else {
        "Unknown"
    };

    NetworkInfo {
        module: meta.network_module_display(&file.format()),
        network_type: meta.network_type(),
        family: family.to_string(),
        rank,
        alpha,
        layers: LayerCounts {
            total: total_layers,
            unet,
            text_encoder,
        },
    }
}

/// The only constructor of [`Report`].
pub fn build_report(
    path: &Path,
    file: &LoRAFile,
    meta: &Metadata,
    opts: &ReportOptions,
) -> inspector::Result<Report> {
    let size_bytes = std::fs::metadata(path)?.len();

    let mut total_layers = 0usize;
    let mut unet = 0usize;
    let mut text_encoder = 0usize;
    let mut base_names = file.base_names();
    base_names.sort();
    for base_name in &base_names {
        total_layers += 1;
        match parse_block_key(base_name).family() {
            BlockFamily::Unet => unet += 1,
            BlockFamily::TextEncoder => text_encoder += 1,
            BlockFamily::Other => {}
        }
    }

    let network = build_network_info(file, meta, total_layers, unet, text_encoder);
    let metadata = if opts.metadata {
        meta.metadata.clone()
    } else {
        None
    };

    // The header-only path is exactly `--no-blocks` without `--rank`: neither
    // the block pass nor the rank pass runs, so nothing payload-derived is
    // computed. Every other combination loads the tensor payload.
    let load = if !opts.blocks && !opts.rank {
        Load::HeaderOnly
    } else {
        Load::Full
    };

    // When the block pass does not run, `blocks`, `ungrouped` and `skipped`
    // are all `null` (not computed), matching the `null` vs `[]` contract.
    let (blocks, block_weight, ungrouped, skipped) = if opts.blocks {
        let (blocks, ungrouped, skipped, summary) = build_blocks(file, opts)?;
        (Some(blocks), Some(summary), Some(ungrouped), Some(skipped))
    } else {
        (None, None, None, None)
    };

    let (rank, rank_summary) = if opts.rank {
        let (layers, summary) = build_rank(file, opts.jobs)?;
        (Some(layers), Some(summary))
    } else {
        (None, None)
    };

    Ok(Report {
        schema_version: SCHEMA_VERSION,
        load,
        file: FileInfo {
            path: path.to_string_lossy().into_owned(),
            size_bytes,
            precision: file.precision().map(|precision| precision.to_string()),
        },
        network,
        metadata,
        blocks,
        ungrouped,
        skipped,
        rank,
        summary: Summary {
            block_weight,
            rank: rank_summary,
        },
    })
}

/// Median of `values`; an empty slice is `0.0`, an even length averages the
/// middle two entries.
fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n.is_multiple_of(2) {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Outlier threshold shared by the block and rank rules: `1.5 × median`,
/// floored to [`OUTLIER_FLOOR`] when the median is too close to zero. The two
/// rules differ only in which median they pass in, never in this formula.
fn outlier_threshold(median: f64) -> f64 {
    if median < OUTLIER_FLOOR {
        OUTLIER_FLOOR
    } else {
        1.5 * median
    }
}
