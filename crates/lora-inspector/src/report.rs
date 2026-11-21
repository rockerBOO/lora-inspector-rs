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
use inspector::svd::RankHealth;
use serde::Serialize;

pub const SCHEMA_VERSION: u32 = 1;

const SKIPPED_REASON: &str = "no low-rank factor (LoHA/LoKr/OFT/BOFT)";
const UNGROUPED_PREFIX_CAP: usize = 10;

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

/// The top-level JSON document emitted by `--json`.
#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub schema_version: u32,
    pub file: FileInfo,
    pub network: NetworkInfo,
    pub metadata: Option<HashMap<String, String>>,
    pub blocks: Option<Vec<BlockStat>>,
    pub ungrouped: Ungrouped,
    pub skipped: Skipped,
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
    let threshold = if median_block_mean < 1e-10 {
        1e-10
    } else {
        1.5 * median_block_mean
    };

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
                ratio_to_median: round_sig(
                    if median_block_mean > 0.0 {
                        agg.mean / median_block_mean
                    } else {
                        0.0
                    },
                    6,
                ),
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

    let (blocks, block_weight, ungrouped, skipped) = if opts.blocks {
        let (blocks, ungrouped, skipped, summary) = build_blocks(file, opts)?;
        (Some(blocks), Some(summary), ungrouped, skipped)
    } else {
        (
            None,
            None,
            Ungrouped {
                layers: 0,
                prefixes: Vec::new(),
            },
            Skipped {
                layers: 0,
                reason: None,
            },
        )
    };

    // The rank pass is built by a later task; until then the key is always
    // present and `null`, never absent or an empty list.
    let rank: Option<Vec<RankLayer>> = None;
    let rank_summary: Option<RankSummary> = None;

    Ok(Report {
        schema_version: SCHEMA_VERSION,
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
