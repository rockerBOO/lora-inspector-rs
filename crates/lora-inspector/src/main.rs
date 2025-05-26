use clap::{Parser, Subcommand};
use inspector::metadata::compare_metadata;
use inspector::{file, metadata, norms, statistic, InspectorError};
use serde::{Deserialize, Serialize};
use std::io;
use std::{collections::HashMap, fs::File, io::Read, path::PathBuf};

#[derive(Parser, Debug)]
#[clap(author, version, about = "Inspect LoRA file weights and norms")]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Inspect block weights
    BlockWeights {
        /// Path to the safetensors file
        #[clap(short, long)]
        file: PathBuf,

        #[clap(short, long, default_value = "json")]
        output_format: String,
    },

    /// Compare metadata
    CompareMetadata {
        /// Path to the first safetensors file
        #[clap(long)]
        file1: PathBuf,

        /// Path to the second safetensors file
        #[clap(long)]
        file2: PathBuf,
    },
}

#[derive(Serialize, Clone, Debug)]
struct WeightStatistics {
    l1_norm: f64,
    l2_norm: f64,
    matrix_norm: f64,
    max: f64,
    min: f64,
    std_dev: Option<f64>,
    median: Option<f64>,
}

#[derive(Serialize, Debug, Clone)]
struct BlockData {
    weights: HashMap<String, WeightStatistics>,
    average_l2_norm: f64,
}

#[derive(Serialize, Debug)]
struct OutputData {
    metadata: Option<HashMap<String, String>>,
    base_names: Vec<String>,
    norms: HashMap<String, WeightStatistics>,
    blocks: HashMap<String, BlockData>,
    non_block_weights: HashMap<String, WeightStatistics>,
}

pub type Result<T> = std::result::Result<T, LoraInspectorError>;

#[derive(Debug)]
pub enum LoraInspectorError {
    Json(serde_json::Error),
    Io(io::Error),
    Inspector(InspectorError),
}

impl From<serde_json::Error> for LoraInspectorError {
    fn from(err: serde_json::Error) -> LoraInspectorError {
        LoraInspectorError::Json(err)
    }
}

impl From<io::Error> for LoraInspectorError {
    fn from(err: io::Error) -> LoraInspectorError {
        LoraInspectorError::Io(err)
    }
}

impl From<InspectorError> for LoraInspectorError {
    fn from(err: InspectorError) -> LoraInspectorError {
        LoraInspectorError::Inspector(err)
    }
}

// Parses a weight name to determine if it belongs to a block and which one
fn parse_block_info(weight_name: &str) -> Option<(String, String)> {
    // Check for "blocks_XX" pattern
    if let Some(blocks_idx) = weight_name.find("blocks_") {
        // Find where the block number starts
        let num_start = blocks_idx + "blocks_".len();

        // Find where the block number ends (next underscore)
        if let Some(num_end) = weight_name[num_start..].find('_') {
            let block_num = &weight_name[num_start..num_start + num_end];

            // Determine if it's a single or double block
            let block_type = if weight_name.contains("single") {
                "single"
            } else if weight_name.contains("double") {
                "double"
            } else {
                "unknown"
            };

            let block_key = format!("{}_{}", block_type, block_num);
            return Some((block_key, weight_name.to_string()));
        }
    }

    None
}

// Calculate average L2 norm for a block
fn calculate_block_average_l2(norms: &HashMap<String, WeightStatistics>) -> f64 {
    if norms.is_empty() {
        return 0.0;
    }

    let sum: f64 = norms.values().map(|norm| norm.l2_norm).sum();
    sum / (norms.len() as f64)
}

// Generate ASCII bar chart
fn generate_ascii_bar(value: f64, max_value: f64, width: usize) -> String {
    let bar_width = if max_value > 0.0 {
        ((value / max_value) * width as f64).round() as usize
    } else {
        0
    };

    let bar = "█".repeat(bar_width);
    let padding = " ".repeat(width - bar_width);
    format!("[{}{}] {:.6}", bar, padding, value)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::BlockWeights {
            file,
            output_format,
        } => {
            // Handle block weights command
            println!("Block weights command with file: {}", file.display());
            parse_block_weights(file, output_format.as_str())
        }

        Command::CompareMetadata { file1, file2 } => {
            // Handle compare metadata command
            println!(
                "Compare metadata command with files: {} and {}",
                file1.display(),
                file2.display()
            );
            let diff = compare_metadata(&metadata_from_file(file1)?, &metadata_from_file(file2)?);

            println!("Added:");
            for (k, v) in diff.added.iter() {
                println!("{}", k);
                println!("{}", v);
            }

            println!("Removed:");
            for (k, v) in diff.removed.iter() {
                println!("{}", k);
                println!("{}", v);
            }

            println!("Changed:");
            for (k, v) in diff.changed.iter() {
                println!("{}", k);
                println!("old: {}", v.old);
                println!("new: {}", v.new);
            }
            Ok(())
        } // Handle other subcommands...
    }
}

fn parse_block_weights(file: PathBuf, output_format: &str) -> Result<()> {
    let device = if let Ok(device) = candle_core::Device::cuda_if_available(0) {
        device
    } else {
        candle_core::Device::Cpu
    };

    // Read file
    let mut f = File::open(&file)?;
    let mut data = vec![];
    f.read_to_end(&mut data)?;

    // Parse metadata
    let metadata = metadata::Metadata::new_from_buffer(data.as_slice())
        .map_err(|e| e.to_string())
        .ok();

    // Create LoRA file
    let file = file::LoRAFile::new_from_buffer(data.as_slice(), file.to_str().unwrap(), &device);

    // Get base names
    let base_names = file.base_names();

    // Setup progress bar
    let pb = indicatif::ProgressBar::new(base_names.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} weights ({eta})")
            .unwrap()
            .progress_chars("#>-")
    );

    // Calculate norms for each base name
    let mut norms_map = HashMap::new();

    // Maps to store our organized data
    let mut blocks: HashMap<String, HashMap<String, WeightStatistics>> = HashMap::new();
    let mut non_block_weights: HashMap<String, WeightStatistics> = HashMap::new();

    for base_name in &base_names {
        pb.set_message(format!("Processing: {}", base_name));

        let scale_weight = file
            .scale_weight(base_name)
            .unwrap()
            .to_dtype(candle_core::DType::F64)
            .unwrap();

        // Calculate all norms
        let l1 = norms::l1::<f64>(&scale_weight).unwrap();
        let l2 = norms::l2::<f64>(&scale_weight).unwrap();
        let matrix = norms::matrix_norm::<f64>(&scale_weight).unwrap();
        let max_val = norms::max(&scale_weight).unwrap();
        let min_val = norms::min(&scale_weight).unwrap();

        // Some norms might not be implemented yet, so we make them optional
        let std_dev = Some(statistic::std_dev(&scale_weight)?);
        // let median = statistic::median(&scale_weight).ok();

        norms_map.insert(
            base_name.to_string(),
            WeightStatistics {
                l1_norm: l1,
                l2_norm: l2,
                matrix_norm: matrix,
                max: max_val,
                min: min_val,
                std_dev,
                median: None,
            },
        );

        // Organize weights into blocks or non-blocks
        if let Some((block_key, weight_name)) = parse_block_info(base_name) {
            blocks.entry(block_key).or_default().insert(
                weight_name,
                norms_map
                    .get(base_name)
                    .expect("To get norm map for base name")
                    .clone(),
            );
        } else {
            non_block_weights.insert(
                base_name.to_string(),
                norms_map.get(base_name).expect("To get norm map").clone(),
            );
        }

        pb.inc(1);
    }

    pb.finish_with_message("Completed norm calculations");

    // Create final block data with average L2 norms
    let blocks_data: HashMap<String, BlockData> = blocks
        .into_iter()
        .map(|(block_key, weights)| {
            let average_l2_norm = calculate_block_average_l2(&weights);
            (
                block_key,
                BlockData {
                    weights,
                    average_l2_norm,
                },
            )
        })
        .collect();

    // Create output data structure
    let output = OutputData {
        metadata: metadata.map(|m| m.metadata).expect("No metadata"),
        base_names,
        norms: norms_map,
        blocks: blocks_data.clone(),
        non_block_weights,
    };

    // Output based on format
    match output_format {
        "json" => {
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
        "text" => {
            println!("Metadata: {:?}", output.metadata);
            println!("Base Names: {:?}", output.base_names);
            println!("Norms:");
            for (name, norm) in &output.norms {
                println!("  {name}:");
                println!("    L1 Norm: {}", norm.l1_norm);
                println!("    L2 Norm: {}", norm.l2_norm);
                println!("    Matrix Norm: {}", norm.matrix_norm);
                println!("    Max: {}", norm.max);
                println!("    Min: {}", norm.min);
                if let Some(std) = norm.std_dev {
                    println!("    Std Dev: {}", std);
                }
                if let Some(med) = norm.median {
                    println!("    Median: {}", med);
                }
            }

            println!("\nBlocks:");
            for (block_name, block_data) in &output.blocks {
                println!(
                    "  {} (Average L2 Norm: {})",
                    block_name, block_data.average_l2_norm
                );
                println!("  Weights in block: {}", block_data.weights.len());
            }

            // Find max L2 norm for scaling the ASCII bars
            let max_l2_norm = blocks_data
                .values()
                .map(|block| block.average_l2_norm)
                .fold(0.0_f64, |a, b| a.max(b));

            // Sort blocks by name for better visualization
            let mut block_names: Vec<String> = blocks_data.keys().cloned().collect();
            block_names.sort_by(|a, b| {
                // First sort by block type (single/double)
                let a_parts: Vec<&str> = a.split('_').collect();
                let b_parts: Vec<&str> = b.split('_').collect();

                let a_type = a_parts[0];
                let b_type = b_parts[0];

                if a_type != b_type {
                    return a_type.cmp(b_type);
                }

                // Then sort by block number
                let a_num = a_parts[1].parse::<i32>().unwrap_or(0);
                let b_num = b_parts[1].parse::<i32>().unwrap_or(0);
                a_num.cmp(&b_num)
            });

            // Display ASCII chart
            for block_name in block_names {
                let block = &blocks_data[&block_name];
                let bar = generate_ascii_bar(block.average_l2_norm, max_l2_norm, 40);
                println!(
                    "{:20} {} ({} weights)",
                    block_name,
                    bar,
                    block.weights.len()
                );
            }

            println!("\nNon-Block Weights:");
            for (name, norm) in &output.non_block_weights {
                println!("  {name}:");
                println!("    L2 Norm: {}", norm.l2_norm);
            }
        }
        _ => {
            eprintln!("Unsupported format: {}. Using JSON instead.", output_format);
            println!("{}", serde_json::to_string_pretty(&output)?);
        }
    }

    Ok(())
}

use std::io::{Seek, SeekFrom};

const HEADER_LENGTH_BYTES: u64 = 8;

fn metadata_from_file(file_path: PathBuf) -> Result<metadata::Metadata> {
    let mut file = File::open(file_path)?;
    file.seek(SeekFrom::Start(0))?;

    // Read the length of the header
    let mut buf = [0; 8];
    file.read_exact(&mut buf)?;
    let length_of_header = u64::from_le_bytes(buf);

    // Seek to the start of the header data
    file.seek(SeekFrom::Start(HEADER_LENGTH_BYTES))?;

    // Read the header data
    let mut header_data = vec![0; length_of_header as usize];
    file.read_exact(&mut header_data)?;

    #[derive(Deserialize, Debug)]
    struct MD {
        __metadata__: HashMap<String, String>,
    }

    let metadata = serde_json::from_slice::<MD>(header_data.as_slice())?;

    Ok(metadata::Metadata {
        metadata: Some(metadata.__metadata__.clone()),
    })
}
