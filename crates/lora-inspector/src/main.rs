use std::collections::HashMap;
use std::fs::File;
use std::io::{self, IsTerminal, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use candle_core::Device;
use clap::{Args, Parser, Subcommand};
use inspector::file::LoRAFile;
use inspector::metadata::{compare_metadata, Metadata};
use inspector::InspectorError;
use lora_inspector::render::{render, render_block_section};
use lora_inspector::report::{build_report, ReportOptions};
use lora_inspector::terminal::{ColorChoice, Terminal};
use serde::{Deserialize, Serialize};

/// Mirrors `inspector::header::MAX_HEADER_SIZE`, kept here so the header-only
/// loader can reject an absurd declared header length before allocating it.
const MAX_HEADER_SIZE: usize = 100_000_000;
const HEADER_LENGTH_BYTES: u64 = 8;

#[derive(Parser, Debug)]
#[clap(author, version, about = "Inspect LoRA file weights and norms")]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Analyze a LoRA file and print a terminal report
    #[clap(long_about = "\
Analyze a LoRA file and print a width-adaptive terminal report.

The header and block sections are always printed; --rank adds the
effective-rank section. --json replaces the terminal report with a single
machine-readable object whose floats are rounded to six significant digits.
--no-blocks without --rank loads only the file header, never the tensor
payload.

Examples:
  lora-inspector inspect model.safetensors
  lora-inspector inspect model.safetensors --rank --width 60
  lora-inspector inspect model.safetensors --json --no-blocks")]
    Inspect(InspectArgs),

    /// Inspect block weights (deprecated)
    BlockWeights {
        /// Path to the safetensors file
        #[clap(short, long)]
        file: PathBuf,

        /// Output format: json or text
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

    /// Extract a lightweight JSON fixture (key names, shapes, dtypes, and
    /// raw __metadata__) from a safetensors file's header, without loading
    /// tensor data. Used to build committed test fixtures from large local
    /// LoRA files.
    ExtractFixture {
        /// Path to the safetensors file
        #[clap(short, long)]
        file: PathBuf,

        /// Path to write the JSON fixture to
        #[clap(short, long)]
        out: PathBuf,
    },
}

#[derive(Args, Debug)]
struct InspectArgs {
    /// Path to the safetensors file to inspect
    file: PathBuf,

    /// Compute and render the effective-rank section (off by default)
    ///
    /// `balance` and the Good/Ok/Weak/Poor health counts are `effective_rank /
    /// nominal_rank`, so they fall as `dim` rises even when the LoRA is unchanged:
    /// compare `effective_rank` (absolute) across files, not health. `outlier` is
    /// `eff_scale > 1.5 x the median eff_scale of this file`, so it is relative to
    /// the file and its count does not travel between files of different overall
    /// strength. Both baselines are reported in `summary.rank`.
    #[clap(long)]
    rank: bool,

    /// Emit one row per layer in --json, including layers with no resolvable factors
    #[clap(long, requires = "json")]
    layers: bool,

    /// Emit a single machine-readable JSON object instead of the terminal report
    #[clap(long)]
    json: bool,

    /// Skip the block section and load only the file header
    #[clap(long)]
    no_blocks: bool,

    /// Include the raw safetensors metadata map in --json output
    #[clap(long)]
    metadata: bool,

    /// Allow the materializing fallback for layers without an up/down pair
    #[clap(long)]
    slow: bool,

    /// Override the detected terminal width in columns
    #[clap(long)]
    width: Option<usize>,

    /// Number of rows in the most-loaded blocks table
    #[clap(long, default_value_t = 8)]
    top: usize,

    /// Worker threads for the rank pass (0 = available cores)
    #[clap(long, default_value_t = 0)]
    jobs: usize,

    /// Use ASCII glyphs only, no Unicode bar characters
    #[clap(long)]
    ascii: bool,

    /// When to use ANSI colour: auto, always, or never
    #[clap(long, value_enum, default_value_t = ColorChoice::Auto)]
    color: ColorChoice,

    /// Suppress the progress indicator on stderr
    #[clap(long)]
    quiet: bool,
}

#[derive(Debug)]
enum ErrorKind {
    Io,
    Format,
    Unsupported,
}

impl ErrorKind {
    fn as_str(&self) -> &'static str {
        match self {
            ErrorKind::Io => "io",
            ErrorKind::Format => "format",
            ErrorKind::Unsupported => "unsupported",
        }
    }
}

/// A runtime error reported to the user. `json` selects the `--json` error
/// contract: one `{"error": {"kind", "message"}}` object on stderr and exit 1.
struct CliError {
    json: bool,
    kind: ErrorKind,
    message: String,
}

impl CliError {
    fn io(json: bool, err: io::Error) -> CliError {
        CliError {
            json,
            kind: ErrorKind::Io,
            message: err.to_string(),
        }
    }

    fn format(json: bool, message: impl Into<String>) -> CliError {
        CliError {
            json,
            kind: ErrorKind::Format,
            message: message.into(),
        }
    }

    fn report(&self) {
        if self.json {
            let value = serde_json::json!({
                "error": {
                    "kind": self.kind.as_str(),
                    "message": &self.message,
                }
            });
            eprintln!("{value}");
        } else {
            eprintln!("error: {}", self.message);
        }
    }
}

/// Map an [`InspectorError`] onto the JSON error contract's `kind` field.
///
/// Only filesystem access is `io`; a network type the report cannot decompose
/// is `unsupported`; every malformed-header, truncation, tensor and
/// serialization failure is `format`. The `usage` kind is reserved for clap,
/// which exits 2 before any of this runs.
fn inspector_error(json: bool, err: InspectorError) -> CliError {
    let kind = match &err {
        InspectorError::Io(_) => ErrorKind::Io,
        InspectorError::UnsupportedNetworkType => ErrorKind::Unsupported,
        InspectorError::Candle(_)
        | InspectorError::SafeTensor(_)
        | InspectorError::Load(_)
        | InspectorError::Msg(_)
        | InspectorError::SerdeJson(_)
        | InspectorError::SerdeWasmBindgenError(_)
        | InspectorError::NotFound => ErrorKind::Format,
    };
    let message = match &err {
        InspectorError::Io(io_err) => io_err.to_string(),
        other => other.to_string(),
    };
    CliError {
        json,
        kind,
        message,
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Command::Inspect(args) => exit(run_inspect(args)),
        Command::BlockWeights {
            file,
            output_format,
        } => exit(run_block_weights(file, &output_format)),
        Command::CompareMetadata { file1, file2 } => exit(run_compare_metadata(file1, file2)),
        Command::ExtractFixture { file, out } => exit(run_extract_fixture(file, out)),
    }
}

fn exit(result: Result<(), CliError>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            err.report();
            ExitCode::from(1)
        }
    }
}

fn run_inspect(args: InspectArgs) -> Result<(), CliError> {
    let json = args.json;
    // Load rule: `--no-blocks` without `--rank` and without `--layers` is the
    // one header-only load; every other combination needs the tensor payload
    // for the block, rank or per-layer pass.
    let header_only = args.no_blocks && !args.rank && !args.layers;

    let opts = ReportOptions {
        blocks: !args.no_blocks,
        rank: args.rank,
        layers: args.layers,
        top: args.top,
        jobs: args.jobs,
        slow: args.slow,
        metadata: args.metadata,
    };

    let (file, meta) = load_file(&args.file, header_only, json)?;
    let report = {
        let _progress =
            RankProgress::new(args.rank && !json && !args.quiet && io::stderr().is_terminal());
        build_report(&args.file, &file, &meta, &opts).map_err(|e| inspector_error(json, e))?
    };

    if json {
        let output =
            serde_json::to_string(&report).map_err(|e| CliError::format(json, e.to_string()))?;
        println!("{output}");
    } else {
        let term = Terminal::detect(args.width, args.color, args.ascii);
        print!("{}", render(&report, &meta, &term, &opts));
    }
    Ok(())
}

/// Load the file according to the header-only rule.
///
/// The header-only path reads just `8 + header_len` bytes and never touches the
/// tensor payload, so a multi-gigabyte file costs only its header. The full
/// path reads the whole buffer; `Metadata::new_from_buffer` also validates that
/// the payload is not truncated, which is what turns a cut-short file into an
/// error instead of a report with empty blocks.
fn load_file(path: &Path, header_only: bool, json: bool) -> Result<(LoRAFile, Metadata), CliError> {
    let filename = path.to_string_lossy();
    if header_only {
        let buffer = read_header_bytes(path, json)?;
        let file = LoRAFile::new_from_header_buffer(&buffer, &filename)
            .map_err(|e| inspector_error(json, e))?;
        let meta =
            Metadata::new_from_header_buffer(&buffer).map_err(|e| inspector_error(json, e))?;
        Ok((file, meta))
    } else {
        let bytes = std::fs::read(path).map_err(|e| CliError::io(json, e))?;
        let file = LoRAFile::new_from_buffer(&bytes, &filename, &Device::Cpu);
        let meta = Metadata::new_from_buffer(&bytes).map_err(|e| inspector_error(json, e))?;
        Ok((file, meta))
    }
}

/// Read only the first `8 + header_len` bytes of a safetensors file.
fn read_header_bytes(path: &Path, json: bool) -> Result<Vec<u8>, CliError> {
    let mut file = File::open(path).map_err(|e| CliError::io(json, e))?;
    let mut len_bytes = [0u8; HEADER_LENGTH_BYTES as usize];
    file.read_exact(&mut len_bytes)
        .map_err(|e| CliError::io(json, e))?;
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    if header_len > MAX_HEADER_SIZE {
        return Err(CliError::format(
            json,
            format!(
                "safetensors header length {header_len} exceeds maximum of {MAX_HEADER_SIZE} bytes"
            ),
        ));
    }

    let mut buffer = vec![0u8; HEADER_LENGTH_BYTES as usize + header_len];
    buffer[..HEADER_LENGTH_BYTES as usize].copy_from_slice(&len_bytes);
    file.read_exact(&mut buffer[HEADER_LENGTH_BYTES as usize..])
        .map_err(|e| CliError::io(json, e))?;
    Ok(buffer)
}

/// A spinner shown on stderr while the rank pass runs, cleared when dropped.
///
/// It is constructed only when `--rank` is active, stderr is a TTY and neither
/// `--json` nor `--quiet` was passed; `--json` must never put progress on
/// stdout, and this never touches stdout at all.
struct RankProgress(Option<indicatif::ProgressBar>);

impl RankProgress {
    fn new(enabled: bool) -> RankProgress {
        if !enabled {
            return RankProgress(None);
        }

        let bar = indicatif::ProgressBar::new_spinner();
        bar.set_draw_target(indicatif::ProgressDrawTarget::stderr());
        bar.set_style(
            indicatif::ProgressStyle::with_template(
                "{spinner} computing effective rank {elapsed_precise}",
            )
            .expect("progress template is valid")
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
        );
        bar.enable_steady_tick(std::time::Duration::from_millis(120));
        RankProgress(Some(bar))
    }
}

impl Drop for RankProgress {
    fn drop(&mut self) {
        if let Some(bar) = self.0.take() {
            bar.finish_and_clear();
        }
    }
}

fn run_block_weights(file: PathBuf, output_format: &str) -> Result<(), CliError> {
    eprintln!("warning: `block-weights` is deprecated; use `inspect FILE` instead");

    let opts = ReportOptions {
        blocks: true,
        rank: false,
        layers: false,
        top: 8,
        jobs: 0,
        slow: false,
        metadata: false,
    };

    let (lora, meta) = load_file(&file, false, false)?;
    let report = build_report(&file, &lora, &meta, &opts).map_err(|e| inspector_error(false, e))?;

    match output_format {
        "json" => {
            let output = serde_json::to_string(&report)
                .map_err(|e| CliError::format(false, e.to_string()))?;
            println!("{output}");
        }
        "text" => {
            let term = Terminal::detect(None, ColorChoice::Auto, false);
            if let Some(section) = render_block_section(&report, &term, &opts) {
                println!("{section}");
            }
        }
        other => {
            return Err(CliError::format(
                false,
                format!("unsupported output format `{other}` (expected `json` or `text`)"),
            ));
        }
    }

    Ok(())
}

fn run_compare_metadata(file1: PathBuf, file2: PathBuf) -> Result<(), CliError> {
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
}

fn metadata_from_file(file_path: PathBuf) -> Result<Metadata, CliError> {
    let mut file = File::open(&file_path).map_err(|e| CliError::io(false, e))?;
    file.seek(SeekFrom::Start(0))
        .map_err(|e| CliError::io(false, e))?;

    // Read the length of the header
    let mut buf = [0; HEADER_LENGTH_BYTES as usize];
    file.read_exact(&mut buf)
        .map_err(|e| CliError::io(false, e))?;
    let length_of_header = u64::from_le_bytes(buf);

    // Seek to the start of the header data
    file.seek(SeekFrom::Start(HEADER_LENGTH_BYTES))
        .map_err(|e| CliError::io(false, e))?;

    // Read the header data
    let mut header_data = vec![0; length_of_header as usize];
    file.read_exact(&mut header_data)
        .map_err(|e| CliError::io(false, e))?;

    #[derive(Deserialize, Debug)]
    struct MD {
        __metadata__: HashMap<String, String>,
    }

    let metadata = serde_json::from_slice::<MD>(header_data.as_slice())
        .map_err(|e| CliError::format(false, e.to_string()))?;

    Ok(Metadata {
        metadata: Some(metadata.__metadata__),
    })
}

#[derive(Serialize, Deserialize, Debug)]
struct FixtureTensorInfo {
    dtype: String,
    shape: Vec<usize>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Fixture {
    source: String,
    keys: HashMap<String, FixtureTensorInfo>,
    metadata: Option<HashMap<String, String>>,
}

/// Write a header-only JSON fixture (key names, dtypes, shapes and the raw
/// `__metadata__` map) for a safetensors file, so a large local LoRA can become
/// a small committed test fixture without copying any tensor data.
fn run_extract_fixture(file: PathBuf, out: PathBuf) -> Result<(), CliError> {
    let buffer = read_header_bytes(&file, false)?;
    let header_data = &buffer[HEADER_LENGTH_BYTES as usize..];

    #[derive(Deserialize)]
    struct RawTensorInfo {
        dtype: String,
        shape: Vec<usize>,
    }

    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(header_data).map_err(|e| CliError::format(false, e.to_string()))?;

    let mut keys = HashMap::new();
    let mut metadata = None;

    for (key, value) in header {
        if key == "__metadata__" {
            metadata = serde_json::from_value(value).ok();
            continue;
        }
        let info: RawTensorInfo =
            serde_json::from_value(value).map_err(|e| CliError::format(false, e.to_string()))?;
        keys.insert(
            key,
            FixtureTensorInfo {
                dtype: info.dtype,
                shape: info.shape,
            },
        );
    }

    let fixture = Fixture {
        source: file.display().to_string(),
        keys,
        metadata,
    };

    let json = serde_json::to_string_pretty(&fixture)
        .map_err(|e| CliError::format(false, e.to_string()))?;
    std::fs::write(&out, json).map_err(|e| CliError::io(false, e))?;

    println!(
        "Wrote fixture with {} keys to {}",
        fixture.keys.len(),
        out.display()
    );

    Ok(())
}
