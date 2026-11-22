# Enriching the training-set LoRA index with inspector-derived facts

## Intent and why

`/mnt/900/training/sets/` holds every LoRA training run produced here — 3376
`.safetensors` files, 197.8 GB. `index.py` already indexes it into DuckDB, but it reads
only the safetensors `__metadata__` block: what the *trainer recorded*.

The point of the index is to answer "which run was the one that…" quickly. Today it can
only answer questions the trainer happened to write down.

`lora-inspector inspect --json` derives facts from the tensor keys, shapes and weights
instead: the network format each layer actually uses, real per-file rank and alpha
sets, precision, layer counts, per-block weight norms, per-layer effective-rank health.
Feeding that into the index is the whole of this work.

Three things make it worth doing beyond "more columns":

1. **The keys are the truth; the metadata is a hint.** Generation tooling reads tensor
   keys — `hada_w1_a` loads as LoHa, `lokr_w1` as LoKr, `lora_up/down` as LoRA —
   regardless of what `ss_network_module` claims. A file whose metadata says
   `networks.lora` while its keys say LoHa will be treated as LoHa at inference time.
   Nothing in the current index, and nothing in `inspect --json` today, records what
   the keys actually say.
2. **Rank divergence.** Sampling 25 files found declared-vs-actual rank disagreements
   the index currently records as truth. `women-flux-2025-03-24-174551` declares
   `ss_network_dim=8` and actually holds rank 16. `retro-anime-2026-03-31-021415`
   declares rank 4 / alpha 8 and actually holds mixed ranks `[4, 16]` / alphas `[8, 32]`.
3. **Non-LoRA files sitting in the tree.** `pov_sd15-2025-10-04-012201` is a 2.1 GB
   full SD1 checkpoint (`modelspec.architecture: stable-diffusion-v1`, no `/lora`
   suffix). The `*-proj.safetensors` files are ~300 MB dumps of two tensors
   (`0.weight`, `2.weight`) with no metadata at all. Both index today as ordinary rows
   with a null title.

## Goals

- Record what the tensor keys and shapes say the network actually is, per file and per
  layer, and compare it against what metadata claims.
- Every other derived fact `lora-inspector` can produce lands in queryable, typed
  DuckDB columns — one row per layer included, not just per-block aggregates.
- Per-layer rows cover *all* layers, including those with no resolvable scale, so
  "missing" shows up as a null rather than an absent row.
- Declared-vs-actual disagreement is precomputed into boolean columns, so it is a cheap
  `WHERE`, not a per-query JSON comparison.
- Re-enrichment is cheap, idempotent, and re-runs automatically when a file changes
  *or* when the inspector's own output contract changes.

## Non-goals

- Not changing the terminal report, the WASM frontend, or any existing CLI behaviour.
- Not re-implementing norms or rank metrics in Python.
- **Not deleting anything.** Enrichment never prunes, never cleans, never reconciles.
  See "No pruning" below.
- Not building a general-purpose directory scanner in Rust (see Approaches).
- Not moving the 197.8 GB of weights.

## Anti-goals — this fails even if it works

- Storing the new data as an undifferentiated JSON blob that has to be `json_extract`-ed
  per query. That is the thing being fixed.
- A full re-read of 197.8 GB to answer a question about one run.
- A "mismatch" column that compares a declared value to itself, and is therefore
  structurally always false or always null.
- Treating metadata as evidence of structural format. Metadata is a hint about intent;
  the keys are what a loader will act on.

## Measured constraints

Measured against the real tree and the real binary.

| Fact | Value |
|---|---|
| Files / total size | 3376 files, 197.8 GB, median 41.6 MB |
| Disk headroom | `/mnt/900` is 99% full — **11 GB free** |
| Throughput, blocks + rank | 167 MB/s → **~20 min** for a full serial sweep |
| `--no-blocks` (header only) | 26 ms/file, but **drops `alpha`** (comes back `[]`) |
| Existing DB | `loras.duckdb` is 1015 MB, 6252 rows, **3193 of which point at deleted files** |
| Alive by declared module | `lora_flux_2` 1116, `lora_flux` 987, `lora` 396, `NULL` 381, `lora_lumina` 163, other 16 |
| LyCORIS fixture | `ahetableh_9_lokr`: 1086 layers → 193 with a block scale, 238 with rank metrics, 893 with neither |

Consequences:

- Do **not** use a header-only fast path. It loses `alpha`, which is a divergence column
  we care about. Always run the block pass.
- The binary at `target/release/lora-inspector` is **stale** — it predates the `load`
  field current source emits. Rebuild before any sweep.
- `--slow` stays **off**. It would resolve the 893 unresolved layers of a LyCORIS file
  by materializing dense matrices (the terminal-report spec rejected that path for
  memory). Those layers ship as rows with null scales instead.
- Peak RSS ≈ `workers × file size`, because each worker process reads its whole file.
  One 2.1 GB checkpoint times eight workers is the budget that matters, and the only
  measured headroom figure is disk.

## Current state

### Structure is already known to the loader, never reported

`weight.rs` resolves each base name by branching on key suffixes it actually finds:

| Keys present | Format |
|---|---|
| `lora_down` + `lora_up` | LoRA (Kohya) |
| `lora_A` + `lora_B` | LoRA (PEFT) |
| `hada_w1_a/_b` + `hada_w2_a/_b` | LoHa |
| `lokr_w1` + `lokr_w2`, or `lokr_w1_a/_b` + `lokr_w2_a/_b` | LoKr |
| `oft_diag` | OFT (diagonal) |
| `oft_blocks` | OFT / BOFT |
| `a1` `b1` `a2` `b2` | GLoRA |
| plus `dora_scale` alongside any of the above | DoRA weighting |

That detection happens inside `BufferedLoRAWeight` and is thrown away. Meanwhile
`Metadata::network_type` derives the reported type from `ss_network_module` and
`ss_network_args` **only** (`metadata.rs:40-58`) — so `inspect --json` reports what
metadata claims, not what the loader actually did. The observed `LoRAC3Lier` type comes
from a declared `conv_dim`, not from tensor structure.

By contrast `network.rank` and `network.alpha` **are** key-derived:
`HeaderIndex::dims()` reads `lora_down` / `hada_w1_b` / `lokr_w1` / `b1.weight` /
`lora_A` / `oft_*` shapes (`header.rs:83-101`).

### Per-layer norms are computed then discarded

Two passes both iterate `file.base_names()`:

- **Block pass** (`report.rs:203`): resolves a `scale` per layer via
  `low_rank_effective_scale`, appends it to a per-`(group, index)` bucket, computes
  `mean`/`max`/`median`, then discards the individual scales.
- **Rank pass** (`report.rs:352`): rayon-parallel, one `RankLayer` per layer that has a
  low-rank up/down pair. Every other layer is counted in `skipped.layers` and lost.

### Three different notions, not one

On `lora_unet_down_blocks_1_resnets_1_conv2.safetensors` (a 3×3 conv LoRA) the report
gives `skipped.layers: 1`, `blocks: []`, and one `rank` row with `eff_scale: null`. On
`ahetableh_9_lokr` the block pass resolves 193 layers while the rank pass resolves 238.
Neither set contains the other. Any design or test that conflates them is wrong.

## Approaches considered

**A. Python drives the CLI; results go into new DuckDB tables. (chosen)**
`index.py enrich` subprocesses `lora-inspector inspect --json --layers --rank`, with a
bounded worker pool, and writes three tables. A per-file JSON cache means rebuilds never
re-read weights. The work is disk-bound, so parallelism buys little but costs nothing if
bounded.

**B. A `lora-inspector scan <dir> --json-lines` subcommand in Rust.**
One process, rayon across the tree, NDJSON out; Python only loads it. Rejected for now:
the sweep is disk-bound at 167 MB/s, so it would not be meaningfully faster here, and it
is materially more work (new subcommand, tests, a second place that knows how to walk a
tree). Worth revisiting if this must run on other machines; A's cache format carries
over unchanged.

**C. Dump the raw JSON into one column, query with `json_extract`.**
Least code, but it is exactly the anti-goal: no typed columns, no cheap `WHERE`, and
disagreement stays buried in JSON. Rejected.

**D. Re-implement norms/rank in Python to avoid the CLI.** Rejected: duplicates
algorithms that are the whole point of the Rust crate.

## Stage 1 — CLI: report the structural format, and emit per-layer rows

Two additions to `inspect --json`, both additive.

### 1a. `network.structure` — what the keys say

A new key beside the existing `network.type`:

```json
"network": {
  "module": "kohya-ss/lora",
  "type": "LoRa",
  "family": "UNet",
  "rank": [4], "alpha": [4.0],
  "layers": { "total": 264, "unet": 192, "text_encoder": 72 },
  "structure": {
    "primary": "LoRA",
    "by_format": { "LoRA": 260, "LoHa": 4 },
    "formats": ["LoHa", "LoRA"],
    "has_conv": false,
    "has_dora": true
  }
}
```

- `network.type` keeps its current, metadata-derived meaning. It is **not** retyped —
  that would violate the contract rule. The doc and README must state plainly that
  `type` is the declared hint and `structure` is what the loader will act on.
- `primary` is the format with the most layers; ties break by a fixed format order so
  output is deterministic.
- `formats` is sorted; `by_format` counts layers.
- `has_conv` is true when any structural up/down factor is 4-dimensional — that is the
  LoCon distinction, which is a property, not an algorithm.
- `has_dora` is true when any layer carries `dora_scale`.
- Per-layer `format` is also emitted inside `layers[]` (1b), so a mixed file shows where.

Derivable from the header alone, so `structure` is populated in `--no-blocks` mode too.
That makes a cheap structural sweep possible later if it is ever wanted.

Implementation: lift the suffix classification out of `BufferedLoRAWeight` into a small
`fn structural_format(keys: &[String]) -> Option<StructureFormat>` over one base name's
suffixes, and have both the loader and the report call it. One classifier, two callers —
so the report cannot drift from what the loader does.

### 1b. `--layers` — one row per layer

Name: `--layers`. Not `--full`: that does not say what it adds, and would then have to
grow if a later feature needs "everything".

`--layers` is additive and orthogonal:

- Adds a top-level `layers` array. Does **not** change `blocks`, `rank`, `summary`, or
  `network` beyond adding `structure`.
- `--rank` still behaves exactly as today; alone, `--layers` leaves `rank: null` while
  rank fields appear inside `layers` rows.
- **No `schema_version` bump.** This repo's contract rule
  (`docs/superpowers/specs/2026-11-21-cli-terminal-report-design.md:446-447`) is that the
  version changes only on rename, retype, or removal.
- `--layers` needs the tensor payload, so it forces a full load. With `--no-blocks` it
  emits `blocks: null` while `load` stays `"full"`; `--no-blocks` keeps meaning "omit the
  blocks array".

Row shape — one row per base name, in `base_name` sort order (matching `RankLayer`
determinism), so output is byte-identical for any thread count:

```json
{"base_name": "lora_unet_blocks_0_attn_gate",
 "group": "Blocks", "family": "UNet", "label": "TB00", "index": 0,
 "format": "LoRA",
 "scale": 0.123456,
 "nominal_rank": 16, "effective_rank": 5.93623, "balance": 0.371014,
 "top1_energy": 0.52932, "dominance": 1.94931, "health": "Weak",
 "eff_scale": 0.449574, "outlier": false}
```

Rules:

- Every base name appears, always. A layer with no resolvable scale gets `scale: null`;
  one with no rank metrics gets `nominal_rank: null` and every other rank field `null`.
  Those nulls are evidence: they are how a query tells "LoKr layer, no low-rank factor"
  from "LoRA layer that failed to parse".
- There is **no** `low_rank` boolean. Measured above, it cannot be made to mean one
  thing. `nominal_rank IS NOT NULL` is the honest test for "rank metrics exist";
  `scale IS NOT NULL` is a separate, independent test.
- Ungrouped layers (`BlockGroup::Other`) appear with `"group": "Other"`, `label: null`,
  `index: null` and their scale, instead of being rolled into `ungrouped.layers`.
- `health` is `null` when rank metrics are absent — never a sixth invented value.
- Floats go through the existing `round_sig(_, 6)`; `null` means not computed, `[]`
  keeps meaning computed-and-empty.

Implementation: keep the two passes separate. Give each an optional per-layer sink, then
merge the sinks by `base_name` into the sorted `layers` array. Merging the passes would
restructure working, tested code and change the block outlier rule's inputs, for CPU
time the disk-bound sweep cannot notice. Cost is small: a 1086-layer LyCORIS file keeps
about 1086 rows in memory versus a few hundred block aggregates today.

### Not in scope for Stage 1

- Un-capping `ungrouped.prefixes` (still `UNGROUPED_PREFIX_CAP = 10`). `layers` supersedes
  it.
- Per-layer skip reasons. A null `scale` already carries that.
- Detecting BOFT vs OFT beyond `oft_blocks` presence.

## Stage 2 — `index.py enrich`

A new subcommand in the existing `index.py`, sharing `DB_DEFAULT`, `SETS_DEFAULT` and
`read_safetensors_metadata`. Same DB file, same `uv run --with duckdb` entry point; no
new script.

```
uv run --with duckdb index.py enrich                    # new/changed files only
uv run --with duckdb index.py enrich --force            # everything
uv run --with duckdb index.py enrich --jobs 8
uv run --with duckdb index.py enrich --no-layers        # skip the per-layer table
uv run --with duckdb index.py enrich --from-cache       # replay cached JSON only
```

### Driving the CLI

- Invoke `lora-inspector inspect <file> --json --layers --rank --quiet` with `--jobs 1`
  per process, running N processes concurrently.
- **Pass `--rank` as well.** It costs nothing extra (the metrics are computed for
  `--layers` anyway) and makes `rank_layers`, `rank_median_balance`,
  `rank_weak_or_worse` and the `health_*` counts verbatim CLI output rather than a second
  Python implementation of the same summary.
- Concurrency is bounded by **bytes in flight**, not worker count: default 8 workers, but
  a file over 500 MB is processed alone.
- Capture **stderr**. Parse warnings arrive there even under `--quiet` (observed:
  `Error("Parse sequence failed", line: 1, column: 128)` on two fixtures, both of which
  still exit 0 with a valid report). Store them in `lora_inspect.parse_warnings`.
- Empty or unparseable stdout, or a non-zero exit, records the file with `enrich_error`
  set and no other derived columns, and the run continues.
- Binary lookup: `--inspector` flag, then `$LORA_INSPECTOR`, then
  `target/release/lora-inspector` relative to this repo, then `PATH`.

### Which files

`enrich` enumerates from **disk** (`SETS_DEFAULT.rglob("*.safetensors")`), then left joins
`loras` on `file_path`. Files with no `loras` row are still enriched — `declared_*` come
out null and `file_mtime` comes from `os.stat`. This keeps `enrich` usable without a
prior `index` run.

### No pruning

`enrich` never deletes a row — not from `loras`, not from its own tables, not for files
that have vanished. If a file is gone it simply is not in the enumeration and is not
enriched; if it disappears mid-run the inspect call fails and only an `enrich_error` is
recorded for that file. Stale rows accumulate and stay. This is deliberate.

The consequence to be aware of (and the reason it is stated here rather than fixed): the
existing index already carries 3193 rows for files that no longer exist, so `v_runs` and
any aggregate will include phantom files. `v_issues` will show them with a null
`enriched_at`. That is a reporting concern for whoever reads the views, not something
enrichment should resolve.

### Cache

Write each file's full report, including `--metadata`, to `sets/.inspect/<relpath>.json.gz`,
keyed by the same relative path `lora_index.json` uses.

The cache is **not** authoritative on its own. Each entry records a fingerprint — the
inspector version string, the binary's size and mtime, and `schema_version`. On `enrich`,
an entry whose fingerprint matches the current binary may be replayed; one that does not
is re-inspected, never replayed. `inspector_version` alone is a weak signal: versions are
bumped by hand (`scripts/bump_release.py`), so a rebuilt `target/release` during
development keeps the same version string and would otherwise stamp new provenance on old
metrics.

`--from-cache` relaxes this to replay whatever is cached, and is only for rebuilding the
DB tables after a schema change.

### Schema

```sql
CREATE TABLE IF NOT EXISTS lora_inspect (
  file_path        TEXT PRIMARY KEY,
  inspector_version TEXT,          -- `lora-inspector --version`
  fingerprint      TEXT,           -- version + binary size/mtime, see Cache
  schema_version   INTEGER,
  enriched_at      TEXT,           -- ISO-8601 UTC
  load_kind        TEXT,           -- 'full' | 'header_only'

  -- DECLARED: metadata-derived, the hint
  type_declared    TEXT,           -- network.type, e.g. 'LoRA', 'LoRAC3Lier', 'LoHa'
  module_declared  TEXT,           -- ss_network_module, e.g. 'networks.lora_flux_2'
  module_reported  TEXT,           -- network.module, e.g. 'musubi-tuner/lora_flux_2'
  family_reported  TEXT,

  -- ACTUAL: key/shape-derived, the truth
  type_actual      TEXT,           -- network.structure.primary
  formats_present  TEXT[],         -- network.structure.formats
  format_counts    JSON,           -- network.structure.by_format
  has_conv         BOOLEAN,
  has_dora         BOOLEAN,

  -- tensor-derived scalars
  precision        TEXT,           -- 'fp16' | 'bf16' | 'fp32'
  layers_total     INTEGER,
  layers_unet      INTEGER,
  layers_te        INTEGER,
  ungrouped_layers INTEGER,
  skipped_layers   INTEGER,        -- block pass could not resolve a scale
  rank_actual      INTEGER[],      -- distinct, from tensor shapes
  alpha_actual     DOUBLE[],       -- may contain NULL (LoKr)
  rank_min         INTEGER,
  rank_max         INTEGER,

  -- block summary
  block_median     DOUBLE,
  block_peak_label TEXT,
  block_peak_value DOUBLE,
  block_peak_ratio DOUBLE,
  block_outliers   INTEGER,

  -- rank summary
  rank_layers      INTEGER,
  rank_median_balance DOUBLE,
  rank_weak_or_worse  INTEGER,
  health_good      INTEGER,
  health_ok        INTEGER,
  health_weak      INTEGER,
  health_poor      INTEGER,
  health_collapsed INTEGER,

  -- declared scalars, parsed from loras.metadata (see below)
  declared_rank    INTEGER,
  declared_alpha   DOUBLE,

  -- derived verdicts
  kind             TEXT,           -- 'lora' | 'checkpoint' | 'artifact' | 'unknown'
  is_lora          BOOLEAN,
  type_mismatch    BOOLEAN,
  rank_mismatch    BOOLEAN,
  alpha_mismatch   BOOLEAN,

  parse_warnings   TEXT[],
  enrich_error     TEXT,
  raw              JSON            -- report minus `metadata`, `layers`, `rank`
);

CREATE TABLE IF NOT EXISTS lora_blocks (
  file_path   TEXT, group_name TEXT, family TEXT, label TEXT,
  block_index INTEGER, layers INTEGER,
  mean DOUBLE, max DOUBLE, median DOUBLE, outlier BOOLEAN
);

CREATE TABLE IF NOT EXISTS lora_layers (
  file_path   TEXT, base_name TEXT, group_name TEXT, family TEXT,
  label TEXT, block_index INTEGER, format TEXT,
  scale DOUBLE,
  nominal_rank INTEGER, effective_rank DOUBLE, balance DOUBLE,
  top1_energy DOUBLE, dominance DOUBLE, health TEXT,
  eff_scale DOUBLE, outlier BOOLEAN
);
```

`lora_blocks` keeps every `(group, index)`; `--top` only ever trimmed the terminal table.
Measured block rows per file range from 7 (SD1) to 49 (the LyCORIS fixture), so expect
roughly 10–200 per file.

`group` is a reserved-ish word; the column is `group_name` so ad-hoc queries need no
quoting.

Expected sizes for the headroom check: `lora_layers` about 500k–1M rows (~100 MB),
`lora_blocks` about 200k rows (~10 MB), the cache about 200 MB gzipped, on top of the
existing 1015 MB DB.

### Divergence rules

Four verdicts. All are `NULL`-safe: a null verdict means "cannot compare", never
"agrees".

**`type_mismatch`** — declared algorithm vs structural algorithm.

Both sides reduce to one of `lora`, `loha`, `lokr`, `oft`, `glora`, `glokr`, else null:

- Declared side, mapped from `network.type`: `LoRA`/`LoRAC3Lier`/`LoCon`/`LoRAFA`/`DyLoRA`
  → `lora`; `LoHa` → `loha`; `LoKr` → `lokr`; `OFT`/`DiagOFT`/`BOFT` → `oft`; `GLoRA` →
  `glora`; `GLoKr` → `glokr`.
- Actual side, from `network.structure.primary`: lowercased directly.

The mapping exists because LoCon and LoRA share up/down structure — the difference is
`has_conv`, which is its own column. Comparing raw type strings would flag every LoCon
file as a mismatch. DoRA is likewise a modifier (`has_dora`), not an algorithm.

**`rank_mismatch`** — `array_length(rank_actual) > 0 AND declared_rank IS NOT NULL AND
declared_rank <> ALL(rank_actual)`. Two guards, both load-bearing:

- Membership, not equality. Flux LoRAs legitimately declare a max dim while holding
  per-block ranks `[4, 8, 16]`, and `16 ∈ [4, 8, 16]` must not flag. A file declaring 8
  with actual `[16]` does flag.
- The empty-array guard. `x <> ALL('{}')` is **vacuously true**, so without it a file with
  no resolvable dims flags on every declared rank. `dims()` collects nothing for GLoRA
  (`glora_a1`), so this is reachable.

**`alpha_mismatch`** — the same two guards over `alpha_actual`, additionally ignoring
NULL entries in the actual array.

Store both sides of every comparison. The verdict columns are a convenience; the raw pair
is what a human checks when a flag fires.

### Declared scalars — and an existing bug to work around

`declared_alpha` **cannot** come from `loras.network_alpha`. That column is `INTEGER`
(`index.py:110`) and is filled by `to_int()` (`index.py:69-73, 83`), so `int("4.0")`
raises and stores NULL. Real files carry float strings (`ss_network_alpha: "4.0"`,
`"16.0"`, `"32.0"`), which means `network_alpha` is null for essentially every row today
and `alpha_mismatch` would be dead on arrival.

`enrich` therefore parses `declared_rank`, `declared_alpha` and the declared module/type
directly out of `loras.metadata`, with float-tolerant parsing. It does not modify `loras`.
Fixing `extract_common` to store alpha as `DOUBLE` is a sensible follow-up but out of
scope, and nothing here depends on it.

### Verdict columns

- `is_lora` — `array_length(rank_actual) > 0`. The type branch is deliberately not used:
  `network.type` is metadata-derived and should not be trusted to decide what a file is.
  The observed `lumina` LoRAs report `type: null` while holding `rank: [8]`, and this rule
  gets them right.
- `kind` — `'lora'` when `is_lora`; `'checkpoint'` when not, and `layers_total > 100`, and
  the declared architecture has no `/lora` suffix; `'artifact'` when not and
  `layers_total <= 100`; else `'unknown'`. This is what separates `pov_sd15-…`
  (checkpoint) and the `*-proj` files (artifact) from real LoRAs.

### Incremental behaviour

Re-enrich a file when any of these differ from the stored row:

- `file_mtime`
- `fingerprint` (see Cache)
- `schema_version`
- the row is missing entirely

`--force` re-enriches everything regardless.

### Convenience views

```sql
CREATE OR REPLACE VIEW v_runs AS ...   -- one row per run_name: file counts, summed size,
                                       -- kind mix, min/max training_date
CREATE OR REPLACE VIEW v_issues AS ... -- rows where type_mismatch, rank_mismatch,
                                       -- alpha_mismatch, kind <> 'lora',
                                       -- block_outliers > 10, parse_warnings <> [],
                                       -- or rank_weak_or_worse > rank_layers / 2
```

`v_issues` is the payoff: "what is wrong in here" becomes `SELECT * FROM v_issues`.

### Extending `stats`

`index.py stats` gains sections for `type_actual`, `kind`, `is_lora`, `precision`, and the
rank-health distribution, so the existing summary command answers the new questions too.

## Verification

- Stage 1a, `insta` snapshots of `network.structure`:
  - `boo.safetensors` → `primary: LoRA`, `has_conv: false`
  - `ahetableh_9_lokr.safetensors` → `primary: LoKr` (and note: metadata declares
    `lycoris.kohya` + `algo=lokr`, so declared and actual agree here — a snapshot that
    asserts agreement is as valuable as one that asserts divergence)
  - `lora_unet_down_blocks_1_resnets_1_conv2.safetensors` → `has_conv: true`
  - a fixture whose metadata disagrees with its keys. If none exists in the repo, synth
    one by copying a LoHa file's tensors under LoRA-shaped metadata keys — the whole
    point is that this case is invisible today.
- Stage 1b, `insta` snapshots of `layers`, one per network family, using the fixtures
  already in the repo.
- Stage 1b invariants, one per notion — the three-way split is the point:
  - `COUNT(*)` of `layers` == `network.layers.total`
  - rows with `scale IS NULL` == `skipped.layers`
  - rows with `health IS NOT NULL` == `len(rank)`
  - on the conv fixture: 1 layer row, `scale: null`, `health: "Good"`, `skipped: 1`
- Stage 1: `--layers` alone leaves `rank: null`, leaves `blocks` and `summary`
  byte-identical to a run without it, and keeps `schema_version` at 1.
- Stage 1: `--layers --no-blocks` emits `blocks: null` with `load: "full"`.
- Stage 2: enrich a 10-file sample and assert per file that
  `COUNT(*) FROM lora_layers WHERE file_path = ?` == `layers_total`, and that
  `COUNT(*) FROM lora_blocks` == `len(blocks)` in the cached JSON.
- Stage 2: add a row, delete the file, re-run `enrich` — the row must still be there
  and nothing may be deleted.
- Stage 2: re-running `enrich` with no changes touches zero rows.

## Left to the builder

- Thread-pool sizing beyond the bytes-in-flight cap, and progress output style — match the
  existing script's plain `print` counters.
- The rule `v_runs` uses to pick a "final" file per run directory: names carry
  `-000028`-style step suffixes, `-proj` artifacts and checkpoints share the directory.
  `MAX(file_mtime)` or highest step number are both defensible.
- Cache directory naming and whether `--cache-dir` is worth a flag.
- Whether `raw` should keep the `blocks` array, given `lora_blocks` already holds it.
- Whether `structure` should also count layers per format *per family* (UNet vs
  TextEncoder), or whether `lora_layers.format` grouped by family is enough.
