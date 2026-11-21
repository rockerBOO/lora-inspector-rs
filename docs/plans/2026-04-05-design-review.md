# Design Review: LoRA Inspector UI

**Date:** 2026-04-05\
**Branch:** layout-review\
**Scope:** Frontend redesign of `crates/lora-inspector-wasm/`

---

## Current State Analysis

### Visual Identity

The app has a strong dark teal/monospace aesthetic that suits its technical audience well. Color palette:

- Background: `#020f0f`
- Accent background: `#021e19`
- Accent color: `#2aaa93` (teal)
- Text: `#ccf8d5` / `#9ddfab` (green)
- Font: monospace throughout

The landing page (file drop zone with background image) is clean and purposeful. The aesthetic is worth keeping.

### Landing page structure

Three-panel header: `[file dropbox] | [Compare LoRA] | [Support]`

### Post-load structure (problems)

After loading a file, all content renders as a single unstructured vertical scroll. The `Main.jsx` component renders 13 child components sequentially with no visual grouping or labeling. There is no way to know where "network" ends and "training" begins.

---

## Identified Issues

### 1. No section structure

All components render as flat `div.row.space-apart` CSS grids with no labels, dividers, or grouping. A user cannot scan the page to find a specific category of information.

### 2. Empty/null attributes render labels without values

`MetaAttribute` always renders its label (`caption` div) even when `value` is `undefined` or `null`. This produces rows like:

```
Date          Title         Author        Prediction type
Invalid Date  (blank)       (blank)       (blank)
```

Most LoRAs do not have modelspec fields (title, author, license, etc.), so these rows are nearly always blank.

**Decision:** Show `—` (em-dash) for keys that are absent from metadata. If a key is present but explicitly set to the string `"None"` or `"null"`, show that value as-is — it is meaningful (the training tool explicitly stored it).

### 3. Training comment duplicated

`ModelSpec.jsx` renders `ss_training_comment` twice: once in the `training` array (lines 37–46) and once in the main return block (lines 123–130).

### 4. "Invalid Date" on missing modelspec date

`new Date(metadata.get("modelspec.date")).toLocaleString()` evaluates to `"Invalid Date"` when the key is absent. Needs a `metadata.has()` guard.

### 5. Optimizer section is too thin

`Optimizer.jsx` only shows optimizer name and seed. Related fields — CLIP skip, max grad norm, scale weight norms, precision settings (fp16/fp8/mixed), gradient checkpointing — are scattered across adjacent components. Logically they all belong together.

### 6. Training timestamps are low priority

`ss_training_started_at`, `ss_training_finished_at`, and elapsed time currently render at the very top of the page. These are lower-priority metadata and should appear lower within the Metadata section.

---

## Proposed Design

### Page Layout

```
┌──────────────────────────────────────────────────────────┐
│  [file dropbox area]  │  [Compare LoRA]  │  [Support]   │  always visible
├──────────────────────────────────────────────────────────┤
│  boo.safetensors                        [Show raw]       │  scrolls with page
│  Metadata │ Network │ Training │ Optimizer │ Dataset │ ↓ │  becomes sticky on scroll
├──────────────────────────────────────────────────────────┤
│  ## Metadata                                             │
│    ...                                                   │
│  ## Network                                              │
│    ...                                                   │
│  (etc.)                                                  │
└──────────────────────────────────────────────────────────┘
```

### Sticky nav behavior

- Initially the filename row (`boo.safetensors [Show raw]`) and section nav row are visible below the header.
- Once the user scrolls past the filename row, the filename + section nav pin to the top as a compact sticky bar.
- "Show raw" is **not** carried into the sticky bar — it is low priority and only needed once.
- The active section is highlighted in the nav via `IntersectionObserver` scroll-spy.

### Section structure

| Section       | Content                                                                                                                                                                                                                                                                                                         |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Metadata**  | Trigger phrase, SD model name, model hash, session ID, sd-scripts commit hash, architecture, title, author, description, license, tags, prediction type, implementation. Training timestamps + comment lower in the section.                                                                                    |
| **Network**   | Module, type, rank, alpha, dropout, module dropout, rank dropout, DoRA weight decomposition, rank-stabilized, network args. Sub-components: LoRANetwork / LoKr / BOFT / DiagOFT                                                                                                                                 |
| **Training**  | LR scheduler, warmup steps, UNet LR, text encoder LR, epochs, steps, max train steps, batch size, gradient accumulation steps, noise offset, adaptive noise scale, IP noise gamma, multires noise, loss type, debiased estimation, min SNR gamma, zero terminal SNR, masked loss, wavelet loss, caption dropout |
| **Optimizer** | Optimizer name, seed, CLIP skip, max grad norm, scale weight norms, precision (fp16 / fp8 base), mixed precision, full fp16, gradient checkpointing                                                                                                                                                             |
| **Dataset**   | Num train images, num batches per epoch, subsets, bucket info, tag frequency                                                                                                                                                                                                                                    |
| **Advanced**  | Key counts (base names, text encoder, unet, all), block weights (Get block weights), statistics (Calculate statistics), effective rank (Calculate effective rank)                                                                                                                                               |

### Value display rules

| Condition                                         | Display       |
| ------------------------------------------------- | ------------- |
| Key not present in metadata                       | `—` (em-dash) |
| Key present, value is string `"None"` or `"null"` | Show as-is    |
| Key present, value is any other value             | Show normally |

**Rationale:** "Not set" is meaningful information. It tells the reader either that the training tool used its default, or that it did not expose that option. Hiding absent fields would obscure this signal.

---

## Implementation Notes

### Sticky nav implementation

Use CSS `position: sticky` on the nav bar combined with an `IntersectionObserver` to detect which section is currently in the viewport. No JavaScript scroll listeners needed.

```
<div class="file-header">          ← scrolls away
  <h1>boo.safetensors</h1>
  <button>Show raw</button>
</div>
<nav class="section-nav sticky">   ← pins to top
  <a href="#metadata">Metadata</a>
  <a href="#network">Network</a>
  ...
</nav>
<main>
  <section id="metadata"><h2>Metadata</h2> ...</section>
  <section id="network"><h2>Network</h2> ...</section>
  ...
</main>
```

### Component reorganization

- Move CLIP skip, max grad norm, scale weight norms, and all precision fields from their current locations into an **Optimizer** section component.
- Create a `Section` wrapper component that renders a semantic `<section>` with an `<h2>` heading.
- Fix `ModelSpec`: remove duplicate training comment, guard `modelspec.date` with `metadata.has()`.
- Update `MetaAttribute` to accept an explicit `notSetLabel` prop (default `"—"`) rendered when value is nullish.

### CSS additions

- Section headings: left border accent, or underline with `--accent-color`
- Sticky nav: `position: sticky; top: 0; z-index: 10; background: var(--bg-color)`
- Active section link in nav: highlighted with `--accent-color-b` (yellow-green)
- Use CSS nesting to scope section-level styles

---

## Ground Truth: boo.safetensors Metadata

Extracted from the live app after loading the file. This is the reference dataset for tests and visual verification.

### Present values (non-empty)

| Field                       | Value                                                                                                                                                          |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Started                     | Mon Feb 20 2023 04:22:16 GMT-0500                                                                                                                              |
| Training comment            | "dimension is resized from 128 to 4; Trained by: rockerBOO. Identifier: cp-ede Class: anime Learning Rate: 1.0, Batch Size: 1, Gradient Accumulation Steps: 1" |
| SD model name               | v1-5-pruned-emaonly.ckpt                                                                                                                                       |
| Model hash                  | 74b62bfc40e092a4…9731fcfe16 (full 64-char hash)                                                                                                                |
| Legacy model hash           | 9e689848                                                                                                                                                       |
| Session ID                  | 682850588                                                                                                                                                      |
| sd-scripts commit hash      | b612d0b091213f39…1e1974d7                                                                                                                                      |
| Network module              | kohya-ss/lora                                                                                                                                                  |
| Network type                | LoRA                                                                                                                                                           |
| Network Rank/Dimension      | 4                                                                                                                                                              |
| Network Alpha               | 4.0                                                                                                                                                            |
| Module dropout              | None                                                                                                                                                           |
| Rank dropout                | None                                                                                                                                                           |
| Weight decomposition (DoRA) | False                                                                                                                                                          |
| Rank-stabilized             | False                                                                                                                                                          |
| Gradient Checkpointing      | False                                                                                                                                                          |
| Network args                | None                                                                                                                                                           |
| LR Scheduler                | cosine                                                                                                                                                         |
| Warmup steps                | 10                                                                                                                                                             |
| UNet learning rate          | 1.0                                                                                                                                                            |
| Text encoder learning rate  | 0.5                                                                                                                                                            |
| Optimizer                   | dadaptation.dadapt_adam.DAdaptAdam                                                                                                                             |
| Seed                        | 42                                                                                                                                                             |
| CLIP Skip                   | 1                                                                                                                                                              |
| Precision                   | fp16                                                                                                                                                           |
| Mixed precision             | fp16                                                                                                                                                           |
| Full fp16                   | False                                                                                                                                                          |
| Epoch                       | 1                                                                                                                                                              |
| Max Train Steps             | 20000                                                                                                                                                          |
| Num train images            | 30300                                                                                                                                                          |
| Num batches per epoch       | 60600                                                                                                                                                          |
| Batch size                  | 1                                                                                                                                                              |
| Gradient Accumulation Steps | 1                                                                                                                                                              |
| Noise offset                | -0.1                                                                                                                                                           |
| Debiased Estimation         | False                                                                                                                                                          |
| Max token length            | None                                                                                                                                                           |

### Absent from metadata (should display `—`)

These keys were checked but not found in the file:

- `modelspec.date`, `modelspec.title`, `modelspec.author`, `modelspec.prediction_type`
- `modelspec.license`, `modelspec.description`, `modelspec.architecture`
- `modelspec.implementation`, `modelspec.trigger_phrase`, `modelspec.tags`
- `ss_network_dropout`, `ss_max_grad_norm`, `ss_scale_weight_norms`, `ss_fp8_base`
- `ss_steps`, `ss_ip_noise_gamma`, `ss_adaptive_noise_scale`
- `ss_multires_noise_iterations`, `ss_multires_noise_discount`
- `ss_loss_type`, `ss_min_snr_gamma`, `ss_zero_terminal_snr`, `ss_masked_loss`
- `ss_caption_dropout_rate`, `ss_caption_dropout_every_n_epochs`, `ss_caption_tag_dropout_rate`

**Key observation:** This LoRA has no modelspec fields at all (title, author, etc.) and no dataset/bucket metadata. These are common absences — the design must handle them gracefully without looking broken.

---

## Testing Process

### Stack

- **Unit/component tests**: Vitest + React Testing Library (jsdom), existing pattern in `tests/*.test.jsx`
- **E2E tests**: Playwright in Docker (`yarn e2e-test` using `PW_TEST_CONNECT_WS_ENDPOINT`)
- **Visual review**: Playwright MCP tool used sparingly during development for key milestones

### Unit test approach (Vitest + RTL)

Use `boo.safetensors` metadata as the canonical test fixture. Build a shared mock metadata Map:

```js
// tests/fixtures/boo-metadata.js
export const booMetadata = new Map([
  ["ss_training_started_at", "1676884936"],
  ["ss_training_comment", "dimension is resized from 128 to 4; ..."],
  ["ss_sd_model_name", "v1-5-pruned-emaonly.ckpt"],
  ["ss_network_module", "kohya-ss/lora"],
  ["ss_network_dim", "4"],
  ["ss_network_alpha", "4.0"],
  ["ss_lr_scheduler", "cosine"],
  ["ss_lr_warmup_steps", "10"],
  ["ss_unet_lr", "1.0"],
  ["ss_text_encoder_lr", "0.5"],
  ["ss_optimizer", "dadaptation.dadapt_adam.DAdaptAdam"],
  ["ss_seed", "42"],
  ["ss_clip_skip", "1"],
  ["ss_mixed_precision", "fp16"],
  ["ss_full_fp16", "False"],
  ["ss_epoch", "1"],
  ["ss_max_train_steps", "20000"],
  ["ss_num_train_images_per_concept", "30300"],
  ["ss_num_batches_per_epoch", "60600"],
  ["ss_batch_size_per_gpu", "1"],
  ["ss_gradient_accumulation_steps", "1"],
  ["ss_noise_offset", "-0.1"],
  // ... absent keys intentionally omitted
]);
```

For each component under change, write tests that:

1. Render with `booMetadata` — assert present values appear
2. Render with a minimal/empty Map — assert `—` appears for absent fields
3. Render with a Map where a key is `"None"` — assert `"None"` appears as-is (not `—`)

### What to test per component

| Component           | Key assertions                                                                      |
| ------------------- | ----------------------------------------------------------------------------------- |
| `MetaAttribute`     | `—` when value is null/undefined; shows "None" when value is string "None"          |
| `ModelSpec`         | No "Invalid Date"; training comment appears once; absent modelspec fields show `—`  |
| Section nav         | All 6 section links render; active section changes on scroll (IntersectionObserver) |
| `Metadata` section  | Shows SD model name, hash, trigger phrase; timestamps at bottom                     |
| `Optimizer` section | Shows optimizer, seed, CLIP skip, precision fields together                         |

### E2E test approach (Playwright / Docker)

E2E tests live in `e2e-tests/`. Run via:

```
yarn e2e-test
# which expands to:
PW_TEST_CONNECT_WS_ENDPOINT=ws://127.0.0.1:3030/ BASE_URL=http://hostmachine:4173 npx playwright@1.51.1 test
```

E2E tests cover:

1. File upload → page renders with filename visible
2. Section nav renders and all 6 links are present
3. Scrolling past the filename causes nav to become sticky (check `position: sticky` is active)
4. Clicking a nav link scrolls to the correct `<section>`
5. Present values from boo.safetensors are visible in the correct sections
6. Absent fields show `—` not a blank space

### Playwright MCP (visual review)

Used only at section completion milestones, not after every change:

- After sticky nav is implemented
- After each major section is reorganized
- After the full layout is complete

Use `browser_snapshot` (accessibility tree) as first check; only `browser_take_screenshot` when visual layout needs confirmation.

---

## Out of Scope (this review)

- Dark/light mode toggle
- Comparison view layout (separate concern)
- Mobile layout (existing responsive grid handles basics)
- Any new data/analysis features
