# Network Module Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report the toolkit family and network module of a LoRA from evidence in the file, so MOSS-Image (`networks.lora_generic`) and every future module stop showing `no_module_found`.

**Architecture:** A `NetworkFamily` enum plus a detection chain in `Metadata` (metadata string first, `modelspec.implementation*` next, key-shape `LoRAFormat` last), and a `network_module_display()` that owns all display strings. Both live in the `inspector` crate so they are covered by fast `cargo test`; the wasm worker only delegates. `Network.jsx` gains a family row and keeps its module row.

**Tech Stack:** Rust (`inspector` crate, serde), wasm-bindgen worker, React + Vitest, AVA, Playwright.

**Spec:** `docs/superpowers/specs/2026-11-20-network-module-identification-design.md`

## Global Constraints

- Known modules keep their exact current strings: `kohya-ss/lora`, `kohya-ss/lora_flux`, `musubi-tuner/lora_flux_2`, `kohya-ss/lora_lumina`, `kohya-ss/lora_sd3`, `musubi-tuner/lora_krea2`, `kohya-ss/lora_fa`, `kohya-ss/dylora`, `kohya-ss/oft`, `lycoris`.
- `#network-module` in `e2e-tests/index.spec.js:33,145` asserts `kohya-ss/lora`; that text must not change.
- Unmapped modules get no `NetworkType` (`network_type()` returns `None`). Only `lora_generic` joins the typing table, as `NetworkType::LoRA`.
- `LoRAFormat::Kohya` is the fallback format and is **not** evidence of the kohya family; only `Peft` and `Lycoris` formats are.
- No changes to weight/norm math or to any key parser.
- All display-string logic lives in `crates/inspector/src/metadata.rs`, never in `worker.rs`: the wasm test suite fetches fixtures over the network from a Linode bucket, so anything only covered there is effectively untested in CI-less runs.

## Review Focus

- Empty `ss_network_module` value (`""`) — must resolve to unknown, never to `kohya-ss/`.
- `ss_network_module` with an unknown vendor prefix (e.g. `foo.bar`) — must resolve to unknown family and `unknown` display, without panicking.
- Metadata present but with neither `ss_network_module` nor `modelspec.implementation*` — must resolve to unknown.
- A PEFT file with no metadata at all — family from `LoRAFormat::Peft` is `Peft`, display `peft`.
- `modelspec.implementation` holding an unrecognised URL — unknown, not guessed.

---

### Task 1: `NetworkFamily` and the metadata-only detection steps

**Files:**

- Modify: `crates/inspector/src/network.rs` (add enum after `NetworkModule`, ~line 30)
- Modify: `crates/inspector/src/metadata.rs:99-115` (add methods beside `network_module()`)
- Test: `crates/inspector/src/metadata.rs` (`#[cfg(test)] mod tests`, after `musubi_tuner_flux2_network_module`)

**Interfaces:**

- Consumes: nothing.
- Produces:
  - `pub enum NetworkFamily { Kohya, MusubiTuner, Lycoris, Diffusers, Peft, Unknown }` deriving `Deserialize, Serialize, Clone, Debug, PartialEq`
  - `impl NetworkFamily { pub fn display(&self) -> &'static str }` returning `"kohya-ss"`, `"musubi-tuner"`, `"lycoris"`, `"diffusers"`, `"peft"`, `"unknown"`
  - `Metadata::network_family(&self) -> NetworkFamily` — steps 1–2 only
  - `Metadata::network_module_name(&self) -> Option<String>` — the module without its vendor prefix

- [ ] **Step 1: Write the failing tests** (add to `metadata.rs`'s test module)

```rust
fn meta(pairs: &[(&str, &str)]) -> Metadata {
    Metadata {
        metadata: Some(
            pairs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        ),
    }
}

#[test]
fn network_family_from_module_string() {
    use crate::network::NetworkFamily;

    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_generic")]).network_family(),
        NetworkFamily::Kohya
    );
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_anima")]).network_family(),
        NetworkFamily::Kohya
    );
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_flux_2")]).network_family(),
        NetworkFamily::MusubiTuner
    );
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_krea2")]).network_family(),
        NetworkFamily::MusubiTuner
    );
    assert_eq!(
        meta(&[("ss_network_module", "lycoris.kohya")]).network_family(),
        NetworkFamily::Lycoris
    );
}

#[test]
fn network_family_from_implementation() {
    use crate::network::NetworkFamily;

    assert_eq!(
        meta(&[(
            "modelspec.implementation_version",
            "sd-scripts/11d8cedd0c67b93a10f8b2d75f43554f6f12c794",
        )])
        .network_family(),
        NetworkFamily::Kohya
    );
    assert_eq!(
        meta(&[(
            "modelspec.implementation",
            "https://github.com/OpenMOSS/MOSS-Image",
        )])
        .network_family(),
        NetworkFamily::Kohya
    );
    assert_eq!(
        meta(&[(
            "modelspec.implementation",
            "https://github.com/huggingface/diffusers",
        )])
        .network_family(),
        NetworkFamily::Diffusers
    );
}

#[test]
fn network_family_unknown_cases() {
    use crate::network::NetworkFamily;

    assert_eq!(meta(&[]).network_family(), NetworkFamily::Unknown);
    assert_eq!(
        meta(&[("ss_network_module", "")]).network_family(),
        NetworkFamily::Unknown
    );
    assert_eq!(
        meta(&[("ss_network_module", "foo.bar")]).network_family(),
        NetworkFamily::Unknown
    );
    assert_eq!(
        meta(&[(
            "modelspec.implementation",
            "https://github.com/nobody/some-trainer",
        )])
        .network_family(),
        NetworkFamily::Unknown
    );
    assert_eq!(
        Metadata { metadata: None }.network_family(),
        NetworkFamily::Unknown
    );
}

#[test]
fn network_module_name_strips_vendor_prefix() {
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_generic")]).network_module_name(),
        Some("lora_generic".to_string())
    );
    assert_eq!(
        meta(&[("ss_network_module", "lycoris.kohya")]).network_module_name(),
        Some("kohya".to_string())
    );
    assert_eq!(meta(&[]).network_module_name(), None);
    assert_eq!(meta(&[("ss_network_module", "")]).network_module_name(), None);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inspector metadata::tests::network_family 2>&1 | tail -20`
Expected: compile error — `NetworkFamily` not found.

- [ ] **Step 3: Add `NetworkFamily` to `crates/inspector/src/network.rs`**

Place it directly after the `NetworkModule` enum with derives `Deserialize, Serialize, Clone, Debug, PartialEq` (no `wasm_bindgen`), plus `display()`.

- [ ] **Step 4: Add `network_family()` and `network_module_name()` to `crates/inspector/src/metadata.rs`**

`network_module_name()`: read `ss_network_module`, trim, return `None` when empty; when the value has two or more `.`-separated segments, return everything after the first (`.split_once('.')` on the remainder), otherwise the whole string.

`network_family()`, first match wins:

1. `ss_network_module` starting with `networks.` → `MusubiTuner` when the module is `lora_flux_2` or `lora_krea2`, else `Kohya`
2. `ss_network_module` starting with `lycoris.` → `Lycoris`
3. `modelspec.implementation_version` starting with `sd-scripts/` → `Kohya`
4. `modelspec.implementation` containing `kohya-ss/sd-scripts` or `OpenMOSS/MOSS-Image` → `Kohya`; containing `diffusers` → `Diffusers`
5. otherwise `Unknown`

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p inspector metadata::tests 2>&1 | tail -20`
Expected: the 4 new tests pass and pre-existing metadata tests still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/inspector/src/network.rs crates/inspector/src/metadata.rs
git commit -m "Add NetworkFamily and metadata-based family detection"
```

---

### Task 2: Key-shape fallback, display strings, and `lora_generic` typing

**Files:**

- Modify: `crates/inspector/src/network.rs` (`NetworkModule`: add `KohyaSSLoRAGeneric`)
- Modify: `crates/inspector/src/metadata.rs:46-115`
- Test: `crates/inspector/src/metadata.rs` (`mod tests`)

**Interfaces:**

- Consumes: `NetworkFamily`, `network_family()`, `network_module_name()` from Task 1.
- Produces:
  - `Metadata::resolve_network_family(&self, format: &crate::weight::LoRAFormat) -> NetworkFamily`
  - `Metadata::network_module_display(&self, format: &crate::weight::LoRAFormat) -> String` — every display rule, including the removal of `no_module_found`
  - `Metadata::network_module(&self) -> Option<NetworkModule>` also returns `Some(NetworkModule::KohyaSSLoRAGeneric)` for `networks.lora_generic`

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn lora_generic_maps_to_kohya_module_and_lora_type() {
    use crate::network::{NetworkModule, NetworkType};

    let m = meta(&[("ss_network_module", "networks.lora_generic")]);
    assert_eq!(m.network_module(), Some(NetworkModule::KohyaSSLoRAGeneric));
    assert_eq!(m.network_type(), Some(NetworkType::LoRA));
}

#[test]
fn unmapped_kohya_module_is_not_typed() {
    let m = meta(&[("ss_network_module", "networks.lora_anima")]);
    assert_eq!(m.network_module(), None);
    assert_eq!(m.network_type(), None);
}

#[test]
fn resolve_family_uses_format_only_as_last_resort() {
    use crate::network::NetworkFamily;
    use crate::weight::LoRAFormat;

    assert_eq!(
        meta(&[]).resolve_network_family(&LoRAFormat::Peft),
        NetworkFamily::Peft
    );
    assert_eq!(
        meta(&[]).resolve_network_family(&LoRAFormat::Lycoris),
        NetworkFamily::Lycoris
    );
    // LoRAFormat::Kohya is the fallback format, not evidence.
    assert_eq!(
        meta(&[]).resolve_network_family(&LoRAFormat::Kohya),
        NetworkFamily::Unknown
    );
    // Metadata outranks the format.
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_generic")])
            .resolve_network_family(&LoRAFormat::Peft),
        NetworkFamily::Kohya
    );
}

#[test]
fn module_display_keeps_known_strings() {
    use crate::weight::LoRAFormat;

    assert_eq!(
        meta(&[("ss_network_module", "networks.lora")]).network_module_display(&LoRAFormat::Kohya),
        "kohya-ss/lora"
    );
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_krea2")])
            .network_module_display(&LoRAFormat::Kohya),
        "musubi-tuner/lora_krea2"
    );
    assert_eq!(
        meta(&[("ss_network_module", "lycoris.kohya")]).network_module_display(&LoRAFormat::Lycoris),
        "lycoris"
    );
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_generic")])
            .network_module_display(&LoRAFormat::Kohya),
        "kohya-ss/lora_generic"
    );
}

#[test]
fn module_display_never_says_no_module_found() {
    use crate::weight::LoRAFormat;

    // Unmapped module under a known vendor prefix keeps the family prefix.
    assert_eq!(
        meta(&[("ss_network_module", "networks.lora_anima")])
            .network_module_display(&LoRAFormat::Kohya),
        "kohya-ss/lora_anima"
    );
    // Family known, module unknown.
    assert_eq!(
        meta(&[("modelspec.implementation", "https://github.com/huggingface/diffusers")])
            .network_module_display(&LoRAFormat::Kohya),
        "diffusers"
    );
    // Nothing known, but the key shape is PEFT.
    assert_eq!(
        Metadata { metadata: None }.network_module_display(&LoRAFormat::Peft),
        "peft"
    );
    // Nothing known and the format is the fallback.
    assert_eq!(
        Metadata { metadata: None }.network_module_display(&LoRAFormat::Kohya),
        "unknown"
    );
    // Empty and unrecognised vendor prefixes are unknown, not "kohya-ss/".
    assert_eq!(
        meta(&[("ss_network_module", "")]).network_module_display(&LoRAFormat::Kohya),
        "unknown"
    );
    assert_eq!(
        meta(&[("ss_network_module", "foo.bar")]).network_module_display(&LoRAFormat::Kohya),
        "unknown"
    );
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p inspector metadata::tests::module_display 2>&1 | tail -20`
Expected: compile error — `KohyaSSLoRAGeneric` / `network_module_display` not found.

- [ ] **Step 3: Implement**

- `NetworkModule::KohyaSSLoRAGeneric` after `KohyaSSLoRA`.
- `metadata.rs:110` table: `"networks.lora_generic" => Some(NetworkModule::KohyaSSLoRAGeneric),`
- `network_type()` (~line 74): `Some(NetworkModule::KohyaSSLoRAGeneric) => Some(NetworkType::LoRA),`
- `resolve_network_family(&self, format)`: return `self.network_family()` unless `Unknown`; then `Peft` → `Peft`, `Lycoris` → `Lycoris`, `Kohya` → `Unknown`.
- `network_module_display(&self, format)`: `network_module()` hit → its existing string (`lycoris` for `NetworkModule::Lycoris`, matching today's output); no hit and `network_module_name()` is `Some(m)` with family `Kohya`/`MusubiTuner`/`Lycoris` → `format!("{}/{}", family.display(), m)`; family known without a module → `family.display()`; otherwise `"unknown"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p inspector 2>&1 | tail -10`
Expected: all inspector tests pass, including the pre-existing `musubi_tuner_flux2_network_module`, `no_network_type`, `no_meta_network_type`.

- [ ] **Step 5: Commit**

```bash
git add crates/inspector/src/network.rs crates/inspector/src/metadata.rs
git commit -m "Type lora_generic as LoRA and own module display strings in inspector"
```

---

### Task 3: Worker delegation

**Files:**

- Modify: `crates/lora-inspector-wasm/src/worker.rs:279-294`

**Interfaces:**

- Consumes: `Metadata::network_module_display`, `Metadata::resolve_network_family`, `NetworkFamily::display`.
- Produces on `LoraWorker`:
  - `network_module(&self) -> String` → `self.metadata.network_module_display(&self.file.format())`
  - `network_family(&self) -> String` → `self.metadata.resolve_network_family(&self.file.format()).display().to_string()`

- [ ] **Step 1: Replace the match block in `network_module()` with the delegation, and add `network_family()`**

The existing `match` and its `"no_module_found"` arm are deleted; both methods become one-liners per Interfaces.

- [ ] **Step 2: Verify it compiles for the browser target**

Run: `cargo check -p lora-inspector-wasm --target wasm32-unknown-unknown 2>&1 | tail -5`
Expected: `Finished`, no errors, no warnings about unused variants.

- [ ] **Step 3: Commit**

```bash
git add crates/lora-inspector-wasm/src/worker.rs
git commit -m "Delegate network module and family reporting to the inspector crate"
```

---

### Task 4: Family row in the UI, unit tests, and one end-to-end assertion

**Files:**

- Modify: `crates/lora-inspector-wasm/assets/js/worker.js:220-229, 508-510` (message routing + helper)
- Modify: `crates/lora-inspector-wasm/assets/js/components/network/Network.jsx:9-40, 75-90`
- Modify: `crates/lora-inspector-wasm/tests/network-components.test.jsx`
- Modify: `crates/lora-inspector-wasm/e2e-tests/index.spec.js:33`

**Interfaces:**

- Consumes: wasm `network_family()` (Task 3) and the existing `network_module()`.
- Produces: worker message `network_family` replying `{ messageType: "network_family", networkFamily }`; DOM id `#network-family`; `#network-module` text unchanged for known modules.

- [ ] **Step 1: Write the failing tests**

```jsx
it("renders the network family alongside the module", async () => {
    const { Network } = await import(
        "../assets/js/components/network/Network.jsx"
    );

    render(
        <Network
            metadata={
                new Map([
                    ["ss_network_module", "networks.lora_generic"],
                    ["ss_network_dim", "8"],
                    ["ss_network_alpha", "16"],
                ])
            }
            filename="moss.safetensors"
            worker={createMockWorker({
                networkModule: "kohya-ss/lora_generic",
                networkFamily: "kohya-ss",
            })}
        />,
    );

    expect(screen.getByText("Network family")).toBeDefined();
    expect(screen.getByText("kohya-ss")).toBeDefined();
    expect(screen.getByText("kohya-ss/lora_generic")).toBeDefined();
});

it("shows the family alone when the module is unknown", async () => {
    const { Network } = await import(
        "../assets/js/components/network/Network.jsx"
    );

    render(
        <Network
            metadata={new Map([["ss_network_dim", "8"]])}
            filename="diffusers.safetensors"
            worker={createMockWorker({
                networkModule: "diffusers",
                networkFamily: "diffusers",
            })}
        />,
    );

    expect(screen.getByText("diffusers")).toBeDefined();
});

it("shows unknown for a file with no metadata", async () => {
    const { Network } = await import(
        "../assets/js/components/network/Network.jsx"
    );

    render(
        <Network
            metadata={new Map()}
            filename="bare.safetensors"
            worker={createMockWorker({
                networkModule: "unknown",
                networkFamily: "unknown",
            })}
        />,
    );

    expect(screen.getAllByText("unknown").length).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd crates/lora-inspector-wasm && yarn test:vitest tests/network-components.test.jsx 2>&1 | tail -15`
Expected: FAIL — "Network family" not found / mock lacks `network_family`.

- [ ] **Step 3: Implement**

`assets/js/worker.js`: add `else if (e.data.messageType === "network_family")` beside the `network_module` branch, replying with `networkFamily`, and the helper `async function getNetworkFamily(e) { return withWorker(e.data.name, (w) => w.network_family()); }`.

`Network.jsx`: add `const [networkFamily, setNetworkFamily] = useState("unknown");`, a `trySyncMessage({ messageType: "network_family", name: filename }, worker)` in the existing effect, and a `MetaAttribute` with `containerProps={{ id: "network-family" }}`, `name="Network family"`, `valueClassName="attribute"`, placed **before** the `Network module` attribute. Leave the module attribute untouched.

`createMockWorker`: accept an overrides object and answer `network_module` / `network_family` from it, defaulting to `"kohya-ss/lora"` / `"kohya-ss"` so existing tests keep passing.

`e2e-tests/index.spec.js:33`: after the existing `#network-module` assertion, add
`await expect(page.locator("#network-family")).toContainText("kohya-ss");`

- [ ] **Step 4: Run the JS suites**

Run: `cd crates/lora-inspector-wasm && yarn test:vitest 2>&1 | tail -5 && yarn test:ava 2>&1 | tail -3`
Expected: all pass; the existing `#network-module` → `kohya-ss/lora` e2e assertion still holds.

- [ ] **Step 5: Commit**

```bash
git add crates/lora-inspector-wasm/assets/js/worker.js crates/lora-inspector-wasm/assets/js/components/network/Network.jsx crates/lora-inspector-wasm/tests/network-components.test.jsx crates/lora-inspector-wasm/e2e-tests/index.spec.js
git commit -m "Show network family in the network panel"
```

---

### Task 5: Whole-workspace verification

**Files:**

- Add: `docs/superpowers/specs/2026-11-20-network-module-identification-design.md`
- Add: `docs/superpowers/plans/2026-11-20-network-module-identification.md`

- [ ] **Step 1: Format**

Run: `make fmt`
Expected: no changes (`git status --short` shows only the two doc files as untracked).

- [ ] **Step 2: Full test gate**

Run: `make test`
Expected: cargo tests, wasm-bindgen tests, `yarn test:ava` and `yarn test:vitest` all pass.

- [ ] **Step 3: Verify against the real file**

Run: `make dev-wasm` (or reuse the running dev server at http://localhost:5173/) and load `/mnt/900/lora/testing/mj-base-test-moss/mj-base-test-moss-1.safetensors`.
Expected: `Network family` = `kohya-ss`, `Network module` = `kohya-ss/lora_generic`, and no `no_module_found` anywhere in the console or panel.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/specs/2026-11-20-network-module-identification-design.md docs/superpowers/plans/2026-11-20-network-module-identification.md
git commit -m "Document network module identification design and plan"
```
