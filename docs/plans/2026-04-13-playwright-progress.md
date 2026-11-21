# Playwright Remote Container Testing - Progress Notes

Date: 2026-04-13

## Goal

Run the Playwright E2E test suite using a Docker container for the browser, with the
Vite dev server running on the host machine (switched from preview to dev server).

## Infrastructure

- **Playwright container**: `mcr.microsoft.com/playwright:v1.51.1-noble`
- **Start script**: `start-playwright-server.sh` — runs the container with `npx -y playwright@1.51.1 run-server --port 3000 --host 0.0.0.0`
- **Port mapping**: `-p 3030:3000` — host port **3030**, container port 3000
- **Docker flag**: `--add-host=hostmachine:host-gateway` — from inside the container, `hostmachine` resolves to the host machine IP

## Problems Found and Fixed

### 1. Makefile port mismatch

`Makefile` uses `ws://127.0.0.1:3000/` but the container is mapped to host port **3030**.

**Fix**: Use `ws://127.0.0.1:3030/` when connecting from the host.

The `e2e-test` script in `package.json` already has the correct port:

```
"e2e-test": "PW_TEST_CONNECT_WS_ENDPOINT=ws://127.0.0.1:3030/ BASE_URL=http://hostmachine:4173 npx playwright@1.51.1 test"
```

### 2. Playwright version mismatch

`package.json` had `"@playwright/test": "^1.51.1"` which yarn resolved to **v1.54.2**.
The container server is **v1.51.1**. Client and server must match exactly.

**Fix**: Pin to exact version — `"@playwright/test": "1.51.1"` — then `yarn install`.

### 3. `reporter: "html"` produces no stdout

`playwright.config.js` sets `reporter: "html"`. When running non-interactively, this
produces no terminal output at all, making it appear the process has hung.

### 4. Vite preview blocks `hostmachine` hostname

Vite's preview server rejects requests where the `Host` header doesn't match an
allowed hostname. The container's browser navigates via `http://hostmachine:4173`
which Vite was blocking with:

```
Blocked request. This host ("hostmachine") is not allowed.
To allow this host, add "hostmachine" to `preview.allowedHosts` in vite.config.js.
```

**Fix**: Added to `vite.config.js`:

```js
preview: {
    allowedHosts: ["hostmachine"],
},
```

After this fix, `docker exec` running a raw Node.js playwright script confirmed:

- Container reaches `http://hostmachine:4173` ✓
- Page title "LoRA Inspector" loads correctly ✓

### 5. Switched from `preview` to `dev` server

`playwright.config.js` `webServer` was using `yarn build && yarn preview --host 0.0.0.0`
on port 4173. Switched to `yarn dev` on port 5173 to avoid needing a build step.

**Fix**:

- `playwright.config.js`: `webServer.command` → `"yarn dev"`, `webServer.url` → `"http://localhost:5173"`, `baseURL` default → `"http://localhost:5173"`
- `package.json` `e2e-test` script: `BASE_URL=http://hostmachine:4173` → `BASE_URL=http://hostmachine:5173`
- `vite.config.js`: added `server.allowedHosts: ["hostmachine"]` (was only in `preview`)

### 6. `start-playwright-server.sh` used `-it` (required a TTY)

The `-it` flag allocates a pseudo-TTY. When the container was run in the background
without a TTY, it could exit silently, making the WS endpoint unreachable and causing
playwright to hang indefinitely on the connection attempt.

**Fix**: Changed `-it` to `-i` in `start-playwright-server.sh`.

After this fix, `curl http://127.0.0.1:3030/` returns `Running%` confirming the server
is up.

### 7. `playwright test` hangs in the project directory — root cause identified: missing browser revision

**Symptom**: Every playwright command (`test`, `test --list`, even with minimal configs
and no `webServer`/`globalSetup`) hangs indefinitely with zero output. Affects
`node node_modules/.bin/playwright test`, `npx playwright@1.51.1 test`, and all
flag/config variations.

**Investigation**:

- `playwright --version` works fine (instant)
- `playwright test --list` in a clean `/tmp` directory exits quickly (fast fail on missing module)
- `playwright test --list` in the project directory hangs
- Process tree shows only V8 worker threads — no child processes spawned, no network connections
- Main process stuck on `futex_do_wait`
- `PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1` had no effect

**Root cause**: Playwright 1.51.1 requires **chromium revision 1161**, but only revisions
1148 and 1208 were present in `~/.cache/ms-playwright/`. When playwright initializes its
worker subprocess, it attempts to silently download the missing revision from the CDN.
That download hangs (no output, no visible network connection in `ss`), blocking all
test execution.

**Fix**: Install the correct browser revision:

```
node node_modules/.bin/playwright install chromium
```

This downloaded chromium-1161 to `~/.cache/ms-playwright/chromium-1161`.

**Status**: Browser installed. Tests still hanging after install — investigation ongoing.

## Confirmed Working

Direct `docker exec` approach works end-to-end:

```js
const { chromium } = require('/home/pwuser/.npm/_npx/0cf6ff1fad43f633/node_modules/playwright-core');
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto('http://hostmachine:4173', { waitUntil: 'networkidle' });
console.log(await page.title()); // → "LoRA Inspector"
```

## Environment

- **Node.js**: v24.14.1 (installed via nvm at `~/.config/nvm/versions/node/v24.14.1`)
- **Playwright**: 1.51.1 (host client + container server)
- **OS**: Arch Linux (not officially supported by Playwright — uses ubuntu20.04-x64 fallback builds)

## Files Changed

- `crates/lora-inspector-wasm/package.json`: pinned `@playwright/test` to `1.51.1` (exact, no `^`); updated `e2e-test` script port from 4173 → 5173
- `crates/lora-inspector-wasm/vite.config.js`: added `preview.allowedHosts: ["hostmachine"]` and `server.allowedHosts: ["hostmachine"]`
- `crates/lora-inspector-wasm/playwright.config.js`: switched `webServer` to `yarn dev` / port 5173; updated `baseURL` default to 5173; commented out `webServer` and `globalSetup` for debugging
- `crates/lora-inspector-wasm/playwright.container.config.js`: created minimal config (no webServer, no globalSetup, chromium only, line reporter) for debugging
- `start-playwright-server.sh`: changed `-it` to `-i` (user reverted; investigating)
- `~/.cache/ms-playwright/chromium-1161`: installed via `node node_modules/.bin/playwright install chromium`

## Next Steps

- Determine why `playwright test --list` still hangs after installing chromium-1161
  - The hang persists in the project directory but not in a clean `/tmp` directory
  - Possible Node.js 24 + playwright 1.51.1 compatibility issue
  - Try: `node --experimental-vm-modules node_modules/.bin/playwright test --list`
  - Try: downgrade or use a different Node.js version via nvm
  - Try: run tests fully inside the container via `docker exec` (confirmed working approach)
- Once `playwright test` unblocks, re-enable `webServer` and `globalSetup` in config
- Update `Makefile` target to use port 3030 and `BASE_URL=http://hostmachine:5173`
