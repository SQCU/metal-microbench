# UI Surface Map — user-personas plugin

**Generated:** 2026-05-24  
**Branch:** toolcards HEAD  
**Sources:** `sillytavern-fork/plugins/user-personas/static/*.html`, `sillytavern-fork/public/scripts/extensions/user-personas/index.js`, `docs/*.md` (mtime −7d), Playwright specs 80–84 + 99

---

## 1. Hamburger Menu + Drawer Layout

### Anchor element

`installUserPersonasToolsDrawer()` (index.js:852) probes for the anchor in order:

1. `#persona-management-button`
2. `#user-agent-designer-button`
3. `#top-settings-holder`

If none is present it reschedules itself via `setTimeout(..., INSTALL_RETRY_MS=500ms)` with ±10% jitter (index.js:856–857). It is idempotent — it short-circuits if `#user-personas-tools-button` already exists (index.js:860).

### Installed DOM structure

```
div#user-personas-tools-button.drawer          ← hamburger wrapper (position:relative)
  div.drawer-toggle                             ← click target
    div.drawer-icon.fa-user-gear               ← icon
  div#UserPersonasToolsMenu.drawer-content.closedDrawer.user-personas-tools-menu
    button.user-personas-tools-menuitem × 4   ← populated below

div#user-personas-surface-{key}.drawer.user-personas-surface  ← sibling, ×4
  div.drawer-toggle (display:none)             ← hidden from top-row
  div.drawer-content.closedDrawer.wide100p
    iframe title="{title}"                     ← lazy-loaded
```

Sibling drawers are inserted immediately after the hamburger wrapper (index.js:921). Their `.drawer-toggle` is `display:none` so they do not claim top-row real estate. Only the hamburger itself is visible.

### Activation pattern

Click hamburger toggle → popover appears. Click a menu item → `openSurface(key)` runs: calls `closeAllOpen()` (mirrors ST's close-others-on-open behavior), sets `iframe.src` on first open (lazy-load), swaps `closedDrawer` → `openDrawer` on the surface wrapper (index.js:937–958). Clicking the already-active surface's menu item closes it (toggle semantics, index.js:947). Clicking outside closes all (index.js:992).

### Menu items (index.js:814–850)

| `key` | `label` | `iconClass` | `src` |
|---|---|---|---|
| `suggester` | Suggester | `fa-magnifying-glass-arrow-right` | `/api/plugins/user-personas/static/suggester.html` |
| `corpus` | Corpus | `fa-chart-column` | `/api/plugins/user-personas/static/corpus.html` |
| `fixed-point` | Fixed-Point | `fa-arrows-rotate` | `/api/plugins/user-personas/static/fixed_point.html` |
| `designer` | Designer | `fa-wand-magic-sparkles` | `/api/plugins/user-personas/static/designer.html` |

Note the key is `fixed-point` (hyphen), while the HTML file is `fixed_point.html` (underscore). This asymmetry is load-bearing for any DOM selector that targets `#user-personas-surface-fixed-point`.

---

## 2. Per-Surface Contract

### 2a. Suggester

| Field | Value |
|---|---|
| Menu key | `suggester` |
| Container ID | `user-personas-surface-suggester` |
| HTML file | `plugins/user-personas/static/suggester.html` |
| Problem solved | Context-driven persona ranking: given recent chat turns, ranks the K_top+K_side most contextually relevant bios using `/yapper-seed` (LLM-as-judge sparse-NN). Surfaces a live `/poll` candidate preview per ranked persona. Operator clicks a candidate to insert it into `#send_textarea`. |
| CSRF status | `<script src="csrf-fetch.js">` at suggester.html:9. All POSTs use `csrfFetch()` from that wrapper (see §2f). |

**Unsafe-method endpoints:**
- `POST /api/plugins/user-personas/yapper-seed` — context ranking + loadout pick
- `POST /api/plugins/user-personas/poll` — per-persona candidate generation
- `POST /api/plugins/user-personas/dispatch-missing-agent-synth` — auto-synth on empty-agents condition (see §4)

**Interaction patterns:**
- On load: `pollBridgeAndAgents()` fires immediately and every 5s (suggester.html:1598–1599). If bridge is up + agents empty + bios present, auto-dispatches synth and shows "Synthesizing…" banner. Bridge-down shows a non-imperative error banner (no "Restart ST" text — see spec 80).
- Ranked rows with `.bio-without-agent-badge` badge: clicking the orange `.bio-without-agent-redirect` button calls `openDesignerTab({ bio: bioId })` (suggester.html:1056–1096), which rewrites `#user-personas-surface-designer iframe`'s `src` to `designer.html?bio=<key>` and programmatically opens the designer drawer. This is the canonical cross-surface navigation path.
- `_meta` strip shows ContextJudge reasoning; ranked rows show distance pills (near/mid/far CSS classes).

### 2b. Corpus

| Field | Value |
|---|---|
| Menu key | `corpus` |
| Container ID | `user-personas-surface-corpus` |
| HTML file | `plugins/user-personas/static/corpus.html` |
| Problem solved | Single scrollable surface merging (a) effective-dimensionality dashboard showing PR-score, active axes, bios/agents/compositions counts + per-axis variance-contribution chart; and (b) axis registry with lineage, inline edit, and delete. Also hosts the coordinate-picker — operator dials axis sliders to synthesize a new bio at a target point in behavioral space. |
| CSRF status | `<script src="csrf-fetch.js">` at corpus.html:6. All unsafe calls go through `fetchJSON()` which calls `csrfFetch()` (corpus.html:451–454). |

**Unsafe-method endpoints:**
- `POST /api/plugins/user-personas/axes/{id}` — save axis definition edits (corpus.html:851–852)
- `DELETE /api/plugins/user-personas/axes/{id}` — delete axis from registry (corpus.html:898)
- `POST /api/plugins/user-personas/corpus-snapshot` — refresh snapshot (corpus.html:1058–1059)
- `POST /api/plugins/user-personas/synthesize-bio-from-coordinates` — coordinate-picker synth (corpus.html:1217–1218)

**Interaction patterns:** Refresh button drives both dashboard sections simultaneously (shared `loadAll()`). Axis card has inline expand → edit form → Save. Delete triggers a confirm dialog. Coordinate picker: slider per bio-space axis → "Synthesize" dispatches run, streams completion via `/synthesize-bio-from-coordinates` polling, shows preview, offers Save-to-corpus CTA. The axes.html and corpus_dashboard.html files exist separately but are not wired into the menu; `corpus.html` is the unified replacement.

### 2c. Fixed-Point

| Field | Value |
|---|---|
| Menu key | `fixed-point` |
| Container ID | `user-personas-surface-fixed-point` |
| HTML file | `plugins/user-personas/static/fixed_point.html` |
| Problem solved | Runs and monitors the fixed-point iteration harness: nested outer (bio) × inner (agent) loops that converge bio prose and agent overlays toward target behavioral signatures measured by an LLM judge. Displays experiment cards, dispatch controls, live log tail, per-bio result trees with outer/inner attempt nesting, and invariant validation results. |
| CSRF status | `<script src="csrf-fetch.js">` at fixed_point.html:6. Dispatch, validate, and save calls use `csrfFetch()`. |

**Unsafe-method endpoints:**
- `POST /api/plugins/user-personas/experiments/{id}/run` — dispatch the iteration loop (fixed_point.html:998)
- `POST /api/plugins/user-personas/experiments/{id}/validate` — run invariant checks without re-running loop (fixed_point.html:1181)
- `POST /api/plugins/user-personas/experiments/{id}` — save experiment card edits (fixed_point.html:1675)

**Interaction patterns:** See §5 for the `lock_in_tetrad` card walkthrough. Clicking an experiment card (but not its buttons) opens the Trajectory view (fixed_point.html:980–988). The name is also a clickable edit-trigger. "Dispatch run" button disables itself during dispatch and re-enables on return; polls `/experiments/runs/{run_id}` every 2s, auto-scrolls log pane, stops polling on `done`/`failed`, then fires `doValidate()` automatically (fixed_point.html:1050–1056). Past results button shows historical per-bio JSON result files.

### 2d. Designer

| Field | Value |
|---|---|
| Menu key | `designer` |
| Container ID | `user-personas-surface-designer` |
| HTML file | `plugins/user-personas/static/designer.html` |
| Problem solved | "Selection IS design" — operator selects a bio from the left panel, the designer synthesizes K=3 candidate agent overlays for it via `/synthesize-agents-for-persona/{key}`. When arrived at via `?bio=<key>` redirect from suggester, auto-fires synthesis if the bio has no existing agents. Also exposes `/compare-agents` to compute axis-signature deltas between pairs. |
| CSRF status | `<script src="csrf-fetch.js">` at designer.html:6. All POSTs use `csrfFetch()` (designer.html:340–342, 418). |

**Unsafe-method endpoints:**
- `POST /api/plugins/user-personas/synthesize-agents-for-persona/{key}` — trigger K=3 candidate synthesis (designer.html:340–342)
- `POST /api/plugins/user-personas/compare-agents` — compare two agent cards by axis signature (designer.html:418–419)

**Interaction patterns:** On load, parses `?bio=` querystring (designer.html:221–232). If a bio param is present and the bio has zero derived agents, auto-fires synthesis with a visible status line before the POST returns (designer.html:241–248). Bio list sorts agentless bios to the top. Axis sliders allow composing a target signature to steer synthesis. Result shows K=3 candidate cards; operator selects one to persist.

### 2e. CSRF wrapper (all surfaces)

`plugins/user-personas/static/csrf-fetch.js` is served as a static asset from the same directory. Each iframe surface loads it via `<script src="csrf-fetch.js">` at line 6–9 of each HTML file. The wrapper:
1. Lazily fetches `/csrf-token` on first unsafe-method call and caches the token in the iframe's closure.
2. Injects `X-CSRF-Token: <token>` on every POST/PUT/DELETE/PATCH.
3. On 403 + `EBADCSRFTOKEN`: refetches a fresh token, retries once.

The ST parent page's `csrfRecoveringFetch` does NOT apply across iframe document boundaries (each iframe has its own `window.fetch`). Without this wrapper, all POSTs 403 under the default CSRF middleware. st-debug runs `--disableCsrf` which causes `/csrf-token` to return `{token: 'disabled'}`, but the wrapper still sends that as a header, so HEADER PRESENCE is testable regardless of server enforcement (spec 82).

---

## 3. Native ST Persona Panel Integration

ST's native Persona Management drawer calls `getUserAvatars()` (personas.js:235), which POSTs to `/api/avatars/get` and receives a filesystem scan of `User Avatars/` as an array of PNG filenames. `addMissingPersonas()` then ensures every returned filename has an entry in `power_user.personas`; for any file with no entry it fabricates `[Unnamed Persona]` (personas.js:222–226).

Each bio renders via `getUserAvatarBlock()` (personas.js:192–215), which produces an `.avatar-container` template block showing:
- `.ch_name` — display name from `power_user.personas[avatarId]`
- `.ch_description` — bio text from `power_user.persona_descriptions[avatarId]?.description`
- `.ch_additional_info` — title from `power_user.persona_descriptions[avatarId]?.title`
- `.avatar img` — thumbnail via `getThumbnailUrl('persona', avatarId)`
- `.default_persona` CSS class if it is the active persona

The operator clicks a persona card to select it (sets active persona). The "edit" affordance in the native panel leads to a textarea form that reads from `power_user.persona_descriptions`.

**Write path (chokepoint):** The plugin's canonical write is `_writePlayerCardPng()` (index.mjs:826–847), which writes a chara_card_v3 PNG to `User Avatars/<canonical_key>.png` via atomic rename. There is NO `settings.json` write from this path. The plugin's `src/persona-file-store.js` + `src/endpoints/settings.js` chokepoints project the PNG tEXt store into the `power_user.*` shape that ST's `/settings/get` endpoint returns (index.mjs:833–834). If the operator edits a bio description in the native ST persona panel and saves, that write goes to `settings.json` directly through ST's own `/settings/save` endpoint — bypassing the plugin's PNG store entirely. The result is a silent round-trip loss: the next plugin read from the PNG tEXt chunk returns the old value, not the operator's edit. This is not guarded by any current Playwright spec.

---

## 4. Auto-Synth Dispatch Loop in Suggester

`pollBridgeAndAgents()` (suggester.html:1513) runs on page load and every `BRIDGE_CHECK_INTERVAL_MS = 5000ms` (suggester.html:1504, 1598–1599).

**When it fires the dispatch:** The condition at suggester.html:1543–1546 is:
- `_bridgeStatus.reachable === true` AND
- `_agentCount === 0` AND
- `_bioCount > 0` AND
- `Date.now() - _lastDispatchAt > 4500ms` (per-tick guard against hammering)

When all conditions are met it POSTs `{ reason: 'suggester-poll' }` to `/dispatch-missing-agent-synth`. The plugin's own idempotency set (`_inFlightSynthBios`) skips bios that already have agents or are in-flight, so repeated calls are safe.

**When it stops:** The 5s interval never stops — it is a permanent heartbeat. But the dispatch within the tick stops firing as soon as `_agentCount > 0` (agents have landed and been loaded by the plugin). The banner then self-hides (renderBridgeBanner at suggester.html:1566 produces no visible DOM when `_agentCount > 0` and bridge is reachable).

**Telemetry produced:** The `in_flight` count returned by the dispatch endpoint is stored in `_inFlightCount` and rendered in the banner as "Synthesizing K=2 agents for N bios… (M in-flight)". Each 5s poll cycle updates this count. When the dispatch succeeds and in-flight drops to 0 and `_agentCount > 0`, the banner clears entirely.

---

## 5. The Fixed-Point Iteration Drawer — lock_in_tetrad Card

The `lock_in_tetrad` experiment card is the canonical demo. It is defined in `plugins/user-personas/experiments/lock_in_tetrad.json` and verified by spec 83.

**DOM selector for the card:**
```
.experiment-card[data-eid="lock_in_tetrad"]
```
This attribute is set at fixed_point.html:943: `data-eid="${escapeHtml(e.id)}"`. The `selected` CSS class is added when `e.id === selectedExperimentId` (fixed_point.html:939).

**Dispatch button:**
```
button.compact.run-btn[data-eid="lock_in_tetrad"]
```
(fixed_point.html:955) — label "Dispatch run". Clicking calls `dispatchRun('lock_in_tetrad', btn)` which POSTs to `/experiments/lock_in_tetrad/run`, receives `{ run_id, started_at }`, populates `activeRunId`, switches the right pane from `#run-empty` to `#run-active`, and starts the 2s polling loop.

**Telemetry stream in the right pane:**
- `div.run-banner` — status badge (`RUNNING` / `DONE` / `FAILED`) with run_id, pid, started_at, exit_code
- `div.log` — raw stdout tail from the harness child process, auto-scrolled when the user is at the bottom (fixed_point.html:1085–1087). Each poll tick calls `GET /experiments/runs/{run_id}` and replaces `runLog.textContent` with the cumulative log.
- `.bio-result` cards in `#run-results` — written at outer-loop completion; each shows `bio_slug`, `target_bio` signature, nested `.outer-attempt` blocks (converged/stalled left-border color), and within each `.inner-attempt` showing agent overlay text + distance metric.

On completion (`status === 'done'` or `'failed'`), polling stops and `doValidate()` is called automatically (fixed_point.html:1050–1056) to compute invariants without re-running the loop.

---

## 6. Playwright Spec Coverage Map

| Spec | Surface(s) exercised | What is covered | What is NOT covered |
|---|---|---|---|
| `80_suggester_no_defeatist_banner.spec.js` | Suggester | No "Restart ST" text in banner; `/dispatch-missing-agent-synth` endpoint responds with `dispatched`/`in_flight`; banner uses "Synthesizing" framing | Visual render; ranked rows; polling cycle completing |
| `81_suggester_fills_from_empty.spec.js` | Suggester | Full lifecycle: agents moved aside → ST restarted → banner appears "Synthesizing" → waits up to 180s for rows to appear → screenshot | Cross-surface navigation (suggester→designer) |
| `82_suggester_csrf_header.spec.js` | Suggester | `X-CSRF-Token` header present on every POST from iframe; `/csrf-token` GET fires before first unsafe POST | Corpus/Designer/Fixed-Point CSRF coverage |
| `83_lock_in_tetrad_demo.spec.js` | Fixed-Point | Experiment card present in DOM with correct `data-eid`; 4 axes + 2 bios present via API; "Dispatch run" button exists; dispatch returns 200 + run_id | Actual iteration convergence; log tail rendering; invariant results |
| `84_no_unnamed_personas.spec.js` | Native ST persona panel | Zero `[Unnamed Persona]` entries; every persona has non-empty name; corpus non-empty | Native panel write-path correctness; round-trip after edit |
| `99_session_surfaces_screenshot.spec.js` | Suggester, Fixed-Point, (attempts Corpus/Designer via stale selectors) | Pixel evidence via PNG captures | Structural assertions on Corpus or Designer surfaces |

**Corpus surface:** has no structural DOM assertion spec. Spec 99 references `iframe[src*="corpus_dashboard.html"]` (stale — the merged surface is `corpus.html`) so the screenshot step likely silently skips. No spec validates the axis editor, delete confirm dialog, or coordinate picker.

**Designer surface:** has no spec of any kind beyond a stale reference in spec 99.

---

## 7. Known Gaps

### Design-doc features with no UI surface

- **Outer-outer selector** (feature_factorization_design.md:59, 116, 146) — the target-picking layer above the current outer fixed-point loop is documented as MISSING. No surface exists.
- **Axis splitter** (feature_factorization_design.md:68) — diagnostic that decomposes a correlated axis into orthogonal sub-axes. Documented as MISSING. No surface.
- **Phase 3 multi-user dialogue** (multi_user_agent_chat_interface_spec.md:149, index.js:8–13) — N-persona turn-taking with round-robin / signature-weighted scheduling. The FE extension comment says "Phase 3 will live in this same extension, behind additional UI." No UI exists.
- **Novelty ranking over trajectories** (multi_user_agent_chat_interface_spec.md:433) — surfaces K interesting trajectories from an intractable posterior. Referenced in the spec; no surface.

### UI surfaces with no Playwright coverage

- **Corpus surface** — zero structural specs. The axis editor, delete confirm, coordinate picker, and corpus-snapshot refresh path are untested.
- **Designer surface** — zero specs. The `?bio=` auto-synth path, candidate K=3 rendering, and `/compare-agents` call are untested.
- **CSRF on Corpus/Designer/Fixed-Point** — spec 82 only covers the suggester iframe. The other three surfaces each load `csrf-fetch.js` and call `csrfFetch()` but no spec captures the header on their POSTs.

### User flows that cross multiple surfaces but are not end-to-end tested

- **Suggester → Designer redirect:** clicking `.bio-without-agent-redirect` on an agentless bio in the suggester's ranked rows calls `openDesignerTab({ bio: bioId })`, which rewrites the designer iframe's `src` and opens the designer drawer. The designer then auto-fires synthesis. This entire cross-surface navigation path has no Playwright test.
- **Coordinate picker → corpus refresh:** corpus.html's coordinate picker POSTs `/synthesize-bio-from-coordinates`, polls until done, then shows a Save CTA. Saving would write a new bio PNG; the native ST persona panel would then need to refresh to surface it. The full round-trip (picker → save → native panel update → suggester sees new bio) has no coverage.
- **Fixed-point completion → agent reloading:** after `lock_in_tetrad` completes, the plugin calls `loadAgents()` internally and the suggester's next 5s poll should see the new `_agentCount > 0`. This cross-process signal chain (harness child → plugin reload → suggester poll) has no end-to-end spec; spec 83 only asserts the dispatch returns 200.
- **Native ST persona panel edits vs. plugin PNG store:** the write-path mismatch described in §3 (ST saves to `settings.json`, plugin reads from PNG tEXt) is a silent data-loss path that no test exercises.
