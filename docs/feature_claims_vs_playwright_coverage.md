# Feature claims vs Playwright coverage

**Grounding document for future agents.**
**Scope:** docs in `metal-microbench/docs/` with mtime in last 7 days;
commits on `sillytavern-fork:toolcards` since 2026-05-17;
commits on `metal-microbench:main` since 2026-05-17;
specs at `tools/st-debug/tests/*.spec.js`.
**Date produced:** 2026-05-24.

---

## 1. Feature claim ledger

Each claim is expressed as a testable assertion. Commit hash cited is the
primary landing commit; multiple hashes are given when later commits revised
the claim.

| ID | Feature claim (testable assertion) | Primary commit(s) |
|----|------------------------------------|-------------------|
| FC-01 | Plugin bio store is exclusively PNG chara_card_v3 tEXt chunks in `User Avatars/`; `settings.json.power_user.persona_descriptions` is empty at all times and the plugin never writes to it. | `2e73f7f` (Sub-14) |
| FC-02 | A server-side tripwire in `init()` asserts that `settings.json` persona keys are empty at boot; any violation is a fatal startup error. | `2e73f7f` (Sub-14) |
| FC-03 | A source-level lint script (`lint_settings_persona_access.mjs`) prevents reintroduction of `power_user.persona_descriptions / personas / default_persona / character_persona_overrides` references in plugin source outside the allow-listed chokepoint. | `ca98c74` (Sub-14) |
| FC-04 | `User Avatars/` contains ONLY bio-bearing PNG cards (chara_card_v3 with tEXt); all vanilla PNGs (no tEXt) have been deleted; ST's native persona drawer shows zero `[Unnamed Persona]` ghosts. | `de653b6` (Sub-14 followup) |
| FC-05 | Iframe POSTs from plugin surfaces (suggester and others) carry a non-empty `X-CSRF-Token` header on every unsafe-method request; the iframe issues a `GET /csrf-token` before its first POST. | `0403bb0` (Sub-13) |
| FC-06 | The suggester surface does NOT show "Restart ST" / "use the Designer" / "./scripts/run.sh" in any user-visible banner. | `761fb54` (Sub-13) |
| FC-07 | When `bridge_up && agents_empty && bios>0`, the suggester auto-dispatches via `POST /dispatch-missing-agent-synth`; the banner reads "Synthesizing K=2" (status framing, not imperative). | `761fb54` (Sub-13) |
| FC-08 | After auto-dispatch completes (up to 180s), the banner self-hides and the suggester ranked-list container transitions to a healthy post-synth state (rows or "Awaiting active chat" empty-state, never synth-pending copy). | `761fb54` (Sub-13) |
| FC-09 | The canonical sorceror-rogue tetrad (`lock_in_tetrad`) experiment card is present on disk, reachable via `GET /experiments/lock_in_tetrad`, and its spec matches the canonical configuration (2 bios at antipodes on astrology axes, 2 agent targets). | `193dbd7` (Sub-14 followup) |
| FC-10 | The Fixed-Point Iteration drawer renders the `lock_in_tetrad` experiment as a selectable card showing both bio slugs and both agent_target slugs; a "Dispatch run" button is visible. | `193dbd7` (Sub-14 followup) |
| FC-11 | Axes/corpus/fixed_point surfaces have NO bare primary `<input>` or `<textarea>` elements on first paint (P-EMPTY-FORM); the banned affordances (`#add-axis-btn`, `#add-form`, `#new-experiment-btn`) have been deleted; the seed-input CTA (`#seed-tab-cta-btn`) is present and its rows are pre-filled. | `e2ea2dd` (Sub-13) |
| FC-12 | ST openai-CUSTOM source auto-reconnects mid-stream when the upstream emits `X-Stream-Id`; the reconnect `GET /v1/streams/{id}/sse?since=N` uses `N = lastSeenOffset+1`; the stitched response contains every token from the full sequence. | `1f94e1b` + `a99b0f1` (bridge refactor) |
| FC-13 | When upstream does NOT emit `X-Stream-Id`, ST makes no reconnect attempt. | `1f94e1b` |
| FC-14 | Lineage-badge styles (`.lineage-badge`, `.lineage-badge.root`, `.lineage-badge.derived`) are present in `suggester.html`; `renderLineageBadge()` helper exists and produces correct HTML; `renderRankedRow()` includes a badge. | `133a4bc` (R-4) |
| FC-15 | The `/personas` endpoint returns a `derived_from` field on every persona object. | `133a4bc` (R-4) |
| FC-16 | First-launch auto-synthesis (`bootAutoSynthMissingAgents`) fires in parallel for all bios lacking agents at boot; synth experiments are dispatched with spread ≤ ~2s across N bios; the endpoint is idempotent on second boot. | `761fb54` |
| FC-17 | The user-personas drawer panel contains no `+ Add user-agent` button, no inline `create new user-agent` form, no `Name` text input, no `Voice` textarea, no `[no personas in inventory yet]` text, and no `.user-personas-add-btn` / `.user-personas-add-form` class usage. | `cbc5d45` + `445ccdd` |
| FC-18 | The sub-13 `experiments.set` fix resolves the broken experiment-card save path; port-hardcode lint prevents port literals in plugin source; spawn-env URL propagation ensures child processes inherit the correct plugin URL. | `610dae4` (Sub-12) |
| FC-19 | Cold-prefill tolerance: ST delivers a non-empty assistant reply within 60s even when the engine requires a cold-prefill for the system prompt. | `a1d71a7` |
| FC-20 | Prefix-cache second-message cache-hit ratio ≥ threshold on identical successive user turns. | `84dd5f6` + `e338cc2` |

---

## 2. Coverage map

For each spec that is relevant to the recent feature ledger, the testable
assertions it makes are listed with file:line references.

### `80_suggester_no_defeatist_banner.spec.js`

- `80:37` — `POST /dispatch-missing-agent-synth` returns `{ok:true, dispatched:N, bios:[], experimentIds:[], bios_in_corpus:N, in_flight:N}`.
- `80:50` — `suggester.html` source contains no "Restart ST" in `bridgeBanner.innerHTML` blocks.
- `80:65` — `bridgeBanner.innerHTML` blocks contain no "use the Designer" escape-hatch text.
- `80:69` — `bridgeBanner.innerHTML` blocks contain no `./scripts/run.sh`.
- `80:78` — Source-level: banner contains `Synthesizing K=2` or `Dispatching K=2 agent synthesis`.
- `80:80` — Source-level: auto-dispatch fetch wired in `pollBridgeAndAgents`.

**Observations:** Three of these six assertions are source-level reads (`request.get` on the HTML file and regex on the body), not browser-rendered interaction. They pass even if the banner DOM never renders. The source-level assertions are useful for catching typos but do not prove the banner renders correctly in a real browser session.

### `81_suggester_fills_from_empty.spec.js`

- `81:114` — `/agents` is empty before the test opens the suggester.
- `81:123` — `#bridge-status-banner` becomes visible within 30s.
- `81:127` — Banner text matches `/Synthesizing K=2|Dispatching K=2/`.
- `81:130` — Banner text does NOT match `/Restart ST/i`.
- `81:131` — Banner text does NOT match `/use the Designer/i`.
- `81:155` — `/agents` fills within 180s.
- `81:161` — Banner self-hides within 15s of `/agents` filling.
- `81:225` — `drawerIsHealthy` = ranked-list visible AND (rows > 0 OR visible `.empty-state`), with forbidden text check.

**Observations:** This is a real end-to-end render test. It exercises the user-facing surface through browser interaction, not just source inspection. It is gated `desktop-only` and requires a live LLM bridge — it will be skipped in CI environments without the bridge.

### `82_suggester_csrf_header.spec.js`

- `82:154` — Iframe issued ≥ 1 `GET /csrf-token`.
- `82:161` — Iframe issued ≥ 1 unsafe-method POST.
- `82:167` — Every iframe POST carries `X-CSRF-Token`.
- `82:173` — Every `X-CSRF-Token` value is non-empty.

**Observations:** Network-layer capture via `page.on('request')`. This validates the header on the wire from the iframe. Critically it operates under `--disableCsrf` where `GET /csrf-token` returns `{token:'disabled'}` — so it proves the wrapper sends the header but does NOT prove the token is valid under real CSRF. Gated `desktop-only`.

### `83_lock_in_tetrad_demo.spec.js`

- `83:48` — `GET /experiments/lock_in_tetrad` → 200, id/schema/bios/agent_targets/bio_axes/agent_axes match canonical.
- `83:62` — `wizard.target_bio.astrology_sagittarian === 5`, `astrology_cancerian === 1`.
- `83:65` — `rogue.target_bio.astrology_sagittarian === 1`, `astrology_cancerian === 5`.
- `83:75` — All 4 required axes present in `/axes`, correct `kind` values.
- `83:87` — Both bios present in `/personas` with `bio.length > 100`.
- `83:113` — `.experiment-card[data-eid="lock_in_tetrad"]` visible in Fixed-Point Iteration drawer within 30s.
- `83:117` — Card name contains "Wizard" and "Rogue".
- `83:122` — Meta text contains both bio slugs and both agent_target slugs and counterparty.
- `83:133` — "Dispatch run" button is visible.

**Observations:** The API sub-tests are not tautological — they check actual file content that could have been deleted. The UI render test is a genuine browser interaction. The dispatch is NOT invoked (intentional — full loop is 10+ min). No coverage of what happens after dispatch is clicked.

### `84_no_unnamed_personas.spec.js`

- `84:61` — `[Unnamed Persona]` occurrence count in native ST persona drawer text === 0.
- `84:70` — Persona drawer text length > 50 (corpus non-empty).
- `84:97` — "Despotic Miscreant" visible in persona drawer text.
- `84:99` — "scringlo" visible in persona drawer text.

**Observations:** The spec evaluates `.innerText` of a DOM block located by a fallback chain of selectors. If the selector chain returns null, `drawerText` is empty and the `unnamedCount === 0` assertion trivially passes. This is a tautology risk: if `#user_avatar_block` / `#persona-management-block` don't match, the test passes vacuously. The test does have the `.length > 50` guard, but that only catches the total-empty case — a partial selector match on a wrapper element that doesn't include persona rows would still pass.

### `78_no_bare_inputs_on_axes_corpus_fixed_point.spec.js`

- `78:166` — Zero bare primary text inputs on first paint for `axes.html`, `corpus.html`, `fixed_point.html`.
- `78:177` — `#add-axis-btn` count === 0 on `axes.html`.
- `78:178` — `#add-form` count === 0 on `axes.html`.
- `78:196` — `#new-experiment-btn` count === 0 on `fixed_point.html`.
- `78:201` — `#seed-tab-cta-btn` visible on `fixed_point.html`.
- `78:215` — Seed-input bio rows are pre-filled (non-empty value).
- `78:222` — Seed-input motive rows are pre-filled.

**Observations:** Navigates the surface URLs directly (no drawer chrome). The bare-input audit is a good functional test. The exemption list is explicit and documented. This covers P-EMPTY-FORM for the three flagged surfaces but not for the Designer surface (UX-T1 from `ux_debt_followup_tickets_2026_05_21.md`).

### `79_st_reconnect_via_stream_id.spec.js`

- `79:175` — Flaky upstream received initial POST.
- `79:178` — ST issued `GET /v1/streams/{id}/sse` after mid-stream drop.
- `79:187` — Reconnect `?since` parameter === 3 (= lastSeenOffset+1).
- `79:195` — Stitched response contains every expected token (hello/from/flaky/upstream/and/finally).
- `79:198` — Response terminates with `data: [DONE]`.
- `79:254` — Without `X-Stream-Id`, no reconnect attempt.

**Observations:** No browser — pure Node HTTP. The test stands up its own "flaky upstream" server and routes requests through ST's `/api/backends/chat-completions/generate` API. This is a genuine integration test of the reconnect path. Skips if ST is not reachable on :8002.

### `69_lineage_badges.spec.js`

- `69:10` — `.lineage-badge`, `.lineage-badge.root`, `.lineage-badge.derived` CSS present in `suggester.html` source.
- `69:13–16` — Specific color constants present in source.
- `69:44–47` — `renderLineageBadge('root',...)` and `renderLineageBadge('derived',...)` produce correct HTML via `page.evaluate`.
- `69:72–74` — `renderRankedRow()` produces HTML containing a `.lineage-badge.root` element.
- `69:82–89` — `/personas` API returns `personas` array; at least one persona has a `derived_from` field.

**Observations:** Tests 1–3 (CSS constants, helper in-browser evaluation) are source-inspecting rather than interactive. Test 4 is a browser-side evaluation of a helper function, not a rendered UI scenario. Test 5 is an API contract check. None of these tests verify that a user SEES lineage badges when they open the suggester drawer and look at ranked rows. The visual rendering path is not covered.

### `74_no_prohibited_surfaces.spec.js`

- `74:29` — No `+ Add user-agent` button anywhere in DOM.
- `74:33` — No inline `create new user-agent` form.
- `74:35` — No `Name` text input.
- `74:37` — No `Voice` textarea.
- `74:41` — No `no personas in inventory yet` text.
- `74:43` — No `design one with + Add user-agent` text.
- `74:84` — No element uses `.user-personas-add-btn` class.
- `74:86` — No element uses `.user-personas-add-form` class.

**Observations:** Strong negative-space test. Covers FC-17 well. Gated `desktop-only`.

### `68_first_launch_synth.spec.js`

- `68:205` — Boot log contains `first-launch-synth` dispatch line.
- `68:220` — N dispatched > 0.
- `68:233` — N synth experiment JSON files appear in `experiments/` dir.
- `68:256` — Mtime spread across synth files ≤ ~2s (parallel dispatch).
- `68:269` — `/experiments` responds 2xx.
- `68:308/352` — Agent count ≥ 2 within 5 min.
- `68:361` — Suggester rows non-empty after first-launch synth.

**Observations:** This is the strongest coverage of FC-16. The parallel-dispatch timing assertion is meaningful and would catch serial execution. The synth-file mtime check is a side-effect observable, not a UI render — but the subsequent suggester-rows assertion does go through the browser.

---

## 3. Coverage matrix

| Feature claim | Spec file | Coverage status |
|---|---|---|
| FC-01: plugin writes only to PNG tEXt, never to `settings.json` persona keys | `63_bios_visible_in_st_persona_ui.spec.js` (line 93: calls `/personas` twice from same endpoint) | ⚠️ WEAK — spec 63's "no parallel store" test hits the same `/personas` endpoint twice instead of cross-checking the plugin against the actual `settings.json` file content; it does not prove `settings.json` is clean |
| FC-02: tripwire asserts settings.json persona keys empty at boot | (none) | ❌ NOT TESTED — the tripwire is source-visible in `index.mjs:1783–1784` but no spec starts ST with non-empty persona keys and asserts a fatal boot error |
| FC-03: source-level lint blocks persona refs in plugin source | (none) | ❌ NOT TESTED — `lint_settings_persona_access.mjs` exists (commit `ca98c74`) but is not wired into any playwright spec or CI gate visible in this test suite |
| FC-04: no `[Unnamed Persona]` ghosts; User Avatars/ has only bio-bearing PNGs | `84_no_unnamed_personas.spec.js` (lines 61, 70, 97, 99) | ⚠️ PARTIAL — assertion passes vacuously if the DOM selector chain returns null (see tautology audit §4); the canonical-anchor checks (Despotic Miscreant, scringlo) provide a partial guard |
| FC-05: iframe POSTs carry `X-CSRF-Token` header | `82_suggester_csrf_header.spec.js` (lines 154, 161, 167, 173) | ✓ covered — header presence on the wire; NOT covered under real CSRF (see §5) |
| FC-06: suggester never shows "Restart ST" / escape-hatch copy | `80_suggester_no_defeatist_banner.spec.js` (lines 64–69) | ⚠️ SOURCE-ONLY — three checks are regex on raw HTML source, not rendered DOM; a banner that appears via JS insertion after `DOMContentLoaded` would not be caught |
| FC-07: auto-dispatch fires when agents_empty; banner is status-framed | `80_suggester_no_defeatist_banner.spec.js` (lines 34–47, 78–80); `81_suggester_fills_from_empty.spec.js` (lines 127–132) | ✓ spec 81 covers the rendered path; spec 80 adds source-level reinforcement |
| FC-08: banner self-hides; drawer healthy post-synth | `81_suggester_fills_from_empty.spec.js` (lines 161, 225) | ✓ covered — gated on live bridge + real LLM; will SKIP in bridge-down environments |
| FC-09: `lock_in_tetrad` experiment card exists and is correctly configured | `83_lock_in_tetrad_demo.spec.js` (lines 48–88) | ✓ covered — API contract check is genuine |
| FC-10: Fixed-Point Iteration drawer renders the tetrad card | `83_lock_in_tetrad_demo.spec.js` (lines 113–134) | ✓ covered — real browser interaction; dispatch not exercised |
| FC-11: P-EMPTY-FORM on axes/corpus/fixed_point; banned affordances deleted | `78_no_bare_inputs_on_axes_corpus_fixed_point.spec.js` (lines 166, 177, 178, 196, 201, 215, 222) | ✓ covered for the three flagged surfaces; Designer surface NOT covered |
| FC-12: ST reconnects mid-stream with `?since=N` and no token gap | `79_st_reconnect_via_stream_id.spec.js` (lines 178, 187, 195, 198) | ✓ covered — Node-level integration test against a live ST instance |
| FC-13: no reconnect attempt without `X-Stream-Id` | `79_st_reconnect_via_stream_id.spec.js` (line 254) | ✓ covered |
| FC-14: lineage badge styles + helper in `suggester.html` | `69_lineage_badges.spec.js` (lines 10–16, 44–47, 72–74) | ⚠️ SOURCE/EVAL-ONLY — CSS constants and helper in-browser eval; rendered UI path not validated |
| FC-15: `/personas` returns `derived_from` field | `69_lineage_badges.spec.js` (lines 82–89) | ✓ API contract check |
| FC-16: first-launch synth fires in parallel; idempotent on second boot | `68_first_launch_synth.spec.js` (lines 205, 220, 233, 256, 352, 361) | ✓ covered — strongest test in the suite; mtime spread proves parallelism |
| FC-17: drawer contains no add-user-agent affordances | `74_no_prohibited_surfaces.spec.js` (lines 29, 33, 35, 37, 41, 43, 84, 86) | ✓ covered |
| FC-18: experiments.set fix; port-hardcode lint; spawn-env URL | (none for experiments.set); port-hardcode lint ships as source tool not spec | ❌ NOT TESTED for experiments.set fix specifically; lint tool not wired to playwright |
| FC-19: cold-prefill tolerance ≤ 60s first byte | `77_cold_prefill_tolerance.spec.js` (line 84) | ✓ covered — 60s timeout, real bridge required |
| FC-20: prefix-cache second-message hit ratio | `76_st_cache_validation.spec.js` (line 209) | ✓ covered — gated desktop-only, real bridge required |

---

## 4. Tautology audit

### `84_no_unnamed_personas.spec.js` — PARTIALLY TAUTOLOGICAL

The test locates the persona drawer block via a fallback chain:
```js
document.querySelector('#user_avatar_block')
  || document.querySelector('#persona-management-block')
  || document.querySelector('[id*="persona"]')
```
If all three queries return `null`, `block.innerText` is never evaluated and `drawerText` is `''`. The `unnamedCount === 0` assertion then trivially passes because there are zero occurrences in an empty string. The `drawerText.length > 50` guard is the only protection against this null-selector path, but `> 50` is weak: a half-rendered drawer or a wrapper element that doesn't include persona rows would pass.

**Verdict:** The canonical-anchor assertions on lines 97 and 99 (`Despotic Miscreant`, `scringlo`) provide a partial guard against the vacuous pass, because those names must appear in the same `drawerText`. If the selector returns the wrong element, those names are unlikely to be present and the test would fail for the right reason. This is better than nothing, but the selector chain should be hardened to assert that the located element is actually the persona list.

### `69_lineage_badges.spec.js` — PARTIALLY TAUTOLOGICAL

Tests 1 (CSS constants) and 2 (helper in-browser eval) inspect source content and evaluate a JS function in isolation. They do NOT:
- Navigate to a page where the suggester is rendered with actual persona data.
- Assert that lineage badges appear in the rendered DOM after polling `/personas`.
- Assert that a user who opens the suggester and sees ranked rows will see a badge on any row.

A complete source rewrite that moves badge rendering to a different function name, or a CSS refactor that changes the constant values, would break these tests — but a bug where badges are generated but never inserted into the DOM would pass them. This is the canonical source-inspection tautology: the spec validates that the code COULD produce badges, not that users SEE badges.

### `80_suggester_no_defeatist_banner.spec.js` — PARTIALLY TAUTOLOGICAL

Three of six assertions in this spec use `request.get(suggester.html)` and regex-scan the raw HTML source. This approach:
- Does NOT open the page in a browser.
- Does NOT trigger `DOMContentLoaded` or `pollBridgeAndAgents()`.
- Would pass if the forbidden text was injected into the DOM via a JS variable that isn't embedded in the source as a static string.
- Would pass against an entirely broken ST that returns the right HTML source but fails to mount the plugin.

Spec 81 is the real test for this feature. Spec 80's source-level assertions are defense-in-depth, not primary coverage.

### `63_bios_visible_in_st_persona_ui.spec.js` — TAUTOLOGICAL for the "no parallel store" test

The second test (`plugin /personas endpoint returns the same bios as settings.json (no parallel store)`) at line 93 fetches the plugin `/personas` endpoint twice (lines 103 and 110 call the same URL). It never reads the actual `settings.json` file. The comment at line 108 says "Read settings.json's persona_descriptions via fetch" but the fetch URL is the plugin endpoint, not a `settings.json` reader. This test cannot detect a parallel store because it compares the plugin endpoint to itself. It would pass even if `settings.json` still held persona data, as long as the plugin `/personas` endpoint returned the expected keys.

**Verdict:** This test is vacuous for its stated purpose. It should be rewritten to actually read `settings.json` (e.g., via the bridge's file-read API or by exec-reading the file directly) and compare against the plugin endpoint.

### `79_st_reconnect_via_stream_id.spec.js` — NOT TAUTOLOGICAL

This test stands up a real HTTP server, routes traffic through a running ST instance, and asserts on actual network behavior. It exercises the user-facing surface (the chat completion response visible to the client) through the FE-adjacent API path. No tautology concern.

### `78_no_bare_inputs_on_axes_corpus_fixed_point.spec.js` — NOT TAUTOLOGICAL

The test navigates the actual surface URLs in a browser and evaluates DOM input elements with visibility checks. The exemption list is documented. The seed-row pre-fill check reads actual `.inputValue()` via Playwright, which requires real DOM interaction. No tautology concern.

---

## 5. Known production-vs-test gaps

### 5a. Port 8002 (st-debug) coverage vs port 8008 (root)

All playwright specs run against port 8002 exclusively (hardcoded in every spec via `ST_URL = 'http://127.0.0.1:8002'` or the playwright config baseURL). No spec runs against root ST on port 8008. This means:

- FC-05 (CSRF header presence), FC-07 (auto-dispatch), FC-08 (banner self-hide), FC-10 (tetrad drawer), FC-16 (first-launch synth), FC-17 (no add-affordances) — all validated on st-debug only.
- Root ST (`~/sillytavern-fork`) runs a different sync state (not necessarily identical plugin commit) and is never exercised by any spec.

### 5b. CSRF disabled (st-debug `--disableCsrf`) vs real CSRF

Spec 82 explicitly acknowledges that `--disableCsrf` causes `/csrf-token` to return `{token:'disabled'}`. The spec proves the wrapper sends the header with the received value, but the token is not validated server-side. The following behaviors are NEVER validated under real CSRF:

- 403+`EBADCSRFTOKEN` handling (the retry path in `csrf-fetch.js`).
- Whether a stale token causes a request to silently fail vs retry.
- Whether any surface other than the suggester (axes.html, corpus.html, fixed_point.html) sends the CSRF token on its POSTs.

### 5c. Basic-auth absent (st-debug) vs auth-gated root

st-debug runs with no auth. No spec validates any behavior under HTTP Basic Auth, including whether the plugin's fetch wrappers include credentials.

### 5d. First-launch empty-state spec pattern

Spec 68 (`68_first_launch_synth.spec.js`) covers the first-launch auto-synthesis boot path. However, the spec's `beforeAll` cleans up agents but does NOT test the absolute zero-state (no `settings.json`, no `User Avatars/`, no agents, fresh bootstrap). It tests the "agents directory emptied" path. The true first-launch (fresh `bootstrap.sh` install) is not covered.

Specs 80/81/82 use the same "move `agents/` aside" pattern. They do not test the even-earlier state where the corpus (`players/`) is also absent.

### 5e. Surfaces that fire only on first-launch

The `UX-T4` ticket (`ux_debt_followup_tickets_2026_05_21.md`) identifies that st-debug's seed includes no default chat content, so the suggester first-paint in a brand-new install may have no chat context. No spec tests the "suggester with a non-empty default chat already present" scenario (UX-T2 claims this should auto-fire top-K streaming rows without any button click). This first-paint streaming behavior has no spec at all.

---

## 6. Prioritized missing coverage

Items ordered by: blast radius of regression × likelihood of regression ÷ cost to write.

| Priority | Missing coverage item | Blast radius | Regression likelihood | Writing cost |
|---|---|---|---|---|
| 1 | **FC-01 / settings.json excision is not tested at runtime**: no spec asserts that `settings.json.power_user.persona_descriptions` is empty after a round of bio synthesis. A future settings.json write re-introduction (the exact bug that cost the May-2026 corpus) would pass all current specs. | High — data loss that's silent and cumulative | High — the revert at `0ea824617` shows the pattern recurs | Low — read the file via exec, compare to `{}` |
| 2 | **FC-04 tautology in `84_no_unnamed_personas.spec.js`**: the selector chain can silently return null and make the `[Unnamed Persona]` check pass vacuously. A vanilla PNG in `User Avatars/` (the bug this spec was written to catch) could go undetected if the selector fails. | High — ghost personas corrupt the drawer UX | Medium — vanilla PNGs reappear any time an avatar is saved without tEXt encoding | Low — add `expect(block).not.toBeNull()` before the text scan |
| 3 | **FC-14 lineage badges never validated in rendered UI**: `69_lineage_badges.spec.js` proves the helper function exists but not that users see badges in the suggester drawer. A CSS `display:none` or a missing `renderLineageBadge()` call in the render path would pass the spec. | Medium — visual regression, operator can't see lineage | Medium — any suggester refactor could drop the call | Medium — requires opening suggester with live data and checking `.lineage-badge` elements in ranked rows |
| 4 | **FC-07/FC-08 coverage is bridge-gated**: specs 81 and 82 skip if the bridge is down. The auto-dispatch + banner self-hide flow is not exercisable in a bridge-absent environment. Any change to the dispatch-missing-agent-synth endpoint that breaks the status framing would be undetected without a running LLM. | High — broken auto-dispatch forces operator to manually restart (the exact UX regression spec 80/81 was written to prevent) | Low — endpoint is stable | High — requires a mock bridge that simulates agent synthesis completion |
| 5 | **UX-T2 (suggester auto-streams top-K on first paint)** — completely uncovered: no spec validates that the top-K rows auto-fire `/poll` on first open without an operator button click. The `ux_debt_followup_tickets_2026_05_21.md` ticket identifies this as P-NO-EMPTY-FIRST-PAINT, and the feature is not even claimed as shipped. | High — first-paint is the operator's first impression | N/A — not shipped, but writing the failing spec documents the gap | Medium |
| 6 | **FC-03 (lint not wired to any automated gate)**: `lint_settings_persona_access.mjs` exists but is not invoked by any playwright spec or CI hook. A future plugin commit that re-adds a `settings.json` persona read would not fail any automated check until the runtime tripwire fired at ST boot. | High — the lint is the early-warning layer before the tripwire | Medium — see the revert history | Low — add a spec that runs `node scripts/lint_settings_persona_access.mjs` and asserts exit code 0 |
| 7 | **FC-18 (`experiments.set` fix) has no spec**: the Sub-12 commit (`610dae4`) fixed broken experiment-card save behavior. There is no spec that creates an experiment card, saves it, and verifies the on-disk state matches the posted data. | Medium — corrupted experiments cause Fixed-Point Iteration to silently fail | Low — the fix was straightforward | Medium |
| 8 | **FC-05 CSRF under real enforcement**: spec 82 proves the header is present but does not validate the 403+retry path. Running a single test under a temporarily CSRF-enabled st-debug instance would catch the entire retry code path. | Medium — all iframe POSTs fail silently under real CSRF if the retry is broken | Low — the token plumbing is simple | High — requires re-enabling CSRF in the test environment |
| 9 | **FC-63 "no parallel store" test is vacuous** (`63_bios_visible_in_st_persona_ui.spec.js:93`): the test fetches `/personas` twice and compares the result to itself. It should read `settings.json` directly and verify it contains no `persona_descriptions` entries. | Medium — the parallel-store bug is what caused the May-2026 corpus loss | Medium — the revert history shows it can recur | Low — exec-read `settings.json`, assert `power_user.persona_descriptions === {}` |
| 10 | **UX-T1 (Designer surface P-EMPTY-FORM) has no spec**: spec 78 covers axes/corpus/fixed_point but the Designer surface is explicitly called out in `ux_debt_followup_tickets_2026_05_21.md` as still showing bare JSON field inputs. A spec that navigates `designer.html` and asserts zero bare inputs would formalize this debt. | Medium — operator-visible UX regression on a tool-creation surface | Low — Designer is used infrequently | Low — extend spec 78's pattern to `designer.html` |
