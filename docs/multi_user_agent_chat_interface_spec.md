# Multi-user-agent chat interface — specification

**Date:** 2026-05-21
**Author of record:** Claude Opus 4.7, working from the project archeology + operator-stated revisions
**Status:** Restorative specification — captures peak accepted state + last wave of revisions + extrapolated next layer
**Companion documents:** `project_archeology.md` (timeline), `ux_debt_followup_tickets_2026_05_21.md` (ticket backlog)

---

## What this is

A normative chat-LLM-API user chat client (forked SillyTavern), bedazzled with
modern-ML features until it is recognizable as a multi-agent-inference
environment, regardless of whether it's ever used for MARL or multi-agent
launches in any specific downstream deployment. The chat surface remains the
central interface; multi-agent capability is added by composition over that
chat surface, not by replacing it.

The system rides on a custom Metal inference engine on Apple Silicon
(`metal-microbench/`) that decodes K parallel sessions in batch with
prefix-cache and KV-pool reuse. The chat client exposes the engine's
batched-decode capability through interface affordances that visibly demonstrate
parallel decoding to operators who would not otherwise see it.

---

## Thesis (load-bearing)

Cheap batched autoregressive decoding on consumer Apple Silicon is realistic
for chat-interface users. The product surface must demonstrate this excessively
relative to historic baselines:

- Multi-stream decode must be VISIBLE in the UI — auto-firing K candidates in
  parallel is the canonical demo of batched throughput, not just a latency
  optimization.
- KV-cache + prefix-reuse must be LAVISH in feature design — every place
  the engine can amortize a prefix across multiple users, the UI should
  exercise that path so the resulting throughput is part of the operator's
  observable experience.
- Visible-residue contract for tool calls + forked-agent summaries is product
  evidence of K=N concurrent streams running for negligible additional cost.

Regressions that strip parallelism, hide concurrency, or replace
multi-stream calls with single-stream calls are thesis-negative even when
locally neutral on principles. Acceptance review must include a
"does this strengthen or weaken the thesis" beat.

---

## Principles (the load-bearing 11)

1. **P-VISIBLE-RESIDUE** — Every tool invocation, forked-agent call, and
   synthesis dispatch leaves a card in chat DOM. Silent deletion is forbidden.
2. **P-FINITE-K-DRAWER** — Chat-suggestion drawer surfaces K_1
   immediately-polling + K_2 suggested-disabled candidates. Never the full
   corpus, never empty, never paginated past K_1+K_2.
3. **P-NO-EMPTY-FIRST-PAINT** — Every interactive surface has non-empty,
   operator-meaningful content visible on first paint. "Click here to begin"
   empty states are forbidden.
4. **P-EMPTY-FORM** — Never ask the operator to fill a form-with-bare-fields
   without contextual suggestions, pre-filled defaults from corpus, AND
   visible worked examples. JSON-fields-as-strings is the canonical
   forbidden anti-pattern.
5. **P-SELECTION-IS-DESIGN** — Operator clicks ARE training signal. Selection
   of a bio auto-chains to its agent; agentless bios redirect to the designer;
   selection events leave residue the system can consume as supervision.
6. **P-ONTOLOGICAL-CLOSURE** — Bios without agents are unusable. Usable bios
   MUST have agents. Synthesis pipelines auto-derive missing agents at
   first-launch or persona-create time.
7. **P-CANONICAL-NOT-MIRRORED** — One store per concept. Bios = ST personas
   (one settings.json store); agents = chara_card_v3 PNGs (one agents/
   dir); axes = JSON cards (one axes/ dir). Mirror writes forbidden.
8. **P-PER-CHAT-CACHE** — `/poll` and `/yapper-seed` results complete + cache
   per chatKey. A→B→A is O(1). No abort-on-navigation. No DoS cascades.
9. **P-API-EQUALS-GUI** — Every API endpoint call sequence must be
   exercisable through GUI affordances for e2e validation. Playwright
   pixel-space tests are the only acceptance evidence.
10. **P-NO-CHAT-DISPLACEMENT** — New UI elements must not hide, cover, or
    replace the central chat interface. Drawers cover the chat ONLY when
    the operator explicitly chose to view that drawer. Hamburger popovers,
    menus, tooltips are small + transient + chat-stays-visible.
11. **P-LLM-DISCRETION-DEFAULT** — Sorting, ranking, summarizing, and
    selecting use LLM API calls whenever possible. Deterministic ordering
    is the fallback, not the default.

---

## Component specification

### 1. Toolcard system

**Files:** `plugins/toolcards/` (server-authoritative plugin),
`public/scripts/extensions/toolcards/` (FE renderer)

**What it does:**
- Every tool call invoked by the assistant leaves a visible card in the
  chat DOM. The card carries: tool name, arguments summary, in-flight
  spinner, result text (when complete), per-card summary (from a
  spur-caption forked summarizer running in parallel with the tool's
  primary work).
- Cards are inline collapsibles. The default state shows the per-card
  summary; the operator can expand to see the full result.
- Async tool calls (shape D) leave a placeholder card immediately, then
  patch in their result + summary when the work finishes.
- Forked-agent descendant tool calls leave nested cards under their
  parent card. Summaries compress nested work so the parent context
  stays grounded.

**Demonstration shapes (4, all live in the demo install):**
- `random-choice` — context-fresh, sync; samples one of K options.
- `python-exec` — context-copy, sync; runs python in sandbox.
- `extended-thinking` — context-copy, sync; sub-LLM call for reasoning.
- `async-lookup` — context-fresh, async; database query + patch-on-complete.

**Acceptance:**
- A playwright spec opens an ST chat, sends a message that triggers each
  of the 4 tool shapes, asserts the card renders + summary is non-empty
  + result appears under the card.
- A playwright spec triggers an async tool, navigates away + back, asserts
  the patch-on-complete still landed in the right card.

**Current state:** ✅ Working at peak. The visible-residue contract is intact
in the toolcards plugin. Watch for regression: any "what if result is empty,
skip render" code path violates the contract.

---

### 2. User-personas plugin — bio + agent architecture

**Files:** `plugins/user-personas/index.mjs` (plugin backend),
`public/scripts/extensions/user-personas/` (FE extension),
`plugins/user-personas/static/*.html` (iframe surfaces)

**Data model:**
- A **bio** is a user-persona (= what ST calls a `power_user.persona`):
  name, description, optional voice_anchor + signature.
- An **agent** is an elicitation overlay: name, system_prompt,
  injection_mode (`author_note` / `system_prefix`), optional
  designed_for_bio_id, target_axis_signature, derived_from chain.
- A **chat participant** = bio × agent. Selecting both fully specifies
  the participant.
- An **axis** is a behavior dimension with 1-5 scale + judge rubric
  (def field). bios + agents both carry sparse signatures over axes.

**Phases delivered (peak state — 2026-05-11):**
- Phase 1: server plugin + FE extension + suggestion-mode panel.
- Phase 2: autonomous-tick "yapping" — system runs N personas mid-chat
  on a timer.
- Phase 3: multi-user dialogue (N personas + 1 assistant) — turn-taking
  among multiple bios in a single chat.
- Phase 4: unified panel + chara-card-style manifest enrichment.

**Acceptance:**
- `/personas` endpoint returns bios sourced from
  `<dataRoot>/<user>/User Avatars/` + `settings.json` (canonical).
- `/agents` endpoint returns agents from `plugins/user-personas/agents/`
  as chara_card_v3 PNGs.
- A playwright spec creates a persona via ST's native UI, asserts the
  plugin's PERSONAS_UPDATED event fires, asserts K=2 agents are
  auto-synthesized for the new persona within ~minutes.

**Current state:** 🟡 Canonical-store unification landed (T2 + T6 passing).
Auto-synth wiring exists via PERSONAS_UPDATED hook. Missing: first-launch
auto-synth for the 3 prefab bios — currently /agents = 0 on a fresh install,
which makes the suggester empty (P-NO-EMPTY-FIRST-PAINT violation).

---

### 3. Suggester / yapper

**File:** `plugins/user-personas/static/suggester.html` + the FE extension
hooks in `public/scripts/extensions/user-personas/index.js`

**What it does:**
- Reads the active SillyTavern chat via
  `window.parent.SillyTavern.getContext().chat`.
- Extracts a behavior signature from the chat context (via
  `extractSignatureInline`).
- Ranks the (bio × agent) corpus by L2 distance from the chat signature.
- Surfaces K_1 = 2-3 "top picks" (auto-polling immediately on first
  paint) and K_2 = 2-3 "side picks" (suggested-but-disabled, click to
  poll).
- "+ More" bumps both K values and appends new rows below existing ones.
  Originals stay in the DOM (preserved across re-renders).
- Per-row "Suggest" button POSTs to `/poll` with `{persona_id, agent_id,
  chat: getActiveChat().slice(-12)}` and renders the completion inline
  under the row.
- Per-row cache keyed `(chatKey, personaId, agentId)`. Second click on
  the same row hits cache with a `.cache-badge` marker, no `/poll` fires.
- Per-chat cache for the rank itself: switching chats A→B→A is O(1).

**Auto-polling contract (the load-bearing P-FINITE-K-DRAWER manifestation):**
- On first paint of the suggester surface with a non-empty chat present,
  the top-K_1 rows auto-fire `/poll` in parallel. Their prose streams
  into the row-completion slots WITHOUT operator action.
- This is the canonical demo of batched-AR-decoding. K_1 parallel streams
  visibly running for the latency of one stream is the thesis-evidence.
- If the chat is empty, no auto-poll fires (no-chat-no-event).

**Acceptance:**
- Playwright spec opens suggester with chat present, asserts K_1 rows
  visible within 5s with streaming text appearing under each top-K row
  without operator interaction.
- Playwright spec clicks +More, asserts row count grows AND original
  rows are still present.
- Playwright spec clicks per-row Suggest twice, asserts exactly 1 POST
  to /poll AND cache-badge visible on the 2nd render.
- Playwright spec switches chat A→B→A, asserts no re-fetch on the
  2nd A (per-chat cache hit).

**Current state:** 🟡 Mostly working (T1 passes 3 viewports). Missing:
auto-polling top-K_1 rows on first paint. Currently rows render with
inert Suggest buttons; the operator must click each. This is a
thesis-negative regression — the parallelism is no longer visibly
demonstrated.

---

### 4. Designer system — selection IS design

**Files:** `plugins/user-personas/static/designer.html` (currently deleted —
needs restoration per archeology event 2026-05-17 `cbc5d45e6`)

**What it does:**
- When the operator selects a bio in the suggester or persona drawer:
  - If the bio has an associated agent (a derived agent in agents/ with
    `designed_for_bio_id` matching), the selection auto-chains: bio +
    agent are both set as active.
  - If the bio has NO agent (agentless), selection redirects to the
    designer. The designer offers: "synthesize agents for this bio"
    (K=2 candidates pre-filled from the bio's signature), OR "edit
    this bio manually."
- The designer never presents bare JSON fields. Every input is
  pre-filled from context (corpus average for the kind, or signature
  of the currently-selected reference bio) and every input has a
  causal description.
- Designer surfaces:
  - Bio designer: starts from chat context or selected reference bio.
    Operator can dial axis coordinates; system synthesizes a candidate
    bio matching those coordinates.
  - Agent designer: starts from selected bio + chat context. System
    synthesizes K=2-3 candidate agents the operator can compare via
    Compare button (`/compare-agents` endpoint).

**Acceptance:**
- Playwright spec selects an agentless bio in the suggester, asserts
  the designer opens, asserts K candidate agents render with prose
  (not empty form fields).
- Playwright spec uses the agent designer's Compare button, asserts
  side-by-side diff of 2 candidate agents renders.
- Playwright spec dials a bio designer axis slider, asserts a
  synthesized candidate bio's prose updates within bounded time.

**Current state:** ❌ Designer.html is deleted. Agentless bios currently
have no redirect path. This is the most severe accepted-feature regression.
The "selection IS design" invariant from 2026-05-17 (`cbc5d45e6`) is not
enforced.

---

### 5. Fixed-point iteration

**File:** `plugins/user-personas/static/fixed_point.html`

**What it does:**
- Operator picks an experiment card (or composes a new one).
- Experiment card specifies: bios (or bio target signatures to
  synthesize), agent_targets (target signatures + motive_hint each),
  loop control (k_max inner, k_max outer).
- System dispatches `lock_in_iterative.mjs` or `outer_outer.mjs` as a
  detached child process.
- UI shows: run progress (iteration counter, k_max progress bar,
  per-bio trajectory cards), inline streaming output from the harness
  log, "stop run" button.
- On convergence: synthesized agents land in `plugins/user-personas/agents/`
  with `derived_from` chain pointing at the parent bio + experiment.

**Form coherence (P-EMPTY-FORM):**
- Every input has a `<label>` AND a 1-sentence causal description
  ("how does this affect the solver if changed?").
- Multi-cue dumps (one textarea, "type your bios here separated by
  whatever") are forbidden. Add structured per-bio fieldsets via
  "+Add bio" button.

**Acceptance:**
- Playwright spec opens fixed-point tab, asserts every visible input
  has a label + description sibling element.
- Playwright spec clicks Dispatch on a probe experiment, asserts
  EITHER a POST to /experiments/:id/run fires OR a run-banner with
  non-empty status appears within bounded time.

**Current state:** ✅ T4 desktop passes. Form labels + descriptions present.
Run dispatch wired through outer_outer.mjs.

---

### 6. Axes registry + factorization

**Files:** `plugins/user-personas/axes/*.json`,
`tools/user-agent-harness/axis_splitter.mjs`,
`tools/user-agent-harness/outer_outer.mjs`

**Data model (axis-v1):**
- `{axis_schema, id, name, def, kind, scale_min, scale_max, derived_from, created_at}`
- kind ∈ {bio, agent, either, meta}
- `derived_from` = `null` (root axis) OR `{parent, sibling, hypothesis, contexts}`

**Current registry (precollapsed canonical set):**
- `rpg_class` (bio, root): wizard ↔ rogue
- `star_sign` (bio, root): cancer ↔ sagittarius
- `money_orientation` (agent, root): pure-theft ↔ romance-leveraged-theft

**Factorization invariant (P-SELECTION-IS-DESIGN + spec T6):**
- Any non-root axis MUST have `derived_from.parent` referencing an
  existing axis in the registry. Dangling-parent axes are rejected at
  POST time.
- Every non-root axis must trace back to a spec root via the
  `derived_from` chain. Rogue roots (axes with `derived_from=null` not
  in the spec set) are caught by the T6 static invariant test.
- Axis splitting only happens via the `axis_splitter.mjs` harness OR
  via explicit operator action through POST `/axes/:id`. Auto-discovery
  of axes outside the operator's spec is forbidden.

**Surface in corpus dashboard:**
- Axes registry rendered with lineage tree (root axes + their derived
  children).
- Per-axis variance contribution shown alongside the registry (in the
  same panel, not split across two tabs).
- "Split this axis" button per axis row — invokes `axis_splitter` for
  the chosen parent with bounded k_max.

**Acceptance:**
- T6 spec passing (5 tests × 3 viewports = 15 cells).
- A playwright spec demonstrates an axis split via the GUI: clicks Split
  on `rpg_class`, asserts within bounded time a new derived axis appears
  in the registry with `derived_from.parent === 'rpg_class'`.

**Current state:** ✅ Registry + T6 + POST validation working. Missing:
GUI affordance to invoke axis_splitter (currently only via CLI / harness).

---

### 7. Bridge stream-lifecycle safety

**Files:** `server/bridge.py`, `lm_engine.swift`, `ffi_batch.swift`

**The safety triple (memory: `bridge_lifecycle_safety_triple.md`):**
- **Disconnect polling** in `_consume_engine_stream` — inbound `request.is_disconnected()` checked between yields.
- **Bounded response_q + driver-side cancel** — overflow on the bridge's queue triggers explicit cancel.
- **Engine forward-progress deadline** — Session reaped if no token consumed for >60s (memory: `kv_pool_vs_model_size.md`).

These three are INDEPENDENT mechanisms. None is a primary signal; together they bound the worst-case stale-stream duration.

**Explicit AbortSignal propagation (per the operator's "explicit > timeout backstop" principle):**
- Inbound HTTP disconnect → `req.on('aborted')` → AbortController → outbound bridgeCall fetch aborted → bridge sees its `is_disconnected` go true → opcode-2 cancel to engine → closeSession runs immediately.
- Timeouts ONLY fire when the explicit propagation chain has broken. They are the backstop, not the primary mechanism.

**Acceptance:**
- A playwright spec opens a chat, sends a message that triggers a long generation, closes the tab mid-stream, asserts (via /v1/engine/state polling) that active_streams drops to 0 within ~1s.

**Current state:** ✅ Working at peak.

---

### 8. Storage canonical contracts

| Concept | Canonical store | Read API | Write API |
|---|---|---|---|
| Bios = ST personas | `<dataRoot>/<user>/User Avatars/<key>.png` (avatar) + `settings.json → power_user.persona_descriptions[<key>]` (bio text) | `/personas` GET, ST native persona drawer | `/personas` POST (writes both atomically), ST's own persona-create flow |
| Agents | `plugins/user-personas/agents/<id>.png` (chara_card_v3 PNG) | `/agents` GET | `/agents/:id` POST |
| Axes | `plugins/user-personas/axes/<id>.json` | `/axes` GET | `/axes/:id` POST (validates `derived_from.parent`) |
| Experiments | `plugins/user-personas/experiments/<id>.json` | `/experiments` GET | `/experiments/:id` POST |
| Trajectories | `plugins/user-personas/data/runs/<run_id>.log` + per-bio JSON outputs | `/runs/:id` GET | written by harness on completion |

**Invariants:**
- One canonical store per concept. No mirror writes (the
  `_mirrorPersonaToSettingsJson` anti-pattern is permanently forbidden).
- Atomic writes to settings.json: tmp file + rename. Settings.json is
  shared with ST's own writes; partial writes corrupt the application.

---

## Recent wave of revisions (2026-05-20/21)

These are the specific revisions surfaced in the recent regression-cycle:

1. **Bio↔persona unification** (✅ T2 passing) — bios = ST personas, one canonical store, mirror code expunged.
2. **Axes precollapsed to 3** (✅ T5 passing) — rpg_class, star_sign, money_orientation.
3. **Factorization stays on-spec** (✅ T6 passing) — POST /axes/:id enforces parent-must-exist.
4. **Persona-create → auto-synth K agents** (✅ T3 passing) — PERSONAS_UPDATED event in ST, `/synthesize-agents-for-persona/:key` endpoint, FE hook dispatches synthesis.
5. **Suggester resynth + per-row cache** (✅ T1 passing) — +More preserves originals, per-row Suggest renders inline, cache-badge on 2nd click.
6. **Fixed-point form coherence** (✅ T4 desktop) — every input has label + causal description.
7. **Top-bar hamburger consolidation** (✅) — 4 drawer buttons → 1 hamburger + small popover; chat stays visible.

**Pending from this wave (re-articulated with corrected framing):**

- **R-AUTO-POLL-K1** — Suggester top-K_1 rows must auto-fire `/poll` in parallel on first paint. Currently rows render with inert Suggest buttons; the parallelism is no longer visibly demonstrated. **Thesis-negative — restore.**
- **R-DESIGNER-RESTORE** — Designer.html (or equivalent) restored as the redirect target for agentless bios. Selection-IS-design auto-chain reconnected. Currently agentless bios have no path.
- **R-FIRST-LAUNCH-SYNTH** — On plugin boot, any prefab bio with 0 derived agents triggers auto-synthesis. Currently /agents = 0 on a fresh install with 3 prefab bios; suggester is empty.
- **R-LINEAGE-BADGES** — Every persona row in ST's native drawer + every suggester ranked row shows lineage: "root persona" / "derived from X via axis_Y=N". Currently invisible.
- **R-SPLIT-DEMO-PRESTAGED** — Corpus tab ships with a pre-staged derived-axis demo so first-open shows the factorization machinery in operation. Currently registry shows only 3 root axes with no demonstration of splitting.
- **R-COORDINATE-PICKER** — Synthesize-bio-from-coordinates widget in the Corpus tab. Operator picks (rpg_class=2, star_sign=4, money_orientation=3), gets a candidate bio. Currently no such affordance.

---

## Extrapolated features (the next layer)

Building on the restored peak state, these are the features the project is
extrapolating toward. None of these block the restoration above; they're the
next iteration once peak is restored.

### EXT-1 — Multi-participant scene composer

**What:** A surface where the operator drops N bio×agent participants into a
chat simultaneously. Each participant's turns interleave via configurable
turn-taking (round-robin / signature-weighted / operator-driven). The chat
DOM shows K participants with distinct avatars + name colors.

**Why:** Demonstrates the engine's K=N batch-decode in the most legible way
possible — N visible streams interleaving in one chat. Thesis-positive.

**Acceptance:** Playwright spec drops 4 participants into a chat, sends a
prompt, asserts 4 distinct streams emit content interleaved in the DOM
within bounded total time (not 4× single-stream time).

### EXT-2 — Trajectory novelty mode

**What:** Per-experiment trajectory store with novelty ranking. Operator
can ask "show me trajectories where bio_X went OFF-script" — system surfaces
trajectories with high distance from bio_X's median signature.

**Why:** Makes the operator's curation effort efficient. Hand-reading 100
trajectories is intractable; novelty ranking surfaces the K interesting ones.
Maps to existing task #135.

**Acceptance:** Playwright spec opens a trajectory store with K trajectories,
filters by novelty > threshold, asserts the displayed set is non-empty +
sorted descending by computed novelty.

### EXT-3 — Per-stream tok/s in `/v1/engine/state` + UI surface

**What:** Bridge `/v1/engine/state` returns per-stream tok/s alongside total
active_streams. UI surface (tetraplex extension or new sidebar) shows live
per-stream rates so the operator sees batched-decode efficiency in real time.

**Why:** The thesis ("batched AR decode is cheap") is currently asserted
verbally + implied by responsiveness. EXT-3 makes it numerically observable.
Maps to existing task #178.

### EXT-4 — Session export for downstream RL/simulator use

**What:** Per-chat export to a structured format containing: bio + agent
metadata per participant, full turn history with timestamps + per-turn
signatures, operator selection events (selection-IS-design supervision
signal), tool-call cards + summaries. Format is consumable by an external
simulator/training pipeline.

**Why:** This is where the chat workbench becomes RL-environment
infrastructure. The operator's curation work in the chat becomes the
training signal for a downstream system that instantiates these bios + agents
in a longer-running simulator. Maps to existing task #135.

### EXT-5 — Generation-config violation cleanup

**What:** Triage the 75 violations exposed by the comprehensive linter.
Each violation is either: a hardcoded max_tokens cap, a hardcoded temperature,
or a prompt-prefix-mangling string interpolation. Caps are forbidden;
temperatures must be forwarded from the request, not set internally;
prefix-manglers must use LINT-OK-PREFIX-SAFE escapes or be refactored.

**Why:** Each violation is a place where bridge admission backpressure is
overridden, KV-cache prefix-reuse is broken, or the model is sent off-policy.
All of these are thesis-negative when they fire. Maps to existing task #181.

---

## Acceptance protocol — paired-agent fixed-point review

For any future code change touching this interface, use the paired-agent
fixed-point review pattern (mirrors the user-personas plugin's own
fixed-point iteration design):

1. **Proposer agent** is dispatched with a goal (e.g., "restore auto-polling
   on suggester first-paint"). Prompt is goal-focused; do not enumerate
   constraints in the proposer prompt. Proposer outputs a numbered diff
   proposal + reasoning sketch + open-questions list, then PAUSES without
   applying or closing.
2. **Reviewer agent** is dispatched with the proposal + the principle list
   from this spec + the thesis statement. Reviewer either:
   - Approves explicitly with a one-sentence justification per principle
     touched, OR
   - Returns specific objections referencing principle IDs + thesis-impact
     analysis.
3. **Proposer resumes** (via SendMessage) with the reviewer's feedback.
   Iterates until reviewer approves.
4. On approval, proposer applies the diff + writes the playwright spec that
   locks in the principle the change advanced. Spec must fail against the
   pre-change state.

This pattern is itself an instance of the principle P-LLM-DISCRETION-DEFAULT:
the acceptance decision is made by an LLM call against the principle list,
not by the proposer's self-evaluation.

---

## How to use this spec

1. Before designing anything in this interface, read this doc + the
   archeology + the principle list. Most "obvious" designs have already been
   tried and accepted or rejected here.
2. To restore a regressed feature, find its row in components 1-8 above. The
   acceptance criteria are testable. Write the playwright spec, watch it
   fail against current code, implement until green.
3. To extend a feature, propose via the paired-agent protocol. The reviewer
   agent's job is to catch thesis-negative changes that look principle-neutral.
4. To delete or simplify anything, you must explicitly justify against the
   principle list AND the thesis. Default-deny: if you can't articulate why
   the deletion is principle-positive AND thesis-positive, don't delete.

The codebase contains more accepted-to-spec features than any single
context-window can hold. This spec is the compressor: it lossy-encodes the
peak state at one moment in time so future agents can decompress to the
right design intent without having to re-derive it from the archeology
every session.
