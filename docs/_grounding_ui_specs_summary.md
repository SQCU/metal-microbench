# Grounding document — UI specification summary (7 specs)

Generated from read-only analysis of `docs/ui_spec_*.md`. No implementation files were consulted.

---

## 1. Axis Registry + Lineage View (V1)

**Title:** UI spec — B4 axis registry + lineage view (V1)
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> Today, only discovery exists in code. Inspection is `cat … | jq` followed by `rm` if you don't like one. No client surface. V1 of this panel is the inspection half.

### Specified surface

New file `sillytavern-fork/plugins/user-personas/static/axes.html`. Opened via a new top-level drawer button `#user-axes-button` (sibling to other tabs). The axis list and edit/delete forms live inside this iframe.

### Specified behaviors

- Tree layout: axes with `derived_from == null` are roots; descendants render indented under their parent using ASCII tree branches.
- Per-axis row: id (monospace), kind badge (bio/agent/meta/either), scale range, one-line rubric, `scored on: N bios, M agents` count.
- Edit button: opens an inline edit form for name/def. Submits `POST /axes/:id`. Kind and scale are immutable — POST validates and rejects changes to those fields.
- Delete button: opens a confirm dialog showing `orphaned_signatures` count. If orphaned > 0, warning: > "Deleting this axis will leave N bios + M agents with dangling references to it. The L2 distance code in /yapper-seed handles missing axes via the neutral-3 baseline, so this is non-fatal, but the references will persist in their signature blobs."
- Add axis button: opens an inline form for id, name, def, kind. POSTs `/axes/:id`. id must match `[A-Za-z0-9._-]+`.
- Orphaned signatures section at page bottom: lists any axis id that appears in a bio's or agent's `signature` field but does NOT have a registry entry.
- Refresh and + Add axis buttons in the header row.

### Specified API/data contracts

Backend additions: **none**. Uses existing endpoints only:

- `GET /axes` — fetch full registry
- `GET /axes/:id` — single axis card
- `DELETE /axes/:id` — response shape includes `orphaned_signatures` count
- `POST /axes/:id` — create or edit; id must match `[A-Za-z0-9._-]+`; kind/scale immutable on edit

### Open questions / TODOs / known gaps

Explicitly deferred to V2 (not in this spec):
- "Propose new axis" button calling a model (needs `POST /axes/propose` — new endpoint).
- Bulk operations (delete all dormant axes).
- Lineage diff showing what `axis_splitter` actually split.
- Live cluster-collapse alerts.
- If no derived axes exist in the current corpus, the lineage test case is skipped and annotated.

---

## 2. Context-Driven Suggester Refactor

**Title:** UI spec — context-driven suggester refactor
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> The current `suggester.html` is bio-list-driven: it shows every biography in the corpus, and the operator clicks "Suggest" per bio to invoke `/poll`. That inverts the user-stated mental model.

The user's mental model is quoted directly:
> Given the current chat context, the suggester ranks all `(bio, agent)` compositions in the corpus by L2 distance from a target signature extracted from that chat context.

### Specified surface

`sillytavern-fork/plugins/user-personas/static/suggester.html` — rewrite of the right panel. Left panel (history scratchpad + candidate feed) stays unchanged.

### Specified behaviors

- Replace bio-list-driven right panel with "Ranked compositions" driven by `/yapper-seed`.
- Add a "Rank for this context" button. On click: POST `/yapper-seed` with `chat_context_summary` (concatenated history), `counterparty_card_id` (from parent ST state), `K_top: 3, K_side: 3`.
- Per ranked row: bio name, agent name, `L2=<float>` distance pill (green ≤1, amber 1–2, red >2), "why" line, "Suggest" button (calls existing `/poll` with that row's bio_id + agent_id).
- If `top[i].agent_id` is missing: render a red badge "BIO WITHOUT AGENT — corpus bug" and continue. The raw-bio path is removed.
- `_meta` strip: horizontal strip above the list showing `target_signature` as `axis_id=value` pills (axes sorted alphabetically), `target_completed_axes` / total, `candidates_considered` / `bios_total × agents_total`, K_top, K_side, `pending_synthesis` as clickable bio_ids (open FP tab via `#user-fixed-point-button` with `?prefill_bio=<bio_id>`), `pending_count` badge. Placeholder when no rank yet: "Click 'Rank for this context' to query the suggester."
- `+ More` button below ranked list: increments `K_top` and `K_side` by 3, re-POSTs `/yapper-seed`, re-renders. Disabled while in-flight. If new response row count equals previous, disabled permanently with label "(no more compositions)".
- Synthesize CTA: when `top[0].distance > SYNTHESIZE_THRESHOLD` (configurable constant, default `2.0`), render a CTA row. The button opens the FP tab via `#user-fixed-point-button` and passes `?target_bio_signature=<urlencoded JSON of axis→value>`.

> **This CTA is the ONLY path to "make me an agent for this kind of context" that's permitted to exist.** … The CTA opens the fixed-point editor; it does not call a one-shot synthesis endpoint, because no such endpoint exists and never will.

- Delete: the "show overlay only" checkbox, per-bio agent dropdowns, and right-panel bio list are removed.

### Specified API/data contracts

Backend: **do not modify**.

`POST /api/plugins/user-personas/yapper-seed`

Request body (relevant fields):
```jsonc
{
  "chat_context_summary": "string",
  "counterparty_card_id": "the-rock.png",  // optional
  "K_top": 3, "K_side": 3,                 // optional
  "axes": [...]                             // optional override
}
```

Response `top` = nearest by L2. `side` = farthest of remainder with distinct bios. Each entry has `bio_id`, `agent_id`, `why`, `distance`, `persona`, `agent`. `_meta` block: `K_top`, `K_side`, `target_signature`, `target_completed_axes`, `candidates_considered`, `bios_total`, `agents_total`, `pending_synthesis`, `pending_count`.

### Open questions / TODOs / known gaps

- Persistence of ranking history across page reloads is out of scope (caller can re-rank).
- If the corpus is too small to exercise the `+ More` K-ceiling path, the spec documents the gap and explicitly asserts the disabled-state path.
- Blocking dependency: Doc B (experiment editor) consumes the `?target_bio_signature=` query param this doc emits; ship in parallel.

---

## 3. Corpus Effective-Dim Dashboard

**Title:** UI spec — A1 corpus effective-dim dashboard
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> The outer-outer loop's objective is to grow the *effective dimensionality* of the corpus's behavioral coverage. `effDimParticipationRatio` exists in `harness_lib.mjs:279` … but never surfaces. The operator can run experiments forever without seeing whether the corpus's behavioral coverage is widening, narrowing, or saturating. Without this view, the search has no operator-visible objective.

### Specified surface

New file `sillytavern-fork/plugins/user-personas/static/corpus_dashboard.html`. New drawer button `#user-corpus-button` (sibling to `#user-fixed-point-button` and `#user-suggester-button`), installed via the `installFixedPointDrawer` pattern in `index.js`.

### Specified behaviors

- PR number, active axis count, bio/agent/composition counts in a summary header row.
- Per-axis variance bar chart: axes sorted by variance contribution descending. Bar width proportional to `normalized[axis]`. Colors: green (top quartile), amber (middle), gray (bottom quartile / dormant). Tooltip: axis_id, def, variance value, count of compositions scored on it.
- PR uses COMPOSITION signatures (each agent card's `agent.signature`), not bio-only signatures. PR is computed over agents.
- Saturation history: reads `GET /corpus-snapshot`, renders a JSONL-backed snapshot list with timestamp, PR, n_compositions per row. Status line: "climbing" if last Δ ≥ 0.1; "stalled" if last 3 snapshots all within Δ 0.1 (with recommendation text); otherwise "climbing" with Δ.
- Refresh button: re-fetches `/personas`, `/agents`, `/axes` and appends a snapshot via `POST /corpus-snapshot`.
- Empty state (0 compositions): renders an empty-state message, does not crash.
- All computation is **client-side**; no new computation endpoints.

`effDimPR` function is specified verbatim (copy from `harness_lib.mjs:279`, adapt for browser context):
```js
function effDimPR(sigsByComposition, axisNames) { … }
```

### Specified API/data contracts

All existing, read-only:
- `GET /personas` — bio signatures
- `GET /agents` — agent signatures (composition signatures via `agent.signature`)
- `GET /axes` — axis registry

New endpoints (2, simple file-system passthrough):
- `POST /corpus-snapshot` — idempotent, appends one row to `data/corpus_dashboard/snapshots.jsonl` with timestamp, PR, n_compositions
- `GET /corpus-snapshot` — returns parsed JSONL

### Open questions / TODOs / known gaps

V2 deferrals:
- Auto-snapshot on experiment-run completion.
- PCA over the full signature matrix (V1 uses participation-ratio only).
- Comparing multiple corpora.
- Drilling into specific axes to see which bios contribute.

---

## 4. Experiment Editor Form

**Title:** UI spec — fixed_point.html experiment editor form
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> `fixed_point.html` currently lists experiments and runs them. There is no way to create or edit experiment-spec cards from the client. The only path is to write `experiments/<id>.json` on disk and reload. That's not an operator-facing interface; that's a developer artefact.

### Specified surface

`sillytavern-fork/plugins/user-personas/static/fixed_point.html` — add a "New Experiment" modal (or full-iframe section; modal preferred). Editing an existing experiment opens the same form pre-populated by `GET /experiments/:id`. The modal is opened by a "New Experiment" button in the top-right of the Experiments section.

### Specified behaviors

- `id` field: text input, `pattern="[A-Za-z0-9._-]+"`, immutable on edit (disabled with tooltip).
- `bio_axes`: multi-select from `GET /axes` filtered to `kind == "bio"`. Required: at least one.
- `agent_axes`: multi-select from `GET /axes` filtered to `kind == "agent"`. Required: at least one.
- When an axis is picked, a 1–5 stepper materializes inside each bio/agent-target row for that axis. Un-picking removes it.
- `bios[]` repeating-row block: slug, name, canonical_key (readonly, derived as `user-personas-<slug>.png`), per-bio-axis 1–5 steppers, design_brief textarea. Blank target value = "no preference" (neutral-3 rule in harness handles it). Minimum 1 row.
- `agent_targets[]` repeating-row block: slug, per-agent-axis 1–5 steppers, motive_hint textarea. Minimum 1 row.
- `counterparty_avatar`: dropdown from ST character list via `parent.window`; fallback to free text input.
- Loop control (`loop_control`): six numeric inputs in a `<details>` collapsed block. If not opened by operator, FE omits the object entirely and server applies defaults.
- Save: POST `/experiments/:id`. On success close form, refresh list. On error render server validation message inline in red above footer.
- Cancel: close without saving.
- Delete (edit mode only): confirm dialog, then DELETE `/experiments/:id`.

Pre-population from `?target_bio_signature=<urlencoded JSON>`:
1. Auto-open the New Experiment form.
2. Pre-populate `bio_axes` with the axis IDs present as keys.
3. Pre-populate `bios[0].target_bio` with the signature values.
4. Pre-populate `bios[0].slug` with `from-chat-context-<short_hash>`.
5. Pre-populate `bios[0].design_brief` with "Synthesized from chat context. Adjust the brief before saving."
6. Leave `agent_targets` empty. Operator must add at least one before save (the validator enforces non-empty `agent_targets`).

### Specified API/data contracts

Backend: **do not modify**.

- `GET /axes` — populate axis selectors
- `GET /experiments/:id` — pre-populate edit form
- `POST /experiments/:id` — save; validated by `validateExperimentCard` in `index.mjs:848`
- `DELETE /experiments/:id` — delete

Card schema must match `experiments/lock_in_tetrad.json` exactly. `experiment_schema` field is server-filled; FE never sends it.

### Open questions / TODOs / known gaps

- A free-form JSON paste-in mode is explicitly out of scope (operator can edit the card file directly on disk).
- Blocking dependency: Doc A (suggester refactor) emits `?target_bio_signature=` which this editor consumes; ship in parallel.

---

## 5. Iteration Timeline View

**Title:** UI spec — D10 iteration timeline view
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> After running an experiment, the operator sees only the converged card. The fixed-point loop's *learning curve* … lives in `/Users/mdot/metal-microbench/data/lock_in_iterative/<exp_id>/<bio_slug>.json` and is invisible to the FE. This is half the value of running the loop in the first place; the loop's iteration trace IS the experimental finding, not just the final card.

### Specified surface

New "Trajectory" subsection inside `fixed_point.html`. Reached by clicking any row in the Experiments list that has at least one result file. The Trajectory view replaces or expands below the Experiments list. A back button or breadcrumb returns to the list.

### Specified behaviors

- Bio toggle row: one button per bio in the experiment's results. Click to scroll into view or load that bio's trace. Disabled if no result file exists.
- Per-bio header: target_bio pills, stop_reason badge (converged=green / max_outer=amber / stall=amber / error=red), elapsed seconds.
- Sparkline of `max_off`: K_max_outer dots, each colored by `max_off` vs `eps_per_axis` (≤eps green, ≤2×eps amber, >2×eps red). Empty dots for iterations not reached.
- Outer attempt accordion: collapsed by default if `iter !== result.best.iter`; expanded for the converged iteration. Header shows iter #, status, max_off, elapsed.
- Bio prose: monospaced `<pre>` with overflow-scroll.
- Measured signature line: `axis=value(↓distance)` per axis. ↓ for under-target, ↑ for over, no arrow for exact.
- Inner blocks: one per agent_target. Per inner attempt row: iter, agent_text snippet (collapsible to full), measured agent-axis values, converged check ✓/✗.
- Chat preview: `[Show chat turns]` toggle inside each inner attempt reveals the actual chat as a 2-column or message-bubble layout.

### Specified API/data contracts

Two new read-only endpoints (file-system passthrough):

`GET /api/plugins/user-personas/experiments/:id/results`
```jsonc
{
  "experiment_id": "lock_in_tetrad",
  "results": [
    { "bio_slug": "rpg-wizard-sagittarius", "size_bytes": 12834, "mtime": "..." }
  ]
}
```
Returns `{ results: [] }` (not 404) if directory doesn't exist.

`GET /api/plugins/user-personas/experiments/:id/results/:bio_slug`
Returns the full result JSON as written by `lock_in_iterative.mjs`. 404 if file doesn't exist. Shape includes top-level keys: `bio`, `agent_targets`, `bio_axes`, `agent_axes`, `result` (with `stop_reason`, `best`, `attempts[]`), `elapsed_ms_total`.

Backend: add `LOCK_IN_ITERATIVE_DIR` constant in `index.mjs` next to `EXPERIMENTS_DIR`. Path: `/Users/mdot/metal-microbench/data/lock_in_iterative/<id>/*.json`.

### Open questions / TODOs / known gaps

V2 deferrals:
- Per-turn signature trajectory chart (C7, separate spec).
- Judge feedback turns from K-shot retries (D11, separate spec).
- Live polling for in-progress runs.
- Compare-side-by-side mode.
- Export trace as markdown or CSV.
- Filtering / searching inside a trace.
- If `lock_in_tetrad` result files are not present on disk, the Playwright spec skips with an annotation explaining the seeding requirement.

---

## 6. Provenance Tagging + View Filter

**Title:** UI spec — F16 provenance tagging + view filter (no auto-deletion)
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> Currently they [lock_in_tetrad bios and lock_in_iterative agent outputs] show up in `/yapper-seed` rankings as if they were canonical operator-curated personas.

> The earlier "transient: true" framing was the wrong abstraction. It conflated "view-filter property" with "auto-delete license" and would have shipped a destruction-of-evidence pattern under the cover of an experiment-output cleanup. Provenance tagging is the right separation.

**Hard constraint (normative):**

> **NEVER auto-delete a card.** Filtering hides cards from views; the filesystem retains them indefinitely. Cleanup is always operator-clicked + confirmed.

### Specified surfaces

- Schema addition to bio (`extensions.provenance`) and agent JSON (`provenance`) cards. Missing field treated as `{ kind: "legacy" }` on read.
- Filter row added to `suggester.html` above or below the `_meta` strip.
- `harness_lib.mjs:saveAgent` and `saveBio` extended to accept and forward `provenance`.
- New one-shot script `scripts/tag_existing_corpus.mjs`.
- Corpus dashboard and axis registry filter toggles: V2 (structurally ready, same localStorage key).

### Specified behaviors

Provenance schema:
```jsonc
"provenance": {
  "kind": "canonical" | "manual" | "experiment_output" | "seed_demo" | "legacy",
  "experiment_id": "...",   // when kind=experiment_output
  "run_id": "...",           // ditto
  "iter": { "outer": 0, "inner": 1 },  // ditto
  "seed_phrase": "...",      // when kind=seed_demo
  "operator_note": ""
}
```

Default visibility:

| kind | shown by default | rationale |
|---|---|---|
| `canonical` | shown | explicitly promoted |
| `manual` | shown | operator created |
| `legacy` | shown | backward compat |
| `experiment_output` | **hidden** | intermediate cards, mostly noise |
| `seed_demo` | **hidden** | forcing-function probe |

Filter row in `suggester.html`:
```
Show: [✓ canonical] [✓ manual] [✓ legacy] [ ] experiment_output [ ] seed_demo   [13 hidden]
```
- Toggling re-filters client-side (no re-fetch).
- "[N hidden]" count shows how many ranked candidates the filter suppresses.
- Toggle state persists to localStorage under `user-personas/suggester-filter-state`.
- Defaults on first load: canonical=on, manual=on, legacy=on, experiment_output=off, seed_demo=off.

Backend principle (normative):

> `/personas`, `/agents`: return all cards including their `provenance` field as written. **NO server-side filtering.** Clients filter.
> `/yapper-seed`: ranks all candidates; includes `provenance` on each `persona` and `agent` in the response. **NO server-side filtering.** The FE filter row decides what to render.

What writes provenance:
- `harness_lib.mjs:saveAgent` (now accepts `provenance` arg): writes `{ kind: 'experiment_output', experiment_id, run_id, iter }`.
- `harness_lib.mjs:saveBio`: writes `{ kind: 'seed_demo', seed_phrase }` when called from seed-textarea materialize path; `{ kind: 'manual' }` from structured editor.
- `scripts/tag_existing_corpus.mjs`: one-time idempotent retroactive tagging, agent filename pattern matching → `experiment_output`; player PNGs matching experiment `canonical_key` → `seed_demo`; anything else → `legacy`.

### Specified API/data contracts

No new endpoints. Existing endpoints extended:
- `validateBioCard` / `validateAgentCard` in `index.mjs`: allow optional `provenance` field.
- `/yapper-seed` response: each candidate's `persona` and `agent` objects now include `provenance`.
- `/personas`, `/agents`: passthrough `provenance` as written; no filtering.

### Open questions / TODOs / known gaps

V2 deferrals:
- Corpus dashboard filter (PR computation toggle "all cards" vs "canonical-only").
- Axis registry filter (scored-on counts filtered by provenance).
- "Pin as canonical" promotion action button.
- Operator-driven cleanup UI (multi-select delete, always confirmed). Described as a distinct surface; explicitly NOT automatic.
- Spec note: checking tag-on-write for new experiment runs during acceptance may be deferred (would require waiting on a 10-min run).

---

## 7. Verbatim-Seed Textarea (Contrast-Spec Input)

**Title:** UI spec — verbatim-seed textarea (contrast-spec input)
**Status:** Approved for implementation
**Surface file date:** Not present in document

### Problem statement

> The 6-word seed phrase is the operator's actual mental representation; the structured editor's `design_brief` field requires the operator to pre-paraphrase.

> This surface eliminates the pre-paraphrase requirement. The operator types verbatim seeds, the FE pairs them combinatorially on operator-picked axes, an experiment-spec card materializes, and `/experiments/:id/run` dispatches the same fixed-point loop the structured editor uses.

Basis: seed ablation result shows verbatim seed `"rpg wizard but he a sagittarius"` was sufficient to drive bio convergence; Claude-paraphrased expansion did not improve convergence in the trial.

### Specified surface

`sillytavern-fork/plugins/user-personas/static/fixed_point.html` — new "Seed input" tab, sibling to the existing "Experiments" section. Opened via existing `#user-fixed-point-button`.

### Specified behaviors

- Textarea syntax: `bios:` / `motives:` section headers (or aliases `agents:`, `agent_targets:`). Lines under each header become seeds. Blank lines and `# comments` ignored. Leading/trailing whitespace trimmed.
- "Parse seeds" button: renders N bio chips and M motive chips, shows "Will produce N × M compositions" count.
- Bio axes multi-select (`GET /axes?kind=bio`) and agent axes multi-select (`GET /axes?kind=agent`). Warning if N > 2^K_b.
- Target-assignment algorithm: **maximally-distant corners** — place each bio/motive at a vertex of `{1, 5}^K` such that pairwise Hamming distance is maximized. Operator can override per-cell in the preview grid.
- Operator names the experiment (default: `seed_<short_hash>_<timestamp>`).
- "Materialize and run": POSTs `/experiments/<id>` (with `design_brief` = verbatim seed line, no paraphrase), then POSTs `/experiments/<id>/run`. Switches to the existing run-progress section.
- Materialize button disabled while N > 2^K_b warning is active.

Parser contract (pure JS, no LLM) specified verbatim:
```js
function parseSeeds(text) { … }
```
Output: `{ bios: string[], motives: string[], warnings: string[] }`.

Slugification: lower-cased, `[^a-z0-9]+` → `-`, truncate to 40 chars, append `-<sha1[:6]>`. `canonical_key = user-personas-<slug>.png`.

Materialized card fields specified verbatim:
- `bios[i].design_brief` = verbatim seed line (no paraphrase)
- `agent_targets[j].motive_hint` = verbatim seed line
- `description` = `"Seed-input experiment. Bios: N seeds. Motives: M seeds. Auto-assigned targets via maximally-distant-corners."`
- `loop_control` omitted; server fills from `validateExperimentCard` defaults.

### Specified API/data contracts

Backend: **no new endpoints**. Uses existing:
- `GET /axes?kind=bio`, `GET /axes?kind=agent` — axis multi-selects
- `POST /experiments/<id>` — save card; must satisfy `validateExperimentCard`
- `POST /experiments/<id>/run` — dispatch

### Open questions / TODOs / known gaps

V2 deferrals:
- LLM proposes new axes from seeds (`POST /experiments/seed-to-axes` — new endpoint, not in V1).
- LLM expands seeds at runtime.
- Editing parsed seed chips inline (V1: edit by retyping in textarea + re-parsing).
- Saving seed-input text as a reusable template.
- Spec item 8 (parser tolerance for seeds without section headers) is left to the implementer to decide — the spec says "implementer picks; document it."

---

## Cross-cutting section

### Vocabulary / conventions defined across multiple specs

- **`canonical_key`** — filename key for bio PNG cards, pattern `user-personas-<slug>.png`. Defined in experiment editor (spec 4) and seed textarea (spec 7).
- **`signature`** — axis→integer-1-5 map on each bio and agent card, used by provenance filter (spec 6), corpus dashboard (spec 3), and suggester (spec 2). Composition signature = agent card's `agent.signature` (spec 3 establishes this).
- **`eps_per_axis`** — convergence threshold used in iteration timeline sparkline coloring (spec 5) and experiment card schema (spec 4).
- **`stop_reason`** vocabulary — `"converged" | "max_outer" | "stall" | "error"` — defined in the results schema (spec 5) and referenced for badge colors.
- **`effDimParticipationRatio` / PR** — the corpus effective-dimensionality metric. Defined and ported from `harness_lib.mjs:279` in spec 3; informally referenced as the corpus objective elsewhere.
- **`provenance.kind`** — `"canonical" | "manual" | "experiment_output" | "seed_demo" | "legacy"` — schema ownership is spec 6; referenced in filter behavior.
- **Neutral-3 rule** — absent axis values default to 3 in L2 distance computation. Mentioned in spec 1 (axis delete warning) and spec 4 (bio target-bio blank = "no preference").
- **Drawer button pattern** (`installFixedPointDrawer`, `#user-fixed-point-button`, etc.) — the shared installation idiom across specs 1 (axes button), 3 (corpus button), and implicitly specs 4/7 (FP-tab button).
- **`SYNTHESIZE_THRESHOLD`** — configurable constant in `suggester.html`, default `2.0`, triggers the Synthesize CTA (spec 2).
- **Maximally-distant corners** algorithm — specified only in spec 7; consumed only by the seed textarea surface.

### Inter-spec dependencies / query param contracts

- Spec 2 (suggester) emits `?target_bio_signature=<urlencoded JSON>` → consumed by spec 4 (experiment editor) for New Experiment pre-population.
- Spec 2 (suggester) emits `?prefill_bio=<bio_id>` → consumed by the FP tab (spec 4/7) to open a pre-filled experiment.
- Spec 6 (provenance) extends the `/yapper-seed` response → consumed by the filter row in spec 2 (suggester).
- Spec 7 (seed textarea) tags bios with `provenance: { kind: 'seed_demo', seed_phrase }` → visible in spec 6 filter and spec 6 tagging script.
- Spec 5 (timeline) requires the `LOCK_IN_ITERATIVE_DIR` constant and two new endpoints; spec 4 (experiment editor) adds the New Experiment / edit modal to the same `fixed_point.html` file.

### Apparent conflicts or disagreements between specs

- Spec 4 says the structured editor passes `provenance: { kind: 'manual' }` when creating a bio directly, and that when the experiment is dispatched the resulting agents will get `experiment_output` via `saveAgent`. Spec 6 elaborates the same point but adds that `saveBio` receives `kind: 'manual'` from the structured editor. This is consistent but distributed — neither spec is a single source of truth for the full write-path.
- Spec 2 says the Synthesize CTA is "the ONLY path" to agent synthesis. Spec 4 and spec 7 both also provide paths to dispatch experiments. These are not in conflict (specs 4/7 are experiment creation surfaces, not ad-hoc one-shot synthesis), but the normative statement in spec 2 is stronger than necessary and could be read as contradicting the existence of the experiment editor surfaces if taken out of context.
- None of the seven specs define a `POST /axes/:id` for editing (as opposed to creating); spec 1 uses that same route for both. The distinction between create and edit on the same POST route is implicit (handled by server-side validation: kind/scale immutable on edit). This is not a conflict between specs, but is a gap in explicit contract definition.
