# UI spec — F16 provenance tagging + view filter (no auto-deletion)

**Status:** approved for implementation
**Acceptance:** Playwright spec passes against `tools/st-debug`; existing specs continue to pass
**Surfaces:** schema additions in the plugin + tag-on-write in harness + suggester filter toggle (corpus dashboard + axis registry are V2 extensions, structurally ready to receive the same toggle)
**Hard constraint:** **NEVER auto-delete a card.** Filtering hides cards from views; the filesystem retains them indefinitely. Cleanup is always operator-clicked + confirmed.

---

## Why this exists, restated

The lock_in_tetrad bios and the lock_in_iterative agent outputs are not "canonical user bios" — they were forcing-function probes designed to make the harness expose its feature-factorization machinery. Currently they show up in `/yapper-seed` rankings as if they were canonical operator-curated personas. The fix is structured tagging on each card, applied automatically by the writer that produced the card, surfaced in views via a filter the operator can toggle. The cards stay on disk; the views auto-collapse them out of default surfacing.

The earlier "transient: true" framing was the wrong abstraction. It conflated "view-filter property" with "auto-delete license" and would have shipped a destruction-of-evidence pattern under the cover of an experiment-output cleanup. Provenance tagging is the right separation.

## Schema: `provenance` field on bio + agent cards

Added as an OPTIONAL field on both bio (player PNG card `extensions.provenance`) and agent JSON (`provenance`). Missing field = treated as `{ kind: "legacy" }` on read.

```jsonc
"provenance": {
  "kind": "canonical" | "manual" | "experiment_output" | "seed_demo" | "legacy",
  "experiment_id": "lock_in_tetrad",        // present when kind=experiment_output
  "run_id": "lock_in_tetrad-2026-...",      // ditto
  "iter": { "outer": 0, "inner": 1 },       // ditto
  "seed_phrase": "rpg wizard but he a sagittarius",   // present when kind=seed_demo
  "operator_note": ""                       // free text, edited later
}
```

### Default kind-to-visibility mapping (in views, not in storage)

| kind | shown by default in suggester / corpus dashboard / axis registry | rationale |
|---|---|---|
| `canonical` | ✓ shown | explicitly promoted by operator |
| `manual` | ✓ shown | operator created via editor; ready to use |
| `legacy` | ✓ shown | untagged; backward compat — we don't know its purpose |
| `experiment_output` | ✗ hidden by default | intermediate fixed-point iteration cards; lots of them, mostly noise |
| `seed_demo` | ✗ hidden by default | forcing-function probe; not for production chat use |

Default state of the filter is a UI choice — operator's toggle preference is persisted to localStorage and re-applied on next page load.

## What writes provenance, and when

### 1. `harness_lib.mjs:saveAgent` — auto-emits `experiment_output`

`saveAgent` is called from `lock_in_iterative.mjs` during the fixed-point loop. It POSTs to `/agents/:id`. Currently it sends `{name, agent_text, designed_for_bio_id, injection_mode, injection_depth, signature}`. Extend to also send `provenance`.

Signature change:
```js
// Was: saveAgent(agent_id, name, agent_text, designed_for_bio_id)
// Now: saveAgent(agent_id, name, agent_text, designed_for_bio_id, provenance)
//   where provenance = { kind: 'experiment_output', experiment_id, run_id, iter }
```

`lock_in_iterative.mjs` already knows its `EXPERIMENT_ID` (line 50) and constructs `run_id` (via the dispatch from the plugin). Plumb both into the saveAgent call sites; add an `iter: {outer, inner}` field per inner-loop attempt.

### 2. Seed textarea materialization in `fixed_point.html`

When the operator clicks "Materialize & dispatch" in the seed input tab, the FE assembles the experiment-spec card and POSTs it. The harness then dispatches lock_in_iterative which will eventually produce agent cards. But the **bio PNG cards** are written by the harness too (via `saveBio` in harness_lib). These should carry `provenance: { kind: 'seed_demo', seed_phrase: <verbatim design_brief> }`.

Extend `saveBio` similarly:
```js
// Was: saveBio({canonical_key, name, prose})
// Now: saveBio({canonical_key, name, prose, provenance})
```

The seed textarea passes `provenance: { kind: 'seed_demo', seed_phrase: line }` when materializing. The structured experiment editor passes `provenance: { kind: 'manual' }`.

### 3. Manual creation (designer.html / structured editor)

If/when the designer creates a bio/agent directly (no experiment run), the card gets `provenance: { kind: 'manual' }`. The structured editor in `fixed_point.html` already POSTs experiment-spec cards; when its experiment is dispatched, the resulting agents will get `experiment_output` via path (1). No change needed at the editor itself.

### 4. Retroactive tagging script

A one-time idempotent script `scripts/tag_existing_corpus.mjs` walks `players/` and `agents/` directories and assigns provenance based on filename pattern:

- Agent filenames matching `<bio_slug>-<motive_slug>-iter<N>` → `experiment_output`. Inferred fields: `iter.outer = N`, `experiment_id` if we can match against existing experiment-spec cards by checking which experiment has a matching bio_slug + motive_slug combination; otherwise omit.
- Player PNG cards whose `canonical_key` matches the pattern in any experiment-spec's `bios[].canonical_key` → `seed_demo` with `seed_phrase` pulled from that experiment's `bios[i].design_brief`.
- Anything else → `legacy`.

Script writes the provenance back to each card via POST `/personas/:id` and POST `/agents/:id`. The plugin's POST routes preserve and persist the new field.

The retroactive tagging runs once: this session has 20 agents + 4 players that need tagging. After that, all new writes carry provenance from creation.

## Suggester filter UI (the only V1 view-side surface)

In `suggester.html`, add a filter row above the ranked list (in the `_meta` strip area or below it — either is fine):

```
Show: [✓ canonical] [✓ manual] [✓ legacy] [ ] experiment_output [ ] seed_demo   [13 hidden]
```

- Toggling re-runs the ranked-list filter client-side (no re-fetch — the FE already has all candidates from `/yapper-seed`).
- The "[N hidden]" count shows how many ranked candidates the current filter is suppressing.
- Toggle state persists to localStorage under `user-personas/suggester-filter-state`.
- Defaults on first load: canonical=on, manual=on, legacy=on, experiment_output=off, seed_demo=off.

The candidate rendering already has `row.persona` and `row.agent` available; the `provenance` field is on each. If the FE doesn't currently get provenance from `/yapper-seed`'s candidate payload, extend that endpoint's response to include it (`candidate.persona.provenance`, `candidate.agent.provenance`). The L2 distance computation is unaffected; only the surfacing is.

## What the plugin endpoints return

- `/personas`, `/agents`: return all cards including their `provenance` field as written. NO server-side filtering. Clients filter.
- `/yapper-seed`: ranks all candidates; includes `provenance` on each `persona` and `agent` in the response. NO server-side filtering. The FE filter row decides what to render.

This is the principle: storage and ranking are unfiltered; views filter on top.

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/47_provenance_filter.spec.js`

Pre-condition: the retroactive tagging script has run (against the current corpus of 20 agents + 4 players). The spec can run it idempotently as a setup step, OR assume it ran at install time — implementer picks.

Asserts:

1. **Schema persistence**: `GET /agents/rpg-wizard-sagittarius-steals-iter0` includes `provenance: { kind: 'experiment_output', ... }` after the retroactive tagging runs.
2. **`/yapper-seed` includes provenance**: each candidate's `persona` and `agent` objects have a `provenance` field (possibly `{kind: 'legacy'}` for untaggable cards).
3. **Filter row renders**: suggester sidebar shows the filter row with 5 checkboxes; the experiment_output and seed_demo are unchecked by default.
4. **Default filter hides experiment outputs**: with the populated corpus tagged, ranking returns N candidates; the FE renders only those with kind ∈ {canonical, manual, legacy} — assert the visible count < total returned, and the hidden count badge matches.
5. **Toggle reveals**: clicking the experiment_output checkbox makes the hidden cards visible (assert visible count increases by exactly the previously-hidden experiment_output count).
6. **Persistence**: reload the suggester drawer; assert the toggle state matches what was set before reload.
7. **Tag-on-write for new experiment runs**: dispatch an experiment via the existing fixed_point.html flow (use the existing `lock_in_tetrad_verbatim` seed — don't run to convergence, just verify the dispatched card's provenance is recorded somewhere observable; or skip this assertion and annotate as deferred to a future test if checking it would require waiting on a 10-min run).
8. **No card was deleted**: spec compares `ls plugins/user-personas/agents` before and after, asserts every original file still exists.

Run: `cd /Users/mdot/metal-microbench/tools/st-debug/tests && npx playwright test 47_provenance_filter.spec.js --project=desktop`

Must pass green. No cleanup needed (no new cards created); the spec must NOT delete or otherwise modify cards.

## Out of scope (V2 deferrals — structurally ready)

- **Corpus dashboard filter**: PR computation toggle between "all cards" and "canonical-only." Easy add later; just respect the same localStorage key.
- **Axis registry filter**: scored-on counts can optionally filter. Same key.
- **"Pin as canonical" promotion action**: a button on a ranked candidate row that POSTs `/agents/:id` (or `/personas/:id`) with `provenance.kind: "canonical"`. Trivial endpoint addition; the V1 surfaces don't need to ship it.
- **Operator-driven cleanup UI**: a "Cleanup" tab listing cards by kind with multi-select delete (always confirmed). Distinct surface; comes later. NEVER automatic.

## File touch list

- `sillytavern-fork/plugins/user-personas/index.mjs` — extend `validateBioCard` / `validateAgentCard` to allow optional `provenance`. Extend `/yapper-seed` to surface `provenance` on each ranked candidate's `persona` + `agent`.
- `metal-microbench/tools/user-agent-harness/harness_lib.mjs` — `saveAgent` + `saveBio` accept `provenance` arg and forward to plugin POST.
- `metal-microbench/tools/user-agent-harness/lock_in_iterative.mjs` — pass `provenance: { kind: 'experiment_output', experiment_id, run_id, iter }` to `saveAgent`. The `run_id` comes from `process.env.RUN_ID` (the plugin's dispatch should set it; if not currently set, the plugin's `/experiments/:id/run` handler should be extended to pass it as env).
- `sillytavern-fork/plugins/user-personas/static/fixed_point.html` — seed-textarea materialize path tags bios with `provenance: { kind: 'seed_demo', seed_phrase }`.
- `sillytavern-fork/plugins/user-personas/static/suggester.html` — filter row + toggle persistence + filter logic in the render path.
- `sillytavern-fork/plugins/user-personas/scripts/tag_existing_corpus.mjs` — NEW. One-shot idempotent retroactive tagging. Operator runs it once.
- `metal-microbench/tools/st-debug/tests/47_provenance_filter.spec.js` — NEW.

## What this surface UNBLOCKS

- The suggester finally surfaces what a real operator wants (canonical personas) instead of the 20 lock_in_tetrad iteration cards.
- The corpus dashboard's PR can later compute "production PR" vs "exploratory PR" by filtering, giving a much more useful metric.
- Cleanup work becomes safe: an operator can delete cards explicitly by selecting them in a future cleanup UI; nothing happens behind their back.
- Resumability concerns (a separate spec, R1) become tractable because nothing's auto-shredded — partial run state on disk is the foundation for "resume from iteration N."
