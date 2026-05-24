# UI spec — D10 iteration timeline view

**Status:** approved for implementation
**Acceptance:** Playwright spec passes against `tools/st-debug` with the existing `lock_in_tetrad` output JSON on disk
**Surface:** new "Trajectory" subsection inside `sillytavern-fork/plugins/user-personas/static/fixed_point.html`, triggered from a row click in the Experiments list
**Backend additions:** 2 new GET endpoints (read-only, file-system passthrough)

---

## Why this exists

After running an experiment, the operator sees only the converged card. The fixed-point loop's *learning curve* — which prose was tried each outer-k, which inner-k attempts hit/missed target, what stop_reason fired — lives in `/Users/mdot/metal-microbench/data/lock_in_iterative/<exp_id>/<bio_slug>.json` and is invisible to the FE. This is half the value of running the loop in the first place; the loop's iteration trace IS the experimental finding, not just the final card.

## Backend additions

Two new plugin endpoints. Read-only, file-system passthrough; no new state.

### `GET /api/plugins/user-personas/experiments/:id/results`

Lists bio slugs that have output JSONs for this experiment.

```jsonc
{
  "experiment_id": "lock_in_tetrad",
  "results": [
    { "bio_slug": "rpg-wizard-sagittarius", "size_bytes": 12834, "mtime": "..." },
    { "bio_slug": "rpg-rogue-cancer",       "size_bytes": 14209, "mtime": "..." }
  ]
}
```

Path: `/Users/mdot/metal-microbench/data/lock_in_iterative/<id>/*.json`. Hard-coded constant `LOCK_IN_ITERATIVE_DIR` at the top of `index.mjs` next to `EXPERIMENTS_DIR`. Returns `{ results: [] }` (not 404) if directory doesn't exist — empty is a valid state.

### `GET /api/plugins/user-personas/experiments/:id/results/:bio_slug`

Returns the full result JSON. 404 if file doesn't exist.

```jsonc
{
  "bio": { "target_bio": {...}, "design_brief": "...", "slug": "...", "name": "..." },
  "agent_targets": [...],
  "bio_axes": [...],
  "agent_axes": [...],
  "result": {
    "stop_reason": "converged" | "max_outer" | "stall" | "error",
    "best": { "iter", "prose", "measured", "dist_per_axis", "max_off_axis", "innerResults" },
    "attempts": [
      { "iter", "prose", "measured", "dist_per_axis", "max_off_axis",
        "innerResults": [
          { "agentTarget": {"slug","target_agent","motive_hint"},
            "attempts": [
              { "iter", "agent_text", "agent_id",
                "chat": [...],
                "measured": {...}, "dist": {...}, "max_off": <num>,
                "converged": <bool>, "wall_ms": <num> }
            ]
          }
        ],
        "bioTurnJudgments": [...],
        "elapsed_ms": <num>,
        "stop_reason"?: "..."
      }
    ]
  },
  "elapsed_ms_total": <num>
}
```

Format is exactly what `lock_in_iterative.mjs` writes; the endpoint just streams the file.

## Surface

A new subsection inside `fixed_point.html` named **Trajectory**. Reached by:
1. Operator clicks any row in the Experiments list with at least one result file.
2. The Trajectory view replaces (or expands below) the Experiments list.
3. A back button or breadcrumb returns to the list.

### Layout

```
┌─────────────────────────────────────────────────────────────────┐
│ ← back to Experiments         lock_in_tetrad                     │
│                                                                  │
│ Bios in this run:                                                │
│   [rpg-wizard-sagittarius] [rpg-rogue-cancer]                    │
│                                                                  │
│ ── rpg-wizard-sagittarius ─────────────────────────────────────  │
│ target_bio: astrology_sagittarian=5 astrology_cancerian=1        │
│ stop_reason: converged · elapsed: 121s                           │
│                                                                  │
│ max_off across outer iterations:                                 │
│   k=0 ● 0.75   k=1 (not reached)                                 │
│   ─────────────                                                  │
│   (small inline sparkline of max_off values, color green/amber/red) │
│                                                                  │
│ ▼ Outer k=0 [CONVERGED max_off=0.75 elapsed=121s]                │
│   ┌ bio prose ──────────────────────────────────────────────────┐│
│   │ A relentless seeker of cosmic truths, this wizard views ... ││
│   └────────────────────────────────────────────────────────────┘│
│   measured: sag=4.25(↓0.75) can=1.25(↓0.25) max_off=0.75        │
│                                                                  │
│   Inner: steals                                                  │
│     ▶ iter 0  agent_text [...] measured the_agg=4 rom_adv=1   ✓ │
│     ▶ iter 1  agent_text [...] measured the_agg=5 rom_adv=1   ✓ │
│   Inner: romances-and-steals                                     │
│     ▶ iter 0  agent_text [...] measured the_agg=5 rom_adv=5   ✓ │
│                                                                  │
│ ▶ Outer k=1 [not reached - converged at k=0]                     │
└─────────────────────────────────────────────────────────────────┘
```

### Rendering rules

- **Bio toggle row** (top): one button per bio in the experiment's results. Click to scroll-into-view (or load) that bio's trace. Disabled if no result file exists.
- **Per-bio header**: target_bio pills, stop_reason badge (color: converged=green / max_outer=amber / stall=amber / error=red), elapsed seconds.
- **Sparkline of max_off**: K_max_outer dots in a row, each colored by max_off value vs `eps_per_axis` (≤eps green, ≤2×eps amber, >2×eps red). Empty dots for iterations not reached.
- **Outer attempt accordion**: collapsed by default if `iter !== result.best.iter`; expanded for the converged-at iteration. Header shows: iter #, status, max_off, elapsed.
- **Bio prose**: monospaced, in a `<pre>` with overflow-scroll. Always shown when accordion expanded.
- **Measured signature line**: `axis=value(↓distance)` per axis. ↓ symbol for under-target, ↑ for over, no arrow for exact. The arrow is informational; the distance is what counts.
- **Inner blocks**: one per agent_target. Inside, one row per inner-loop attempt with: iter, agent_text snippet (collapsible to full), measured agent-axis values, converged-check `✓` / `✗`.
- **Chat preview**: inside each inner attempt, a `[Show chat turns]` toggle reveals the actual chat (counterparty + bio's drafted turns) that was scored. Renders as a 2-column or message-bubble layout. This is the raw evidence the judges saw.

### What's NOT rendered (out of V1 scope)

- Per-turn signature trajectory chart (that's C7, separate spec).
- Judge feedback turns from K-shot retries (that's D11, separate spec).
- Live updates while a run is in progress (only finished runs render). The endpoint can serve in-progress files since the harness writes them as it goes, but the FE polling for live updates is V2.
- Compare-side-by-side mode (two experiments next to each other).

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/44_iteration_timeline.spec.js`

The spec uses the existing `lock_in_tetrad` results already on disk (left over from this session's runs). If those aren't present, the spec skips with an annotation explaining the seeding requirement.

Asserts:

1. **Endpoint contract**: `GET /experiments/lock_in_tetrad/results` returns `{ results: [{bio_slug, ...}, {bio_slug, ...}] }` with both bio slugs.
2. **Per-bio endpoint**: `GET /experiments/lock_in_tetrad/results/rpg-wizard-sagittarius` returns JSON whose top-level keys include `bio`, `agent_targets`, `result`, `elapsed_ms_total`.
3. **404 path**: `GET /experiments/lock_in_tetrad/results/nonexistent-bio` returns 404 (not 500).
4. **FE renders**: clicking the `lock_in_tetrad` row in the Experiments list reveals the Trajectory section with both bio buttons.
5. **Bio header**: clicking `rpg-wizard-sagittarius` button shows target_bio pills, stop_reason badge with `converged` text, elapsed time.
6. **Outer accordion**: at least one outer-attempt accordion renders, with bio prose visible (or expandable), measured signature pills present, max_off shown.
7. **Inner blocks**: at least one inner block (per agent_target) renders with at least one attempt row inside.
8. **Chat preview**: clicking `[Show chat turns]` on an inner attempt reveals at least one chat message bubble.
9. **Back button** returns to the Experiments list.

Run: `cd /Users/mdot/metal-microbench/tools/st-debug/tests && npx playwright test 44_iteration_timeline.spec.js --project=desktop`

Must pass green. No cleanup needed (endpoint is read-only; doesn't create state).

## Out of scope (V2)

- Live polling for in-progress runs.
- Comparing two experiments side-by-side.
- Exporting a trace as markdown or CSV.
- Filtering / searching inside a trace.

## File touch list

- `sillytavern-fork/plugins/user-personas/index.mjs` — add `LOCK_IN_ITERATIVE_DIR` constant, two new GET routes (`/experiments/:id/results` and `/experiments/:id/results/:bio_slug`).
- `sillytavern-fork/plugins/user-personas/static/fixed_point.html` — add Trajectory subsection + click-handler on Experiments list rows + rendering JS for the trace shape.
- `metal-microbench/tools/st-debug/tests/44_iteration_timeline.spec.js` — new spec.
