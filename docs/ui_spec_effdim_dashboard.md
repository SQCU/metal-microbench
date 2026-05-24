# UI spec — A1 corpus effective-dim dashboard

**Status:** approved for implementation
**Acceptance:** Playwright spec passes against `tools/st-debug` with the current populated corpus (5+ agents from `lock_in_tetrad`)
**Surface:** NEW file `sillytavern-fork/plugins/user-personas/static/corpus_dashboard.html` + drawer button `#user-corpus-button` (sibling to `#user-fixed-point-button` and `#user-suggester-button`)
**Backend additions:** none — all computation is client-side over existing endpoints

---

## Why this exists

The outer-outer loop's objective is to grow the *effective dimensionality* of the corpus's behavioral coverage. `effDimParticipationRatio` exists in `harness_lib.mjs:279` and is computed by `lock_in_iterative.mjs`'s outer-outer loop but never surfaces. The operator can run experiments forever without seeing whether the corpus's behavioral coverage is widening, narrowing, or saturating. Without this view, the search has no operator-visible objective.

## Surface

A new top-level drawer button `#user-corpus-button` installs in the ST top row (same `installFixedPointDrawer` pattern in `extensions/user-personas/index.js`). Clicking it opens a drawer hosting `corpus_dashboard.html` in an iframe.

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Corpus dashboard                                          [Refresh] │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    Effective dim (PR):  3.42                                         │
│    Active axes:         22                                           │
│    Bios:                2     Agents: 5     Compositions: 5          │
│                                                                      │
│  ── Per-axis variance contribution ────────────────────────────────  │
│  astrology_sagittarian   ▮▮▮▮▮▮▮▮▮▮▮▮▮  0.18 (20%)                  │
│  astrology_cancerian     ▮▮▮▮▮▮▮▮▮▮▮▮   0.16 (18%)                  │
│  theft_aggressiveness    ▮▮▮▮▮▮▮▮       0.10 (11%)                  │
│  romantic_advance        ▮▮▮▮▮▮         0.08 (9%)                   │
│  ...                                                                 │
│  goal_clarity            ·              0.00 (dormant)               │
│  in_character            ·              0.00 (dormant)               │
│                                                                      │
│  ── Saturation indicator ────────────────────────────────────────── │
│  PR snapshot history:                                                │
│   2026-05-19 09:24  PR=3.42  (5 compositions)  ← current             │
│   2026-05-19 02:11  PR=2.18  (1 composition)                         │
│   2026-05-18 03:50  PR=1.00  (0 compositions, target-only baseline)  │
│                                                                      │
│  Status: corpus PR climbing (Δ +1.24 in last 2 experiments)          │
│  ─ OR ─                                                              │
│  Status: corpus PR stalled (no change in last 3 experiments) — try   │
│           an experiment seed off the current behavioral cluster      │
└─────────────────────────────────────────────────────────────────────┘
```

### Computation

All client-side. Fetch:
- `GET /personas` → bio signatures (each bio's `signature` field, axis→value)
- `GET /agents` → agent signatures
- `GET /axes` → axis registry

For PR computation, USE COMPOSITION SIGNATURES (bio × agent), not bio-only signatures. The composition signature is what `saveAgent` writes to each agent card (the result of `/signature-extract` on the combined prose). Each agent card already has its composition signature in `agent.signature`. PR is over agents.

Port `effDimParticipationRatio` from `harness_lib.mjs:279` to JS in the HTML (it's already JS — copy verbatim and adapt for the in-browser context):

```js
function effDimPR(sigsByComposition, axisNames) {
    const ids = Object.keys(sigsByComposition);
    if (ids.length < 2) return { effDim: null, perAxisVar: {}, n: ids.length, note: 'need ≥2 compositions' };
    const perAxisVar = {};
    for (const a of axisNames) {
        const vals = ids.map(i => sigsByComposition[i][a]).filter(Number.isFinite);
        perAxisVar[a] = vals.length >= 2 ? variance(vals) : 0;
    }
    const totalVar = Object.values(perAxisVar).reduce((a, b) => a + b, 0);
    if (totalVar <= 0) return { effDim: null, perAxisVar, n: ids.length, note: 'zero total variance' };
    const p = {};
    for (const a of axisNames) p[a] = perAxisVar[a] / totalVar;
    const effDim = 1 / Object.values(p).reduce((s, pi) => s + pi * pi, 0);
    return { effDim, perAxisVar, normalized: p, totalVar, n: ids.length };
}
```

### Per-axis variance bar chart

Sort axes by variance contribution descending. Bar width proportional to `normalized[axis]`. Color: green (top quartile of variance), amber (middle), gray (bottom quartile / dormant). Tooltip on each row shows: `axis_id`, `def` (rubric), variance value, count of compositions that scored on it.

### Saturation history

Computed at refresh time. Append a row to a JSONL file at `data/corpus_dashboard/snapshots.jsonl` via a new POST endpoint `POST /corpus-snapshot` (idempotent — appends one row with timestamp, PR, n_compositions). The dashboard reads `GET /corpus-snapshot` (returns the JSONL parsed). V1 is fine if snapshots only happen on refresh; V2 can auto-snapshot at end of each run.

If snapshot history is empty (first run): show only the current PR with "no prior snapshots yet."

Saturation status line:
- If last 1 snapshot → climbing iff PR_now > PR_prev by ≥ 0.1
- If last 3 snapshots all within Δ 0.1 of each other → "stalled" status with the recommendation text
- Otherwise → "climbing" with the Δ

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/45_corpus_dashboard.spec.js`

1. **Drawer button installs**: `#user-corpus-button` is visible in the top row.
2. **Iframe loads**: clicking opens the drawer; `corpus_dashboard.html` renders inside.
3. **PR number renders**: with the current populated corpus (5 agents), a numeric PR is displayed; assert it's > 0 and < axis count.
4. **Bio/agent/composition counts** match `/personas`, `/agents` counts.
5. **Per-axis bar chart**: at least one bar with non-zero variance renders; assert sort-order is descending by variance.
6. **Dormant axes**: at least one axis with variance=0 renders with the dormant marker / gray bar.
7. **Refresh button**: clicking triggers a re-fetch (assert a network request to `/agents` and `/personas` after click).
8. **Snapshot history**: clicking refresh appends a snapshot; subsequent fetches show it in the history list.
9. **Empty-state**: if corpus has 0 compositions (mock by query-param or fresh fixture), the dashboard renders an empty-state message instead of crashing.

Run: `cd /Users/mdot/metal-microbench/tools/st-debug/tests && npx playwright test 45_corpus_dashboard.spec.js --project=desktop`

Must pass green.

## Out of scope (V2)

- Auto-snapshot on experiment-run completion.
- PCA over the full signature matrix (V1 uses participation-ratio over per-axis marginal variance; PCA is a different objective and a different spec).
- Comparing multiple corpora (e.g. before/after a cleanup).
- Drilling into specific axes to see which bios contribute.

## File touch list

- `sillytavern-fork/plugins/user-personas/static/corpus_dashboard.html` — NEW.
- `sillytavern-fork/public/scripts/extensions/user-personas/index.js` — install `#user-corpus-button` drawer, sibling to `#user-suggester-button` and `#user-fixed-point-button`.
- `sillytavern-fork/plugins/user-personas/index.mjs` — 2 endpoints for snapshot storage: `POST /corpus-snapshot` (appends one row), `GET /corpus-snapshot` (returns parsed JSONL).
- `metal-microbench/tools/st-debug/tests/45_corpus_dashboard.spec.js` — NEW.
