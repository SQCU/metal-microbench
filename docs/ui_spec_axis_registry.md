# UI spec — B4 axis registry + lineage view (V1)

**Status:** approved for implementation
**Acceptance:** Playwright spec passes against `tools/st-debug` with current axes
**Surface:** NEW file `sillytavern-fork/plugins/user-personas/static/axes.html` + drawer button `#user-axes-button`
**Backend additions:** none (uses existing `GET /axes`, `GET /axes/:id`, `DELETE /axes/:id` from `index.mjs`)

---

## Why this exists

The feature-factorization story has two halves:
- **Discovery**: `axis_splitter.mjs` / `cluster_disambiguator.mjs` run from the CLI and propose new axis cards with `derived_from` set.
- **Inspection / curation**: the operator looks at the axis registry, sees the lineage of splits, accepts or deletes axes.

Today, only discovery exists in code. Inspection is `cat /Users/mdot/sillytavern-fork/plugins/user-personas/axes/*.json | jq` followed by `rm` if you don't like one. No client surface. V1 of this panel is the inspection half. V2 (separate, deferred) will add a live "Propose new axis" button that calls a model.

## Surface

New top-level drawer button `#user-axes-button` (sibling to the other tabs). Opens drawer hosting `axes.html`.

### Layout

```
┌────────────────────────────────────────────────────────────────────┐
│  Axes registry                              [Refresh] [+ Add axis] │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Root axes (no derived_from)                                        │
│                                                                     │
│  ▼ astrology_sagittarian   bio   1-5                                │
│      1: calm/measured · 5: high-arousal/exclamatory                 │
│      scored on: 2 bios, 0 agents                                    │
│                                       [edit] [delete]               │
│                                                                     │
│  ▼ astrology_cancerian     bio   1-5                                │
│      ...                                                            │
│                                                                     │
│  ▼ theft_aggressiveness    agent 1-5                                │
│    │   1: never steals · 5: steals everything not nailed down       │
│    │   scored on: 0 bios, 5 agents                                  │
│    │                                                                │
│    └── theft_via_stealth   agent 1-5    (derived from theft_agg.)   │
│           1: theft is open/brazen · 5: theft is concealed/cunning   │
│           scored on: 0 bios, 0 agents                               │
│                                       [edit] [delete]               │
│                                                                     │
│  Bottom: orphaned signatures (axes that bios/agents reference but   │
│   that don't have a registry entry — surfaces drift)                │
│  (none)                                                             │
└────────────────────────────────────────────────────────────────────┘
```

### Rendering rules

- **Tree layout**: axes with `derived_from == null` are roots; descendants render indented under their parent. ASCII tree branches (`├── └──`) are fine.
- **Per-axis row**: id (monospace), kind badge (bio/agent/meta/either), scale range, def (one-line rubric). Sub-line: `scored on: N bios, M agents` (count of cards whose `signature` object includes this axis).
- **Edit button**: opens an inline edit form for name/def. Submits `POST /axes/:id` with the modified body. Kind/scale are immutable (POST validates and rejects changes to those).
- **Delete button**: opens a confirm dialog showing `orphaned_signatures` count from `DELETE /axes/:id`'s response shape. If orphaned > 0, warning text: "Deleting this axis will leave N bios + M agents with dangling references to it. The L2 distance code in /yapper-seed handles missing axes via the neutral-3 baseline, so this is non-fatal, but the references will persist in their signature blobs."
- **Add axis button**: opens an inline form for id, name, def, kind. POST `/axes/:id`. id is `[A-Za-z0-9._-]+`.
- **Orphaned signatures section**: at bottom of page, lists any axis id that appears in a bio's or agent's `signature` field but NOT in the registry. This is the inverse of "deleting an axis orphans its signatures"; an orphan that *already exists* (e.g. someone deleted an axis without going through the panel) gets surfaced so the operator can either re-create the axis or clean up the signature blobs.

### What's NOT in V1

- **Propose new axis** button that calls a model (V2 — needs new endpoint).
- **Bulk operations** (e.g. delete all dormant axes).
- **Lineage diff** ("what did axis_splitter actually split when it created this?").
- **Live cluster-collapse alerts** (separate from B4, would be its own spec).

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/46_axis_registry.spec.js`

1. **Drawer button installs**: `#user-axes-button` is visible.
2. **Iframe loads**: clicking opens the drawer with `axes.html`.
3. **Root axes render**: all axes from `GET /axes` are present in the page; root axes (derived_from null) render at the top level.
4. **Per-axis row contents**: each axis has id, kind badge, def text, scored-on counts.
5. **Add axis flow**:
   - Click [+ Add axis], fill id=`playwright_test_axis_46`, name=`PW test`, def=`1: foo · 5: bar`, kind=`bio`.
   - Submit. Assert the new row appears in the list. Assert `GET /axes/playwright_test_axis_46` returns 200 with the entered values.
6. **Edit flow**: edit the new axis's def to `1: changed · 5: changed`. Assert the page updates and `GET /axes/playwright_test_axis_46` reflects the change.
7. **Delete flow with orphan warning**:
   - The test axis has 0 orphaned signatures (nothing references it). Click delete, confirm.
   - Assert it vanishes from the list and `GET /axes/playwright_test_axis_46` returns 404.
8. **Derived-axis lineage**: if any axis has `derived_from`, assert it renders indented under its parent. (If no derived axes exist currently, this case is annotated and skipped — derived axes come from `axis_splitter.mjs` CLI runs.)
9. **Orphaned signatures section**: with the current corpus, this is empty. Assert the empty-state message renders.

Run: `cd /Users/mdot/metal-microbench/tools/st-debug/tests && npx playwright test 46_axis_registry.spec.js --project=desktop`

Spec must clean up `playwright_test_axis_46` in afterEach via `DELETE /axes/playwright_test_axis_46`.

## File touch list

- `sillytavern-fork/plugins/user-personas/static/axes.html` — NEW.
- `sillytavern-fork/public/scripts/extensions/user-personas/index.js` — install `#user-axes-button`.
- `metal-microbench/tools/st-debug/tests/46_axis_registry.spec.js` — NEW.

## V2 (NOT this doc, but extension-ready)

A `[ Propose new axis from collapsed bios ]` button would:
1. Identify clusters of bios/agents whose signatures collide on the existing axes.
2. POST to `/axes/propose` with the cluster ids.
3. Plugin runs a bridge call asking the LLM to propose a discriminating axis.
4. Returns proposed axis card(s).
5. Operator confirms/rejects; accepted axes POST to `/axes/:id`.

V1 surface is structured to absorb this without rewrites: the [+ Add axis] section gets a sibling [Propose from clusters] button; the rest of the page stays.
