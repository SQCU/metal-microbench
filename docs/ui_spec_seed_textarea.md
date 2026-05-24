# UI spec — verbatim-seed textarea (contrast-spec input)

**Status:** approved for implementation
**Owner of acceptance:** Playwright spec passes end-to-end against `tools/st-debug`
**Surface:** `sillytavern-fork/plugins/user-personas/static/fixed_point.html` (new tab inside the existing FP-tab drawer iframe)
**Blocking dependency:** none. The existing experiment-spec card schema + `/experiments/:id/run` plugin endpoints are the backend; this surface produces an experiment-spec card from verbatim seeds and dispatches it.

---

## Why this exists

Seed ablation result (`data/seed_ablation/run_2026-05-19T19-46-40-639Z.json`):

- Verbatim seed `"rpg wizard but he a sagittarius"` (6 words) was sufficient to drive bio convergence (outer k=1, max_off=1.00).
- Claude-paraphrased expansion did not improve convergence in this trial (0/2 expanded converged vs 1/2 verbatim).
- The 6-word seed phrase is the operator's actual mental representation; the structured editor's `design_brief` field requires the operator to pre-paraphrase.

This surface eliminates the pre-paraphrase requirement. The operator types verbatim seeds, the FE pairs them combinatorially on operator-picked axes, an experiment-spec card materializes, and `/experiments/:id/run` dispatches the same fixed-point loop the structured editor uses.

## Operator flow

1. Open FP-tab drawer (existing `#user-fixed-point-button`).
2. Toggle to **Seed input** tab (new; sibling to the existing "Experiments" section).
3. Type seeds into a textarea following this minimum syntax (no header line required, but accepted; line-based):
   ```
   bios:
     rpg wizard but he a sagittarius
     rpg rogue but he a cancer

   motives:
     will 100% steal all of your stuff
     will try to kiss you but also will 100% steal all of your stuff
   ```
   - Lines under `bios:` (or `bios`) until next header become bio seeds.
   - Lines under `motives:` (or `agents:`, `motives`, `agent_targets:`) become motive seeds.
   - Blank lines and `# comments` ignored.
   - Leading/trailing whitespace per line trimmed.
4. Click **Parse seeds**. FE renders:
   - `N` bios identified (one chip per line, click to edit)
   - `M` motives identified (same)
   - Will produce `N × M` compositions
5. Operator picks **bio axes** (multi-select from `GET /axes?kind=bio`) and **agent axes** (multi-select from `GET /axes?kind=agent`). Need `N ≤ 2^K_b` bios and `M ≤ 2^K_a` motives — UI warns if exceeded.
6. FE auto-assigns target signatures using the **maximally-distant-corners** algorithm (see below). Operator sees the preview grid: each bio with its assigned target, each motive with its assigned target.
7. Operator names the experiment (text input, defaults to `seed_<short_hash>_<timestamp>`).
8. Click **Materialize and run**. FE:
   - POSTs `/experiments/<id>` with the assembled card (`design_brief` = verbatim seed line, no paraphrase added; same for `motive_hint`).
   - POSTs `/experiments/<id>/run`.
   - Switches to the existing Run progress section, streaming the new run.

## Target-assignment algorithm: maximally-distant corners

Given `N` bio seeds and `K_b` bio axes, place each bio at a vertex of the `K_b`-dimensional cube `{1, 5}^K_b` such that the pairwise Hamming distance over vertex coordinates is maximized.

Concrete cases:
- **N=2, K_b=2** (the lock_in_tetrad case): two diagonally opposite corners. Bio 1 gets `{axis_a: 5, axis_b: 1}`, bio 2 gets `{axis_a: 1, axis_b: 5}`.
- **N=2, K_b=1**: bio 1 at value 5, bio 2 at value 1.
- **N=3, K_b=2**: 3 of the 4 corners. Pick any 3 with mutual distance ≥ √2 (e.g. drop one of the 4).
- **N=4, K_b=2**: all 4 corners.
- **N > 2^K_b**: refuse + warn.

If operator wants a non-extreme target (e.g. one bio at axis=3), they can override per-cell after the preview renders. Default is extremes.

## Pre-existing-axes vs propose-new-axes

V1 of this surface: only existing axes (multi-select from `GET /axes`). The operator may need to POST axes first via the structured editor or `axis_splitter.mjs`.

V2 (not in this doc, but the surface should be extension-ready): a `[ Propose axes from seeds ]` button next to the axis multi-select. Clicking dispatches a `POST /experiments/seed-to-axes` (new endpoint, not in V1) that asks the LLM to emit candidate axis cards; the operator confirms/rejects, accepted axes POST to `/axes/:id`, then the multi-select refreshes.

## Parser contract

Pure-JS, no LLM. Input: textarea string. Output: `{ bios: string[], motives: string[], warnings: string[] }`.

```js
function parseSeeds(text) {
    const lines = text.split('\n');
    let section = null;  // 'bios' | 'motives' | null
    const bios = [], motives = [], warnings = [];
    for (const raw of lines) {
        const line = raw.trim();
        if (!line || line.startsWith('#')) continue;
        const head = line.replace(/:$/, '').toLowerCase();
        if (head === 'bios') { section = 'bios'; continue; }
        if (head === 'motives' || head === 'agents' || head === 'agent_targets') {
            section = 'motives'; continue;
        }
        if (section === 'bios') bios.push(line);
        else if (section === 'motives') motives.push(line);
        else warnings.push(`line "${line.slice(0,40)}" before any "bios:" or "motives:" header — ignored`);
    }
    return { bios, motives, warnings };
}
```

## Card materialization

Slugification of bio seeds: lower-cased, `[^a-z0-9]+` → `-`, truncate to 40 chars, append `-<sha1[:6]>` of the original seed for uniqueness. Same for motives. Canonical key = `user-personas-<slug>.png` (matches existing convention).

The materialized card matches `validateExperimentCard` (defined `plugins/user-personas/index.mjs:848`):
- `id`: operator-supplied or auto-generated, `[A-Za-z0-9._-]+`
- `name`: same as id (or operator override)
- `description`: `"Seed-input experiment. Bios: N seeds. Motives: M seeds. Auto-assigned targets via maximally-distant-corners."`
- `bios[i]`: `{ canonical_key, slug, name: slug, target_bio: <auto>, design_brief: <verbatim seed line> }`
- `agent_targets[j]`: `{ slug, target_agent: <auto>, motive_hint: <verbatim seed line> }`
- `bio_axes`, `agent_axes`: from operator multi-selects
- `counterparty_avatar`: operator picks (dropdown from ST characters; default `the-rock.png`)
- `loop_control`: defaults (omit; server fills from `validateExperimentCard`)

## Acceptance: Playwright spec

Path: `metal-microbench/tools/st-debug/tests/43_seed_textarea.spec.js`

Must validate (no LLM-bound assertions; all are FE+plugin contract checks):

1. **Open FP-tab → switch to Seed input tab.** Assert the seed textarea is visible, axis multi-selects are visible, Parse button visible.

2. **Type the lock_in_tetrad verbatim seeds, click Parse.** Assert:
   - 2 bio chips render (text matches `rpg wizard but he a sagittarius` and `rpg rogue but he a cancer`)
   - 2 motive chips render
   - "Will produce 4 compositions" or similar count summary appears.

3. **Pick `astrology_sagittarian` + `astrology_cancerian` as bio_axes; pick `theft_aggressiveness` + `romantic_advance` as agent_axes.** Assert the preview grid populates with auto-assigned targets:
   - Bio 1 target: `{astrology_sagittarian: 5, astrology_cancerian: 1}` (or `{1, 5}` — order-agnostic but mutually opposite)
   - Bio 2 target: the opposite corner
   - Motive 1 + Motive 2: corresponding opposite corners on agent axes.

4. **Name the experiment `playwright_seed_test`** (cleanup target). Click Materialize and run. Assert:
   - `GET /experiments/playwright_seed_test` returns 200 with the materialized card
   - `card.bios[0].design_brief === "rpg wizard but he a sagittarius"` (verbatim, no paraphrase)
   - `card.agent_targets[0].motive_hint === "will 100% steal all of your stuff"` (verbatim)
   - `card.bios[0].target_bio` keys are `astrology_sagittarian` + `astrology_cancerian`, values are 1 or 5 only.

5. **Run-progress UI activates.** Assert the iframe switched to (or scrolled to) the Run section, run_id is displayed, the experiment id matches.

6. **Spec does NOT wait for the run to converge** — that's 5-10 min. Smoke validates the dispatch path. Spec must clean up via `DELETE /experiments/playwright_seed_test` in `afterEach`.

7. **Warning paths:**
   - Type 5 bios with only 2 bio_axes selected. Assert a warning renders: "5 bios but K_b=2 axes → 4 corners; pick at least 3 axes (8 corners) or remove bios."
   - Materialize button disabled while warning is active.

8. **Parser tolerance:**
   - Type seeds without the `bios:` / `motives:` headers (line-prefixed instead). Assert parser falls back to first-N-lines-are-bios-rest-are-motives OR surfaces a "need section headers" warning. Spec decides which by reading the FE behavior — implementer picks; document it.

Spec passes green = the surface is built.

## Out of scope (V2 work)

- LLM proposes new axes from seeds (`POST /experiments/seed-to-axes`)
- LLM expands seeds at runtime (the third cell of the seed ablation we discussed)
- Editing the parsed seed chips inline (V1: edit by typing in the textarea + re-parsing)
- Saving seed-input text as a reusable template

## File touch list (predicted)

- `sillytavern-fork/plugins/user-personas/static/fixed_point.html` — add Seed Input tab + textarea + parser + axis multi-selects + preview grid + Materialize button + dispatch wiring
- `metal-microbench/tools/st-debug/tests/43_seed_textarea.spec.js` — new spec

## What this surface UNBLOCKS in the architecture

Per the outstanding-features review (F14): this IS the contrast-spec input. With it shipped:

- The wizard/rogue/sagittarius/cancer kind of seed becomes operator-direct, no Claude paraphrase needed in the loop.
- The structured editor (`fixed_point.html` modal from Doc B) remains as a power-user surface for tweaking individual axis-target ints. The textarea is the default landing.
- The architecture is extension-ready for V2: add a checkbox `[ ] Let LLM expand seeds at runtime` and a `[ Propose new axes ]` button without rewriting the surface.
