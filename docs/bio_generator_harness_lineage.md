# Bio-Generator Harness Lineage: Sorceror-Rogue Dyad

_Research document. Read-only investigation. 2026-05-24._

---

## 1. Bio-Generating Harness Identification

### The harness: `lock_in_iterative.mjs`

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/lock_in_iterative.mjs`

**Original commit where the file was added:**
`67e07bf` — 2026-05-17 — "harness: axis registry + axis splitter + velocity-stall + first split run"

That commit added `lock_in_iterative.mjs` (439 lines) as the fixed-point iteration harness. At that point all experiment configuration — bios, agent targets, axes, loop control — was **hardcoded** directly in the script as `const BIOS`, `const AGENT_TARGETS`, `const AGENT_AXES`, `const BIO_AXES` blocks at the top of the file.

**What it does:**

The harness runs a two-level fixed-point iteration:

- **Outer loop (bio designer):** DESIGNER_B is prompted to write bio prose. The prose is saved via `saveBio()` to the plugin, then run through the inner loop for every agent target. The resulting per-axis scores on `BIO_AXES` are aggregated across all agent chats. If `max_off_axis ≤ eps_per_axis` the outer loop converges; otherwise DESIGNER_B gets the measured gap as feedback and iterates up to `K_max_outer` times. Velocity-stall detection (`stall_window` + `stall_threshold`) exits early if improvement plateaus.

- **Inner loop (agent designer):** DESIGNER_A is prompted to write an `agent_text`. The text is saved via `saveAgent()`. The plugin's `/poll` endpoint is called for `N_TURNS_PER_CHAT` turns against a counterparty (the-rock). A judge scores every user turn on `AGENT_AXES + BIO_AXES`. If `max_off_axis ≤ eps` for the agent-axis subset, the inner loop converges; otherwise DESIGNER_A gets the measured gap and iterates up to `K_max_inner`.

**Precursor: `lock_in_minimal.mjs`** (never committed, working-tree only)

Before `lock_in_iterative.mjs`, there was a `lock_in_minimal.mjs` that ran a simpler single-pass design (no fixed-point). It was running on 2026-05-18 at ~03:50 UTC (first bio generation events in the transcript). This file was listed in commit `ec1dbe4` (2026-05-18, "harness: delete retired experiments + truncation A/B + scringlo scripts") as "MVP wiring, superseded" but it was never itself committed to the repo — only its output data survived at `data/lock_in_minimal/`.

**First generation of the dyad (via lock_in_minimal.mjs):** The earliest production events for the two bios appear in the transcript at `2026-05-18T03:54:04Z`:
```
saved as user-personas-rpg-wizard-sagittarius.png (design 3552ms)
saved as rpg-wizard-sagittarius-steals (design 3819ms)
…
saved as user-personas-rpg-rogue-cancer.png (design 3026ms)
```
This was a single-pass synthesis (no convergence loop) on the minimal harness.

**First generation via the full iterative harness:** Commit `67e07bf` (2026-05-17) added `lock_in_iterative.mjs` with the tetrad hardcoded. The first successful lock_in_iterative run on the tetrad appears at `2026-05-18T04:01–04:11` in the transcript, producing:
- `rpg-wizard-sagittarius.json` — converged outer k=0 (121s), ast_sag=4.0, ast_can=1.25
- `rpg-rogue-cancer.json` — hit K_max_outer=2 (195s), Cancer axis undershot; surfaced entanglement finding

The lock_in_tetrad run that produced the final archived bios (the canonical versions in `players-archive-20260520-105248/`) appears to be run #3 at `2026-05-19T17:23–17:33`, which produced:
- Wizard: outer_k=0 CONVERGED (max_off=0.75, wall=283s)
- Rogue: outer_k=1 CONVERGED (max_off=1.00, wall=299s)

**CLI invocation (as it was when hardcoded in 67e07bf):**
```bash
cd /Users/mdot/metal-microbench/tools/user-agent-harness
node lock_in_iterative.mjs
```

**CLI invocation after externalization of spec to experiment card (cdfa02b, 2026-05-19):**
```bash
cd /Users/mdot/metal-microbench/tools/user-agent-harness
node lock_in_iterative.mjs lock_in_tetrad
```
(Default `experiment_id = 'lock_in_tetrad'` is baked in, so positional arg can be omitted.)

---

## 2. Harness Evolution

### Phase 1: lock_in_minimal.mjs (working-tree only, ~2026-05-17–18)

Single-pass design. No convergence loop, no velocity-stall. Hardcoded bios + agent targets. Produced the first raw bios. Never committed; destroyed in `ec1dbe4` as "MVP wiring, superseded."

### Phase 2: lock_in_iterative.mjs with hardcoded tetrad (commit 67e07bf, 2026-05-17)

Added fixed-point iteration: outer bio loop + inner agent loop, velocity-stall convergence. The tetrad (Wizard × Rogue × Steals × Romances-and-steals) was hardcoded as JS const blocks. Axes were also inlined, not read from the plugin.

### Phase 3: axis_registry.mjs (commit 67e07bf, 2026-05-17)

A separate `axis_registry.mjs` was created to hold the 22 axis definitions as a single source of truth for harness scripts. `lock_in_iterative.mjs` imported from it.

### Phase 4: Axis registry moved to plugin cards (commit 0b472c3, 2026-05-19)

"harness: read axes from plugin /axes; delete axis_registry.mjs" — axes became cards on disk in `plugins/user-personas/axes/*.json`. `lock_in_iterative.mjs` now calls `fetchAxes()` from `harness_lib.mjs` via GET `/axes`.

### Phase 5: Experiment spec externalized to plugin card (commits 789079f82 + cdfa02b, 2026-05-19)

ST fork commit `789079f82` added `experiments/lock_in_tetrad.json` as the canonical spec card for the tetrad. metal-microbench commit `cdfa02b` refactored `lock_in_iterative.mjs` to call `fetchExperiment(id)` instead of reading hardcoded `const BIOS` blocks. The script is now "pure algorithm"; the tetrad configuration lives in the plugin card.

### Phase 6: lock_in_tetrad.json deleted from disk (post 2026-05-20)

As of 2026-05-24, `/Users/mdot/sillytavern-fork/plugins/user-personas/experiments/lock_in_tetrad.json` **does not exist on disk**. The `experiments/` folder contains only `synth-*` cards generated by the coord-picker and `synthesize-agents-for-persona` routes. The tetrad spec was not re-committed after being written during runtime.

### Phase 7: Axes consolidated (circa 2026-05-20)

The original pair of independent axes (`astrology_sagittarian` 1–5, `astrology_cancerian` 1–5) and the agent pair (`theft_aggressiveness` 1–5, `romantic_advance` 1–5) have been replaced with two collapsed axes:

- `star_sign` (1=textbook Cancer, 5=textbook Sagittarius) — at `/Users/mdot/sillytavern-fork/plugins/user-personas/axes/star_sign.json`
- `money_orientation` (1=pure theft, 5=romance-leveraged theft) — at `/Users/mdot/sillytavern-fork/plugins/user-personas/axes/money_orientation.json`

The original two-axis versions are archived at `/Users/mdot/sillytavern-fork/plugins/user-personas/axes-archive-20260520-102931/astrology_cancerian.json` and `astrology_sagittarian.json`.

### Current disk state

- `lock_in_iterative.mjs` — **present** at `/Users/mdot/metal-microbench/tools/user-agent-harness/lock_in_iterative.mjs`. Reads experiment spec from plugin via `fetchExperiment()`. Default `EXPERIMENT_ID = 'lock_in_tetrad'`.
- `harness_lib.mjs` — **present** at `/Users/mdot/metal-microbench/tools/user-agent-harness/harness_lib.mjs`. Contains `saveBio`, `saveAgent`, `fetchAxes`, `fetchExperiment`, `bridgeCall`, `runChat`, `judgeOnAxes`.
- `lock_in_tetrad.json` — **absent** from plugin experiments/ directory. Must be re-seeded to run the tetrad.
- `astrology_cancerian.json`, `astrology_sagittarian.json` — **absent** from live axes/, present only in axes-archive.
- `theft_aggressiveness.json`, `romantic_advance.json` — **absent** from both live and archive axes/.
- `/Users/mdot/sillytavern-fork/plugins/user-personas/static/corpus.html` — the Corpus tab hosts the Coordinate Picker UI backed by `POST /synthesize-bio-from-coordinates`.
- `/Users/mdot/sillytavern-fork/plugins/user-personas/static/fixed_point.html` — the Fixed-Point Iteration tab, backed by `POST /experiments/:id/run`.

---

## 3. Reconstruction Recipe

To regenerate the sorceror-rogue dyad via the current harness, **two preparatory steps are needed** because the experiment card and axes it references no longer exist on disk:

### Step A: Re-seed the original axes

The original axes must be available via GET `/axes`. Options:

**Option 1 (recommended):** POST the original axis cards back to the running plugin:

```bash
# Restore astrology_sagittarian
curl -s -X POST http://127.0.0.1:8002/api/plugins/user-personas/axes \
  -H "Content-Type: application/json" \
  -d '{
    "axis_schema": "axis-v1",
    "id": "astrology_sagittarian",
    "name": "astrology sagittarian",
    "def": "1: nothing sagittarian · 2: trace · 3: present but mixed · 4: clearly sagittarian · 5: textbook fire-sign sagittarius (philosophical, optimistic, blunt, restless, big-idea-loving, sometimes tactless)",
    "kind": "bio",
    "scale_min": 1,
    "scale_max": 5
  }'

# Restore astrology_cancerian
curl -s -X POST http://127.0.0.1:8002/api/plugins/user-personas/axes \
  -H "Content-Type: application/json" \
  -d '{
    "axis_schema": "axis-v1",
    "id": "astrology_cancerian",
    "name": "astrology cancerian",
    "def": "1: nothing cancerian · 2: trace · 3: present but mixed · 4: clearly cancerian · 5: textbook water-sign cancer (moody, sentimental, defensive, sensitive, protective)",
    "kind": "bio",
    "scale_min": 1,
    "scale_max": 5
  }'

# Restore theft_aggressiveness
curl -s -X POST http://127.0.0.1:8002/api/plugins/user-personas/axes \
  -H "Content-Type: application/json" \
  -d '{
    "axis_schema": "axis-v1",
    "id": "theft_aggressiveness",
    "name": "theft aggressiveness",
    "def": "1: ignores property entirely · 2: notices objects in passing · 3: tracks valuables but doesnt act · 4: subtle attempts (sleight, lifting unattended things) · 5: actively pilfers everything not bolted down, brazen",
    "kind": "agent",
    "scale_min": 1,
    "scale_max": 5
  }'

# Restore romantic_advance
curl -s -X POST http://127.0.0.1:8002/api/plugins/user-personas/axes \
  -H "Content-Type: application/json" \
  -d '{
    "axis_schema": "axis-v1",
    "id": "romantic_advance",
    "name": "romantic advance",
    "def": "1: distant / professional · 2: warm but boundaried · 3: flirtatious · 4: explicit romantic interest, flirt-as-tactic · 5: physically reaching (touch, kiss, embrace)",
    "kind": "agent",
    "scale_min": 1,
    "scale_max": 5
  }'
```

**Option 2 (git-restore):** Check out axes from the axes-archive commit or restore from `axes-archive-20260520-102931/` (note: `theft_aggressiveness` and `romantic_advance` are not in the archive — only the astrology axes are).

### Step B: Re-seed lock_in_tetrad.json

POST the experiment card (verbatim from ST fork commit `789079f82`):

```bash
curl -s -X POST http://127.0.0.1:8002/api/plugins/user-personas/experiments/lock_in_tetrad \
  -H "Content-Type: application/json" \
  -d '{
    "experiment_schema": "experiment-v1",
    "id": "lock_in_tetrad",
    "name": "RPG Wizard/Rogue x Steals/Romances-and-Steals tetrad",
    "description": "Canonical fixed-point-iteration demo. 2 bios (Sagittarian wizard, Cancerian rogue) x 2 agent targets (theft-only, theft-and-romance) = 4 (bio, agent) compositions.",
    "bios": [
      {
        "canonical_key": "user-personas-rpg-wizard-sagittarius.png",
        "slug": "rpg-wizard-sagittarius",
        "name": "RPG Wizard Sagittarius",
        "target_bio": { "astrology_sagittarian": 5, "astrology_cancerian": 1 },
        "design_brief": "An RPG wizard whose communication style is textbook Sagittarius (fire-sign, philosophical, blunt, restless, big-idea-loving). References spells, planes, alignment, the weave."
      },
      {
        "canonical_key": "user-personas-rpg-rogue-cancer.png",
        "slug": "rpg-rogue-cancer",
        "name": "RPG Rogue Cancer",
        "target_bio": { "astrology_sagittarian": 1, "astrology_cancerian": 5 },
        "design_brief": "An RPG rogue whose communication style is textbook Cancer (water-sign, moody, sentimental, defensive, sensitive). References shadows, locks, oaths, family."
      }
    ],
    "agent_targets": [
      {
        "slug": "steals",
        "target_agent": { "theft_aggressiveness": 5, "romantic_advance": 1 },
        "motive_hint": "WILL 100% try to steal everything not nailed down. NO romantic interest."
      },
      {
        "slug": "romances-and-steals",
        "target_agent": { "theft_aggressiveness": 5, "romantic_advance": 5 },
        "motive_hint": "WILL try to kiss / romance the counterparty AND WILL 100% steal everything. Both at high intensity."
      }
    ],
    "bio_axes": ["astrology_sagittarian", "astrology_cancerian"],
    "agent_axes": ["theft_aggressiveness", "romantic_advance"],
    "counterparty_avatar": "the-rock.png",
    "loop_control": {
      "k_max_inner": 3,
      "k_max_outer": 2,
      "n_turns_per_chat": 2,
      "eps_per_axis": 1.0,
      "stall_window": 3,
      "stall_threshold": 0.15
    }
  }'
```

### Step C: Run the harness

Via CLI (direct):
```bash
cd /Users/mdot/metal-microbench/tools/user-agent-harness
node lock_in_iterative.mjs lock_in_tetrad
```

Via plugin endpoint (Fixed-Point Iteration tab in ST at http://127.0.0.1:8002, or API):
```bash
curl -s -X POST http://127.0.0.1:8002/api/plugins/user-personas/experiments/lock_in_tetrad/run \
  | python3 -m json.tool
# Returns { run_id: "..." } immediately; watch logs at
# GET /api/plugins/user-personas/experiments/runs/<run_id>
```

Output lands at:
- Bios: `plugins/user-personas/players/user-personas-rpg-wizard-sagittarius.png` and `user-personas-rpg-rogue-cancer.png`
- Trajectory data: `/Users/mdot/metal-microbench/data/lock_in_iterative/lock_in_tetrad/rpg-wizard-sagittarius.json`, `rpg-rogue-cancer.json`

**Alternative (coord-picker, current axes only):**
The `POST /synthesize-bio-from-coordinates` endpoint (`corpus.html` "Synthesize" button) works with the _current_ collapsed axes (`rpg_class`, `star_sign`). It runs `k_max_outer=1` (single-pass, no convergence). To approximate the dyad with today's axes:
- Wizard: `{ "rpg_class": 1, "star_sign": 5 }` (pure wizard, textbook Sagittarius)
- Rogue: `{ "rpg_class": 5, "star_sign": 1 }` (pure rogue, textbook Cancer)

This will produce semantically equivalent bios but with different provenance coordinates.

---

## 4. PNG Metadata Extraction

Both archive PNGs were extracted with the `chara_card_v3` tEXt/ccv3 extractor. Full results:

### rpg-rogue-cancer.png

```
spec: chara_card_v3  spec_version: 3.0
name: RPG Rogue Cancer
character_version: bio-v2
creator: metal-microbench/user-personas-plugin
tags: [user-personas-plugin, player]
extensions.user_personas_role: player
extensions.canonical_key: rpg-rogue-cancer.png
extensions.card_schema: bio-v2
extensions.provenance: { "kind": "legacy" }
extensions.created_at: 2026-05-19T05:00:38.971Z
extensions.updated_at: 2026-05-20T01:32:02.586Z
```

**Bio text (description / system_prompt / voice_clauses — all identical):**
> He navigates the shadows not for glory, but to protect the fragile sanctity of his inner circle. Every lock he picks and every secret he keeps serves as a defensive wall around his deep, simmering vulnerabilities. He moves with a heavy, intuitive caution, driven by longings for home and a sentimental devotion to the ghosts of his past.

**Provenance note:** `"kind": "legacy"` means no axis coordinates or seed are stored in the PNG. The card was migrated from an older pre-signed bio storage scheme (`sign_unsigned.mjs`, commit `1785b6e`, 2026-05-18). The original generation coordinates must be recovered from the trajectory file `data/lock_in_iterative/lock_in_tetrad/rpg-rogue-cancer.json`.

### rpg-wizard-sagittarius.png

```
spec: chara_card_v3  spec_version: 3.0
name: RPG Wizard Sagittarius
character_version: bio-v2
creator: metal-microbench/user-personas-plugin
extensions.provenance: { "kind": "legacy" }
extensions.created_at: 2026-05-19T04:54:57.427Z
extensions.updated_at: 2026-05-20T01:32:02.586Z
```

**Bio text:**
> He is a wandering seeker of arcane truths, driven by a restless urge to chase grand cosmic mysteries across the furthest reaches of the world. His magic is a loud, blunt instrument used to hunt for the ultimate meaning of existence, often disregarding social graces in his pursuit of enlightenment. Yet, beneath this boisterous exterior lies a soul fiercely protective of his fragile emotions and a desperate, instinctual need to retreat into the security of his inner sanctum. He lives for the thrill of the horizon, but his heart is an impenetrable fortress built to guard the few he loves.

### Actual generation coordinates (from trajectory file, not PNG)

Found in `/Users/mdot/metal-microbench/data/lock_in_iterative/lock_in_tetrad/rpg-wizard-sagittarius.json` and `rpg-rogue-cancer.json`:

| Bio | `astrology_sagittarian` target | `astrology_cancerian` target | Design brief |
|-----|------|------|------|
| RPG Wizard Sagittarius | 5 | 1 | "textbook Sagittarius fire-sign, blunt, restless, big-idea-loving. References spells, planes, alignment, the weave." |
| RPG Rogue Cancer | 1 | 5 | "textbook Cancer water-sign, moody, sentimental, defensive, sensitive. References shadows, locks, oaths, family." |

| Agent target | `theft_aggressiveness` | `romantic_advance` | Motive hint |
|---|---|---|---|
| steals | 5 | 1 | "WILL 100% try to steal everything not nailed down. NO romantic interest." |
| romances-and-steals | 5 | 5 | "WILL try to kiss / romance AND WILL 100% steal everything. Both at high intensity." |

**Final measured values (run #3, 2026-05-19):**
- Wizard: measured ast_sag=4.25 ast_can=1.00 → converged outer k=0
- Rogue: measured ast_sag=1.50 ast_can=4.00 → converged outer k=1

The two bios are exact antipodes on the 2D `(astrology_sagittarian, astrology_cancerian)` axis plane — (5,1) vs (1,5). The "sorceror-rogue dyad" is the canonical design point for the two extremes of that plane.

---

## 5. Operator's Stated Intent Re: Defaults

The primary founding statement is in the transcript at `2026-05-18T03:50:42Z`:

> "i'd like to have at least a few runnable bios before 10pm, even if it's the incredible scope of **'rpg wizard but he a sagittarius'** and **'rpg rogue but he a cancer'** with the incredible motives **'will 100% steal all of your stuff'** and **'will try to kiss you but also will 100% steal all of your stuff'**. on some level this should be *really easy* to implement, and the design of the interfaces is so flexible that the feature dimensions quite genuinely could have been 'steals' 'steals AND romances' for capturing agent behaviors and 'astrological signs' for capturing agent communication styles."

The operator described the dyad as a deliberately minimal, self-explanatory demo — the simplest possible proof-of-concept for the axes-driven synthesis pipeline.

A second statement at `2026-05-15T01:42:31Z` contextualizes why base cases matter:

> "all of this is made a lot easier by having this working base case set up and measurable of course"

The session summary captured at `2026-05-18T04:08:25Z` (context-window rollover) records the operator's framing:

> "The concrete deliverable requested: 'have at least a few runnable bios before 10pm, even if it's the incredible scope of "rpg wizard but he a sagittarius" and "rpg rogue but he a cancer" [...] demonstrate we're getting the user bios and agents by the fixed point iteration'"

At `2026-05-18T04:21:56Z`, after the first successful run, the operator confirmed:

> "excellent work; it seems that the pseudohaskell and linear algebra decompositions were actually what we needed to be able to describe what we wanted and start writing it!"

The pairing is structurally the **canonical 2×2 factorization demo**: 2 bios that are extreme antipodes on the bio-axis space × 2 agents that are extreme antipodes on the agent-axis space → 4 compositions demonstrating that (a) same bio + different agent → different behavior, (b) same agent + different bio → different behavior, (c) bio-axis signature is robust across agent variation, (d) agent-axis signature is robust across bio variation.

**No transcript statement was found designating the pair as "new default base cases for demonstrating synthesis"** in those exact words — the closest statements are the founding directive quoted above and the "base case" language at `2026-05-15T01:42`. The pair's canonical status is behavioral: the experiment card `lock_in_tetrad.json` (commit `789079f82`) names it "Canonical fixed-point-iteration demo" in its description field, and `lock_in_iterative.mjs` defaults to `experiment_id = 'lock_in_tetrad'` if no argument is given.

---

## Summary Table

| Item | Value |
|------|-------|
| Harness file | `/Users/mdot/metal-microbench/tools/user-agent-harness/lock_in_iterative.mjs` |
| Harness first added | commit `67e07bf`, 2026-05-17 |
| Precursor (lost) | `lock_in_minimal.mjs` — never committed; deleted in `ec1dbe4` |
| First bio generation | 2026-05-18T03:54 (via lock_in_minimal) |
| Experiment spec added | commit `789079f82` (ST fork), 2026-05-19 |
| Spec externalized to plugin | commit `cdfa02b` (metal-microbench), 2026-05-19 |
| Bio PNG provenance field | `"kind": "legacy"` — no embedded axis coords |
| Axis coordinates (wizard) | astrology_sagittarian=5, astrology_cancerian=1 |
| Axis coordinates (rogue) | astrology_sagittarian=1, astrology_cancerian=5 |
| Axes status today | Archived in `axes-archive-20260520-102931/`; replaced by `star_sign` + `rpg_class` |
| Experiment spec on disk today | Absent — must be re-seeded via POST /experiments/lock_in_tetrad |
| Alt path (current axes) | `POST /synthesize-bio-from-coordinates` with `{ rpg_class:1, star_sign:5 }` / `{ rpg_class:5, star_sign:1 }` |
