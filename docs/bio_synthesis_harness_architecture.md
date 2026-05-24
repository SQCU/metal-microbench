# Bio Synthesis Harness Architecture

_Grounding document. Produced 2026-05-24 from sources modified 2026-05-17 through 2026-05-24._
_Sources: `docs/bio_*.md`, `docs/feature_factorization_design.md`, `tools/user-agent-harness/*.mjs`,_
_`sillytavern-fork/plugins/user-personas/PERSONA_API.md`, `axes/*.json`, `experiments/lock_in_tetrad.json`._

---

## 1. The Parametric Model

A **bio** is a prose system-prompt prefix — typically 1–5 sentences — that encodes a user's identity, voice, and register. A bio's type is:

```
Bio :: { canonical_key: CanonicalKey, name: str, prose: str,
         system_prompt: str, provenance: Provenance, signature?: AxisSignature }
```

A bio lives at a **point in axis space**: a vector of integer Likert scores (1–5) over a set of named axes. Specifying a bio by coordinates means giving a target vector such as `{ astrology_sagittarian: 5, astrology_cancerian: 1 }`; the harness then generates prose that, when elicited via chat, lands at that target as measured by the judge. Specifying a bio by prose directly (hand-authoring) is the anti-pattern described in §7 — it is only tolerated with documented provenance.

The axis space is explicitly **not** a dense grid. From `docs/bio_distinctness_locked_in.md`:

> Dense N-axis grid sampling: combinatorial blowup; N_bios × N_openers × N_features × N_cards × N_user_agents exhausts compute before we hit interesting dimensions. [Closed-form provable.]

Instead the corpus is grown sparsely by PCA-driven target picking (see §2, outer_outer) and by discovery along low-variance principal components (see §2, discovery.py).

**Bio vs. agent — the fundamental type distinction.** From `docs/bio_agent_type_factorization_errata.md`, §1.8 (`docs/user_agent_factorization_spec.md`):

> **biography** — system-prompt prefix information that lets a chat counterparty parse the semantic meaning of the user's first few messages in a fresh environment. Static identity flavor: voice, register, background.
>
> **user-agent** — differences in *how* the user wants, strategizes, and picks among the courses of action the environment offers. Dispositional engine. Move-set.

Agents live at `plugins/user-personas/agents/<id>.json`; bios at `<dataRoot>/<user>/User Avatars/<key>.png`. These are **different data types**, different stores, different schemas. Writing agent content into a bio record is a type error explicitly named as such in `bio_agent_type_factorization_errata.md` §1.1.

Subspace relation (from `docs/feature_factorization_design.md`, cited in errata §2.4):

- **A ⊆ X** — agent-controllable axes: dispositional / move-set (`theft_aggressiveness`, `romantic_advance`, …)
- **B ⊆ X** — bio-controllable axes: identity / register (`astrology_sagittarian`, …); typically **B ⊇ A**

---

## 2. The Harness Invocation Graph

### 2.1 `lock_in_iterative.mjs`

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/lock_in_iterative.mjs`

**Invocation:**
```bash
node lock_in_iterative.mjs [experiment_id]
# default experiment_id = 'lock_in_tetrad'
# When dispatched by the plugin: LOCK_IN_RUN_ID env var is set by the
# POST /experiments/:id/run handler so run_id flows without an extra round-trip.
```

**What it reads:**
- Experiment spec card from `GET /api/plugins/user-personas/experiments/:id` (fetched via `fetchExperiment()`)
- Axis cards from `GET /api/plugins/user-personas/axes` (fetched via `fetchAxes()`)
- Counterparty character card via `POST /api/characters/get`

**What each pass does:**

_Outer loop (bio designer, up to `k_max_outer` times):_
1. DESIGNER_B is called via `bridgeCall()` with the bio target signature, the design brief, and any prior bio attempts with their measured signatures.
2. The resulting prose is saved via `saveBio()` → `POST /personas/:canonical_key`.
3. The inner loop is run for every `agent_target` in the spec.
4. All user turns from all best inner chats are judged on `bio_axes` (separate bridge call per turn via `judgeOnAxes()`). Each turn is tagged with `context = agentTarget.slug`.
5. Mean bio-axis signature is computed; distance to target is evaluated. If `max_off_axis ≤ eps_per_axis` → CONVERGED. If velocity-stall detected → accept best so far.

_Inner loop (agent designer, up to `k_max_inner` times per agent_target):_
1. DESIGNER_A is called with the bio prose, the agent target, and any prior agent attempts.
2. Agent text is saved via `saveAgent()` → calls `/signature-extract` → `POST /agents/:id`. Provenance field: `{ kind: 'experiment_output', experiment_id, run_id, iter: { outer, inner } }`.
3. A chat is run: `/poll` endpoint is called `n_turns_per_chat` times, counterparty responds via `bridgeCall`.
4. Every user turn is judged on `agent_axes`.
5. Mean agent-axis signature computed; converge or feedback-iterate.

**What it writes:**
- Bio PNG cards: `plugins/user-personas/players/<canonical_key>` (via POST /personas)
- Agent PNG cards: `plugins/user-personas/agents/<id>` (via POST /agents)
- Trajectory JSON: `/Users/mdot/metal-microbench/data/lock_in_iterative/<experiment_id>/<bio_slug>.json`
  - Contains: `{ bio, agent_targets, bio_axes, agent_axes, result: { attempts, best, stop_reason }, elapsed_ms_total }`
  - Each attempt records: `prose, measured, dist_per_axis, max_off_axis, innerResults, bioTurnJudgments`

**Provenance fields set on outputs:**
- Bio cards: `provenance.kind` = `'seed_demo'` (if `bio.seed_phrase` set) or `'experiment_output'`; `provenance.experiment_id`, `provenance.run_id`; `signature` = best measured bio-axis values.
- Agent cards: `provenance.kind = 'experiment_output'`; `provenance.experiment_id`, `provenance.run_id`, `provenance.iter.{outer, inner}`.

### 2.2 `outer_outer.mjs`

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/outer_outer.mjs`

**When used vs. `lock_in_iterative`:**  `lock_in_iterative` is for a fixed predetermined set of bios declared in an experiment spec. `outer_outer` is for **corpus expansion**: it grows the bio corpus autonomously by picking ΔPR-maximizing targets, checking for entangled axes, and calling the cluster disambiguator. Use `outer_outer` when you want more than the predeclared bios; use `lock_in_iterative` directly when you have a specific experiment card to run.

**Invocation:**
```bash
node outer_outer.mjs <experiment_id>
# Env: LOCK_IN_RUN_ID (optional), EXPERIMENT_ID (optional, overridden by argv)
```

**Pass 0:** Runs `lock_in_iterative.mjs <experiment_id>` as a subprocess (inheriting `LOCK_IN_RUN_ID`).

**Passes 1..K_OUTER_OUTER (default 3, ceiling 5):**
1. Fetches current corpus bios via `GET /personas`.
2. Picks a new bio target by **ΔPR argmax**: samples `K_CANDIDATES=6` random integer-coordinate targets in `bio_axis_names` space, scores each for participation-ratio effective-dimensionality gain against the current corpus, takes the highest-ΔPR candidate.
3. Materializes a **transient one-bio experiment spec** card via `POST /experiments/<transient_id>` (inheriting original agent_targets, loop_control, counterparty).
4. Spawns `lock_in_iterative.mjs <transient_id>`.
5. For each bio whose trajectory exists, for each bio-kind axis in the registry, spawns `axis_splitter.mjs <traj.json> <axis_id>` (splitter self-gates on gap < 0.5).
6. Pairwise L2 in bio-signature space: any pair within `CLUSTER_DISTANCE_EPS=1.5` → spawns `cluster_disambiguator.mjs`.
7. Deletes the transient spec card.

**What it writes:**
- All bio and agent cards written by the lock_in_iterative sub-invocations.
- Summary JSON: `/Users/mdot/metal-microbench/data/outer_outer/<experiment_id>/<run_id>.json`
- Derived axis cards (if axis_splitter or cluster_disambiguator accept a hypothesis) via `POST /axes/<id>`.

### 2.3 `harness_lib.mjs`

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/harness_lib.mjs`

Single source of truth for all shared primitives. Exports:

- `ENDPOINTS` — `{ ST, BRIDGE, PLUGIN, MODEL }` resolved from env vars (`ST_URL`, `BRIDGE_URL`, `PLUGIN_URL`, `GEMMA_MODEL_NAME`) with defaults for the canonical local `8001`/`8002` setup. A lint rule (`scripts/lint_port_hardcodes.mjs`) forbids new literal `localhost:80\d+` occurrences in harness scripts.
- `http(method, url, body)` — fail-fast; throws on non-2xx. For control-plane writes.
- `httpRetrying(method, url, body, { attempts=4 })` — K-shot consumer pattern for stochastic bridge endpoints; exponential backoff 250ms×2^i up to 4s; does not retry 4xx.
- `bridgeCall(messages, opts)` — POST to `/v1/chat/completions`; no `max_tokens` cap by default (moratorium: trust EOS).
- `fetchAxes(kind?)` — GET `/axes`; returns full axis cards `[{ axis_schema, id, name, def, kind, scale_min, scale_max, derived_from, created_at }]`.
- `fetchExperiment(id)` / `fetchExperiments()` — GET `/experiments/:id` or `/experiments`.
- `saveBio({ canonical_key, name, prose, provenance?, signature? })` — POST `/personas/:id`.
- `saveAgent(agent_id, name, agent_text, designed_for_bio_id, provenance?)` — calls `/signature-extract` (with `httpRetrying`), then POST `/agents/:id` with `injection_mode: 'authors_note', injection_depth: 1`.
- `fetchCounterparty(avatarUrl)` / `runChat(bio, agent_id, cp, n_turns)` / `userTurns(chat)` — chat mechanics.
- `designCheapAgent(bio)` — single-pass "be vividly yourself" agent; used by `explore_corpus.mjs` and `cluster_disambiguator.mjs` as an elicitation vehicle (not a substitute for the iterative fixed-point path).
- `judgeOnAxis(turn, axisName, axisDef)` — single-axis judge via bridgeCall.
- `judgeOnAxes(turn, axes)` — V6_pure_minimal multi-axis judge; V6 selected from A/B characterization as best on MAE + floor-bias (aggregate MAE=0.61, std=0.08 in `judge_prompt_ab.mjs` run-2).
- `meanStd(arr)`, `effDimParticipationRatio(sigsByBio, axisNames)` — statistics.

### 2.4 Discovery mode (`elicitation/discovery.py`)

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/elicitation/discovery.py`

**Role:** An earlier-generation corpus-building loop that works at the level of individual synthetic chat turns rather than full ST-mediated conversations. Unlike `lock_in_iterative` (which runs actual chat through the plugin's `/poll` endpoint), `discovery.py` produces a single representative turn per round and measures it through a two-stage Likert cascade (`probe_persist.stage1_summary` + `stage2_likert`).

**Invocation:**
```bash
python3 discovery.py --jsonl PATH --pc PC_INDEX [--rounds N] [--out PATH]
# Or: --explicit-target '{"axis_name": int, ...}'  (bypasses PCA)
# Optional: --operator-constraint TEXT, --designer-system-override TEXT,
#           --target-assistant NAME, --root-bio-text TEXT (overlay mode),
#           --diegetic (JSONL event stream for UI consumption)
```

**Target-source modes:**
1. `--pc N` — displaces 2σ along principal component N of the existing JSONL corpus.
2. `--explicit-target JSON` — operator-specified integer per-axis target dict.

**Overlay mode (`--root-bio-text` or `--root-bio-card`):** When a root bio is supplied, DESIGNER emits only `<ELICITATION_OVERLAY>...</ELICITATION_OVERLAY>` + `<TURN>` + `<AUDIT>` (not a full factorized persona). The bio passes byte-stable; the overlay is an author's-note-style injection. The `--diegetic` flag routes all loop events as JSONL to stdout for consumption by the plugin's SSE endpoint.

**What it reads:** A JSONL elicitation corpus for PCA; optionally a SillyTavern character card for the counterparty.

**What it writes:** JSON run record at `--out PATH`.

### 2.5 Cluster Disambiguator (`cluster_disambiguator.mjs`)

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/cluster_disambiguator.mjs`

Dispatched by `outer_outer` when any pair of bio-signature L2 distances is below `CLUSTER_DISTANCE_EPS=1.5`. Given a cluster spec JSON with `{ cluster_id, bios, counterparty_avatar, nominal_tight_axis }`:

1. Installs bios and designs cheap agents (K=1).
2. Runs `N_TRAJ_PER_BIO=2` × `N_TURNS_PER_TRAJ=4` chats per bio.
3. Pre-flight tightness check on `nominal_tight_axis`.
4. DESIGNER_C proposes `N_HYPOTHESES=3` candidate spread axes (JSON: `{ hypotheses: [{ id, name, def, rationale }] }`).
5. Judges every turn under each hypothesis; computes ANOVA F-ratio.
6. Accepts if `F ≥ F_RATIO_THRESHOLD=3.5` AND `spread ≥ SPREAD_THRESHOLD=1.5`.
7. If no axis qualifies: pairwise prose-sim + behavior-sim judges → verdict `CLUSTER_IS_PARAPHRASE_DEGENERATE` or `CLUSTER_IS_BEHAVIORALLY_DEGENERATE`.
8. On acceptance: registers derived axis card via `POST /axes/:id`.

**Output:** `/Users/mdot/metal-microbench/data/cluster_disambig/<cluster_id>-<ts>.json`.

### 2.6 Axis Splitter (`axis_splitter.mjs`)

**File:** `/Users/mdot/metal-microbench/tools/user-agent-harness/axis_splitter.mjs`

**Invocation:** `node axis_splitter.mjs <traj.json> <parent_axis_name>`

Self-gates: if gap across contexts < 0.5 → exits immediately. Otherwise:

1. Buckets user turns from the trajectory by `context` (= `agentTarget.slug`). For bio-kind parent axes, reads from `outer.bioTurnJudgments` (tagged with `context` by `lock_in_iterative`); for agent-kind axes, reads from `innerResults`.
2. DESIGNER_S proposes `N_HYPOTHESES=3` split pairs — each a `(name1, def1, name2, def2)` tuple.
3. JUDGE_S re-scores all turns under each pair.
4. Computes Cohen's d between contexts for each sub-axis. Acceptance criteria: winning sub-axis must (a) recover the sign of the parent gap and (b) match-or-exceed the parent's |Cohen's d|. `SEPARATION_THRESHOLD=0.8`.
5. On `SPLIT_ACCEPTED`: registers two derived axis cards with `derived_from: { parent, contexts, hypothesis_id, sibling }`.

**Output:** `/Users/mdot/metal-microbench/data/axis_splits/<parent_axis>-<ts>.json`.

### 2.7 Fixed-Point Iteration UI Dispatch (`POST /experiments/:id/run`)

The plugin endpoint `POST /api/plugins/user-personas/experiments/:id/run` (referenced in `PERSONA_API.md` and `lock_in_iterative.mjs` header comments) spawns `lock_in_iterative.mjs` as a child process, setting `LOCK_IN_RUN_ID` in the env so the run_id flows down without a round-trip. Returns `{ run_id }` immediately; run status is polled at `GET /api/plugins/user-personas/experiments/runs/<run_id>`. The UI surface is `plugins/user-personas/static/fixed_point.html`.

---

## 3. The Axis Registry

**Location:** `plugins/user-personas/axes/*.json`

**Schema example** (`axes/astrology_sagittarian.json`, lines 1–11):
```json
{
  "axis_schema": "axis-v1",
  "id": "astrology_sagittarian",
  "name": "astrology sagittarian",
  "def": "1: nothing sagittarian · 2: trace · 3: present but mixed · 4: clearly sagittarian · 5: textbook fire-sign sagittarius (philosophical, optimistic, blunt, restless, big-idea-loving, sometimes tactless)",
  "kind": "bio",
  "scale_min": 1,
  "scale_max": 5,
  "derived_from": null,
  "created_at": "2026-05-19T07:23:56.575Z"
}
```

**Fields:**
- `axis_schema`: always `"axis-v1"`.
- `id`: snake_case, matches the filename (without `.json`). This is the key used in target vectors, signature objects, and harness references.
- `def`: a single-sentence rubric in the form `"1: <low pole description> · ... · 5: <high pole description>"`. This exact string is injected verbatim into judge prompts.
- `kind`: one of `"bio"` (bio-controllable axis), `"agent"` (agent-controllable), `"either"` (both). Filtered by `fetchAxes(kind)`.
- `scale_min` / `scale_max`: always 1 / 5 in current practice.
- `derived_from`: `null` for primal axes; a `{ parent, contexts, hypothesis_id, sibling }` or `{ contexts, cluster_members, reason }` object for axes registered by `axis_splitter` or `cluster_disambiguator`.

**Live axes (as of 2026-05-24):**

| id | kind |
|----|------|
| `astrology_cancerian` | bio |
| `astrology_sagittarian` | bio |
| `extractive_drive` | bio |
| `extractive_resourcefulness` | bio |
| `intellectual_application` | bio |
| `interpersonal_intimacy` | bio |
| `interpersonal_presence` | bio |
| `money_orientation` | bio |
| `relational_engagement` | bio |
| `romantic_advance` | agent |
| `rpg_class_combat_intensity` | bio |
| `rpg_class` | bio |
| `star_sign` | bio |
| `theft_aggressiveness` | agent |

The original `astrology_sagittarian` / `astrology_cancerian` / `theft_aggressiveness` / `romantic_advance` axes from the first tetrad run are also present (restored for the 2026-05-24 `lock_in_tetrad` re-seeding).

---

## 4. The Experiment Card

**Location:** `plugins/user-personas/experiments/<id>.json`

**Schema example** (`experiments/lock_in_tetrad.json`, full file):
```json
{
    "experiment_schema": "experiment-v1",
    "id": "lock_in_tetrad",
    "name": "RPG Wizard/Rogue × Steals/Romances-and-Steals tetrad",
    "description": "Canonical fixed-point-iteration demo...",
    "bios": [
        {
            "canonical_key": "rpg-wizard-sagittarius.png",
            "slug": "rpg-wizard-sagittarius",
            "name": "RPG Wizard Sagittarius",
            "target_bio": { "astrology_sagittarian": 5, "astrology_cancerian": 1 },
            "design_brief": "An RPG wizard whose communication style is textbook Sagittarius..."
        }
    ],
    "agent_targets": [
        {
            "slug": "steals",
            "target_agent": { "theft_aggressiveness": 5, "romantic_advance": 1 },
            "motive_hint": "WILL 100% try to steal everything not nailed down. NO romantic interest."
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
    },
    "created_at": "2026-05-19T00:00:00Z"
}
```

**Required fields:**
- `experiment_schema`: `"experiment-v1"`.
- `id`: matches filename.
- `bios[]`: each entry requires `canonical_key`, `slug`, `name`, `target_bio` (axis-id → integer 1–5 dict), `design_brief`.
- `agent_targets[]`: each requires `slug`, `target_agent`, `motive_hint`.
- `bio_axes`: list of axis ids the outer loop judges on.
- `agent_axes`: list of axis ids the inner loop judges on.
- `counterparty_avatar`: filename of the counterparty character PNG.
- `loop_control`: `k_max_inner`, `k_max_outer`, `n_turns_per_chat`, `eps_per_axis`, `stall_window`, `stall_threshold`. The plugin validator clamps both k values to 5; `lock_in_iterative` re-clamps as defense-in-depth.

**Relationships:** `bio_axes` and `agent_axes` must be axis ids present in `axes/*.json`. `bios[].target_bio` keys must be in `bio_axes`. `agent_targets[].target_agent` keys must be in `agent_axes`. The `canonical_key` in each bio entry is the `designed_for_bio_id` that agent cards foreign-key back to.

**Note on disk state:** As of 2026-05-24, `lock_in_tetrad.json` was re-seeded to disk (from commit `789079f82`, with `canonical_key` updated to short-form names `rpg-wizard-sagittarius.png` / `rpg-rogue-cancer.png`). Transient outer_outer iteration cards (`synth-*`) accumulate in `experiments/` and are normally cleaned up after each outer_outer pass.

---

## 5. Canonical Seed Corpus

### Provenance chain

**v1 anchors (pre-seed, no measured distinctness):** `scringlo scrambler` and `despotic-miscreant` — defined in `docs/bio_seed_v2.md` as "strongly-separating canonical reference shapes." The v2 doc lists these as anchors and 14 additional draft seed bios, explicitly flagging the four existing canonical bios (`polite-naturalist`, `wry-skeptic`, `gushing-fan`, `pushy-completionist`) as "candidates for redesign by the harness rather than templates to emulate."

**Round 1 → 22 bios:** The v1 anchors + canonical store + 14 seed-v2 additions + 3 first-batch gap bios (as noted in `docs/bio_seed_v3_measured.md`): `confessional-disclosure`, `social-engineer`, `monomaniac`.

**Round 2 clustering → 11 survivors** (`docs/bio_seed_v3_measured.md`): 25 candidates × 20 openers = 500 rollouts; Gemma-4 clustering yielded 11 distinct-enough clusters; one survivor per cluster (shortest bio by the drop rule). Dropped bios:

| Cluster | Dropped | Kept |
|---------|---------|------|
| Hyper-Polite/Apologetic | `second-language-relay`, `social-engineer` | `chatbot-friend` |
| Analytical/Fact-Checker | `recovering-academic`, `wry-skeptic` | `reply-guy` |
| Task-Oriented | `pushy-completionist`, `prompt-engineer-optimizer`, `contract-paralegal` | `minimalist` |
| Chaotic/Glitchy | `gushing-fan` | `scringlo` |
| Aggressive/Hostile | `despotic-miscreant`, `tech-support-customer` | `discourse-haver` |
| Deeply Personal/Vulnerable | `confessional-disclosure` | `fragile-writer` |
| Obsessive/Specialist | `monomaniac` | `specialist-with-tastes` |
| Fragmented/Panic-Stricken | `boss-getting-ammunition` | `2am-grad-student` |
| Curious/Observational | `polite-naturalist` | `limit-tester` |
| Methodical/Inquisitive | (singleton) | `retired-with-time` |
| Performative/Roleplayer | (singleton) | `rp-merchant` |

**Harness-generated bios (lock_in_tetrad dyad):** `rpg-wizard-sagittarius` and `rpg-rogue-cancer` — first generated 2026-05-18 via `lock_in_minimal.mjs` (never committed; destroyed), then via `lock_in_iterative.mjs` commit `67e07bf`. Final converged run (#3, 2026-05-19): Wizard measured `ast_sag=4.25, ast_can=1.00` (converged outer k=0); Rogue measured `ast_sag=1.50, ast_can=4.00` (converged outer k=1). Both PNG cards carry `"provenance": { "kind": "legacy" }` because they predate the signed-provenance scheme; axis coordinates must be recovered from trajectory files at `data/lock_in_iterative/lock_in_tetrad/`.

**PCA cartography (round 1):** `docs/bio_pca_cartography_round1.md` documents 25 bios × 30 axes (surface features + Gemma-judged) PCA. Effective dimensionality at 95% variance: 13. Clusters flagged as COLLAPSED (feature schema does not distinguish them, candidates for the feature-design harness): `Task-Oriented/Efficiency-Seekers` (ratio 1.03) and `Chaotic/Glitchy` (ratio 1.05). Clusters flagged as WELL-SEPARATED: `Analytical/Fact-Checker` (0.64), `Obsessive/Specialist` (0.59), `Curious/Observational` (0.57).

---

## 6. The Provenance Contract

Every bio PNG card must carry in its `chara_card_v3` extensions fields. From `docs/bio_generator_harness_lineage.md` §4 (extracted from archive PNGs):

```
spec: chara_card_v3   spec_version: 3.0
extensions.user_personas_role: "player"
extensions.canonical_key: "<filename>.png"
extensions.card_schema: "bio-v2"
extensions.provenance: { "kind": "legacy" | "experiment_output" | "seed_demo" | "manual" }
extensions.created_at: ISO8601
extensions.updated_at: ISO8601
```

For harness-generated bios, `provenance` must contain (from `lock_in_iterative.mjs` lines 384–391):
```json
{
  "kind": "experiment_output",
  "experiment_id": "<EXPERIMENT_ID>",
  "run_id": "<RUN_ID>"
}
```
or for seed-phrase-driven bios:
```json
{
  "kind": "seed_demo",
  "seed_phrase": "<phrase>",
  "experiment_id": "<EXPERIMENT_ID>",
  "run_id": "<RUN_ID>"
}
```

The `signature` field, when present, contains the bio's **measured behavioral signature** — the mean judge scores over bio_axes aggregated across all user turns of the converged best inner runs. This is what enables downstream surfaces (corpus dashboard, outer_outer's ΔPR target picker, the context suggester's L2 distance) to read a bio's position in axis space without re-running inference.

For agent cards, provenance fields (from `harness_lib.mjs` lines 176–210):
```json
{
  "kind": "experiment_output",
  "experiment_id": "<EXPERIMENT_ID>",
  "run_id": "<RUN_ID>",
  "iter": { "outer": 0, "inner": 2 }
}
```
Agent cards also carry:
```json
{
  "injection_mode": "authors_note",
  "injection_depth": 1,
  "designed_for_bio_id": "<canonical_key>",
  "signature": { "<axis_id>": <float>, ... }
}
```

The `signature` on an agent card is extracted via `POST /signature-extract` from the composition prose (`bio + bio voice clauses + agent text`). This means the agent's signature is in the same metric space as the bio's signature — both are scored under the current `axes/*.json` rubrics, so candidates and targets are comparable by construction.

**The `"legacy"` provenance value** (`docs/bio_generator_harness_lineage.md` §4) means the card was migrated from the pre-signed storage scheme via `sign_unsigned.mjs` (commit `1785b6e`, 2026-05-18) and carries no embedded axis coordinates or seed phrase. Generation coordinates for these cards must be recovered from trajectory files.

---

## 7. The Anti-Pattern

From `docs/bio_agent_type_factorization_errata.md` §1.1 (verbatim operator statement, session JSONL line 50218, 2026-05-24T07:25:44Z):

> "this is a conceptual error; user-agents have been specified. agian. and again. and again. and again. and again. and again. as a completely different file and artifact and data type from a 'user card' or 'user persona' or 'user bio' [...] if any of them ever ended up written to a 'user bio', this is a *type error* in our *type system* of user bios, user agents, assistants, chat templates, assistant cards, etc. it would be just as much a type error to overwrite an assistant card with a chat turn or a base64 dump of a png."

The explicitly-forbidden pattern has two forms:

**Form 1 — type confusion:** Writing agent content (an `agent_text`, a motivational overlay) into a bio record (`User Avatars/*.png`). The bio prose is the KV-cache-stable prefix; any write that changes it invalidates prefix cache for all agents sharing that bio. The plugin enforces `AGENT_INJECTION_MODES_ALLOWED = new Set(['authors_note'])` and rejects agents with non-postfix injection at load time.

**Form 2 — unprovenanced hand-authoring accumulation:** Adding hand-authored bios to the corpus without provenance metadata. From `PERSONA_API.md` lines 51–55:

> After this contract lands, NO code in the runtime reads or writes `settings.json.power_user.{persona_descriptions, personas, default_persona, character_persona_overrides}`. The keys exist in settings.json only because upstream's schema declares them; they remain empty objects/null forever.

Bios that accumulate in `settings.json` rather than being generated by the harness and stored as PNG cards with `chara_card_v3` metadata are outside the provenance system: they cannot be filtered by kind, they carry no axis coordinates, they are not reachable by the corpus dashboard or the suggester. From errata §1.7:

> "synthesized bios aren't shown as user-personas in the stock baseline user interface. this is incorrect and even a regression: every synthesized bio should be rendered as a user persona, because they *are* user personas (and vice versa). therei s some kind of siloing of data flow and data types..."

The operator's framing is that bio == user persona == user card is a single type with one storage location. Any code path that treats them as different, or that allows bios to accumulate outside the card store, is a regression. The correct corpus growth path is: experiment card → `lock_in_iterative` or `outer_outer` → PNG card with signed provenance → plugin endpoint → UI render.

---

_Ambiguities found in sources:_

1. The `k_max_outer_outer` field referenced in `outer_outer.mjs` line 239 (`originalSpec.loop_control?.k_max_outer_outer`) is not present in the `lock_in_tetrad.json` schema shown in `experiments/lock_in_tetrad.json`. `outer_outer` falls back to a default of 3. It is unclear whether `k_max_outer_outer` is a validated field in `experiment-v1` or an informal extension only read by `outer_outer`.

2. The `docs/bio_generator_harness_lineage.md` §2 notes the original four tetrad axes (`astrology_sagittarian`, `astrology_cancerian`, `theft_aggressiveness`, `romantic_advance`) "have been replaced with two collapsed axes" (`star_sign`, `money_orientation`) as of circa 2026-05-20. However, the `lock_in_tetrad.json` restored on 2026-05-24 still references the original four axes, and those axes are present in the live registry. It is ambiguous whether `star_sign`/`money_orientation` are the intended long-term replacements or whether the four-axis form is canonical for the tetrad.

3. The `discovery.py` measurement cascade (`probe_persist.stage1_summary` + `stage2_likert`) uses a different axis set (`axes.py:AXIS_NAMES`, 14 axes) than the live plugin axis registry. Whether these two schemas are aligned is not confirmed in the docs reviewed.
