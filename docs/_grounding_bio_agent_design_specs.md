# Grounding doc: bio/persona/agent design specifications

_Compiled 2026-05-24 from docs modified in the last 7 days. Read-only synthesis — no recommendations._

---

## 1. bio_seed_v2.md

**Filepath:** `/Users/mdot/metal-microbench/docs/bio_seed_v2.md`

**Date:** Not explicitly stated.

**Spec scope:** Defines the 18-entity candidate bio pool feeding the distinctness harness. Two anchors are declared strongly-separating canonical reference shapes; four existing canonical bios are flagged as redesign candidates; twelve new drafts are offered for harness review.

**Normative claims:**

> "The two anchors at the top are the strongly-separating canonical reference shapes. The four weakly-separating bios in the canonical store (polite-naturalist, wry-skeptic, gushing-fan, pushy-completionist) are candidates for redesign by the harness rather than templates to emulate."

> "The twelve following the anchors are this substep's draft additions; they live here for review before any wiring into the canonical store or the discovery harness's inductive prior."

> "Each of these aims to be the kind of bio that, when read by a person who has spent time in actual chatrooms or API logs, produces immediate recognition of 'oh, that one.'"

**Defined vocabulary:**

- **anchor** — a strongly-separating canonical reference shape (scringlo, despotic-miscreant)
- **signature** — the behavioral claim carried by the bio prose itself
- **weakly-separating** — bios in the canonical store that do not separate cleanly; redesign candidates

**Canonical entities enumerated:**

Anchors (2): `scringlo`, `despotic-miscreant`

Seed additions (16): `reply-guy`, `discourse-haver`, `chatbot-friend`, `boss-getting-ammunition`, `limit-tester`, `fragile-writer`, `tech-support-customer`, `2am-grad-student`, `recovering-academic`, `rp-merchant`, `specialist-with-tastes`, `retired-with-time`, `contract-paralegal-extracting-language`, `minimalist`, `prompt-engineer-optimizer`, `second-language-relay`

Existing canonical weakly-separating (4, redesign candidates): `polite-naturalist`, `wry-skeptic`, `gushing-fan`, `pushy-completionist`

**Open questions / TODOs:** None explicitly surfaced in the doc.

---

## 2. bio_seed_v3_measured.md

**Filepath:** `/Users/mdot/metal-microbench/docs/bio_seed_v3_measured.md`

**Date:** Not explicitly stated. References round-2 measurement.

**Spec scope:** Reports measured distinctness results from two rounds of rollout-and-cluster. Defines the 11-bio survivor set after a drop rule (shortest bio per cluster) applied to 9 behavioral clusters across 25 candidates.

**Normative claims (the drop rule):**

> "drop rule (shortest per cluster) survived 11 bios"

> "Round 2 gap-targeting: Claude identified 3 user-population shapes that round-1 clusters did not capture (confessional-disclosure / social-engineer / monomaniac) and drafted bios for them."

**Defined vocabulary:**

- **survivor set** — bios that passed the drop rule; the 11 endorsed bios after round-2 clustering
- **drop rule** — per cluster, keep the shortest bio; drop all others
- **cluster** — behavioral grouping from Gemma rollout analysis

**Canonical entities — survivor set (11):**

| Slug | Cluster |
|---|---|
| `chatbot-friend` | The Hyper-Polite/Apologetic |
| `reply-guy` | The Analytical/Fact-Checker |
| `minimalist` | The Task-Oriented/Efficiency-Seekers |
| `scringlo` | The Chaotic/Glitchy |
| `discourse-haver` | The Aggressive/Hostile |
| `fragile-writer` | The Deeply Personal/Vulnerable |
| `specialist-with-tastes` | The Obsessive/Specialist |
| `2am-grad-student` | The Fragmented/Panic-Stricken |
| `limit-tester` | The Curious/Observational |
| `retired-with-time` | The Methodical/Inquisitive |
| `rp-merchant` | The Performative/Roleplayer |

**Explicitly dropped bios (by cluster):**

- Hyper-Polite: `second-language-relay`, `social-engineer`
- Analytical: `recovering-academic`, `wry-skeptic`
- Task-Oriented: `pushy-completionist`, `prompt-engineer-optimizer`, `contract-paralegal-extracting-language`
- Chaotic: `gushing-fan`
- Aggressive: `despotic-miscreant`, `tech-support-customer`
- Vulnerable: `confessional-disclosure`
- Obsessive: `monomaniac`
- Panic-Stricken: `boss-getting-ammunition`
- Observational: `polite-naturalist`

**Open questions / TODOs surfaced:**

> "Not done this round: feature_trope_density implementation (deferred); B5 diagnostic-situation cards (the-wall / ambiguous-prompt); embedding-distance fallback."

---

## 3. bio_distinctness_locked_in.md

**Filepath:** `/Users/mdot/metal-microbench/docs/bio_distinctness_locked_in.md`

**Date:** Not explicitly stated.

**Spec scope:** Formalizes five intractable methods that must never be used again, and specifies the correct architectural replacements: online (not batch) measurement via prefix-sharing, sparse PCA-driven sampling, a feature-design harness for unseparated clusters, and Gemma as a bounded-scope critic only.

**Normative claims:**

> "No more `/tmp` scripts. Anything that constitutes research record lives in `scripts/` and imports `scripts/bio_distinctness/common.py` for shared bits."

> "No more dense-grid runs of N_bios × N_openers × N_features."

> "No more multi-query prompts."

> "No more regex extractors against LLM output; use `bio_feature_starter.judge_prose` + `extract_json`."

> "No more rejection-sampling-as-pruning. Clustered bios are inputs to the feature-design harness, not candidates for the drop rule."

> "No more system-message-as-instruction. Stable instruction templates go at the **start of a user message** so they participate in prefix-cache; variable prose goes at the **end** of the same user message."

**Defined vocabulary:**

- **prefix-maxx** — doctrine: stable template prefix at start of user message; variable suffix at end; chained calls copy the prefix and change only the suffix
- **suffix-query** — the per-call variable instruction appended after a shared stable prefix
- **online measurement** — measurement accumulates during live ST chat flow, not via batch reprocessing
- **sparse-sampling cartography** — compressed-sensing-shaped approach: total measurement count grows as polylog(N)

**Listed intractable methods (enumerated table):**

| Method | Intractability class |
|---|---|
| Dense N-axis grid sampling | Closed-form provable (combinatorial) |
| Batch re-judging without prefix reuse | Closed-form provable (token cost) |
| Multi-query / multi-task prompts | Empirical + theoretical |
| Rejection-sampling-as-pruning | Subjective (fit-for-purpose mismatch) |
| Regex against LLM-generated output | Empirical |

**Open questions / TODOs:** None explicitly surfaced; doc is prescriptive/declarative.

---

## 4. bio_pca_cartography_round1.md

**Filepath:** `/Users/mdot/metal-microbench/docs/bio_pca_cartography_round1.md`

**Date:** Not explicitly stated.

**Spec scope:** A measurement report. 25 bios projected into 30-axis feature space; effective dimensionality 13 at 95% variance. Per-cluster within/outside ratio determines WELL-SEPARATED vs COLLAPSED status under the new richer feature schema.

**Normative claims:**

> "Clusters with `within/outside < 0.7` are WELL-SEPARATED in the new space. Clusters with `>= 1.0` are COLLAPSED — these are the feature-design-harness candidates."

> "Cluster-separation threshold 0.7 / 1.0 are heuristic; intent is to highlight feature-design candidates (collapsed clusters), not to make hard rejections."

**Defined vocabulary:**

- **within/outside ratio** — within-cluster mean distance divided by outside-cluster mean distance; < 0.7 = well-separated; ≥ 1.0 = collapsed
- **COLLAPSED** — cluster where the feature schema cannot distinguish members; candidates for the feature-design harness
- **WELL-SEPARATED** — cluster consistent with round-2 behavioral grouping in the richer PCA space
- **BORDERLINE** — ratio between 0.7 and 1.0

**Cluster status summary:**

| Cluster | Ratio | Status |
|---|---|---|
| Hyper-Polite/Apologetic | 0.78 | BORDERLINE |
| Analytical/Fact-Checker | 0.64 | WELL-SEPARATED |
| Task-Oriented/Efficiency | 1.03 | COLLAPSED |
| Chaotic/Glitchy | 1.05 | COLLAPSED |
| Aggressive/Hostile | 0.96 | BORDERLINE |
| Deeply Personal/Vulnerable | 0.97 | BORDERLINE |
| Obsessive/Specialist | 0.59 | WELL-SEPARATED |
| Fragmented/Panic-Stricken | 0.95 | BORDERLINE |
| Curious/Observational | 0.57 | WELL-SEPARATED |

**PCA numbers:** PC1 21.1%, PC1+PC2 38.3%, PC1+PC2+PC3 49.3%. Effective dimensionality at 95% variance = 13.

**Open questions / TODOs surfaced:** None explicitly surfaced; doc notes thresholds are heuristic.

---

## 5. bio_pipeline_code_review_and_refactor.md

**Filepath:** `/Users/mdot/metal-microbench/docs/bio_pipeline_code_review_and_refactor.md`

**Date:** Not explicitly stated.

**Spec scope:** Code review identifying duplicated parallel structures across 5337 lines in the bio/feature/discovery subdomain, estimating ~2600 lines deletable, and specifying a 7-abstraction minimal core to replace all of them. Requires sign-off before any code is touched.

**Normative claims:**

> "No more multi-query prompts."

> "No more regex extractors against LLM output."

> "Conservative estimated total deletion: ~2600 lines in this subdomain alone."

> "What I will not do this turn: Write new code. Begin refactoring without sign-off on the consolidation list above."

**Defined vocabulary:**

- **surviving abstractions** — the 7 minimal interfaces: `bridge.call`, `json_extract`, `judge.score`, `features/*.toml`, `cartography.pca`, `discovery.loop`, `measurement.script`
- **parallel structures** — the bio-mode copies of discovery.py, signature.py, axes.py that are 80-92% identical to their originals

**Open questions / TODOs surfaced explicitly:**

- Sign-off needed on which consolidation items to do, skip, or reorder.
- Confirmation on whether refactor should reach into the user-personas JS plugin.
- Confirmation on whether `discovery.py` and `bio_discovery_from_discovery.py` should consolidate by flag promotion or clean re-apply.
- Whether the 4000-line deletion target is binding and which adjacent subdomain passes reach the remainder.

---

## 6. path_to_12_distinct_bios.md

**Filepath:** `/Users/mdot/metal-microbench/docs/path_to_12_distinct_bios.md`

**Date:** Not explicitly stated.

**Spec scope:** Co-design plan specifying the 7 barriers (B1–B7) to reaching ≥12 on-policy distinguishable bios, the measurement matrix required (18 bios × 20 openers = 360 cells), and the ordered steps to resolve them. Target: 12 survivors after the distinctness filter.

**Normative claims:**

> "Target ~12 bios surviving the distinctness filter."

> "No more bio_signature.py modifications."

> "No prescriptive 'avoid cliché' instructions to the designer. The new trope_density / predictability_given_opener axes are descriptive scalars; PCA cartography routes around overrepresented regions naturally."

> "No retraining of any kind."

**Defined vocabulary:**

- **B1–B7** — the 7 barriers to the 12-bio target (B1=feature_trope_density, B2=measurement matrix, B3=distinctness metric, B4=gap bios, B5=diagnostic situations, B6=drop rule, B7=iteration loop)
- **diagnostic situation** — a situation card designed to be behaviorally separating by construction (e.g. "The Wall," "The Ambiguous Prompt")
- **on-policy** — measured through actual rollouts against varied openers, not by text analysis alone

**Open questions / TODOs surfaced:**

- B1: `feature_trope_density` function not yet implemented.
- B2: 360-cell measurement matrix not yet run.
- B3: Distinctness metric not yet operationalized (3 options specified).
- B4: Gap bios (3) not yet drafted.
- B5: Two diagnostic-situation cards ("The Ambiguous Prompt," "The Wall") not yet built.
- B6: Drop rule defined but not applied.
- B7: Iteration loop uninstantiated.

---

## 7. feature_factorization_design.md

**Filepath:** `/Users/mdot/metal-microbench/docs/feature_factorization_design.md`

**Date:** References runs at `2026-05-18T04:44Z` and `2026-05-18T05:26–05:31Z`.

**Spec scope:** Formalizes the linear algebra, pseudohaskell types, and ball-and-stick diagram of the two built layers (inner agent designer, outer bio designer) and two missing layers (outer-outer target selector, axis splitter). Also documents first runs of the axis splitter, cluster disambiguator, outer-outer MVP, and context-suggester Phase A.

**Normative claims:**

> "Regressions that strip parallelism, hide concurrency, or replace multi-stream calls with single-stream calls are thesis-negative even when locally neutral on principles."

> "A proposed split is accepted only if: (1) Sign-recovery: winning sub-axis Cohen's d has the SAME SIGN as the parent's. (2) Magnitude-recovery: winning sub-axis |d| MEETS OR EXCEEDS the parent's |d|. (3) Threshold: qualified sub-axis |d| ≥ 0.8."

> "A SPREAD_AXIS_FOUND verdict requires: f_ratio ≥ F_RATIO_THRESHOLD AND spread ≥ SPREAD_THRESHOLD (1.5 Likert points)."

> "PARAPHRASE_THRESHOLD = 4.0 — pairwise prose-similarity ≥ 4/5 to call paraphrase-degenerate."

> "Lazy-re-check for BehaviorallyDegenerate clusters. Never auto-rerun all stale clusters as a dense batch — trickle."

**Defined vocabulary:**

- **A ⊆ X** — agent-controllable axes (dispositional / move-set)
- **B ⊆ X** — bio-controllable axes (identity / register); B ⊇ A
- **outer-outer** — missing layer: target selector maximizing eff-dim of the corpus
- **axis splitter** — tool to decompose an entangled axis into sub-axes across chat contexts
- **cluster disambiguator** — tool to find a new spreading axis for a tight cluster in B-space
- **ΔPR** — change in participation-ratio effective dimensionality from adding a new bio to the corpus
- **k_max_inner** — maximum iterations for the inner agent-design fixed point
- **SPREAD_AXIS_FOUND / CLUSTER_IS_PARAPHRASE_DEGENERATE / CLUSTER_IS_BEHAVIORALLY_DEGENERATE** — the three honest verdicts from the cluster disambiguator

**Open questions / TODOs surfaced:**

- Splitter: `N_TURNS_PER_CHAT` should be increased from 2 to 4–6 for Rogue-Cancer before re-feeding splitter.
- Splitter: `proposeSplits` prompt should require sign-recovery as a hard constraint on the designer.
- Outer-outer: bio designer show-don't-tell needs a few-shot upgrade.
- Outer-outer: counterparty diversity auto-rotation not yet wired.
- Outer-outer: splitter/disambiguator auto-dispatch not yet wired.
- Outer-outer: sparse-sampling controller (`axis_registry.pickSubset`) not yet integrated.
- Outer-outer: noise-aware objective (E[ΔPR | noise-model]) not yet implemented.
- Cluster disambiguator: `CLUSTER_IS_BEHAVIORALLY_DEGENERATE` verdict never fired in the wild; path exercised in code only.
- Context-suggester Phase B: `/suggest-personas` plugin endpoint not yet built.

---

## 8. multi_user_agent_chat_interface_spec.md

**Filepath:** `/Users/mdot/metal-microbench/docs/multi_user_agent_chat_interface_spec.md`

**Date:** 2026-05-21

**Spec scope:** Normative specification for the multi-agent chat interface built on a forked SillyTavern + custom Metal engine. Defines 11 load-bearing principles, 8 component specs with acceptance criteria, the recent revision wave, 6 extrapolated features, and the paired-agent acceptance protocol.

**Normative claims:**

> "P-ONTOLOGICAL-CLOSURE — Bios without agents are unusable. Usable bios MUST have agents. Synthesis pipelines auto-derive missing agents at first-launch or persona-create time."

> "P-CANONICAL-NOT-MIRRORED — One store per concept. Bios = ST personas (one settings.json store); agents = chara_card_v3 PNGs (one agents/ dir); axes = JSON cards (one axes/ dir). Mirror writes forbidden."

> "P-FINITE-K-DRAWER — Chat-suggestion drawer surfaces K_1 immediately-polling + K_2 suggested-disabled candidates. Never the full corpus, never empty, never paginated past K_1+K_2."

> "P-VISIBLE-RESIDUE — Every tool invocation, forked-agent call, and synthesis dispatch leaves a card in chat DOM. Silent deletion is forbidden."

> "P-SELECTION-IS-DESIGN — Operator clicks ARE training signal. Selection of a bio auto-chains to its agent; agentless bios redirect to the designer."

> "Regressions that strip parallelism, hide concurrency, or replace multi-stream calls with single-stream calls are thesis-negative even when locally neutral on principles. Acceptance review must include a 'does this strengthen or weaken the thesis' beat."

> "To delete or simplify anything, you must explicitly justify against the principle list AND the thesis. Default-deny."

**Defined vocabulary:**

- **bio** — a user-persona: name, description, optional voice_anchor + signature
- **agent** — an elicitation overlay: name, system_prompt, injection_mode (`author_note` / `system_prefix`), optional designed_for_bio_id
- **chat participant** = bio × agent
- **axis** — a behavior dimension with 1-5 scale + judge rubric
- **thesis-negative** — any change that strips parallelism, hides concurrency, or replaces multi-stream with single-stream
- **P-EMPTY-FORM** — anti-pattern: form with bare fields and no pre-fills or examples
- **paired-agent fixed-point review** — acceptance protocol: proposer → reviewer → iterate → apply + playwright spec

**Canonical agent injection modes:** `author_note`, `system_prefix`

**Canonical axis registry (3 root axes):** `rpg_class` (bio), `star_sign` (bio), `money_orientation` (agent)

**Open questions / TODOs (pending from revision wave):**

- **R-AUTO-POLL-K1** — Suggester top-K_1 rows must auto-fire `/poll` in parallel on first paint; currently inert.
- **R-DESIGNER-RESTORE** — Designer.html deleted; agentless bios have no redirect path.
- **R-FIRST-LAUNCH-SYNTH** — Prefab bios with 0 agents not auto-synthesized on plugin boot; /agents = 0 on fresh install.
- **R-LINEAGE-BADGES** — Lineage not shown on persona rows or suggester rows.
- **R-SPLIT-DEMO-PRESTAGED** — Corpus tab ships no pre-staged derived-axis demo.
- **R-COORDINATE-PICKER** — No synthesize-bio-from-coordinates widget.

---

## 9. PERSONA_API.md

**Filepath:** `/Users/mdot/sillytavern-fork/plugins/user-personas/PERSONA_API.md`

**Date:** Not explicitly stated.

**Spec scope:** Specifies the replacement of upstream SillyTavern's settings.json persona storage with a per-entity card store (chara_card_v3 PNGs). Defines CRUD endpoints, the runtime invariant forbidding legacy key access, and the one-time migration script.

**Normative claims:**

> "NO code in the runtime reads or writes `settings.json.power_user.{persona_descriptions, personas, default_persona, character_persona_overrides}`. The keys exist in settings.json only because upstream's schema declares them; they remain empty objects/null forever."

> "The `canonical_key` matches `[A-Za-z0-9._-]+\.png` and IS the filename, IS the in-memory id, IS the value `designed_for_bio_id` foreign-keys point to."

> "Anything the upstream Persona Management UI used to do is either gone or re-implemented against these endpoints."

**Defined vocabulary:**

- **canonical_key** — the filename of a persona PNG; also the in-memory id and the `designed_for_bio_id` foreign key target
- **card store** — per-entity chara_card_v3 PNG files in `plugins/user-personas/players/`; replaces settings.json blob
- **player role** — `extensions.user_personas_role = 'player'` on a persona card
- **signature** — lives at `extensions.signature` on the card

**Open questions / TODOs:** None explicitly surfaced; doc is a finalized contract.

---

## Cross-doc consolidation

### Canonical bio corpus

**Spec-endorsed survivor set (11 — per bio_seed_v3_measured.md drop rule):**

`chatbot-friend`, `reply-guy`, `minimalist`, `scringlo`, `discourse-haver`, `fragile-writer`, `specialist-with-tastes`, `2am-grad-student`, `limit-tester`, `retired-with-time`, `rp-merchant`

**Anchors (2 — canonical in bio_seed_v2.md, both survived to v3):**

`scringlo`, `despotic-miscreant` — note: `despotic-miscreant` was an anchor in v2 but was DROPPED by the v3 drop rule (fell inside the Aggressive/Hostile cluster, lost to `discourse-haver`).

**Candidate pool (bio_seed_v2.md, pre-measurement):** 18 bios total (2 anchors + 16 seeds)

**Explicitly dropped (bio_seed_v3_measured.md):** `second-language-relay`, `social-engineer`, `recovering-academic`, `wry-skeptic`, `pushy-completionist`, `prompt-engineer-optimizer`, `contract-paralegal-extracting-language`, `gushing-fan`, `despotic-miscreant`, `tech-support-customer`, `confessional-disclosure`, `monomaniac`, `boss-getting-ammunition`, `polite-naturalist`

**Target corpus size:** path_to_12_distinct_bios.md sets ≥12 survivors as the explicit goal.

### Canonical agent type rules

Per multi_user_agent_chat_interface_spec.md (P-ONTOLOGICAL-CLOSURE):

> "Bios without agents are unusable. Usable bios MUST have agents."

Agent injection_mode MUST be one of: `author_note` / `system_prefix`. Agents are chara_card_v3 PNGs stored in `plugins/user-personas/agents/`.

Per feature_factorization_design.md: agents are author's-note depth-1 overlays. The inner fixed-point (agent designer) operates over agent_text in overlay-space.

Per PERSONA_API.md: agents may carry `designed_for_bio_id` pointing to the persona `canonical_key`.

### Where docs disagree

1. **bio_seed_v2.md vs bio_seed_v3_measured.md on `despotic-miscreant`:** bio_seed_v2.md designates `despotic-miscreant` as one of the two strongly-separating canonical anchors. bio_seed_v3_measured.md drops it via the drop rule, replacing it with `discourse-haver` as the Aggressive/Hostile cluster survivor.

2. **bio_seed_v3_measured.md vs bio_distinctness_locked_in.md on the drop rule:** bio_seed_v3_measured.md applies rejection-sampling-as-pruning (shortest bio per cluster = drop rule). bio_distinctness_locked_in.md explicitly repudiates this approach: "No more rejection-sampling-as-pruning. Clustered bios are inputs to the feature-design harness, not candidates for the drop rule." The drop rule results in v3 are treated as a historical artifact; future rounds use the feature-design harness for unseparated clusters.

3. **path_to_12_distinct_bios.md vs bio_distinctness_locked_in.md on batch measurement:** path_to_12_distinct_bios.md specifies a 360-cell batch measurement matrix (B2). bio_distinctness_locked_in.md states the 500-cell run and the rejudge "are the last of their shape" and mandates online measurement going forward. The 360-cell B2 plan predates the locked-in architectural decision.
