# Bio-distinctness study — locked-in description

## 1. What we are working on

A user-descriptivist feature-explorer over the space of behavioral patterns observable in user-side chat trajectories. An iterative actor-critic system across three coupled design layers — bios (user-as-state), situations (multi-user-mono-assistant cards), and features (per-prose scalar/categorical axes) — all sharing the same shape: generator → measurement → cartography → generator update. Bios are designed against a positive descriptivist specification; bios are evaluated on-policy through situation cards in SillyTavern; resulting trajectories are measured against the feature schema; PCA over feature vectors identifies under-explored directions; the schema itself can be extended when clusters don't separate. All measurement is meant to accumulate; the corpus is a sparse cartography object, not a benchmark snapshot.

## 2. Why

To build a research instrument that gets richer as it is used — over the space of all language trajectories observable in chatbot dialogue. The instrument is meant to surface real types of user behavior (not prescriptivist civics-textbook personas), scale with more probes/cards/registers/dyads, and inform downstream uses (suggester sidebar; multi-user-mono-assistant dynamics; eventually triadic/group dynamics; eventually trajectory-level rather than turn-level measurement).

## 3. Intractable methods (do not try these)

| method | intractability | proof-class |
|---|---|---|
| dense N-axis grid sampling | combinatorial blowup; N_bios × N_openers × N_features × N_cards × N_user_agents exhausts compute before we hit interesting dimensions | **closed-form provable** |
| batch re-judging of captured rollouts | pays prefill cost twice per turn because rollout-system-prompt ≠ judge-system-prompt → zero prefix-cache hit; current 500-cell run wasted ~475k tokens of cold prefill that prefix-maxx would have eliminated | **closed-form provable** |
| multi-query / multi-task prompts (single context holding several instructions) | attention is mixed associative recall; multiple instructions create retrieval interference; output quality degrades for each individual task | **empirically measurable** in this project's history; **theoretical** arguments well-established in attention-mechanism literature; the project's prefix-maxx sweep (#149, #152, #153) explicitly factored these out |
| rejection-sampling-as-corpus-selection (drop bios because they cluster) | only sensible if narrowing for a dense grid we never plan to materialize; once sparse sampling is accepted, dropping clustered bios discards exactly the signal that should drive feature-design | **subjective** (fit-for-purpose, not math) |
| regex against LLM-generated symbolic output | LLM grammars are not reliable; every test passes until the model phrases things slightly differently next month; centralized JSON extraction with delimiter wrapping is the only tractable approach | **empirically measurable** (project-internal failure rate over time) |

## 4. Why intractable, in plain terms

**Dense grids**: cost scales as O(∏ N_i) for k axes. Per-cell wallclock is ~10-30s. Each axis added multiplies the bill. Past 3-4 axes we run out of compute. The math we use to make up the gap is PCA-driven sparse sampling + compressed-sensing reasoning + lazy expansion only along high-uncertainty directions.

**Batch re-judging without prefix reuse**: the rollout call that produced a turn used `system = persona, user = opener`. The judge call uses `system = judge-instructions, user = (context + prose)`. The bridge's cache key includes the system message; different systems means zero cache hit. Every judge call re-prefills `(opener + prose)` from scratch even though those tokens were live in cache 30 seconds ago. With 500 calls and ~400 wasted tokens each that's 200k tokens of redundant prefill — measurable in wallclock, projectable in dollars.

**Multi-query prompts**: empirically never worked in this project; consistent with the wider literature on attention-mechanism task-interference. The fix is the prefix-maxx + suffix-query doctrine: one task per call, with the next task launched as a fresh call that **copies the prefix the first call built up and appends a different suffix instruction**, sharing prefill via the bridge's prefix cache.

**Rejection-sampling-as-pruning**: mathematically it works (fewer bios survive). But the goal is feature cartography, not corpus minimization. Clustered bios are signal for the feature-design harness, not signal for the drop rule. The drop rule was a category error rooted in my dense-grid assumption.

**Regex against LLM grammars**: NLP symbolic AI grammars are too brittle to depend on at scale. Even when a regex works today, it silently breaks when the model's output distribution shifts. One centralized parser with delimiter-based extraction confines the failure surface to one place that can be debugged. Many bespoke regexes scattered through the codebase create a maintenance debt that grows unboundedly.

## 5. Intractability provability summary

| method | subjective vs measurable | provable in closed form |
|---|---|---|
| dense grids | measurable | **yes** (combinatorial product) |
| batch re-judging | measurable | **yes** (per-call prefill tokens × calls) |
| multi-query prompts | empirical + theoretical | **partially** (attention-interference results); empirically demonstrated repeatedly |
| rejection-sampling-as-pruning | subjective | no — it's a fit-for-purpose mismatch with the project goal |
| regex on LLM output | empirical | no — but measurable in failure rate over time |

## 6. Actual choices next

Constrained by all of the above. The remaining design space:

### a) Online (not batch) measurement during real ST chat flow

The correct prefix-share architecture:
- Rollout call: user message contains `[stable template prefix + situation + bio + prior chat]` followed by minimal "produce next user turn" suffix. Produces a turn.
- Judge call: a **separate** bridge call. Its user message is `[same stable template prefix + situation + bio + prior chat + the just-generated turn]` with a different **suffix** instruction — "score the latest turn on this schema, output JSON wrapped in `<json>...</json>`."
- Bridge's prefix cache reuses every byte of the shared prefix; only the per-call suffix (instruction + JSON skeleton) is new prefill.
- Per-turn cost: original rollout + ~150 tokens of judgment-suffix prefill + ~150 tokens of judgment-output AR. Roughly +5-7s per chat turn.
- Measurement accumulates organically as ST gets used. No batch reprocessing ever.
- Hooks: `/poll` (user-personas plugin) for turn-by-turn capture; `/iterate` for multi-turn sessions.

### b) Sparse-sampling cartography with PCA-uncertainty-driven targeting

Start with a small set of measurements (~20-30, not 500). PCA over the resulting feature matrix. Identify cells in PC-coordinate space that are under-sampled (high posterior variance in the projection). Add a small number of measurements specifically along those directions. Repeat until projection uncertainty collapses. Compressed-sensing-shaped: total measurement count grows as polylog(N), not linearly in the grid.

### c) Feature-design harness for unseparated clusters

When PCA distance shows cluster C's members have within/outside ≥ 1.0, prompt the feature designer (Gemma in a bounded critic role): "Cluster C contains bios A/B/D, which behave alike on the current schema. Propose a new feature judgment that (i) has non-vanishing variance within C and (ii) has non-vanishing variance across a random sample of bios outside C. Specify the prompt suffix that would elicit the new judgment." Run the proposed feature on a small witness set (the C members + 5-10 random outside bios). If both variance conditions are met, commit the feature; if not, the designer iterates. This is the actor-critic loop applied to the feature schema itself.

### d) Use Gemma as scoped critic at design moments, not as long-running implementer

The critic call demonstrated: ~52s, 4 well-formed sections, 4 real gap identifications. Use it for design review, judge calls (one schema per call, suffix-shaped per (a)), and bounded artifact generation. Do not use it for long-running orchestration or 32k+-context tasks.

### e) Accept the existing corpus as a sparse measurement object

We have 25 bios, 500 captures, one round of 2-axis judgments, partial 8-axis judgments. Extract value from what's already paid for via PCA / cartography over the existing measurements (we already have `pca_cartography.json`). Don't re-batch. Subsequent measurement happens online per (a).

## What this rules out

- No more `/tmp` scripts. Anything that constitutes research record lives in `scripts/` and imports `scripts/bio_distinctness/common.py` for shared bits.
- No more dense-grid runs of N_bios × N_openers × N_features. The 500-cell rollout matrix and the 500-cell rejudge that this study produced are the last of their shape.
- No more multi-query prompts.
- No more regex extractors against LLM output; use `bio_feature_starter.judge_prose` + `extract_json`.
- No more rejection-sampling-as-pruning. Clustered bios are inputs to the feature-design harness, not candidates for the drop rule.
- No more system-message-as-instruction. Stable instruction templates go at the **start of a user message** so they participate in prefix-cache; variable prose goes at the **end** of the same user message; chained calls copy the prefix and change only the suffix. The `judge_prose` helper as currently written needs to be flipped to this shape before it's used again — same logic, different message placement.
