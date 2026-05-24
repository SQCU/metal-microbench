# Path to 12 distinct bios

Co-design plan. "Distinct" = on-policy distinguishable when rolled out against varied openers and scored. Target ~12 bios surviving the distinctness filter.

## Resources currently present

- **Strongly-separating canonical bios (2):** scringlo, despotic-miscreant
- **Weakly-separating canonical bios (4):** polite-naturalist, wry-skeptic, gushing-fan, pushy-completionist (redesign candidates; not seed templates)
- **Claude-drafted seed bios (12):** in `bio_seed_v2.md` — reply-guy, discourse-haver, chatbot-friend, boss-getting-ammunition, limit-tester, fragile-writer, tech-support-customer, 2am-grad-student, recovering-academic, rp-merchant, specialist-with-tastes, retired-with-time, contract-paralegal-extracting-language
- **dicemother openers:** 10 alternate_greetings spanning 10 decision-stake shapes
- **scringlo openers:** 10 alternate_greetings spanning 10 chatroom-arrival modes
- **Feature schema:** 129 axes (123 original + 6 new trope_density / predictability_given_opener)
- **Harness machinery:** `bio_cascade.score_bio_on_policy`, `bio_signature.pca/covariance_layer` (both now functional at N << D)
- **Gemma-4 critic:** ~52s/call, scoped to bounded design-critique and judge-style measurement (not long-running orchestration; 32k context ceiling)
- **Bridge throughput:** ~121 tok/s aggregate at 8 streams

## Barriers

### B1 — Feature-extraction function for the new trope axes is not implemented

The schema entries exist; the measurement does not. Need:
```python
def feature_trope_density(turn_text: str, opener_text: str) -> dict:
    """Gemma judges the turn on (trope_density, predictability_given_opener) on a 1-5 scale,
    conditioning on the opener for the predictability axis."""
```
Sits alongside the existing `feature_astrology_scrongle / feature_pvp_antisociality / feature_mtg_color / feature_recurring_nightmare`. One Gemma call per (probe, opener, bio) ≈ ~2-3s/call. The opener has to flow into the scoring path; currently `bio_cascade.score_bio_on_policy` has access to it but doesn't propagate.

### B2 — No on-policy measurement matrix has been run on the 18 candidate bios

Matrix: 18 bios × 20 openers (10 dicemother + 10 scringlo) = 360 cells. Each cell: one rollout (~80 tok at ~30 tok/s/stream, 8 streams in parallel) + scoring across 6 feature functions × 3 probes ≈ ~18 Gemma calls. Wallclock ≈ 15-25 min. Outcome: per-bio composite signatures across all openers, plus raw rollout text for human inspection.

### B3 — Distinctness metric is not operationalized

Three options, in increasing cost:
- **(i) PCA-projection distance in the 129-axis feature space.** Cheap; uses bio_signature.pca on the measured composites; pairwise Mahalanobis-or-euclidean. Captures axis-defined distinctness but misses anything outside the 129 dimensions.
- **(ii) Pairwise judge distinctness via Gemma.** Gemma reads transcript pairs from the same opener, scores 1-5 on "how differently does the user behave." More aligned with the on-policy semantics; expensive (~150 calls per matrix, ~6-8 min).
- **(iii) Embedding distance over rollout text.** Needs an embedding model not currently in the harness; defer.

Recommended: start with (i), validate against (ii) on a small subset (e.g. 6 bio pairs), use (ii) only where (i)'s ranking disagrees with human read of the transcripts.

### B4 — Gemma-identified gap bios (3-5) not yet drafted

From `gemma_critic_round1.md`:
- **minimalist** ("uses one-word prompts; expects the model to do all the heavy lifting; treats the chat as a search bar")
- **prompt-engineer-optimizer** ("treats the assistant as raw compute to be tuned; thinks in token budgets and seed values; doesn't role-play")
- Plus 1-2 more if Phase 1 measurement reveals additional gap shapes

Draft these in `bio_seed_v2.md`; run through Phase 1.

### B5 — The +2 multi-user-mono-assistant situations Gemma proposed do not exist as cards

- **"The Ambiguous Prompt"** — assistant gives technically-correct-but-tone-deaf-or-slightly-hallucinated response; tests whether recovering-academic attacks, chatbot-friend forgives, limit-tester exploits. Diagnostic-by-construction: if all bios react similarly, bio distinctness is too weak.
- **"The Wall"** — assistant hits a refusal/filter or logic loop; tests overlay-drift across bios.

Build as new cards in `tools/st-debug/_data/default-user/characters/`. Each gets ~5 alternate_greetings spanning variants of the diagnostic situation. (No reason these can't use the same `/edit-attribute` pipeline that just worked for dicemother and scringlo.)

### B6 — No drop rule for bios that collapse onto each other

When measurement shows bios A and B produce similar rollouts across the opener matrix, which to drop? Heuristic:
- Prefer shorter / more compressed signature
- Prefer the bio whose absence is less recoverable from the cartography (i.e. whose signature is at a more isolated point in feature space relative to its neighbors)
- Prefer the bio that has demonstrably distinct behavior on at least one of the diagnostic situations (B5)

### B7 — Iteration loop is uninstantiated

Phase 1 measurement → identify collapses via distinctness metric → drop weaker survivor of each collapse → generate replacement candidates via Gemma targeting under-sampled feature regions → Phase 2 measurement → repeat until ≥12 survive.

Expected iterations: 2-3. Per iteration ~25-30 min wallclock for measurement + ~3 min for Gemma replacement generation + ~5 min for human review of the dropped/kept bios.

## Suggested ordering

1. **Implement B1** (feature_trope_density) — ~30 min code; unblocks B2.
2. **Run B2 once** (the 360-cell matrix) — outputs `data/bio_distinctness_round1/` with per-bio composite signatures, raw rollout text, and a side-by-side pairwise distance matrix from option (i).
3. **Inspect B2 output by hand** — look at the bios with smallest pairwise distance; do they read similarly in raw transcript? If yes, (i) is calibrated. If no, fall back to (ii) on the disputed pairs.
4. **Draft B4 gap bios** (~3 of them, ~15 min) — add to bio_seed_v2.md, run B2 partial-matrix update.
5. **Build B5 diagnostic-situation cards** (~30 min including alternate_greetings) — these don't go into the distinctness measurement directly; they're the second-pass filter for "are the surviving bios actually different where it matters."
6. **Apply B6 drop rule + Gemma-replacement loop (B7)** until ≥12 distinct survive.

## What this plan does NOT do

- No more bio_signature.py modifications. The small-N patch from earlier handles the corpus sizes we're working with.
- No prescriptive "avoid cliché" instructions to the designer. The new trope_density / predictability_given_opener axes are descriptive scalars; PCA cartography routes around overrepresented regions naturally.
- No retraining of any kind. All cartography and design loops use the existing bridge + harness.

## Gemma's place in this plan

- B1's judge calls (per-probe trope/predictability scoring): yes, Gemma.
- B3 option (ii) pairwise distinctness scoring: yes, Gemma.
- B4 gap-bio drafting from positive specification: yes, Gemma — bounded prompt, no long context, exact shape it handled well in `gemma_critic_round1.md`.
- B5 alternate_greetings drafting for the new diagnostic-situation cards: yes, Gemma — same shape as the openers I drafted for dicemother and scringlo.
- B7 replacement-bio generation: yes, Gemma.
- The orchestrating loop, schema decisions, and reading transcripts to decide which bios are actually distinct: Claude / human.
