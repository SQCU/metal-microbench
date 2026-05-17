# User-agent workshop harness — design for compositionality validation

**Companion to:**
- `docs/user_agent_likert_methodology.md` — the measurement-instrument note (why we Likert-judge user-agent turns, and what the 14-axis basis is for)
- `plugins/user-personas/index.mjs` — the harness implementation (judge-cascade complete; workshop loop not yet built)

This document captures the **architecture and methodological commitments** of the next stage: validating that user-agent personas factorize into compositional axes that an LLM can independently manipulate. It supersedes the earlier template-substitution-based variation plan (which was discussed in chat 2026-05-14 and never implemented) — that approach is rejected as a category error and the reasoning is captured below.

---

## The bench-model-as-proxy framing

The model behind the bridge for this whole substack is **Gemma-4 26B-A4B**, but the *research questions* are not Gemma-specific. Gemma-4 is the proxy for "the capacities of any API model in abstract" because:

- Frontier API models (Gemini 2/3 Pro, Claude 4, GPT-5, etc.) and small open models all share the same *qualitative* failure modes when asked to do schema-constrained-output on top of paragraph-length pragmatic reasoning, to maintain factorized rewrites across multiple intertwined persona dimensions, or to do in-context learning over numerical feedback signals. What differs across the model scale is **the horizon at which the failure becomes observable** — bigger models start failing at 8K-token contexts where Gemma-4 starts failing at 1K, but the failure *type* is the same.
- This means: methodology that surfaces a failure mode in Gemma-4 will surface the same failure mode in a frontier model, just at a longer horizon. The cheap, on-device, iterable harness we build here is the right place to *develop* the methodology. Running it against a frontier model is then a matter of swapping the bridge endpoint, not redesigning the test.
- Conversely, methodology that works robustly with Gemma-4 *should* be even more robust with a frontier model. It's a lower bound, not a sample.

The workshop harness described here is therefore designed to be **model-agnostic at the role boundaries** — the DESIGNER, PERFORMER, and JUDGE roles defined below are all behind the bridge contract, and either or all can be a frontier API at swap time.

---

## The factorization problem, sharpened

The deep methodological issue: when an LLM is asked to *"change the motive of gushing-fan from foam parties to quantum mechanics,"* its training distribution smuggles voice changes in alongside the motive change — because in the source corpora, quantum-mechanics-people *also write differently* than foam-party-people. The two aren't independent in the data, so they don't separate in the rewriter's posterior either.

This means we cannot *assume* the rewriter's claim of "I only changed X" is honest. We have to **measure factorization rather than request it**.

Concretely: for any variation V of base persona B with claimed target axes T, the variation factorizes cleanly iff:

- `||signature(V) − signature(B)||` on the *target* axes T is large (>1.5σ on the axis-wise scale)
- `||signature(V) − signature(B)||` on the *non-target* axes is small (<0.5σ)

The first part says "the requested change happened." The second part says "no drift." A variation that fails the second part is not a clean factorization — even if it's an interesting persona on its own merits, it doesn't count as evidence of compositionality, because the rewriter changed things it wasn't asked to.

This is the only honest test of motive×voice independence. Without it, we are just measuring whether the rewriter can produce plausible-sounding new personas — which any halfway-capable model can do, trivially, and which is not evidence of anything.

---

## Why template substitution is rejected

The earlier proposal — template-based variation, where we parameterize a persona spec with handles for `voice_intensity`, `motive_urgency`, etc., and sample variations by perturbing those handles — is the wrong shape. Two reasons:

1. **Templates don't model how humans (or language models) actually use language.** Real elicitation strategies emerge from interactions among many small choices that aren't enumerable as orthogonal handles. A template restricts the variation space to whatever axes the template designer pre-anticipated, and the LLM has access to a much richer variation space when asked to rewrite freely.
2. **Templates skip the part of the research question that's interesting.** The question isn't "can we enumerate variations along axes we already know about" — it's "can the LLM independently manipulate axes when shown what the axes are, including ones we hadn't pre-anticipated." The whole point is the LLM's capacity for compositional rewriting, not the designer's ability to enumerate.

Templates produce variations that are too clean to falsify anything. The factorization claim is interesting precisely *because* it's hard.

---

## Workshop loop architecture

The harness has three distinct LLM roles, each behind the bridge contract:

```
┌──────────────────────────────────────────────────────────┐
│ DESIGNER (Gemma-or-larger; in-context-learning loop)     │
│   ↓ persona-spec proposal                                 │
│ PERFORMER (the proposed user-agent in a roleplay chat)   │
│   ↓ K turns of generated behavior                         │
│ JUDGE    (the 14-axis cascade)                            │
│   ↓ 14-d Likert signature with per-axis variance         │
│   → fed back to DESIGNER as in-context evidence          │
└──────────────────────────────────────────────────────────┘
```

**DESIGNER role:** at round N, the prompt contains the base persona spec, the target axis shifts requested (e.g., *"raise probe_depth from ~2 to ~4, keep everything else within ±0.5"*), every prior round's `(proposed_spec, measured_signature, target_signature, diff)` tuple, and explicit drift-axis annotations. The DESIGNER outputs round N+1's persona spec *with knowledge of what landed and what didn't.* This is in-context learning + induction over examples — the two things modern LLMs are genuinely good at, and the two things template substitution cannot do.

**PERFORMER role:** unchanged from the existing `/poll` endpoint — the user-agent generates K conversational turns according to the persona spec.

**JUDGE role:** the 14-axis cascade harness validated 2026-05-14 (`docs/user_agent_likert_methodology.md`). One judgment per performer-turn; signatures averaged across the K turns for the round.

**Convergence and termination:** a target-shift is "achieved" when target-axis deltas are within tolerance AND non-target-axis drift is below threshold across all K performer-turns of the proposed variation. Cap rounds at ~5 — if the DESIGNER cannot converge in 5 rounds, the target-shift is *infeasible for this rewriter on this base persona*. That is a real datum, not a failure: it bounds the achievable factorization map.

---

## Rewriter prompt design surface

Given the workshop loop is the harness, the design space narrows to a small set of choices:

- **Spec presentation:** the DESIGNER sees the base persona in full source form (not a summary). The rewriter needs the unabridged input to identify which textual elements to preserve.
- **Target specification:** always a *specific target shift* on one or two axes — never "make it more interesting" or "vary slightly." Open-ended requests collapse the variation space onto whatever the LLM's prior is.
- **Round-history accumulation:** every prior round's `(spec, measured signature, intended target, drift annotation)` lives in the in-context history. The DESIGNER reasons over this history; it is the substrate for induction.
- **Reasoning audit:** the DESIGNER is asked to explain *in one line* which aspect of the persona it modified and why it expects other axes to remain stable. The explanation is not graded — it's there to keep the reasoning structurally legible, and to surface entanglement claims that the measurement can then check.
- **Sampling:** temperature stays at 1.0 (on-policy, as always). Diversity comes from the *target-shift choice*, not from sampling noise. We pre-compute a coverage grid over the 14-d Likert space and dispatch rewrites toward grid points.

---

## What this harness measures

The original validation framework measured *whether motive and voice are independent dimensions of variation in the population of personas we already have*. The workshop-loop framework measures *whether motive and voice can be independently controlled by a competent rewriter using in-context learning*. The second is closer to what "compositional factorization" actually means as a generative claim. The first is a correlation question; the second is a causal one.

Three corollary research outputs the workshop loop produces:

1. **Achievability map** — which target shifts are reachable from which base personas, and which are not. This is the *experimental shape* of the motive × voice space.
2. **Drift matrix** — when a shift on axis A is requested, which other axes does the rewriter accidentally move? This is a measured matrix of the rewriter's internal entanglement — and it travels with the model. Two different DESIGNER models on the same base persona will yield two different drift matrices, and the comparison is informative about what each model "thinks of as" intrinsically coupled.
3. **Sample-efficiency** — how many rounds does convergence take, and how does that scale with DESIGNER model size? A smaller DESIGNER might need 8 rounds where a larger one needs 2. This is direct evidence about the quality of in-context learning on numerical feedback.

All three are answers the original "generate variations and select" pipeline cannot produce.

---

## Dependency-ordered build list

1. **14-axis cascade** — extend Stage 2 prompt + parser + downstream consumers from 6 to 14 axes. Validate at N=15 on the 3 known personas. (Pre-work for everything else; cheap to land first.)
2. **Persistent JSONL store** — append-only, one row per `(round_id, performer_turn_id)` judgment. Required because the DESIGNER's round-N prompt reads from this store.
3. **Per-persona signature estimator** — Python: read JSONL, compute mean + per-axis std per persona per round.
4. **Discriminability smoke test** — confirm base personas occupy distinct regions in 14-d Likert space at chosen N. If they overlap, the workshop has no headroom — fix dimensionality or N before going further.
5. **DESIGNER harness** — round-by-round prompt template, in-context history accumulation, target-shift specification, drift-annotation generator. This is the core new piece.
6. **Single-target-shift convergence test** — pick one base persona + one target shift, run the workshop loop end-to-end, verify it converges (or terminates honestly with "infeasible").
7. **Coverage-grid orchestrator** — pre-compute target shifts spanning the 14-d space, dispatch them through the loop, accumulate the achievability map.
8. **Drift-matrix analysis** — across all converged-or-terminated runs, summarize which axes drift when which other axes are targeted.
9. **Sample-efficiency analysis** — distribution of rounds-to-converge as a function of base persona, target-shift, and DESIGNER model. (Naturally invites a frontier-model comparison swap.)

Estimated human time: ~10 hours of focused work (the DESIGNER harness + drift-matrix analysis are real engineering, the rest is glue). Run-time: a multi-day budget for full coverage, but a useful first result is reachable in an overnight run with a curated target-shift list and one or two base personas.

---

## Open methodology decisions still to make

- **Tolerance and threshold values** for "shift achieved" and "drift small." Probably need a calibration pass against base-persona within-condition variance before fixing these.
- **K (turns per round)** — too small and the signature is noisy; too large and the workshop loop costs explode. Probably K = 3 to start, scale up if signature variance is too high to discriminate.
- **Whether to let the DESIGNER see its own prior reasoning audits** — including them grows the in-context history; omitting them may sacrifice some learning signal. Worth A/B-ing.
- **Cap on DESIGNER context length** — the in-context history grows monotonically. Need either a sliding window, a summarization step, or a hard cap with "give up" semantics. The hard cap is probably right for a first cut.
- **Whether to score the DESIGNER's *audit* statements against the measured drift** — i.e., when the DESIGNER says "I changed X without changing Y" and the measurement shows Y did change, that's a calibration failure of the DESIGNER's introspection, and itself worth recording.

---

## Why this matters as research

If a competent DESIGNER (Gemma-4 on this hardware, or any frontier API at swap time) can iteratively converge on target Likert profiles using only numerical feedback in-context, **we have a procedure for generating arbitrary user-agents to spec**. That is the elicitation tooling the project has been building toward. It also gives the substrate the project needs for downstream work: synthesizing user-agents to stress-test responder behavior under deployment-realistic input distributions, rather than under a hand-picked persona zoo.

If a competent DESIGNER cannot converge — if the drift matrix reveals that the rewriter cannot manipulate axis A without dragging axes B and C along — that is also a publishable finding, because it characterizes the **inherent entanglement of the LLM's persona generation**. That entanglement structure travels with the model and is itself a property worth measuring across models.

Either outcome is informative. The harness is designed so that *no plausible result is uninteresting* — which is the only honest condition under which a research instrument is worth building.

---

## Addendum (2026-05-14): target-shift screening + the off-subspace search

The first measured effective-dimensionality result (3 personas × N=30 = 90 judgments under Gemma-4-A4B Q8_0; see methodology-note addendum) puts the judge's resolved subspace at ~2 dimensions at the 80% threshold. This has two consequences for the workshop loop's design:

**1. Target-shift screening is a hard precondition for the DESIGNER, not a soft hint.** A target like *"raise `warm` from 1 to 5 while holding `affective_intensity` at 1"* asks the rewriter to produce a persona that lives orthogonal to PC1 — i.e., off the principal subspace the judge can resolve. Even if the rewriter succeeds at producing such a persona, the judge will project it back onto PC1 and the measurement will register no clean shift. The loop will then misattribute the failure to the rewriter rather than to the judge's projection. **Before any target shift is dispatched, the harness must compute the shift's projection magnitude onto the judge's principal subspace and either re-express it in PCA coordinates or surface a "judge cannot resolve this direction" warning.**

**2. The effective-dim measurement is also an *opportunity*, not just a constraint.** The off-PC1/PC2 axes carry low variance in the current corpus precisely because no existing persona loads strongly on them. **A directed search procedure can use the harness as a discovery instrument**: identify directions of variation the corpus does *not* span (low-variance PCs), then design (or have the DESIGNER synthesize) personas that deliberately load on those directions. Whether such personas land off-subspace when measured is the test of *"is the basis over-parameterized in the judge, or merely in the corpus."* If a deliberately-orthogonal persona extends the variance distribution, the corpus was the limit. If it gets projected back onto PC1/PC2 anyway, the judge is.

**Corollary: the harness ships as a 2-mode tool.**
- *Realization mode*: given a target persona spec, measure whether the realized agent achieves the spec's claimed signature, with drift quantified per axis. Used by the workshop loop's convergence check.
- *Discovery mode*: given the current corpus, identify low-variance PCs, and propose target shifts that would (if realized) extend the resolved subspace. Used to push the corpus off-subspace deliberately.

Both modes share infrastructure (the cascade judge, the JSONL store, the signature estimator). The discovery mode's analytical output is just the list of PCs whose variance is small in the current corpus, ranked by `1 / variance_explained`, with the named-axis basis decomposition of each component for human-interpretable persona-design hints.

This addendum supersedes the workshop loop's pre-2026-05-14 framing that treated the judge's resolution as fixed. The judge IS fixed (for a given model checkpoint), but the corpus is not, and the harness's job is to drive the corpus into directions the judge can't easily collapse — that's where the actually-novel elicitation strategies live.
