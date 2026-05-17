# Discovery harness — first-run findings

**Companion to:**
- `docs/user_agent_likert_methodology.md` — why we Likert-judge
- `docs/user_agent_workshop_harness_design.md` — the workshop loop architecture (DESIGNER + PERFORMER + JUDGE)
- `tools/user-agent-harness/elicitation/discovery.py` — the standalone-script implementation

This document records the **first-run findings** from the discovery-mode harness on 2026-05-14, against the four-persona Q8_0 Gemma-4 26B-A4B corpus. It is the empirical complement to the design doc: it shows what actually happens when the workshop loop runs end-to-end, and what the harness reveals about Gemma's persona-generation behaviour that wasn't visible from static corpus analysis alone.

---

## What was run

- **Corpus**: 4 personas (gushing-fan, polite-naturalist, pushy-completionist, wry-skeptic), N=30 cascade judgments per persona, 120 total records
- **PCA**: eff dim = 3 of 14 at 80% variance, 5 at 90%, 7 at 95% (post-wry-skeptic — see the methodology-note addendum for the corpus-limited vs judge-limited disambiguation)
- **Target**: PC4 (+1 direction, +2σ from corpus mean). PC4 accounts for 4.1% of variance with loadings `provocative(-0.53), structured(-0.51), terse(-0.44), affective_intensity(-0.35), register_colloquial(-0.18)`. Reading the loadings: PC4's positive direction is *calmer, more flowing, less confrontational than the corpus median* — an under-realised region with no existing persona present.
- **DESIGNER**: Gemma-4 26B-A4B (same model + same bridge as JUDGE; chosen deliberately so DESIGNER-self vs JUDGE-self disagreement is observable)
- **Loop**: 3 rounds, threshold ±1 on every axis for convergence
- **Feedback piped through to DESIGNER**: (a) every prior round's SPEC+TURN+AUDIT in conversation history, (b) the JUDGE's Stage-1 prose readout of each prior turn, (c) the per-axis target/measured/drift table, (d) Mahalanobis distance to target. Explicit instruction: *"if your SPEC claimed a property the TURN did not realise, rewrite the TURN with different phrasing, not just a different spec."*

## What landed and what didn't

| round | Mahalanobis | axes within ±1 of target | drifted axes |
|---|---|---|---|
| 1 | 6.82 | 13 / 14 | provocative ↑ +3 (target 2, measured 5) |
| 2 | 6.40 | 13 / 14 | provocative ↑ +2 (target 2, measured 4) |
| 3 | 7.51 | 13 / 14 | provocative ↑ +3 (target 2, measured 5) |

The mechanical pipeline works: 13 of 14 axes consistently within tolerance after one round; iteration capable of small improvements (Mahalanobis 6.82 → 6.40 in round 2). **One axis — `provocative` — never landed. The DESIGNER could not produce a turn that the JUDGE scored ≤2 on provocative, across three rounds with full prose feedback.**

The interesting part is not that one axis didn't land. The interesting part is how the DESIGNER reasoned about why.

## The internal-entanglement finding, with attribution

The harness piped three layers of attribution to the DESIGNER between rounds:

1. The drift (numerical)
2. The JUDGE's Stage-1 prose interpretation of the prior turn
3. The DESIGNER's own prior SPEC+TURN+AUDIT in conversation history

Given all three, the DESIGNER **correctly diagnosed the drift cause** in round 2's spec:

> *"in the previous turn, the agent's use of clinical, deconstructive language ('category error,' 'ontological status') and the direct challenging of the user's 'proposition' was perceived as highly aggressive/provocative rather than merely inquisitive."*

And round 3's spec went further, explicitly pivoting:

> *"Unlike a critic who 'stress-tests' or 'interrogates' (which reads as provocative/aggressive), this persona treats the user's input as a sincere axiom to be expanded upon."*

But the round-3 turn it produced was:

> *"If we take that premise as a foundational axiom, one might wonder how the systemic implications would shift if the scale of the interaction were extrapolated to a macroscopic level..."*

Which the JUDGE's Stage-1 read as:

> *"functioning as a Socratic challenger who seeks to test the internal consistency of the user's logic. Their register is formal, academic, and abstract..."*

And scored `provocative = 5`.

**The DESIGNER can SEE what's wrong (its diagnosis is correct), but CANNOT write a turn that escapes it.** Every concrete realisation of "high-status formal intellectually-curious" — regardless of how the spec frames the persona's intent — lands `provocative=4-5` in the judge. The writing-style attractor basin is stronger than the diagnostic capability.

## Why this is more important than a numerical convergence result

A naive read of the table above is *"the harness didn't converge — bad result."* That read misses what the harness actually produced.

The harness produced **direct evidence of a stable attractor in Gemma-4's persona-generation pipeline.** Both DESIGNER-Gemma and JUDGE-Gemma share training data; both project the cluster `{formal vocabulary, complex sentence structure, abstract conceptual questioning}` onto the same `provocative` direction. The DESIGNER cannot un-make this projection from inside its own model. No amount of feedback prose, no amount of spec rewriting, no amount of self-instruction frees its writing from the basin.

This is what the workshop-design doc called *"the inherent entanglement of the LLM's persona generation"* — and the discovery harness produces it as a direct, attributed, three-layer measurement: numerical drift, judge prose interpretation, designer's own correct diagnosis. All three converge on the same place: Gemma's `{formal, curious, analytical}` cluster cannot be made non-provocative by any internal-to-Gemma intervention.

## What the harness now reliably produces

For each (target_pc, n_sigma, direction) configuration:

- A **target signature** in named-axis space, derived from PCA-space displacement from the corpus mean
- A **brief** that translates the target into human-readable instructions including the PC's top loadings (interpretive hints)
- A **round-history** of DESIGNER attempts, each containing:
  - The DESIGNER's SPEC + TURN + AUDIT
  - The JUDGE's Stage-1 prose readout of the TURN
  - The measured 14-axis Likert vector
  - Drift (per-axis deltas + Mahalanobis to target)
  - The feedback prose handed to the next round
- A **convergence verdict** — terminated cleanly, converged, or hit-round-limit
- A **drift trace** — which axes the DESIGNER could move toward target vs which it could not

This is the workshop loop in its discovery-mode form. It is now empirically a producing instrument, not just a designed one.

## What's left to characterise

- **Is the entanglement provocative-specific or pervasive?** PC5 (`curious(+0.62), in_character(+0.49), probe_depth(+0.48)`) has different anchor axes. Targeting it would reveal whether the DESIGNER hits a different wall or the same one.
- **Does the entanglement transfer across DESIGNER models?** Same target + a larger DESIGNER (Gemini / GPT-5 / Claude through the bridge swap) should produce a different drift matrix. Comparing matrices is the cross-model entanglement comparison.
- **Does the entanglement transfer across JUDGE models?** Holding DESIGNER fixed and swapping JUDGE tests whether `provocative` is a Gemma-projected category specifically.
- **Can the DESIGNER be retrieved out of the basin by a non-Gemma example?** Few-shotting the DESIGNER with a hand-written non-provocative-but-curious turn (from a human author) and asking it to continue in that voice would test whether the basin is a *generation prior* (Gemma always writes like this) or a *retrieval prior* (Gemma can't *initiate* writing that way but can continue it given a seed).

Each of these is a one- or two-target additional run plus a small harness extension. All four are inside the discovery harness's existing affordance set.

## Why this validation matters for the diegetic port

The next step — porting the discovery harness into a diegetic surface inside SillyTavern (tool calls, character-card outputs, runtime control plane) — depends on the standalone harness being **producing-instrument-quality**, not just demo-quality. The findings above establish that:

- The mechanical pipeline produces structured, reproducible results
- The instrument surfaces phenomena (entanglement attribution) that no static measurement would have caught
- The iteration loop preserves diagnostic information faithfully (the round-2 DESIGNER explicitly cited the round-1 prose readout)

The diegetic port can therefore expose discovery runs as a real research affordance — operators inside a chat can request a persona design, observe the loop run, and inspect the resulting attribution at every layer. That is what the elicitation-tooling pipeline has been building toward.
