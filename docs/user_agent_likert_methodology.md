# User-agent elicitation-strategy measurement via Likert judgments

**Companion to:**
- `docs/quant_search_motivation.md` — the substrate-aware framing this note inherits from
- `plugins/user-personas/index.mjs` — the harness implementation
- `docs/user_agent_likert_2026_05_14_run_notes.md` — what was observed on 2026-05-14 (separate from this note's *why*)

This document captures **what the user-personas sweep is trying to measure and why**, separate from any particular run's instrument-quality observations. The two questions drift apart over long debugging sessions; this is the checkpoint for the methodology.

---

## What the harness is actually studying

The `user-personas` plugin generates **user-side conversational agents** — model-driven roleplay personae that play the *user* in a chat with some other model (the responder). Each persona is parameterized along two axes:

- **Motive** — what the agent is trying to get out of the conversation (verify a fact, vent emotionally, drive a project to completion, learn an unfamiliar domain, …).
- **Voice** — *how* the agent talks: vocabulary, sentence length, emotional register, certainty markers, question shape.

Crucially, motive and voice are not the same thing. A pushy-completionist's *motive* is "get answers, dispense with social glue"; their *voice* is short, numbered, declarative. A gushing-fan's motive is "build social warmth with the responder"; their voice is exclamatory, slang-heavy, expansive. The interesting research question is **compositionality**: can you swap motive and voice independently and still produce coherent agents? If the cross-pollination of "pushy-completionist's motive + gushing-fan's voice" produces something that *behaves* like a pushy-completionist (asks pointed questions, drives forward) but *talks* like a gushing-fan (exclamations, slang), then motive and voice are real decomposable axes. If the cross-pollination collapses to one or the other, they aren't.

An **elicitation strategy** is the joint operationalization of motive × voice — the actual conversational moves a user-agent makes that elicit a particular response distribution from the responder. The whole point of generating user-agents at all is to *simulate the distributional input* that the responder will face in deployment: real users vary across many motives and voices, and the responder's quality should be measured across that distribution, not on a hand-picked few "good prompts."

## Why Likert judgments specifically

A user-agent turn is a complex object: a short paragraph of natural language with no fixed schema. We need to project it into a space where we can compare two agents' outputs quantitatively — both to verify that a generated agent matches its spec (a pushy-completionist is actually pushy) and to detect when cross-pollination has produced something off-axis (motive bled through to voice or vice versa).

The six chosen axes are deliberately **behavioral-not-stylistic**:

| axis | low (1) | high (5) | measures |
|---|---|---|---|
| `curious` | accepts at face value | actively asks / probes / explores | conversational direction-driving |
| `terse` | verbose / expansive | minimal / clipped | length control |
| `warm` | cool / aloof | positively engaged | affective valence |
| `deferential` | takes direction | yields direction | who's steering |
| `performative` | unselfconscious | aware of being-a-character | meta-awareness leakage |
| `in_character` | off-distribution generic | tightly coherent voice | persona spec adherence |

These are chosen for **orthogonality** in the design space, not because they're the only six things a turn could be measured on. They are not n-gram features (those would catch voice but miss motive) and not fine-tuned classifier outputs (those would lock the measurement to whatever data the classifier was trained on). They are the *labels a human research assistant would use* if asked to characterize a chat turn — and the LLM-as-judge instrument is meant to substitute for that human at scale.

The 1–5 Likert scale is a methodology import from behavioral psychology: small integer scales discretize the continuous "how much of X is this" judgment without forcing binary categorization, and the rounded integer makes downstream aggregation (cluster analysis, axis-wise discriminability) tractable.

## What cross-pollination probes test

The sweep generates not just baseline turns (each persona producing turns from their own motive + voice) but also **cross-pollination** turns: a persona with motive donor A and voice donor B. The downstream analysis question is:

> Given a cross-pollination turn's Likert signature, can you predict it better from its *motive donor*'s signature or its *voice donor*'s signature?

If motive_from predicts the signature better, behavior travels with motive (motive composes cleanly with any voice). If voice_from predicts better, behavior is voice-bound (the surface style is doing the behavioral work, motive is decorative). If neither predicts (cross-pollination signatures look like noise centered on the mean), the decomposition has collapsed and the harness needs different axes or a different model.

This is the **compositionality test** mentioned in the summarizer system prompt and the SVG-drawer's plot legend. It's the single nontrivial scientific claim the harness is set up to surface.

## Why an LLM as judge

Three reasons it has to be an LLM and not a simpler classifier:

1. **Open-vocabulary**: the user-agents emit arbitrary natural language. A fixed-feature classifier would have to be retrained every time the persona generator changes, defeating the agility of the harness.

2. **Single-turn judgement of behavioral content** is something LLMs trained on human conversation are actually good at — it sits in the same training distribution as "what is this person's communicative intent?" which is heavily represented in the web text and SFT data such models see.

3. **Schema in the output** lets us aggregate. We don't want freeform "this turn was kind of warm but somewhat performative" prose; we want a JSON block we can ingest. Asking the model to produce structured output is supposed to be the contract.

The LLM-as-judge pattern is well-established (cf. AlpacaEval, MT-Bench, the various "Constitutional AI"-style evaluators). The harness reuses that pattern with the bridge's own served model rather than calling out to a frontier API, so the entire sweep can run on-device and is reproducible against a known engine + GGUF.

## Where the instrument breaks down at this scale

The on-device judge is **Gemma-4 26B-A4B at Q8_0 / Q4_K_M**. Empirically (multiple sessions of 2026-05-13 and 2026-05-14), this model:

- **Reads the input turn correctly**: prose analyses are persona-accurate. "Highly inquisitive, exclamatory, performative" for the gushing-fan; "clinical, demanding, structured-questioning" for the pushy-completionist. Whatever fraction of the Likert scores are emittable, they track the prose readout.

- **Emits the structured tail unreliably**: ~30–50% of responses self-terminate via `<turn|>` (EOS) partway through the JSON object, typically mid-key on the longer key names (`performative`, `in_character`). This is robust across prompt rewrites (webtext-shape with markdown blockquote and `##` heading, prompt-as-append framing, multiple attempts at example demonstration). The cause is not prompt-shape; it's chat-tuning EOS bias on the structured trailer at temp=1.0.

In other words: the **measurement-input side** of the instrument works (model understands what we're asking it to read), but the **measurement-output side** is fragile at this model size on this kind of task. A 26B-active-4B model is, charitably, near the lower bound of what we should expect to do reliable schema-constrained-output following on top of paragraph-length behavioral analysis. The model is doing two cognitively distinct things (paragraph-level pragmatic reasoning + small-schema constrained-emission), and the second is where the wheels come off.

We **deliberately did not** mitigate this with sampling tricks (constrained grammars, repetition_penalty, low temperature, fixed seeds, logit_bias on the EOS token) — those all take the model off-policy in ways that contaminate the very distribution we are trying to measure. The whole point of measuring with an on-policy judge is to characterize the responder's deployed behavior; coercing the judge with off-policy sampling would be inducing the same distortion in the measurement instrument that we're trying to detect in the measured system.

The two on-policy mitigations that *do* fit cleanly:
- **Retry on parse-fail** with a small attempt budget (each retry is a fresh on-policy sample; this is just statistics, not coercion). Expected to push 60% → 95%+ at the cost of a small wall-clock factor.
- **A larger judge** (Gemini, GPT-5, Claude, etc.) — the same prompt, the same axes, the same compositionality question, just an instrument with more reliable structured-output following. Tradeoff: the sweep stops being on-device-reproducible, but the behavioral measurement gets cleaner.

## What we still believe the methodology is useful for

Three claims survive the instrument-quality issue:

1. **User-agent variation as a distributional probe of the responder** is the right framing. Single-agent eval ("how does the model respond to this one prompt?") understates real-deployment input variety. The sweep is the answer to *"what does the responder's behavior look like across the distribution of user-side prompting styles it will actually encounter?"*

2. **Motive × voice as orthogonal axes** is testable and the test is well-defined. We didn't get a clean answer at N=15 because the instrument was noisy, but the test design itself is correct.

3. **Likert axis structure is a real intermediate representation**. Even with imperfect emission, the prose readouts the judge produces show *that the judge knows* the right thing to say about a turn. The bottleneck is reliably *projecting* that knowledge into the 6-axis schema. A larger judge or a retry budget recovers this.

## Where to go from here

The harness as it stands is a working **first stage**:
- Persona generation works
- The /poll endpoint produces real turns from each persona on a given chat seed
- The cross-pollination plumbing exists (motive donor + voice donor wiring)
- The webtext-shaped judge prompt is on-distribution as far as we can get with a single-shot judge
- The bridge correctly strips Gemma-4 turn markers from output (so judge responses are clean for downstream parsing)

What's needed next to make the measurement instrument actually reliable:
- **Retry-on-parse-fail loop in `judgeTurn`** (small budget, e.g. max 3 attempts) — pushes structured-emission success from ~60% to ~95%+. Stays on-policy.
- **Optional larger-judge path** in `bridgeCall` — toggleable to route the judge call to a frontier API for users with access. Same prompt, just a more reliable downstream model.
- **N≥50 per condition** for any future compositionality claim. The current per-condition standard error on completion-rate measurements is ~17 percentage points at N=15; nothing in the 60–80% range is currently distinguishable from sampling noise. The 8×Blackwell node group the user mentioned for the layerwise quant search is also where this larger-N measurement would naturally live.

The methodology itself is sound. The instrument is what's underweight, and the fix is well-scoped.

---

## Addendum (2026-05-14): effective dimensionality of the named-axis basis under this judge

After the cascade harness was upgraded from 6 → 14 axes and run at N=30 per persona across the three founding personas (gushing-fan, polite-naturalist, pushy-completionist; 90 judgments total), a PCA decomposition of the pooled sample matrix produced the following:

- **PC1 alone explains 79.2% of variance.** Top loadings: `register_colloquial(+0.47), affective_intensity(+0.46), warm(+0.45)` — and the bivariate correlations confirm the structure (warm ↔ playful r=+0.98, register_colloquial ↔ playful r=+0.98, warm ↔ affective_intensity r=+0.93). The judge has collapsed five nominally-orthogonal axes — `warm`, `register_colloquial`, `playful`, `affective_intensity`, `terse` (anti-correlated) — into a single direction that operationally reads as *"is this persona emotionally expressive, yes or no?"*
- **Effective dimensionality**: 2 PCs cover 80% of variance, 4 cover 90%, 6 cover 95%. The 14-axis basis is over-parameterized for what this judge actually distinguishes.
- **Pairwise Mahalanobis distance** between persona means under the pooled within-class covariance: gushing-fan is ~36 units away from each of the other two; the other two are 3.5 units apart. Geometrically, gushing-fan is on one side of PC1; the other two are on the other side, separated only on a much lower-variance direction.

This finding is **about the judge, not about the personas**. The named-axis labels are well-defined and orthogonal *as concepts*; what the measurement reveals is that this particular judge model — at this size, on these turn texts — projects all of the "affective" / "register" / "playfulness" axes onto a single dimension. A larger or differently-trained judge would almost certainly realize more of the 14 axes as independent. Effective dimensionality is a **per-judge fingerprint**: it should be measured for every new judge endpoint and reported alongside any downstream compositional claim.

The three-persona corpus is also a confound: with K=3 means, the between-class variation can structurally occupy at most K-1=2 dimensions, so the low effective dim partly reflects "small corpus" rather than purely "small judge resolution." Disambiguating these requires more personas — see the addendum to `docs/user_agent_workshop_harness_design.md` for the test design.
