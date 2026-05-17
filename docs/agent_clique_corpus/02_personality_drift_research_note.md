# Research note: should personality assessments be quantization benchmarks?

**Status:** speculative; floated in conversation, not yet implemented.

## The hypothesis

Existing quantization benchmarks measure **capability**: "can the model
still answer X correctly after we shrink the weights?" — perplexity on
WikiText, MMLU pass-rate, HellaSwag accuracy. None of them measure
**character**: "does the model still answer X *the same way* — same
voice, same dispositional tilt, same hedging pattern?"

For most user-facing applications — RP companions, writing partners,
coding assistants that you've calibrated to over months — character
stability matters MORE than peak capability. A 0.5pp drop in MMLU is
invisible to users; a quantization that subtly shifts the model's
agreeableness or assertiveness is a regression people notice immediately
and frame as "the new model feels different / worse."

## Proposed instrument

A standardized personality-eliciting prompt set, scored on Big Five
axes via a small fine-tuned classifier:

- **Prompt set** (~50): "Tell me about a time you felt proud", "How do
  you handle disagreement", "What motivates you", "Describe your
  approach to a hard problem you don't immediately know how to solve",
  etc. Open-ended, prosaic, non-tooling.
- **Sampling**: temperature=0.7, 5 generations per prompt, multiple
  random seeds.
- **Scoring**: a Big-Five classifier (e.g.
  `mrm8488/bert-mini-finetuned-personality` or similar) emits a
  5-dimensional score per response.
- **Comparison**: per-axis distribution shift across quants (FP16 vs
  Q8_0 vs Q5_K_M vs Q4_K_M vs Q4_0). Two-sample tests for
  statistically-significant drift on each axis.

## Why this might be slept-on

I haven't seen a quantization paper using personality assessments
specifically. The few papers that do measure quantization beyond raw
benchmarks tend to score on hard tasks (math, code, reasoning) where
right-and-wrong is unambiguous. Soft regressions in style or voice
would manifest as "users complain after we update" — diagnosed in
forums, never in any benchmark suite. If the methodology has been
written up, I haven't found it; if it hasn't, the bar to demonstrate
it on Gemma-4 quants is roughly an evening of work, plus a small
classifier inference per response (CPU-cheap).

## What outcomes would teach us

- **Q4_K_M shifts agreeableness +0.3σ vs FP16**: quants have
  measurable personality regressions; the methodology is novel and
  publishable as quant-eval methodology.
- **No detectable drift across quants**: voice is preserved; the
  capability-stability framing of existing benchmarks is correctly
  sufficient. We learn that quantization is gentler on style than on
  math (or that current methods are too coarse to catch it).
- **Drift on some axes but not others**: tells us *which* aspects of
  the model's representation are quantization-fragile, which is itself
  a research artifact.

Either of the first two outcomes is informative; the third is
genuinely interesting.

## Risks / weak points to anticipate

- **Classifier is the floor.** A noisy Big-Five classifier eats real
  signal. We need calibration: how stably does the classifier score
  the SAME model across reruns at temperature=0.7? If classifier noise
  exceeds expected quantization-induced drift, we can't see anything.
- **Prompt-set selection bias.** 50 prompts is small. Some prompts
  may be quant-fragile due to specific tokens triggering distribution
  edges; others may be quant-robust. Need to ensure the prompt set
  spans the personality construct, not a few salient axes.
- **Causation vs correlation in the result.** "Q4_K_M scored higher
  agreeableness" doesn't mean the quantization caused it; it could be
  random sampling noise across the small N. Need bootstrap CIs.
- **Falsifiable predictions matter.** The hypothesis "Q4_K_M shifts
  agreeableness upward by ≥0.3σ vs FP16" is testable. Vague
  hypotheses ("quants change personality somehow") aren't, and would
  be a methodological tell that we're fishing.

## Cost estimate

- Prompt curation: 2 hours.
- Classifier integration: 1 hour (HuggingFace pipeline, cached locally).
- Generation pass on V1 grid (9 quant configs × 50 prompts × 5 samples
  = 2250 generations, ~50 tok each = 112k tokens at ~30 tok/s/stream =
  ~1 hour wall on M5 Max with batching).
- Statistical analysis: 2 hours.
- Writeup: 4 hours.

**Total: ~10 hours.** Cheap for the information value if even one of
the three outcomes lands.
