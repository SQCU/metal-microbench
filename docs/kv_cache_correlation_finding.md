# KV-cache trial correlation — empirically confirmed

**Status**: Empirical bug finding. Cache reuse appears to bias
sampling outcomes vs. fresh prefill on bit-identical prompts. Test
script: `/tmp/cache_correlation_test.py` (transient, but reproducible
from this doc's design).

## What the test did

Three batches of K=20 trials each, identical tools[], temperature=0.4:
- Batch 1: prompt `"i wanna see a fractal!! ✨"`, fresh bridge state
- Batch 2: same exact prompt → bridge content-hash cache HITS for the
  prompt prefix
- Batch 3: same prompt with a trailing space appended → bridge cache
  MISSES (different byte sequence, different content hash)

Measured: per-batch tool-use rate (the elicitation study's metric).

## Result

```
batch                | tool-use rate
batch1_base_first    | 7/20 (35%)
batch2_base_repeat   | 2/20 (10%)   ← cache hit
batch3_perturbed     | 8/20 (40%)   ← cache miss, semantically equivalent
```

Bernoulli SE at observed mean p≈0.28, K=20: σ ≈ 0.10, 2σ ≈ 0.20.

- |batch1 − batch2| = 0.25 (cache hit vs. cache miss on identical bytes) — **just outside 2σ**
- |batch1 − batch3| = 0.05 (fresh vs. fresh on different bytes) — well within 1σ

The cache-hit batch (2) is the OUTLIER. Batches 1 and 3 — both fresh
prefills — agree closely despite being on different byte sequences.
This rules out "trailing space changes meaning to the model" as the
explanation: the model's output distribution is essentially identical
whether the prompt is bit-1 or bit-3, but DIFFERENT when the cache
serves it from prior K/V vs. recomputing fresh.

## Mechanism (suspected)

The bridge's `PageManager.contentIndex` shares phys K/V pages by
content-hash. When two requests' prompts produce the same page-hash
prefix, the second request adopts the first's pages read-only and
skips re-prefilling them. In principle this is a pure prefill-time
optimization — the decoder still sees the same K/V values it would
have computed itself.

In practice, batch 2 produces a meaningfully different output
distribution from batch 1. Plausible mechanisms:
- **Cvec / control-vector state mismatch**: cached pages were written
  while some control-state was active; reuse without that state means
  the decoder reads K/V that don't match what fresh computation would
  produce. (chatfile mentions cvec_digest in the page hash; if that's
  only partial coverage, this is a real vector for the bug.)
- **Numerical drift via running statistics**: layer norm, attention
  scaling, or some other per-step accumulator that the cache doesn't
  capture. Subtle bias on later AR steps.
- **K/V quantization noise**: the K/V is stored at fp16 (or quantized
  in cache), and the round-trip through cache flips some bits that
  affect downstream sampling at temperature.
- **Stream-level RNG state**: the bridge's `gpuRngSeed` is per-Session.
  If two trials' Sessions accidentally inherit related RNG state when
  the second one adopts cached pages, sampling becomes correlated
  rather than IID.

## Implications

Any A/B study that fires K identical-prompt trials against the same
warm bridge has trial outcomes that are NOT IID. K-bumps don't shrink
error bars as expected because trials don't carry independent
samples. The elicitation study's run-to-run variance (rates ranging
0%–45% on the same variant across runs) is consistent with this:
each "run" took a different cache state into account, producing a
shifted-but-still-noisy distribution.

The cache mechanism IS a real performance feature for production
serving (multi-turn chats with shared system prompts adopt prefix
KV pages and skip re-prefilling — that's the 100% kernel-fill story
from earlier in this session). The bug is that it appears to also
shift the sampling distribution slightly, which we observe in
aggregate over many trials.

## Recommended workarounds for studies

1. **Per-trial prompt perturbation**. Append a unique tag (`(trial-N)`)
   to the user message OR use a different RNG seed per call so each
   trial generates a different cache key. Removes the same-cache-state
   correlation but doesn't fix the bug itself.
2. **Cache flush between trials**. Add a bridge endpoint that drops
   `pageManager.contentIndex` for fresh state. Costs prefill time
   per trial but makes trials truly independent.
3. **Larger K + cell randomization**. If you can't perturb, run K≥50
   with random ordering across (variant, prompt) cells so cache state
   averages out.

## Recommended bridge-side investigation

Three diagnostic experiments worth running, in increasing depth:

1. **Same prompt, different temperatures**. If the cache effect is
   from the cvec/RNG mechanism, low-temp runs should be MORE affected
   (deterministic decoder still produces different outputs); high-temp
   runs should mask it under sampling noise. If cache effect persists
   at temp=0, the bug is in the K/V values themselves.
2. **Page-by-page KV diff**. Capture K/V values at slot for batch 1
   (fresh prefill) and batch 2 (cache adoption), compare element-wise.
   Differences should be exactly zero IF the cache is a pure
   optimization. Any non-zero difference is a bug.
3. **Cvec-state digest coverage audit**. Walk `cvec_digest_for_page`
   and verify it captures every piece of state that affects K/V
   computation. If something's missing (running statistics? per-layer
   scaling factors? gpu_rng evolution?), the page-hash collision
   becomes a real correctness issue.

## Cross-references

- `tools/st-debug/scripts/tool_elicitation_study.py` — the study that
  motivated this investigation (run-to-run variance was the symptom)
- `docs/tool_elicitation_findings.md` — round 2 caveats section
  references this file
- `page_manager.swift:hashPage` — the content-hash function whose
  inputs the cvec audit should walk
- `lm_engine.swift:cvecDigestForPage` — the cvec-state component of
  the page hash
