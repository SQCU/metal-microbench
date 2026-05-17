# Product brief: Zed-mediated coding sessions as a quantization-eval source

**Audience:** product / direction owner.
**Status:** brief; needs MVP scoping if pursued.

## The opportunity

Quantization-eval suites today rely on synthetic or static benchmarks:
WikiText-2 perplexity, MMLU multiple-choice, HellaSwag, SVG MSE on
fixed prompts. These have a uniform texture. A real coding session
inside an editor like Zed has a categorically different texture:

- **Dense, repetitive context.** File paths, function signatures,
  type definitions — all repeat across turns. The same buffer state
  is in 30 turns of a refactor session. This is precisely the shape
  prefix-cache machinery is built for.
- **Tool-call accuracy as a sharp behavioral signal.** Zed's agent
  emits structured JSON tool calls (`read_file`, `apply_edit`, etc.).
  A quant that nukes JSON-format fidelity will break visibly: malformed
  paths, broken arg shapes, edits that don't apply. This is the kind
  of bit-rot perplexity-on-WikiText cannot see.
- **Multi-turn iterative refinement** is the dominant pattern. The
  user proposes an edit, the model suggests it, the user pushes back,
  the model revises. This exercises both prefix cache (most context
  unchanged across turns) AND coherence (does the model remember what
  it just suggested 3 turns ago?).

## The product hypothesis

Recording real Zed-mediated coding sessions and replaying them against
multiple quantization configs gives us a behavioral benchmark that:
1. Tests the prefix cache under realistic contention (most existing
   benchmarks fit comfortably in one prefill, missing the cache-hot
   regime entirely).
2. Catches quantization regressions that capability benchmarks miss
   (tool-arg JSON validity, multi-turn coherence, code-edit acceptance
   rates).
3. Generates evaluations whose results map directly to user-perceptible
   behavior, not abstract task scores.

## MVP scope

- **Capture**: bridge logs every chat-completion request (already
  partially landed; needs request-body dumping). The user uses Zed
  normally; sessions accumulate as JSONL.
- **Curate**: select 3-5 canonical sessions from captures that
  represent diverse coding patterns (refactor, debug, test-write,
  doc-write, review).
- **Replay**: pure HTTP-client replay against each quant config.
  Capture (a) full response text, (b) tool-call JSON validity rate,
  (c) cache_hits per turn.
- **Score**: per-quant aggregate of {tool-arg JSON-valid rate,
  edit-suggestion overlap with FP16 baseline using token-level Levenshtein,
  per-turn cache_hits efficiency}.
- **Report**: 3-axis chart (capability via existing oracles, behavior
  via this brief, throughput via tok/s) per quant config.

## What this competes with

Standard answer is "more synthetic benchmarks" — bigger MMLU, code-
specific eval suites (HumanEval, MBPP). These are fine; they measure
correctness on bounded tasks. They don't measure the messier behavioral
properties of an interactive coding session: stability across turns,
adherence to user's actual code style, recoverability from a failed
suggestion.

Behavioral coverage is complementary, not a replacement. The pitch is
"this is the suite that catches what the synthetic suites miss."

## Risk: replay fidelity

A captured Zed session has the timing, ordering, and tool-call
sequence frozen. A different quant might emit tool calls with slightly
different args, which Zed (in the live session) would have routed to
different tool execution paths, which would feed different tool
responses back into context. Replay treats this as fixed; the real
session's branching tree is collapsed.

This is acceptable for V1 because the behavioral metrics we care
about (was the tool call JSON-valid? did it call the right tool?) are
single-turn-local. Cross-turn coherence might suffer in replay because
the tool responses are pinned to FP16's path through the tree. We'd
need a tree-replay where the tool harness is re-executed live to get
that right; out of scope for V1.

## Decision the brief is asking for

Should we authorize the MVP work (~1 week of engineering for
capture + replay + 3 metrics) or punt to a later iteration after the
quant-search V1 numbers come in?

The case to do it now: capture is essentially free (we already log
the chat completions); the user can naturally accumulate a corpus
during evening coding sessions. The downstream replay work can then
proceed in parallel with the existing perplexity / KL / MMLU /
HellaSwag oracle pipeline.

The case to defer: V1 quant search isn't done; adding scope risks
distracting from the existing tier-3 measurements; behavioral metrics
without a baseline of capability metrics are hard to interpret in
isolation.

## Recommendation

Authorize **capture only** now (zero-incremental-cost), defer replay
+ scoring infrastructure to after V1 quant search ships. This banks
the corpus for free and avoids scope-creep on the immediate
deliverable.
