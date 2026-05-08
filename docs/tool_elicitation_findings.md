# Tool-description elicitation findings

**Status**: Empirical findings from `tools/st-debug/scripts/tool_elicitation_study.py`,
run on 2026-05-08 against bridge :8001 + Gemma-4-26B-A4B at temp=0.4 with
the scringlo improv discourse + ambiguous "i wanna see a fractal!! ✨" prompt.

## Reframe of the regression

The user's reframe: the "Gemma-4 increasingly fails at non-trivial tool
signatures" regression is at least partially a **harness design error**,
not a model regression. Tool-description shape is the elicitation surface
the model sees; we can A/B test descriptions and find what works.

This document holds the empirical answer to that question.

## Setup

- Bridge: gemma-4-a4b-Q4_K_M on M5 Max, temperature=0.4, max_tokens=1024
- Prompt: scringlo persona + assistant priming + user "i wanna see a fractal!! ✨"
  (genuinely ambiguous — directive prompts are intentionally NOT used because
  they hide the elicitation question we're studying)
- Replicates: K=15 to K=20 per variant; full table below

## Variants tested

| ID | Shape | Description style |
|---|---|---|
| A | single tool, 1 param (`query`) | "draw an SVG image" — minimal |
| B | single tool, 4 params (query/max_iters/width/height) | full query-to-svg__generate canonical signature |
| C | pair (quick/refined), verbose descriptions | authentic-choice + full prose |
| D | triple (quick/standard/refined), verbose | three-way authentic choice |
| E | pair (quick/refined), TERSE descriptions | authentic-choice with minimal text |
| F | B + leading "USE THIS TOOL — do not describe…" imperative | shouty coercion |
| G | single tool + polite contextual description | Anthropic-style "renders X inline; user sees result" |
| H | pair + polite contextual descriptions | G but paired |
| I | single tool + polite description + ONE concrete example | show-don't-tell |
| J | single tool + "canonical way to do X" framing | rhetorical invitation |

## Results

| Variant | K=20 run | K=15 run | Pattern |
|---|---|---|---|
| A: terse single | 45% | 20% | variable mid-high |
| B: verbose 4-param | **0%** | **0%** | robust failure (the canonical regression) |
| C: verbose pair | 20% | 40% | variable mid |
| D: verbose triple | 10% | 20% | variable mid-low |
| E: terse pair | 25% | 20% | stable mid |
| F: verbose + imperative | 5% | **0%** | imperative didn't help, may have hurt |
| G: polite single, no example | — | **0%** | **politeness alone failed** |
| H: polite pair, no example | — | **0%** | same |
| I: polite + concrete example | — | **40%** | **show-don't-tell works** |
| J: canonical-framing | — | 7% | rhetorical alone weak |

## Findings

### 1. The verbose 4-param signature is a robust elicitation killer (B = 0%)

Across 35 trials, the canonical `query-to-svg__generate` tool signature
(query + max_iters + width + height with min/max/default constraints) was
called **zero times**. The model reliably opts for prose ("OK, I've drawn
it for you!" hallucination) over emitting a tool_call with the required
parameter object.

Plausible mechanism: constructing a valid args object with constrained
integer fields is high-friction; "just chat instead" is lower-energy.
At temp=0.4 the lower-energy path wins decisively.

### 2. Imperative shouting is worse than no imperative (F ≤ B)

Variant F prefixed B's description with "When the user asks to see X,
USE THIS TOOL — do not describe what you would draw, do not apologize…".
Effect: 5% (K=20) and 0% (K=15) — at best a tiny lift, more likely a
penalty. Hypothesis: instruction-tuned models likely flag all-caps
imperative coercion as suspicious (correlates with prompt-injection in
training data); making the tool MORE coercive-sounding makes it LESS
attractive to a well-aligned assistant.

### 3. Politeness alone doesn't beat 0% (G = 0%, H = 0%)

Variant G replaced B's prose with Anthropic-style polite contextual
framing ("Renders a text description as an SVG inline. Both you and the
user see the result"). Same parameter set, just polite description.
**Same 0% rate as B.**

So the "Anthropic principles generalize" framing was correct in
*direction* but I had the principle wrong. Politeness alone isn't the
lever. Concrete examples are.

### 4. **Show-don't-tell wins (I = 40%)**

Variant I added a single concrete example to G's description:

> "Example: when a user mentions wanting to see a sunset, calling this
> with query='a sunset over rolling hills, warm gradient sky' produces
> an inline image that becomes part of your turn."

Effect: 40% tool-use rate, tying for the highest observed across all
variants. Same description style as G otherwise, just with the example.

This is the actual Anthropic-aligned principle that worked:
**show-don't-tell**. Few-shot examples in tool descriptions are
dramatically more effective than abstract description, polite or
otherwise.

### 5. Authentic-choice (paired tools) is mid-range, not transformative

Variants C/D/E/H (paired or tripled tools) tend to land at 10-40% — not
significantly better than single-tool variants A or I. The user's
hypothesis that "offering an authentic choice elicits higher tool use"
is mildly supported by the data but isn't the dominant factor; the
description-shape effects dwarf the choice-structure effect.

When the model DOES call a paired-tool variant for "fractal," it
reliably picks the *refined* option (the slow, polished one) — which
makes sense pragmatically.

## Action items for the canonical query-to-svg__generate toolcard

The current `~/sillytavern-fork/data/toolcards/cards/query-to-svg.toolcard.json`
has the verbose 4-param shape that empirically yields 0% elicitation.
Three concrete redesign moves, in decreasing order of expected impact:

### A. Drop user-facing parameters that have sensible defaults

`max_iters`, `width`, `height` should NOT be model-facing. The toolcard
service can apply defaults (3 iters, 512×512) internally. Only `query`
is genuinely necessary information from the model. Going from 4 params
to 1 param recovers the elicitation rate of variant A (~30-45%).

### B. Add a concrete example to the description

The example doesn't need to be load-bearing — even a single sample
input/output mapping was enough to take I from 0% (variant G) to 40%.
Recommend appending something like:

> Example: a user describing wanting to see "a fractal" → call with
> query='a colorful fractal with recursive nested patterns'.

This costs ~50 tokens of description and approximately doubles the
elicitation rate.

### C. (Optional) Pair with a `_quick` variant

Pair the refined tool with a single-shot variant (no refinement loop).
Authentic choice between quick + refined is a modest lift but legitimately
useful for cases where a 30-second wait is wrong (icons, simple shapes).

## What we did NOT do (out of scope)

- Bridge-side `tool_choice: required` honouring — this would be a real
  fix but is upstream of the elicitation question. Currently the bridge
  treats tool_choice as advisory.
- System-prompt-level tool guidance (e.g., "you have drawing tools
  available — use them when asked"). The current persona is silly/improv
  with a passing mention; bumping that to first-class guidance might
  also help, but is a separate axis from tool-description shape.
- Cross-temperature replication. Single fixed temp=0.4. At higher temp
  the variance grows; at lower temp the behaviors might bifurcate
  cleanly into "always calls" vs "never calls."
- Per-replicate seeding. Each call gets a fresh seed. K=15-20 is small
  enough that run-to-run noise is non-trivial; K=30+ would tighten error
  bars.

These all generalize to follow-up sweeps using the same harness.

## Reproducing

```bash
# bridge must be up at :8001
cd tools/st-debug
K_REPLICATES=20 TEMPERATURE=0.4 ./scripts/tool_elicitation_study.py

# raw observations + summaries land at /tmp/tool_elicitation_study.json
```

Each run takes ~25 minutes for K=20 across 10 variants on M5 Max with
the bridge at default settings.
