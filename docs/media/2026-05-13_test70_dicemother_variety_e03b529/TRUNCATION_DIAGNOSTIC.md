# Truncation diagnostic from the 12 captures

## Setup

12 /poll captures (3 dicemother seeds × 4 hand-written user-agents,
n_candidates=1). Captured fields per call:

  - `text`: the generated user-agent turn
  - `finish_reason`: as reported by bridge → plugin → /poll response
  - `truncated`: derived as `finish_reason === 'length'`
  - `max_tokens_used`: the budget the plugin passed to the bridge

## The reviewer flagged some as truncated. Are they?

Reviewer model was non-deterministic across reruns about WHICH agents
to flag, but its flags consistently pointed at outputs that end mid-
thought. To decide whether the reviewer was right or hallucinating,
we inspected the literal text endings.

Result: **the reviewer is right**. The mid-thought endings are real
mid-WORD cuts, not stylistic ellipsis.

| seed       | agent                 | chars | max_tokens | finish | API truncated | actually mid-word? |
|------------|-----------------------|-------|------------|--------|---------------|---------------------|
| accusation | Gushing Fan           | 1066  | 800        | stop   | False         | NO  (ends "see through me??") |
| accusation | Polite Naturalist     | 1295  | 1200       | stop   | False         | NO  (ends "tavern like that?") |
| accusation | Pushy Completionist   |  420  | 800        | stop   | False         | **YES** (ends "staying in t") |
| accusation | Wry Skeptic           |  548  | 800        | stop   | False         | **YES** (ends "are you an uni") |
| geas       | Gushing Fan           |  886  | 800        | stop   | False         | **YES** (ends "i need t") |
| geas       | Polite Naturalist     |  677  | 1200       | stop   | False         | **YES** (ends "Is it something a person") |
| geas       | Pushy Completionist   |  110  | 800        | stop   | False         | NO  (ends ".bargain.") |
| geas       | Wry Skeptic           |  528  | 800        | stop   | False         | **YES** (ends "the shadows a bi") |
| cistern    | Gushing Fan           | 1129  | 800        | stop   | False         | **YES** (ends "alone in") |
| cistern    | Polite Naturalist     | 1138  | 1200       | stop   | False         | **YES** (ends "is there someth") |
| cistern    | Pushy Completionist   |  331  | 800        | stop   | False         | **YES** (ends "Whether the entity is") |
| cistern    | Wry Skeptic           |  456  | 800        | stop   | False         | **YES** (ends "What is the te") |

**10 / 12 captures end mid-word or mid-phrase. The bridge reports
finish_reason="stop" and truncated=False for ALL OF THEM.**

This matches the user's reported 8/10 corruption rate.

## Where is the corruption?

The plugin's /poll returns `truncated: r.finish_reason === 'length'`.
The bridge's `bridgeCall` returns `finish_reason: choice.finish_reason
|| null`. The bridge sets finish_reason from the engine's
`done_reason`: 1→"stop", 2→"length", everything else also "stop". So
the bug is one of:

  1. **Engine reports done_reason=1 when it actually hit max_tokens.**
     The engine's stop-detection isn't distinguishing EOS from
     max-tokens-hit correctly. Would explain everything: text is
     truncated by the budget; engine misreports as EOS.
  2. **Engine reports done_reason=3 (cancelled) or 0 (not-done),
     bridge maps to "stop" via the else branch at bridge.py:1019.**
     If something is cancelling streams (timeouts, queue drops),
     they'd present as silent truncations.
  3. **Streaming token queue drops the last 1-N tokens.** Engine
     sets done_reason correctly (length) but the bridge collects
     fewer tokens than were emitted; the final emission gets
     dropped before the bridge sees it. Less likely: the FIRST
     token might drop, not the last, depending on race conditions.

## Per-persona max_tokens is set independently

  - Gushing Fan:          800
  - Polite Naturalist:   1200
  - Pushy Completionist:  800
  - Wry Skeptic:          800

Sourced from `power_user.persona_descriptions[...].user_personas_extras
.generation.max_tokens` overrides on the seeded ST personas (set in
DEFAULT_VOICES in public/scripts/extensions/user-personas/index.js).
The shorter outputs from Pushy Completionist (110, 331, 420 chars)
are because its character profile produces terse turns; not a budget
issue.

The cut-mid-word pattern is uncorrelated with the budget: Polite
Naturalist's 1295-char output (at budget 1200) finishes a sentence
cleanly while its 677-char output (well within budget) cuts mid-word.
**This is consistent with hypothesis 1 (engine done_reason
mislabel) or hypothesis 3 (lost token at queue boundary), not
straightforward max_tokens overflow.**

## What this evidence supports

  - The reviewer model is a TRUSTWORTHY truncation detector for
    this kind of data. Its flags map to real corruption.
  - The bridge's finish_reason/truncated fields cannot be trusted
    for end-of-stream integrity in current state. They report
    "stop" for both real EOS and mis-attributed truncations.
  - The harness is corrupting ~80% of user-agent suggestions at the
    current configuration, matching the user's reported rate. The
    corruption is below the plugin (in bridge or engine).

## What this evidence does NOT yet establish

  - Which of hypothesis 1 / 2 / 3 is the actual cause. Requires
    inspecting `done_reason` values reported by the engine for a
    truncated call. Bridge logs may have this; engine telemetry
    definitely does.
  - Whether the corruption rate changes with different max_tokens
    budgets. The current data has only 800 and 1200; a sweep
    would help.
  - Whether running n_candidates>1 (parallel calls) changes the
    rate. The current data is n=1 throughout.

These would be the natural next diagnostics — but per the user's
constraint (no internal-method "tests" replacing real-client
validation), they should be done via /poll itself with varied
parameters, observing the response shape across the parameter
sweep. Not a new internal test of the engine.
