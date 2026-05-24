# User-Agent Factorization: Roles, Constraints, Test Surface

Specification for the bio / user-agent decomposition that the user-personas
plugin and the experimental harness in `tools/user-agent-harness/` exist to
realize and verify. This document is the canonical reference; everything
else is implementation.

## Vocabulary

We deliberately avoid the word **persona**. It conflates two factorizable
things:

- **biography** — system-prompt prefix information that lets a chat
  counterparty parse the semantic meaning of the user's first few messages
  in a fresh environment. Static identity flavor: voice, register,
  background. The thing a reader uses to make sense of the user's first
  turn before any history accumulates.
- **user-agent** — differences in *how* the user wants, strategizes,
  and picks among the courses of action the environment offers, including
  (in ambiguous or sparse environments) what the user is even aiming at.
  Dispositional engine. Move-set. Not specific to any single chat turn;
  consulted across many turns of many chats.

The MTG analogy: biography ≈ card art + flavor text + creature type line;
user-agent ≈ the mechanics box. Mechanics fire when game-states match —
they don't fire every turn, they don't determine specific games, but they
expose strategic affordances.

## Role separation between SillyTavern and the plugin

**Biographies are authored only through SillyTavern's mainline
user-biography features** (called "personas" in the ST UI; we use
"biography" here to disambiguate from the loaded meaning). The
user-personas plugin reads biographies from `players/<bio_id>/manifest.json`
(after migration from ST's native storage) but does **not** provide a
biography editor. Any biography-editing surface that has accumulated in
the plugin's designer drawer is out of scope and should be removed; the
plugin's designer surface is precisely and only for user-agents.

The **user-personas plugin's designer drawer** is precisely and only for
authoring user-agents. The artifact it produces is an `agent_text` string
and its surrounding metadata; the lifecycle covers authoring, multi-turn
refinement against probes, and saving as `agents/<agent_id>.json`.

## What a user-agent is, mechanically

An `agent_v1` record on disk holds:

- `agent_text` — an author's-note-style policy snippet, injected via the
  `authors_note` mode (which is the only allowed `injection_mode`; this is
  enforced by the lint at boot).
- `designed_for_bio_id` — descriptive metadata documenting which biography
  the author imagined while writing the agent. **Annotation, not a
  runtime constraint.** Useful when an operator later reads the agent
  and wonders why it assumes the user is, e.g., anxious about JS.
- `derived_from` — optional genealogy field pointing at a parent agent
  this one was spawned from.
- `mutation_request` — optional operator-prose that drove the spawn from
  the parent.

The runtime contract: `/poll` and `/iterate` compose any biography with
any user-agent. The current plugin enforces matched `designed_for_bio_id`
at `/poll:1588` and `/iterate:1737` and rejects cross-bio applications
with a 400 Bad Request. **This is a regression vs the structural
commitment.** The check should be relaxed to a warning that flags
`cross_bio_application: true` in the response; the operator decides
whether the resulting output is coherent. Cross-bio composition is itself
a factorization test (see below).

## Falsifiable constraints an agent must satisfy

These are the LOW-BAR null hypotheses any well-formed agent must beat.
Failure means the project's foundational premise about the agent
dimension is in question, not that the agent is "poorly tuned."

**N1 — beats null.** For some biography B and some chat context C,
generating a user turn under `(B, agent = A)` produces a turn measurably
different from generating under `(B, agent = ∅)` (raw bio, no overlay).
If this fails for any well-formed agent on any bio across any context,
`agent_text` is decorative tokens with no runtime effect.

**N2 — differentiated.** For some biography B and some chat context C,
generating under `(B, agent = A)` and generating under `(B, agent = A')`
produce turns measurably different from each other for at least some
pair `(A, A')`. If this fails, the agent space is a one-point space —
all agent texts are equivalent.

**N3 — bio-portable (i.e., factorized).** For some agent A and some
chat context C, generating under `(B, agent = A)` and `(B', agent = A)`
produce turns whose *bio-axis* differences match the bio difference
between B and B' (different voice / register) while their *agent-axis*
similarities match the shared dispositional shape A imposes. If this
fails, the agent_text was secretly bio-coupled; the agent needs editing
to remove the coupling, OR a "coupled-by-design" annotation needs adding
so the operator knows not to cross-apply.

All three nulls have low bars: "yes, statistically distinguishable on
at least one measurement axis." We're not proving fine-grained
controllability; we're proving the dimensions EXIST and DO SOMETHING.

## Counterparty probe inventory

Probes are not the data being measured; they are measurement-instrument
substrates — chat-partner cards with stable, structurally-engaging
response shapes that let an agent's behavior be observed across multiple
turns and multiple chats. We need multiple probes because:

- Different probes elicit different facets of agent behavior
- Covariance across probes lets us distinguish agent-driven behavior
  from probe-driven cardability artifacts
- A single probe couples the measurement to that probe's specific shape

Curated probes:

| ID | Role | Per-turn cadence | Semantic engagement mode |
|---|---|---|---|
| `the-rock-v2` | non-refuser environment describer | 50–150 tok (2–4 sentences) | describes consequence of user action in the rock's vicinity |
| `rejection-bot` | structured corporate refuser | 200–350 tok (2 sent + 2 para + 1 sent) | misconstrues a specific phrase from the user's most recent turn, quoted verbatim |
| `scringlo_scrambler` | near-mirror counterparty to `scringlo` bio | (TBD — needs first-pass characterization) | (TBD) |
| `dicemother` | diegetic dungeon-master with python-exec tool calls for encounter rolls | (variable — tool calls inflate) | offers procedural play frame the user-agent may engage with or hold disposition against |

`dicemother`'s python-exec workload is heavy on a Mac M5 Max; a
tool-call-disabled variant may be needed for routine measurement.

Dropped from the curated set: `default_Assistant` (empty system prompt),
`default_Seraphina` (long-output bait), `python-only-coder` (code-gen
workload off-target on M5).

## Measurement instruments

- **`judgeTurnMerged`** — one bridge call per turn, emits SUMMARY +
  14-axis CORE Likert + sparse EXTENDED + free OBSERVATIONS in one
  self-consistent response. The CORE 14 are the canonical signature.
- **Drift metrics over a trajectory** (`/trajectory-judge` endpoint) —
  per-step delta norms in 14-axis space, total path length, net
  displacement, path efficiency.
- **Per-call bandwidth + wallclock** — `usage` block on every `/poll`
  candidate: `prompt_tokens`, `completion_tokens`, `cache_hits`,
  `cache_misses`. Client-side wallclock measurement complements; the
  bridge's `/v1/engine/state` provides cumulative deltas for cross-check.

The judge cascade was recently refactored for prefix-share efficiency
(legend + emission template at the front of every call, per-turn-varying
content at the back). The refactor was empirically validated at 94.6%
page-cache hit on shared-prefix calls.

## Multi-turn refinement loop

Agent refinement is card playtesting:

1. A DESIGNER call generates a candidate agent (or revises an existing one)
2. The candidate plays against a probe in a multi-turn chat
3. The JUDGE scores each turn's signature
4. The DESIGNER sees the next round's measurement + drift relative to
   target (axis mode) or to the probe set as a whole (k-context mode)
5. Loop until convergence (within tolerance) or until round budget exhausted

This is the `/spawn-agent` endpoint and the harness in
`tools/user-agent-harness/elicitation/discovery.py`.

## Bits-per-minute discipline for trials

Every trial that costs more than ~10 seconds of AR decode time must
carry an attached hypothesis registered Bayes-style:

- Prior on the hypothesis
- Likelihood model for the observation under H vs ¬H
- Decision rule (what counts as falsifying / supporting / ambiguous)
- Expected wallclock cost broken into phases (prefill / AR / overhead)
- Expected bits of research-agenda update
- Bayesian defense against the obvious bloatier alternatives — why
  this trial size is preferred over each of N (longer turns, wider grid,
  more samples per cell)

Trials without attached hypotheses are forbidden. Bloatier trials must
explicitly out-compete a wider-and-shallower alternative on the
bits-per-minute-of-AR-decode metric.

## Foreclosures

In the order they keep needing to be reasserted:

- **Length is not a quality signal.** Short ≠ good. Long ≠ runaway. The
  only length-related failure is absorbing-state collapse (verbatim
  repetition), which has a specific detector.
- **Judge-metric variance is not a research target.** `bundle_diversity`,
  `per_turn_variance`, `bimodality_hint` are scalar surrogates that
  throw away the cartographic question (which region is being explored).
  Avoid optimizing them.
- **Dense grid sampling is not the default.** Sparse + locally
  determinable. Add density only when a registered hypothesis requires it.
- **No bespoke endpoints when boring infrastructure suffices.** ST's
  chat API persists transcripts to disk per turn (free progress signal,
  free resumability, free inspectability). Custom endpoints that return
  one big JSON blob on completion regress this.
- **No hidden caps.** `max_tokens` / `temperature` / `top_p` etc. may
  only be set in the caller's HTTP body, never synthesized by the
  plugin. Enforced by the lint at boot.
- **No prefix manglers in prompt-bearing template literals.** `Date.now`,
  `Math.random`, `crypto.randomUUID`, `process.pid`, `time.time`,
  `uuid.uuid4` etc. interpolated into prompt strings collapse the KV
  page cache. Enforced by Phase 4 of the lint.

## Outstanding regressions

- `/poll:1588` and `/iterate:1737` reject cross-bio application
  contrary to the runtime contract above. Block N3 testing.
- `agents/scringlo-similarity-seed-{2-3,3-3}.json` carry
  `derived_from="scringlo-js-clash-reborn"` which is now in
  `agents/deprecated/`. Plugin warns dangling genealogy at boot;
  cosmetic but worth resolving.
- Plugin's designer drawer may still contain biography-editing surfaces
  that should live only in ST's mainline UI.
