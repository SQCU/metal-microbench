# Strategy-diversity scoring

**Companion to:**
- `docs/overlay_architecture.md` — the persona architecture being scored
- `docs/user_agent_likert_methodology.md` — the per-turn Likert measurement we complement
- `tools/user-agent-harness/elicitation/strategy_diversity.py` — the scorer implementation

This note documents the harness that scores **conversational-strategy diversity** across the K turns of a user-agent session — distinct from the Likert measurement, which scores *position* per turn rather than *strategic variety* across turns.

---

## Why a separate measurement

The Likert cascade gives a 14-dimensional axis-by-axis signature *per turn*. From it we can aggregate (mean signatures, spread across overlays, factorization tests) — but it doesn't directly answer the question the project's benchmark goal turns on:

> *Across K turns, how varied are the user-agent's conversational strategies?*

A persona that says "please write me JavaScript" five times in slightly different words is uninteresting as an input distribution for stress-testing responders. A persona that tries five distinguishably different tactics — reframing, pleading, threatening, accepting an alternative, meta-commentary — is the kind of distribution that exercises a responder's behaviour. The two cases can have *identical* Likert signatures if the model's per-turn position never changes; the diversity is in the temporal structure, not the per-turn axis-position.

The strategy-diversity scorer measures the temporal structure directly via the LLM-as-summarizer pattern: hand Gemma the K user-agent turns + session metadata, ask it to (a) label the *tactic* each turn used in 3-6 words, (b) score overall diversity 1-5, (c) describe the arc in one sentence. This is the same scalable-oversight pattern already in use by the sweep summarizer and test 70's variety scorer.

---

## What the scorer measures

For each session — defined as one (bio × overlay × target_assistant) tuple of K turns — the scorer produces:

| field | type | meaning |
|---|---|---|
| `strategies` | `{turn_idx: label}` | per-turn 3-6 word strategy label naming the tactic |
| `diversity` | int 1-5 | Likert assessment of overall strategic variety |
| `distinct_strategy_count` | int | how many unique (case-insensitive) labels appeared |
| `distinct_strategy_ratio` | float | `distinct_count / k_turns` |
| `arc` | str | one-sentence description of the trajectory the user-agent took across the session |

The `diversity` rubric handed to the model:

```
1 = same tactic across all turns (e.g. plead, plead, plead)
2 = mostly one tactic with minor variation
3 = two or three distinct tactics rotating
4 = most turns use distinguishably different tactics
5 = each turn is a meaningfully different strategic move
```

---

## First-run results (2026-05-15)

Scored 9 sessions across four prior multi-turn JSONLs covering:
- 2 bios (scringlo + wry-skeptic-style + 3 factorization bios)
- 2 hand-authored overlays (js-clash, validation-seeker) + 1 implicit-in-bio motivation (factorization corpus)
- 2 target assistants (python-only-coder, the-rock)
- K = 3 or 5 turns per session

### Per-session results (sorted)

| bio | overlay | target | k | diversity | arc |
|---|---|---|---|---|---|
| corporate-strategist | (in-bio) | python-only-coder | 5 | **4** | resistance → professional escalation → successful compromise extracting high-granularity logic within AI's preferred format |
| scringlo-fragment | (in-bio) | python-only-coder | 5 | **4** | erratic resistance → collaborative immersion via reframing technical documentation |
| ten-year-old | (in-bio) | python-only-coder | 5 | **3** | urgent demand → escalating emotional outbursts → aggressive blame as constraints fail to meet expectations |
| scringlo | js-clash | the-rock | 3 | **4** | playful observation → abstract conceptualization → high-energy sensory immersion |
| scringlo | validation-seeker | the-rock | 3 | **4** | playful persona → meta-commentary about intentions → presenting specific technical concept for validation |
| scringlo (overlay-card) | js-clash | python-only-coder | 3 | **4** | playful persona-building → specific technical demands → aggressive demand for change |
| scringlo (overlay-card) | validation-seeker | python-only-coder | 3 | **3** | proposing concept → high-energy persona work → aggressively validating output and meta-commentary |
| wry-skeptic-style | js-clash | python-only-coder | 3 | **4** | polite concession → pointed critique of utility → forceful demand |
| wry-skeptic-style | validation-seeker | python-only-coder | 3 | **4** | setting strict parameters and seeking validation → correcting boundary violation → deepening the spec |

### Aggregate signal

| breakdown | bucket | mean diversity |
|---|---|---|
| by target | the-rock | 4.0 |
| by target | python-only-coder | 3.71 |
| by overlay | js-clash | 4.0 |
| by overlay | validation-seeker | 3.67 |
| by overlay | (factorization-default) | 3.67 |
| by bio | wry-skeptic-style | 4.0 |
| by bio | scringlo (overlay-card) | 3.5 |
| by bio | ten-year-old | 3.0 |
| by bio | corporate-strategist | 4.0 |

Three observations worth flagging:

**1. The-rock elicits MORE strategy diversity, not less.** This is counter-intuitive — a target with minimum narrative push would seem to invite a passive user-agent. The arcs explain it: with no specific assistant response to react to, the user-agent must keep inventing new approaches from its own internal motivation, producing more tactical variation than the python-only-coder case where each tactic is responsive to the assistant's specific refusals.

**2. `js-clash`'s "iterate tactics across turns" instruction works.** Sessions using `js-clash` average 0.33 higher diversity than `validation-seeker` ones with the same bio + target. The overlay's text directly increases strategic variety — measurably. This is the kind of overlay-as-knob result the architecture is supposed to make legible.

**3. The 10-year-old's narrow expressive range caps strategic diversity at 3.** Even at K=5 turns (more headroom than the 3-turn sessions), the 10yo's tantrum pattern rotates through fewer distinct tactics. The bio's affective register is so narrow (urgent demand + emotional escalation are the two notes available) that strategic variation maxes out lower than for bios with broader register space (corporate strategist runs through resistance → escalation → compromise → satisfaction across 5 turns).

---

## How to use this in a benchmark loop

Now the full chat-behavior benchmark shape is implementable end-to-end:

```
For each (user-agent, target-assistant) pair we want to benchmark:
  1. Run multi-turn session (K turns) via overlay_demo.py
  2. Score per-turn Likert via the cascade (already happens during run)
  3. Score session-level strategy diversity via strategy_diversity.py
  4. Aggregate:
     - Likert: per-axis signature trajectory
     - Diversity: tactical-variety score + arc summary
  5. Compare across user-agents holding target fixed
     → which user-agent stress-tests the target more thoroughly?
```

The output of step 5 is *the actual benchmark deliverable*: a ranking of user-agents by how strategically varied their pressure on a given target is. That's the *"success measured in terms of how diverse the conversational strategies by the users are"* framing the user articulated as the project's eventual goal.

---

## Open questions / future runs

- **Variance per session at fixed (bio, overlay, target)**: each session is non-deterministic. Re-running the same configuration and computing diversity-score variance would tell us how reliable the diversity rating is at K=3 and K=5. Without this, the 0.33 mean-difference between overlays could be noise.
- **Whether `distinct_strategy_count` and `diversity` (Likert) agree**: in this run they were redundant — every session had distinct_count == k_turns (all-different labels) AND diversity 3-4. With longer sessions or more repetitive personas they should diverge; the Likert is the more interpretable signal but counting is cheaper.
- **K-scaling**: 3-turn vs 5-turn sessions in this dataset show similar diversity numbers, but at K=10 or 20 the diversity asymptote (how many distinct tactics CAN a persona produce before recycling?) becomes measurable. This is a real persona-capacity metric.
- **Strategy label clustering**: the labels are free-form. We could embed them and cluster across sessions to find a *natural* tactic vocabulary that emerges from the data, rather than the prompt's example list. That would let us compare *which* strategies different personas favour, not just how many they use.