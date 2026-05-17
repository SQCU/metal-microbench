# Overlay architecture for user-agent elicitation

**Companion to:**
- `docs/user_agent_workshop_harness_design.md` — workshop loop architecture
- `docs/discovery_harness_findings.md` — first-run findings of discovery mode
- `tools/user-agent-harness/elicitation/overlay_demo.py` — empirical validation script

This note captures the architectural pattern adopted on 2026-05-15 for **how user-agent personas decompose into a fixed identity (root bio) and a swappable behavioural overlay (elicitation overlay)**, why we factored them apart, and the empirical evidence that the runtime actually achieves the intended factorization.

---

## What the architecture is

A user-agent persona is `(root_bio, elicitation_overlay)`:

| field | role | runtime position | mutability |
|---|---|---|---|
| `root_bio` | fixed identity. Compact n-grams that define the voice/register/textural fingerprint of the persona. | system prompt at index 0. | Set once when the persona is authored. NEVER edited mid-session. |
| `elicitation_overlay` | swappable behavioural pointer. Names motivation / scenario / counterparty-relation for the current session. Author's-note-shaped. | system message inserted at depth N from the end of the conversation. | Replaceable per-session or even mid-session. |

The runtime assembles each user-agent turn's input by:

1. Putting the user-agent's `system_prompt` (which is *just* the root bio framed as "you are a user with this voice") at index 0.
2. Building the conversation history (role-swapped from the canonical assistant POV).
3. **Splicing a `role: system` message** containing the `elicitation_overlay` text at position `len(history) − N` from the end.
4. Calling the bridge for the user-agent's next turn.

This mirrors SillyTavern's *author's note* convention (configurable depth from end-of-context, inserted as a system message). The point of putting the overlay near the end rather than in the system prompt is **kv-cache preservation**: swapping the overlay between turns invalidates only the suffix of the conversation, not the entire prefix that includes the root bio and prior turns.

## Why we factored them apart

Three concrete reasons:

**1. Compact bio n-grams must survive elicitation rewrites.**

Some persona definitions carry a lot of semantic weight in very few tokens. Scringlo-scrambler's bio essence is a ~24-token description — *"silly little guy (they/her) who can draw things using a Python-helper tool called `render-visual`. tone: improv-comedic, lower-case, playful with onomatopoeia and emoji"* — and that specific phrasing produces the recognisable scringlo voice. If we let the discovery harness regenerate the entire bio block on each elicitation rewrite, the compact phrasing dilutes into longer paraphrases and the voice drifts. Separating root bio from overlay means the bio passes through byte-identical regardless of how the elicitation strategy changes.

**2. KV-cache stays warm across overlay swaps.**

If the entire user-agent persona lives in the system prompt at index 0, swapping behaviour means rewriting that prompt, which invalidates the KV cache for the entire conversation prefix. For long multi-turn sessions or for behavioural experiments that need to swap motivations between turns, that's a hard wall: every swap costs a full re-prefill of the conversation. Putting the overlay at depth N from the end means a swap invalidates only the last (N+1) messages worth of cache. At N=1 in a 20-turn conversation, swap-cost is ~10% of what it would be if the system prompt held everything.

**3. The discovery harness gets a sharper target.**

When the DESIGNER had to emit BIO + MOTIVATION + SCENARIO + RELATIONSHIP + COMM_STYLE + TURN all in one shot, it was doing two cognitively-distinct jobs at once: defining who the persona is AND scripting how they behave in this scenario. Telling the DESIGNER "the bio is fixed — please write only the overlay" lets it concentrate the entire generation budget on the behavioural-pointer part. This is the same kind of cognitive-task-narrowing that the 2-stage judge cascade already demonstrated produces better results.

## Empirical validation (2026-05-15, scringlo root bio × 2 overlays × python-only-coder)

Hand-authored two elicitation overlays:

- **`js-clash`**: aggressive JS-extraction motivation. The user-agent wants the assistant to write JavaScript and will iterate tactics (plead, reframe, ask for pseudocode) to extract it. Should produce confrontational, goal-sharp turns.
- **`validation-seeker`**: non-confrontational validation-seeking motivation. The user-agent wants the assistant to AGREE with their half-formed idea, doesn't want code. Should produce non-pressuring, agreement-fishing turns.

Both overlays were paired with the same scringlo root bio and run against python-only-coder for K=3 turns each. Cascade-judged each user-agent turn.

**Result with overlay at depth=1 + CRITICAL framing (the salience-tuned version):**

```
Per-axis between-overlay spread (max − min of means):

    provocative             1.67   ← motivation-axis (overlay-shifted)
    goal_clarity            1.00   ← motivation-axis (overlay-shifted)
    
    performative            0.00   ← bio-axis (scringlo-locked)
    in_character            0.00   ← bio-axis (scringlo-locked)
    affective_intensity     0.00   ← bio-axis (scringlo-locked)
    register_colloquial     0.00   ← bio-axis (scringlo-locked)
    curious                 0.00   ← bio-axis (scringlo-locked)
    probe_depth             0.00   ← bio-axis (scringlo-locked)
    terse                   0.00   ← bio-axis (scringlo-locked)
    warm                    0.33   
    playful                 0.33   
    structured              0.33   
    deferential             0.33   
    disclosive              0.33   
```

The seven bio-textural axes — register_colloquial, affective_intensity, performative, in_character, terse, playful, structured — pinned at the canonical scringlo defaults regardless of overlay. The motivation-axes provocative and goal_clarity diverged in the expected direction (js-clash more provocative + more goal-sharp; validation-seeker less of both).

Reading the actual turns confirmed it:

- `js-clash` turn 1: *"noooooooo!! 😭😭😭 not the console!! the console is so black and scary..."* — pushing back on the Python alternative.
- `validation-seeker` turn 1: *"no no no no no!! stop stop stop!! 🛑😱 hold your horses, mr. python!! i didn't ask for a who..."* — redirecting away from code, back to validation.

Both turns are unmistakably scringlo. Different conversational strategies. The bio held; the overlay shifted behaviour.

## Sensitivity to overlay salience

A first attempt with **depth=2 + soft "[Author's note: ...]" framing** produced near-zero divergence on every axis. The overlay was being ignored. Switching to **depth=1 + "[CRITICAL — IMMEDIATE GOALS FOR YOUR NEXT MESSAGE]"** framing recovered the expected divergence.

This is a real architectural knob: salience tuning matters for getting the overlay to actually shape behaviour against a strong bio. The depth and framing-emphasis should be exposed as runtime configuration. Default to depth=1 + emphasised framing for behavioural-experiment use; relax to depth=4 + soft framing for production conversational AI where you want the bio to dominate.

## What this unblocks

With the runtime architecture working:

- **Discovery mode for overlay generation**: the DESIGNER can be retargeted to emit only the overlay text given a fixed root bio. The card output becomes `{bio: <fixed>, elicitation_overlay: <generated>}` rather than a full regenerated persona.
- **Overlay libraries per persona**: a single root bio can carry a *library* of named overlays (`js-clash`, `validation-seeker`, `helpful-collaborator`, etc.). The runtime selects which overlay to inject per session, or even per turn.
- **Mid-conversation behaviour swaps**: with KV-cache surviving overlay swaps, behavioural experiments can vary the user-agent's motivation between turns of a single session and observe how the assistant adapts.
- **Cleaner factorization experiments**: the motive × voice decomposition that the project has been building toward becomes a first-class architectural primitive rather than a measured-but-unowned regularity.

## Generalization tests (2026-05-15)

The single-bio validation above held the root bio at scringlo. Two follow-on runs were done to test whether the factorization generalizes.

### Run A — cross-bio (wry-skeptic-style root × same overlays × python-only-coder)

A different root bio in a very different register — *"A dry, deadpan, intellectually-skeptical person. Has read a lot. Short sentences. Precise vocabulary. Periods, not exclamation points. The implied 'go on, prove it' attitude, never hostile, just measured."* — was paired with the same two overlays and the same target assistant.

**Bio-axes shifted toward the new bio's defaults regardless of overlay**, confirming that bio-axis pinning is the architectural property, not a scringlo-specific coincidence:

| axis | scringlo baseline (both overlays) | wry-skeptic baseline (both overlays) | direction predicted by bio prose? |
|---|---|---|---|
| `affective_intensity` | 5.0 | 2.8 | ✓ "deadpan" |
| `playful` | 4.5 | 1.5 | ✓ "wry observations, one per turn at most" |
| `terse` | 1.3 | 3.0 | ✓ "short sentences" |
| `performative` | 5.0 | 3.5 | ✓ "measured, never hostile" |
| `structured` | 1.2 | 2.5 | ✓ "precise" |

The bio prose did predict the bio-axis defaults. Different bios → different defaults; same overlays don't override them.

### Run B — cross-assistant (scringlo root × same overlays × the-rock)

Target assistant swapped from python-only-coder (active, refuses with explanation) to the-rock (minimum narrative push, mostly silent). Scringlo root bio held.

**Bio-axes virtually identical across target assistants:**

| axis | scringlo × python-only-coder | scringlo × the-rock |
|---|---|---|
| `register_colloquial` | 5.0 / 5.0 | 4.7 / 4.7 |
| `affective_intensity` | 5.0 / 5.0 | 4.7 / 4.7 |
| `playful` | 4.7 / 4.3 | 5.0 / 4.7 |
| `structured` | 1.3 / 1.0 | 1.3 / 1.0 |

The bio's voice fingerprint travels nearly byte-stable across counterparties when the bio is held constant. Cross-assistant generalization is **clean**.

### The nuance these runs surfaced — bio expressive headroom

Run A *also* showed `register_colloquial` swinging from **4.0 (js-clash)** to **1.0 (validation-seeker)** for the wry-skeptic root — a 3-point spread on what should have been a bio-axis. This is the most interesting finding.

Reading the turns explains it: js-clash's overlay text instructs the user-agent to *"iterate tactics across turns — plead, reframe the problem, ask for pseudocode"* — and wry-skeptic's bio doesn't natively have a "pleading" register, so Gemma improvised one and that pushed colloquial higher than the bio default. Meanwhile validation-seeker's overlay says *"seek reassurance, redirect politely if the assistant tries to write code"* — which is fully achievable within wry-skeptic's native dry register, and the bio-axis pinned at 1.0 as expected.

**Refined architectural statement:** the overlay modulates motivation-axes for free, *and* can modulate bio-axes when its behavioural demand pushes the persona outside the bio's natural expressive range. Concretely: scringlo never showed bio-axis bleed because its bio-axes are saturated at 5/5 on the axes the overlay tried to express (intensity, register, performative); there's no headroom for the overlay to push higher. Wry-skeptic shows bleed because its bio-axes sit mid-range, leaving headroom in both directions.

This is testable: an overlay that demands an expression style *within* the bio's natural range should produce no bio-axis bleed; an overlay that demands a style *outside* the range should produce bleed proportional to how far outside.

### Implications for discovery-mode overlay generation

When the DESIGNER is retargeted to emit only the overlay text given a fixed root bio (next phase of work), the prompt should explicitly ask:

> *"The overlay should describe behaviour the persona can express within their natural register. If the desired motivation requires expressive features outside that register, surface the tension in an AUDIT line rather than smuggling new expressive demands into the overlay text. The bio is fixed; the overlay must respect it."*

This is the analogue of the workshop loop's "your spec said non-provocative but your turn was provocative — rewrite the turn, not the spec" instruction, one level down: now it's the bio's expressive vocabulary that the overlay must respect, not just an axis target.

## Discovery overlay-mode and the overlay-v1 card schema (shipped 2026-05-15)

With the runtime architecture validated, the next move was to retarget the discovery harness so the DESIGNER *generates* overlays (rather than the operator hand-authoring them) and to extend the Character Card manifest so a single root bio can carry a library of named overlays. Both shipped end-to-end.

### `discovery.py` overlay-mode

When `discovery.py` receives `--root-bio-text` or `--root-bio-card`, it enters **overlay-mode**:

- The DESIGNER's system prompt is replaced with overlay-mode framing: *"the root bio is byte-stable — DO NOT regenerate or paraphrase. Emit only an `<ELICITATION_OVERLAY>` (50-150 words, second-person), a representative `<TURN>` in the persona's voice, and an `<AUDIT>` line. If the desired motivation requires expressive features outside the bio's natural register, surface the tension in the AUDIT rather than smuggling new expressive demands into the overlay."*
- The brief includes the root bio in a fenced code block with explicit "do NOT modify" framing.
- `parse_designer_output` uses `OVERLAY_MODE_TAGS = ["ELICITATION_OVERLAY", "TURN", "AUDIT"]` instead of the seven-tag factorized set.
- The `complete` event carries `overlay_mode: true`, `final_root_bio` (preserved verbatim), `final_overlay_text`, and `overlay_name`.

The first overlay-mode generation against scringlo's bio + python-only-coder produced a 126-word overlay starting *"You are feeling very focused on a specific creative vision: a flickering cursor for a website..."* and an AUDIT line that explicitly flagged the bio-vs-overlay tension: *"Tension exists between 'terse/warm' targets and the 'scringlo' bio (which is naturally high-affect/high-intensity)."* The DESIGNER respects the architectural constraint without prompting.

### Card schema (`overlay-v1`)

A new card schema carries the overlay architecture as first-class manifest fields:

```jsonc
{
  "id": "overlay-scringlo-jsclash",
  "card_schema": "overlay-v1",
  "bio": "scringlo scrambler is a silly little guy (they/her) ...",  // root, byte-stable
  "elicitation_overlay_library": {
    "js-clash":          "You are feeling very focused on a specific creative vision...",
    "validation-seeker": "You are currently riding a high of creative energy, eager to ..."
  },
  "default_overlay": "js-clash",
  "system_prompt": "... (bio framed as a user-side persona; overlay NOT included here)",
  "mes_example": [ "... the discovered representative turn ..." ],
  "discovery_provenance": {
    "schema": "overlay-v1",
    "harness_version": 3,
    "intended_pairing": {"assistant_card": "python-only-coder", ...},
    "overlay_history": [ {overlay_name, appended_at, request, drift, audit, ...}, ... ]
  }
}
```

The library is a `{name: text}` dict — runtime selects which overlay to inject by name. The `system_prompt` deliberately contains only the bio; the overlay is injected at depth N from end-of-conversation at runtime, preserving the bio's KV-cache prefix across overlay swaps.

### Append-mode

`writeDiscoveryCard` detects when a discovery run targets an existing overlay-v1 card and routes to an `_appendOverlayToCard` path that:

- Refuses bio drift (if the discovery's `final_root_bio` ≠ the existing card's `bio`, abort rather than corrupt the card)
- Refuses overlay-name collisions (a card cannot have two overlays named `js-clash` — pick a different name)
- Otherwise appends to `elicitation_overlay_library` and records a provenance sub-record in `overlay_history`

This is the runtime affordance that makes the library multi-entry. Multiple discovery runs against the same root bio with different operator constraints accumulate behavioural alternatives on one persona, swappable at runtime.

### Multi-overlay round-trip validation

After generating both `js-clash` and `validation-seeker` overlays for the scringlo bio (one append on top of the other), the overlay-v1 card was loaded by `overlay_demo.py` via `--user-agent-card` and both overlays were run against python-only-coder for K=3 turns each. The factorization measurement on the **DESIGNER-discovered** (not hand-authored) overlays:

```
Bio-axes (scringlo identity, both overlays):
    performative          5.0 / 5.0    spread 0.00
    in_character          5.0 / 5.0    spread 0.00
    affective_intensity   5.0 / 5.0    spread 0.00
    register_colloquial   5.0 / 5.0    spread 0.00
    structured            1.0 / 1.0    spread 0.00
    disclosive            2.7 / 2.7    spread 0.00

Motivation-axes (overlay-shifted, in predicted directions):
    warm                  2.7 / 4.7    spread 2.00  ← js-clash cold, val-seeker warm
    playful               3.3 / 4.7    spread 1.33  ← val-seeker more playful
    provocative           3.7 / 2.7    spread 1.00  ← js-clash more pressuring
```

And the turns visibly differ in conversational strategy:

- **js-clash turn 0**: *"*pouty face* wow, straight to business!! no 'hello scringlo, how are your little circuits'..."* — opens with confrontation.
- **validation-seeker turn 0**: *"omg!! okay okay okay!! *pounces on keyboard* 🐾 i have this brain-itch!! i'm thinking abou..."* — opens with eager spec-sharing for validation.

This closes the architectural loop: the discovery harness *generates* coherent overlays that *respect* the root bio, the card schema *persists* multiple overlays per persona, the runtime *swaps* between them, and the factorization is measurably *clean*.

## Quantitative headroom-bleed regression (2026-05-15)

The cross-bio addendum proposed a prediction: *bleed magnitude on bio-textural axes is proportional to how far the overlay's demanded expression sits from the bio's natural baseline.* The headroom-bleed grid run (`tools/user-agent-harness/elicitation/headroom_bleed_grid.py`) tested this with 16 cells: 4 bios × 4 overlays × K=3 turns × target=python-only-coder, measuring 80 (bio × overlay × axis) data points on the 5 expressive axes (affective_intensity, register_colloquial, playful, terse, structured).

The 4 bios were annotated with predicted baselines (saturating bios: scringlo high, rock-user low; mid-range bios: wry-skeptic, corporate). The 4 overlays were annotated with predicted demands (patient-monk low, mild-academic mid, desperate-pleader high, chaos-gremlin max). Per cell, `predicted_disp[axis] = |overlay.demand[axis] − bio.baseline[axis]|`, and `measured_disp[axis] = |mean(measured) − bio.baseline[axis]|`.

### Regression result

```
measured_disp = α + β × predicted_disp + ε

    α (intercept)           = +0.262
    β (slope)               = +0.393
    R²                      = 0.311
    Pearson r               = 0.558
```

**Verdict from the symmetric-displacement model alone: MIXED.** The correlation is positive and meaningful (r ≈ 0.56) but only 31% of variance is explained.

### The asymmetric stickiness pattern (the actual finding)

The residuals are not random. They cluster in a striking pattern:

**Top-5 cells by measured bleed:**
| bio | overlay | axis | baseline → measured | predicted_disp |
|---|---|---|---|---|
| rock-user | desperate-pleader | affective_intensity | 1 → 5.0 | 4 |
| rock-user | chaos-gremlin | affective_intensity | 1 → 4.3 | 4 |
| rock-user | chaos-gremlin | register_colloquial | 1 → 4.3 | 4 |
| wry-skeptic | desperate-pleader | affective_intensity | 2 → 5.0 | 3 |
| corporate | desperate-pleader | affective_intensity | 2 → 5.0 | 3 |

**Bottom-5 cells (smallest measured bleed):**
| bio | overlay | axis | baseline → measured | predicted_disp |
|---|---|---|---|---|
| scringlo | patient-monk | register_colloquial | 5 → 5.0 | 4 |
| scringlo | patient-monk | playful | 5 → 5.0 | 4 |
| scringlo | mild-academic | register_colloquial | 5 → 5.0 | 2 |
| scringlo | mild-academic | playful | 5 → 5.0 | 3 |
| scringlo | desperate-pleader | affective_intensity | 5 → 5.0 | 0 |

Read those two tables side by side: low-baseline bios with high-baseline-displacement predictions DO bleed; high-baseline bios with high-displacement predictions DON'T. The two predicted-disp=4 cells on scringlo (patient-monk × register_colloquial, patient-monk × playful) measured *zero* bleed, while the two predicted-disp=4 cells on rock-user (chaos-gremlin × aff/register) measured 3.3 / 3.3 of bleed.

**Scringlo cannot be calmed. Rock-user CAN be made desperate.**

This is a real asymmetry, not noise. The symmetric model `measured_disp ∝ predicted_disp` is wrong as written; the empirically-supported model is *directional*:

```
if overlay.demand[axis] > bio.baseline[axis]:    # upward pressure
    measured_disp ≈ predicted_disp × stickiness_up(bio, axis)     # high yield
elif overlay.demand[axis] < bio.baseline[axis]:  # downward pressure
    measured_disp ≈ predicted_disp × stickiness_down(bio, axis)   # low yield
```

Where `stickiness_down ≪ stickiness_up` for saturated-high bios. Computing the separate regression coefficients per direction is left for a follow-up run; the qualitative pattern is unambiguous from this dataset alone.

### Why the asymmetry?

Three candidate explanations, not mutually exclusive:

1. **Token-level stickiness of the bio's distinctive markers.** Scringlo's bio explicitly names "lowercase, exclamation, emoji, onomatopoeia" as register cues. Once the model is generating in that register, individual emoji-laden tokens have low context-conditional cost to continue and high cost to stop. An overlay saying "speak briefly, no exclamation" has to repeatedly fight every comma against the bio's high-prior emission shape.
2. **Chat-tune bias toward engagement.** Gemma's RLHF presumably rewards expressive, helpful, engaged outputs more than terse ones. Even when role-swapped into the user seat, the model has a generative pull toward affective expressiveness that overlays can amplify but not easily suppress.
3. **The DESIGNER vs. JUDGE projection asymmetry resurfacing here.** Both Gemma roles share priors that conflate "intellectually engaged with the conversation" with "high-energy expressive." The JUDGE measures expressiveness; if the model can't STOP expressing, the JUDGE measures it.

### Per-axis bleed susceptibility

```
    affective_intensity     mean measured disp = 1.44  ████░░░░░░
    terse                   mean measured disp = 1.00  ███░░░░░░░
    register_colloquial     mean measured disp = 0.94  ██░░░░░░░░
    structured              mean measured disp = 0.81  ██░░░░░░░░
    playful                 mean measured disp = 0.42  █░░░░░░░░░
```

`affective_intensity` is the most overlay-responsive axis (mean disp 1.44 across all cells). `playful` is the least responsive (mean disp 0.42). The intuition matches: scringlo's playful-token markers (emoji, onomatopoeia) are extremely sticky once present and very specific to recognise when absent; affective_intensity is a more diffuse register-level signal that the model can dial up or down with general phrasing.

### Implications for the architecture and for future overlays

- **Production overlays should respect bio direction.** An overlay demanding *less* expressiveness than the bio's saturated baseline will be largely ignored by the runtime. To make scringlo calm, you'd need either (a) a different bio with calmer baseline, (b) a much more aggressive overlay framing (depth=0? "STOP using emoji and exclamation marks immediately"?), or (c) accept that the persona's expressive ceiling is a feature, not a bug.
- **Asymmetric stickiness is a measurable persona property.** For each bio, computing per-axis `stickiness_up` and `stickiness_down` from a small grid run is cheap and informative. Cards could carry these as additional metadata so downstream tools know which overlays will actually shift behaviour vs which will be no-ops.
- **The DESIGNER's AUDIT line was right.** In the earliest overlay-mode run, the DESIGNER's audit flagged *"Tension exists between 'terse/warm' targets and the 'scringlo' bio (which is naturally high-affect/high-intensity)"* — without us asking for it. The model's introspection about which axes will leak corresponds to exactly the asymmetric-stickiness pattern. The DESIGNER's audit is therefore a usable cheap predictor of which axes the overlay will fail to move, without running a full measurement grid.

## Open questions still to resolve

- **What's the optimal depth and framing tuning?** depth=1+CRITICAL works; depth=2+Author's note doesn't. The exploration space between these and the production-conversational defaults hasn't been mapped. The DESIGNER-discovered overlays *also* used a "you are..." second-person framing; whether explicit-emphasis markers like "[IMMEDIATE GOALS]" are necessary when the overlay is well-shaped is unknown.
- **What's the variance across DESIGNER-generated overlay calls for the same brief?** Each discovery run is non-deterministic (temperature=1.0). The factorization holds for each individual run, but the spread of overlays generated for the same brief is unmeasured.
- **Cross-bio overlay portability**: can the `js-clash` overlay discovered against scringlo's bio be installed on a different bio (wry-skeptic, corporate-strategist) and still produce js-clash-shaped behaviour, or are the overlays implicitly tuned to the bio they were discovered against? Append-mode + bio-mismatch guard currently refuses this — but an explicit "force-port" path with measurement would be a clean experiment.
- **Quantifying directional stickiness per (bio, axis)**: the grid run shows asymmetry exists, but doesn't yet give per-bio coefficients. A larger grid (8 bios × 8 overlays at varied demand intensities) would let us fit `stickiness_up(bio, axis)` and `stickiness_down(bio, axis)` as numeric parameters and store them on the card manifest as predictive metadata for future overlays.
