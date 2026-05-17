# Multi-agent multi-suggestion architecture — consolidation note

**Companion to:**
- `docs/overlay_architecture.md` — root-bio + elicitation-overlay factorization
- `docs/user_agent_likert_methodology.md` — the cascade judge and the 14-axis basis
- `docs/strategy_diversity_scoring.md` — LLM-as-summarizer over multi-turn transcripts
- `docs/discovery_harness_findings.md` — the discovery DESIGNER's first-run record

This note steps back from the individual experiments and inventories the **measurement instruments, harnesses, generation tools, and storage layers** built across the elicitation-tooling project, identifies the right generalizations, and proposes how they should compose into a **multi-agent-multi-suggestion UX** inside the SillyTavern client surface.

The UX target the user articulated: *a single-user-single-assistant chat, with a sidebar of side-commenter user-agent personas that can suggest extrapolations / alternative framings. The sidebar should not have 12 variants of scringlo cluttering its selector list — one scringlo card with a library of overlays it can use to suggest in different modes.*

That target is achievable from the components already in place. This note maps the inventory onto what's still missing.

---

## What we've built

Organised by role rather than file. Some files appear under multiple roles.

### Oracles (measurement instruments)

- **The cascade JUDGE** (`probe_persist.py::stage1_summary` + `stage2_likert` + `parse_elementwise`). Two-stage measurement of a single chat turn: Stage 1 emits a 2-3 sentence behavioural prose readout; Stage 2 emits per-line `axis: integer` over the 14 named axes. Element-wise parser tolerates partial output. **Used by every harness in the project.** Treat it as a stable primitive.
- **The signature analyzer** (`signature.py`). Four sample-size-gated layers: per-axis univariate stats (n≥2), bivariate Pearson correlation (n≥6), pooled within-class covariance + Mahalanobis distances (n≥30 total + ≥5/persona), PCA + effective-dimensionality (n≥2·N_AXES). Skips layers it can't honestly produce.
- **The strategy-diversity scorer** (`strategy_diversity.py`). LLM-as-summarizer over multi-turn transcripts: per-turn 3-6 word tactic labels + 1-5 overall diversity score + one-sentence arc summary. Reads any multi-turn JSONL (header-style or factorization-style).

### Harnesses (orchestration over the oracles)

- **`probe_persist.py`**: cascade probe + JSONL judgment store. Builds the elicitation corpus that the analyzer consumes.
- **`overlay_demo.py`**: single-bio × N-overlay × target multi-turn driver. Supports `--user-agent-card` to load bio+library from an overlay-v1 manifest. Author's-note-style overlay injection at depth-N.
- **`factorization_multiturn.py`**: multi-bio × shared-motivation × target driver. Per-bio signature aggregator + pairwise Mahalanobis between persona means.
- **`headroom_bleed_grid.py`**: 4×4 (bio × overlay) grid harness. Predicted vs measured displacement regression. Asymmetric-stickiness surfacer.
- **`discovery.py`** (run loop): the workshop loop. Factorized or overlay-mode. Iterative refinement with drift feedback + judge Stage-1 prose readout in DESIGNER context. PCA-driven target OR explicit per-axis target. Optional target_assistant pairing. Optional operator constraint passthrough.

### Tools (generation)

- **`discovery.py` factorized-mode**: DESIGNER produces a full factorized persona (BIO / MOTIVATION / SCENARIO / RELATIONSHIP_TO_COUNTERPARTY / COMM_STYLE / TURN / AUDIT) given a target signature. Output → factorized-card manifest.
- **`discovery.py` overlay-mode**: given a FIXED root bio, the DESIGNER emits only an elicitation overlay (50-150 words) + a representative turn + an audit. Output → overlay-v1 card manifest. Append-mode adds new overlays to an existing overlay-v1 card's library.

### Solvers (analysis math)

- Per-axis between-class spread (`overlay_demo.py`)
- Pooled within-class covariance + Mahalanobis distances (`signature.py`)
- PCA + variance-explained curve + effective dimensionality at thresholds (`signature.py`)
- Predicted-vs-measured displacement regression (`headroom_bleed_grid.py`)
- Asymmetric-stickiness pattern recognition (in `docs/overlay_architecture.md` — manual analysis, but the regression code is in place)

### Storage layers

- **`data/elicitation_judgments.jsonl`**: append-only judgment store, schema-versioned. Captures `judge_model` and `judge_gguf_basename` per record so cross-condition filtering is possible.
- **`data/headroom_bleed_grid.jsonl`**: 16-cell grid transcripts.
- **`data/strategy_diversity_scores*.jsonl`**: per-session diversity scores.
- **Plugin players/** (`/Users/mdot/sillytavern-fork/plugins/user-personas/players/`): hand-authored cards (`wry-skeptic`, etc.), factorized discovery output cards (`discovery-tidepool-botanist`, `factorization-corporate-strategist`, etc.), and overlay-v1 cards with libraries (`overlay-scringlo-jsclash`).

### Diegetic surfaces (server-side)

- `POST /api/plugins/user-personas/discovery` — SSE event stream of a workshop-loop run
- `POST /api/plugins/user-personas/discovery/invoke` — synchronous variant (tool-call dispatch target)
- `GET /api/plugins/user-personas/discovery/tool-spec` — OpenAI function-call schema for in-chat invocation
- `GET /api/plugins/user-personas/discovery/runs` — lists materialised discovery cards
- `GET /api/plugins/user-personas/static/discovery.html` — control plane page (target + operator constraint + bio + overlay-name + run, watch live SSE)
- `GET /api/plugins/user-personas/personas` — existing card-list endpoint (does NOT yet surface overlay libraries)
- `POST /api/plugins/user-personas/poll` — existing single-turn user-agent generator (does NOT yet inject overlays; treats every card as legacy `system_prompt`-only)

### Cards already in the persona zoo

Plugin players: `gushing-fan`, `polite-naturalist`, `pushy-completionist`, `wry-skeptic` (hand-authored); `discovery-tidepool-botanist`, `discovery-js-aiieeee` (factorized discovery); `factorization-corporate-strategist`, `factorization-scringlo-fragment`, `factorization-ten-year-old` (3-bio experiment); `overlay-scringlo-jsclash` (overlay-v1 with library: js-clash + validation-seeker).

ST characters: `python-only-coder` (assistant target), `scringlo_scrambler`, `the-rock`, `dicemother` (assistant counterparties available for adversarial pairing).

---

## The multi-agent multi-suggestion UX

The product target:

> A single-user-single-assistant chat is happening. In a sidebar, the user has access to ~6 user-agent personas (scringlo, wry-skeptic, corporate-strategist, polite-naturalist, ten-year-old, pushy-completionist, the-rock-as-user, ...). For any persona, the user can click "suggest" and a side commenter chimes in with what THAT persona would say next given the current conversation. Each persona-card carries a small library of overlays (motivational modes); the user (or the system) picks which overlay to apply.

The architectural payoff of this UX:

- **One card per persona in the selector.** Scringlo appears once. The library of scringlo-modes (js-clash, validation-seeker, neutral, chaos-gremlin, ...) lives *inside* the card and is exposed via a secondary control, not as 12 separate selector entries.
- **The user picks INTENT** (which persona × which mode), the runtime injects the overlay at depth-N, the bio's KV prefix survives swaps, and the side commenter speaks in the persona's voice with the requested intent.
- **The overlay library grows over time** via the existing discovery+append pipeline. Operators can run a workshop loop to discover a new overlay for an existing persona and it lands as a new entry in that card's library — no card duplication.

---

## What composes naturally vs what needs work

### Already composes

- **Card → overlay library → runtime overlay injection**: discovery overlay-mode produces v1 cards with libraries. `overlay_demo.py --user-agent-card` already reads libraries from cards. So the WRITE side of the pipeline is complete.
- **Cascade judge → signature → factorization analysis**: a side commenter's suggestion can be cascade-judged on the fly to display its Likert signature alongside the suggestion ("scringlo×validation-seeker says this would land at curious=5, terse=1, warm=5, ..."). That's a single 2-call cascade per suggestion.
- **Strategy-diversity scorer over an accumulating suggestion-history**: if the user invokes multiple suggesters across a chat session, the scorer can summarise *the diversity of suggestions* the user has explored. Free behavioural-meta-readout for the chat.

### Needs work for the UX

1. **`/poll` needs overlay awareness.** Currently `/poll` loads a card and uses `card.system_prompt` as the user-agent's system. For overlay-v1 cards it should:
   - Use `card.bio` (or the existing `card.system_prompt` which wraps the bio) as the system message at index 0
   - Inject the selected overlay at depth-N from the end of the chat history
   - Accept an `overlay_name` parameter (defaulting to `card.default_overlay` if absent)
   - Fall back to legacy behaviour for non-overlay-v1 cards (no library = use `system_prompt` directly)

   This is a small extension of `/poll`. The overlay-injection logic already lives in `overlay_demo.py::inject_overlay`; it just needs to be lifted into the plugin.

2. **`/personas` needs to surface the library.** Currently lists card id/name/bio/motivation. Should also return `card_schema`, `elicitation_overlay_library` (or just the *names* + the `default_overlay` if we want to keep response size small), so a UI can render the per-card overlay dropdown.

3. **UI surface — sidebar suggester.** New affordance (HTML page or SillyTavern extension panel) that:
   - Fetches the persona list with library names
   - Renders one row per persona: name + bio-summary + overlay-name dropdown + "suggest" button
   - On click: calls `/poll/with-overlay` (or extended `/poll`) with the current chat + selected (card, overlay) → renders the returned suggestion in the sidebar
   - Optionally: per-suggestion, kicks off a cascade-judge call and shows the measured signature ("how the JUDGE read this suggestion")

4. **Sticky-coefficient metadata on cards.** The headroom-bleed regression found that overlays can fail when they push a saturated bio downward. Cards could carry an `axis_stickiness` block in their manifest so the UI can warn or filter: *"this overlay aims for affective_intensity=1 but the scringlo bio is sticky at 5 on that axis; the overlay will likely be ignored."* Cheap to compute from any (bio × overlay-library) grid run; just needs the storage convention + a UI consumer.

### Generalizations worth doing while we're consolidating

- **Unify the three multi-turn drivers** (`overlay_demo.py`, `factorization_multiturn.py`, `headroom_bleed_grid.py`). They share the same canonical-message-list-with-role-swap pattern. A `runtime/session_driver.py` library that all three import from would eliminate ~200 lines of duplication and make future drivers cheap.
- **Multi-corpus signature comparison.** `signature.py` operates on one JSONL at a time. Compare-two-stores would let us answer questions like *"does scringlo's signature change between python-only-coder pairings and the-rock pairings?"* without manual cross-tabulation.
- **In-chat discovery tool call.** The OAI tool spec is already published at `/discovery/tool-spec`. SillyTavern can register that tool so the *assistant* can ask "what kind of user is asking me this?" mid-conversation and get a discovered user-agent card back. This closes the loop: a chat assistant can request that the harness build a stress-test persona on the fly.

---

## Concrete next moves, ranked by payoff

1. **Extend `/poll` with overlay-mode handling** (~30 min). This is the keystone — without it, the discovery+library investment can't be used in real chat. Add `overlay_name` parameter, inject at depth-N, fall back to legacy behaviour.

2. **Extend `/personas` to expose overlay-library names** (~5 min). One field added to the existing response.

3. **Sidebar UI panel** (~1-2 hours). Either a new static HTML in the plugin or an integration into SillyTavern's existing extensions surface. Renders the persona list, exposes overlay dropdowns, fires `/poll`-with-overlay on click, displays suggestion + measured signature.

4. **Migrate factorized-discovery cards to overlay-v1 if appropriate** (~1 hour). Several existing cards (`factorization-corporate-strategist`, etc.) are factorized format with embedded motivation. They could be re-shaped as `bio` (the BIO field only) + a library with one entry for their original motivation. Optional — legacy format still works under (1)'s fallback.

5. **Multi-turn driver consolidation** (~2 hours, no urgency). Extract the shared driver pattern, leave individual scripts as thin callers. Worth doing before adding any new comparable harness.

6. **Card-level stickiness metadata** (~1 hour). Re-run the headroom-bleed grid per-bio at a larger scale to derive per-(bio, axis) `stickiness_up` / `stickiness_down` coefficients; write them onto each card's `axis_stickiness` block. The UI can then show overlay-effectiveness hints.

---

## The compositional grammar

The whole system, expressed as a grammar:

```
persona = root_bio × elicitation_overlay
suggestion_call(persona, chat_history, target_assistant) =
    runtime.inject_overlay_at_depth_N(
        system_prompt = render(root_bio),
        history = role_swap(chat_history),
        overlay = elicitation_overlay,
    )
    → bridge.complete(...) → suggestion_turn
measurement(suggestion_turn) = cascade_judge(suggestion_turn)
                              = (stage1_prose, 14_axis_signature)
session(persona, target_assistant, K) = [suggestion_call(persona, history_so_far, ...)
                                          for k in K_turns]
diversity_score(session) = llm_summarize(session.turns)
                          = (per_turn_strategy_labels, 1-5_diversity, arc)
discovery(target_signature, root_bio?, target_assistant?, op_constraint?) =
    workshop_loop:
        propose overlay (or full persona) → measure → drift feedback → revise
    → final spec + final signature + final drift
discovery_promote_to_card(discovery_result, card_id, overlay_name) =
    if card exists:  append_overlay(card_id, overlay_name, ...)
    else:            new_card(card_id, ...)
```

Every primitive in this grammar exists in code. The remaining UX work is wiring them up to a sidebar.

---

## Why this consolidation matters

Without it, the project has six oracles, four harnesses, two tools, several solvers, and a growing zoo of cards — all of which currently require running CLI scripts to use. The diegetic surface only exposes the workshop-loop control plane; the actual product application (multi-agent side-commenter UX) is not yet exposed even though every server-side piece needed to support it is in place.

Landing the keystone (overlay-aware `/poll` + sidebar UI) converts the project from a research-instrument zoo into an interactive elicitation-design surface, which is the form a downstream researcher or product engineer can actually use to author new user-agent personas, watch them react to a real chat, and pick the modes that suggest most usefully.
