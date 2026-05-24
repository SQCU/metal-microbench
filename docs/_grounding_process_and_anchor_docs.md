# Grounding document: process and anchor docs
_Compiled 2026-05-24. Scope: six process/operational/anchor files modified 2026-05-17 through 2026-05-21._

---

## 1. project_archeology.md

**Filepath:** `/Users/mdot/metal-microbench/docs/project_archeology.md`
**Last modified:** 2026-05-21
**Document type:** Archeology survey — indexed timeline of design events and commits, assembled by parallel subagents scanning a 211 MB session transcript (1094 user messages) cross-referenced against 57 sillytavern-fork commits and 210 metal-microbench commits. 277 indexed events total.

**What it specifies:** The document surfaces every version of the multi-user-chat-agent-suggestion interface that has been designed, delivered, or regressed, mapped against the commit sequence. Its stated purpose is to prevent greenfield redesign of already-accepted-to-spec work. It defines the vocabulary of the interface (toolcard, forked agent, user-agent drawer, fixed-point iteration, axes registry, etc.) and records a canonical list of ten load-bearing design principles the operator has repeatedly asserted.

**Normative claims:**

> "Use this when you (the assistant) feel like you're greenfield-designing — you almost certainly aren't."

The ten principles explicitly enumerated in the archeology:

> 1. Finite K-affinity picker — drawer surfaces k_1 immediately-polling + k_2 suggested-disabled, never the full corpus, never empty.
> 2. Per-chat caching, no DoS patterns — /poll completes + caches, never aborts on navigate. A→B→A is O(1).
> 3. Ontological closure — bios without agents are unusable; usable bios must have agents (selection IS design).
> 4. Vectors persist; no runtime recomputation — /yapper-seed uses stored signatures, not cold extraction.
> 5. API client = GUI client contracts — every endpoint exercisable via GUI affordance for e2e validation.
> 6. Visible residue contract — every tool invocation leaves a card in DOM; silent deletion forbidden.
> 7. End-to-end validation only — Playwright pixel-space tests; unit tests not valid evidence.
> 8. No empty/JSON-field forms — interfaces always carry contextual suggestions + examples.
> 9. Chat is the central interface — drawers cover the chat ONLY when the operator explicitly chose that surface.
> 10. Subagent/LLM-driven sorting — discretion via API calls is the default; deterministic ordering is the fallback.

**Named events (critical 15 commits):** The doc identifies fifteen load-bearing commits spanning 2026-05-09 through 2026-05-19: toolcards genesis (`ec16007a7`), inline-collapsibles refactor (`6b5f74d9c`), user-personas phases 1–4 (`c94f950a6`, `abdd6c4af`, `d1ba7fd6c`), agentless-bios-as-broken-state (`317ebf5cd`), selection-is-design (`cbc5d45e6`), agents-as-PNG-cards (`4019f9aa1b6`), bridge stream-lifecycle safety triple (`87254e8`), axes-as-cards (`9ced4d081`), fixed-point iteration tab (`497f4416d`), context-suggester Phase A (`4dece95`), st-debug fixed-point smoke spec (`63d4291`), delete obsolete persona-library specs (`0f6214e`).

**Outstanding questions / gaps the doc flags:** The regression cycle period (2026-05-18 through 2026-05-21) closes with three regressions still open at the time of assembly: no k_1/k_2 streaming suggestions on startup, no axis split or bio-synthesis affordance visible in the client, and empty form pattern still present. The document instructs that future agents should append new period narratives + events when the operator reports regression.

---

## 2. user_descriptivism_brief.md

**Filepath:** `/Users/mdot/metal-microbench/docs/user_descriptivism_brief.md`
**Last modified:** 2026-05-17
**Document type:** Research orientation brief — defines the project's user modeling methodology and specifies the seed corpus production task for the current substep.

**What it specifies:** Establishes the `(bio × situation × overlay)` decomposition as the canonical modeling frame for user-side agents. Defines Bio as user-as-state (dispositional signature, 1–3 sentences, compressed organizing principle), Situation as the shared assistant/user frame, and Overlay as user-as-action (wants and conversational moves, injected at runtime as author's notes). Names two canonical strongly-separating bios (scringlo scrambler, Despotic Miscreant) and four weakly-separating redesign candidates. Specifies the goal for the current substep: produce `docs/bio_seed_v2.md` feeding the discovery harness.

**Normative claims:**

> "Bio = user-as-state. Who this person looks like to other people; what's dispositionally true about them. Texture, register, recognizable type. One to three sentences, up to about sixty words. The signature is itself a behavioral claim — not a Wikipedia infobox of facts, but a compressed organizing principle that explains how everything else about this person tends to go."

> "Overlay = user-as-action. Wants, conversational moves, goal-shape, what this user wants out of THIS interaction with THIS assistant right now. Lives at the runtime layer (author's-notes injected at depth-N), separate from bio."

> "The Claude-written entries currently in the canonical store stay or go on per-bio merit measured on-policy in subsequent runs."

**Outstanding questions / gaps the doc flags:** Two additional situations still need to be added beyond the existing dicemother and scringlo anchors, so the roster "covers enough territory that user-variety draws on chat-room dynamics rather than narrowing to private/secret-elicitation as its sole distinguishing axis."

---

## 3. ux_debt_followup_tickets_2026_05_21.md

**Filepath:** `/Users/mdot/metal-microbench/docs/ux_debt_followup_tickets_2026_05_21.md`
**Last modified:** 2026-05-21
**Document type:** Debt register / ticket list — enumerates UX regressions against load-bearing principles the operator has stated multiple times. Explicitly not a fix plan; each ticket is a confirmed regression with acceptance criteria.

**What it specifies:** Defines seven load-bearing UX principles (P-codes) and four open tickets against them. Establishes that the principles are not discoverable from CLAUDE.md, code comments, or test names — they are diegetic project rules that must be obeyed even when no test enforces them. Provides the motivating rationale (SillyTavern ships with an Assistant character to eliminate empty state; this project must match that).

**Normative claims (operator quote, load-bearing framing):**

> "without these traces/residues present in the user interface, the code looks fake and underdeveloped or like it might not work or never have worked in any useful capacity, parasitically stealing attention and investment of time from the user of the client interface. this is incorrect: We are never allowed to sap joy and motivation from users of our clients, or present contorted ambiguities at first startup."

**The seven P-code principles:**

> **P-EMPTY-FORM** — Never ask the operator to fill out a form-with-bare-fields without contextual suggestions, pre-filled defaults pulled from the existing corpus, or visible worked examples. JSON-fields-as-strings is the canonical forbidden anti-pattern.

> **P-NO-EMPTY-FIRST-PAINT** — Every interactive surface must have non-empty, operator-meaningful content visible on first paint. "No items yet, click X to begin" is forbidden. Pre-staged demos / defaults / already-streaming candidates fill the space.

> **P-NO-FAKE-LOOKING-CODE** — Every advertised mechanism (axis splitting, bio synthesis, agent lineage, factorization, signature distance, etc.) must be DEMONSTRATED by visible UI residue / traces on a fresh install. Operators must never have to take it on faith that the machinery exists; they must see it operating.

> **P-NO-CHAT-DISPLACEMENT** — New UI elements must not hide, cover, or replace the core chat interface. Drawers cover the chat ONLY when the operator explicitly chose to view that drawer.

> **P-NO-DOS-CASCADE** — Client-side request patterns must not constitute a DoS if the bridge were a remote API. Cache-hit on repeated context; no abort-on-navigation; no fire-and-retry loops that starve the model.

> **P-CANONICAL-NOT-MIRRORED** — Bios are personas. Personas are bios. There is ONE canonical store for any concept. Mirror writes between parallel stores are forbidden.

> **P-COMMITS-ARE-NOT-A-GUARD** — Do not condition feature work on commit state. Commits are not synchronization points or design criteria. Validation by passing end-to-end test is the only acceptance.

**Named tickets:**

- **UX-T1** — Replace "Add user agent: fill in JSON fields" form with a guided affordance. Violation: P-EMPTY-FORM. The agent-creation surface exposes raw card fields as bare inputs with no examples, defaults, or synthesis. Required: synthesis from context, candidate preview, K alternatives, tweak-dimension affordance, Save CTA gated on first candidate. Estimated effort: 1–2 days.
- **UX-T2** — Suggester must first-paint with K=2 high-affinity candidates already streaming. Violation: P-NO-EMPTY-FIRST-PAINT. Currently shows "No suggestions yet — click Suggest on a ranked row" on first paint. Required: top-K rows streaming in parallel immediately; no button-click required to see any candidate content. Estimated effort: 1–2 days.
- **UX-T3** — Interface must demonstrate feature-dimension splitting + bio-from-axes synthesis (residue/traces). Violation: P-NO-FAKE-LOOKING-CODE / P-DEMONSTRATE-MECHANISMS. No pre-staged split demo, no coordinate-picker synthesis widget, no lineage display per persona. Estimated effort: 3–5 days.
- **UX-T4** — First-paint defaults: a working chat must exist immediately. Status: implicit from ST-ships-with-Assistant principle. The st-debug `_data/` seed includes no default chat, so the suggester first-paint can be empty. Required: bootstrap creates a default welcome chat with 2–3 pre-written turns. Estimated effort: 0.5 day.

**Consumption protocol the doc specifies:**

> "Picking one up means: 1. Write the playwright spec that encodes the acceptance criteria. The spec should fail against the current code. 2. Implement until the spec passes. 3. The principle (P-...) the ticket maps to becomes a permanent invariant the spec enforces."

> "Do NOT skip step (1). Per the project's anti-vacuous-test-suite stance (2026-05-20), 'an e2e test through the GUI is equivalent to a curl test against the public API; if your test would pass against a broken implementation, it isn't a real e2e test.'"

**Outstanding questions / gaps the doc flags:** None flagged explicitly; the doc is self-contained as a register.

---

## 4. gemma_critic_round1.md

**Filepath:** `/Users/mdot/metal-microbench/docs/gemma_critic_round1.md`
**Last modified:** 2026-05-17
**Document type:** Critic round / external review — Gemma-4's first-pass critique of the user-descriptivism brief and bio seed corpus, structured as COHERENCE / GAPS / RISKS / CONCRETE_SUGGESTIONS.

**What it specifies:** An externally-generated evaluation of the `(bio × situation × overlay)` decomposition and the bio seed corpus. Identifies structural gaps (silent-user axis absent, bio treated as static rather than temporal, no prompt-engineer/optimizer archetype, no INTERFACE_LITERACY dimension). Flags risks around stereotype collapse, overlay leakage into bio, and evaluation bias toward legibility over authenticity. Proposes diagnostic situations (The Ambiguous Prompt, The Wall) for testing bio distinctiveness.

**Normative claims (Gemma's assessments):**

> "The distinction between Bio and Overlay is the project's strongest architectural feature; it prevents the 'one-dimensional agent' trap where a user's personality is conflated with their immediate goal."

> "By relying heavily on 'shared cultural shorthand,' you risk the discovery harness merely rediscovering internet tropes (Reddit/Twitter archetypes) rather than actual user behavior."

> "If the 'scoring along behavioral feature axes' relies on LLM-as-a-judge, the judge will likely reward 'legibility' (how well the agent performs the trope) rather than 'authenticity' (how well the agent adheres to the nuances of the bio)."

**Outstanding questions / gaps the doc flags:** Missing INTERFACE_LITERACY dimension; silent/low-entropy user tier absent; no temporal drift modeling; Prompt Engineer/Optimizer archetype absent.

---

## 5. NOTES_FIXED_POINT_TAB.md

**Filepath:** `/Users/mdot/sillytavern-fork/NOTES_FIXED_POINT_TAB.md`
**Last modified:** 2026-05-19
**Document type:** Implementation handover note — design rationale and endpoint contract for the fixed-point iteration tab feature delivered in commit `497f4416d`.

**What it specifies:** Documents the four endpoints added to `plugins/user-personas/index.mjs` (POST run, GET runs, GET run status, POST validate), the child-process architecture for dispatching `lock_in_iterative.mjs`, the four validation invariants computed over agent card signatures, and the drawer installation mechanism. Explicitly documents what the tab does NOT do (no live cancel, no persistent run history beyond disk logfiles, no experiment editor).

**Normative claims:**

> "Process isolation. The harness loop is long-running (multiple minutes per bio). If it threw inside the plugin event loop the ST server would stall."

> "The hard constraint in the task wording was explicit on this." (re: not refactoring the harness to expose a JS entry-point; the child-process path reuses the existing HTTP contract)

The validation arithmetic is normative: four named invariants with specific thresholds — PASS_DIFFERENT_AGENT_AXIS = 1.0 (min pairwise L2 on agent axes ≥ 1.0), PASS_SIMILAR_AGENT_AXIS = 1.5 (max pairwise L2 on agent axes ≤ 1.5), PASS_DIFFERENT_BIO_AXIS = 1.0 (min pairwise L2 on bio axes ≥ 1.0), plus an L2 ranking invariant that near pairs must be strictly closer than cross pairs.

**Outstanding questions / gaps the doc flags:** Thresholds described as "conservative starting values; adjust if false-positive/false-negative rates miscalibrate against real runs." Cancel-mid-iteration is explicitly deferred with an escape hatch (add DELETE endpoint sending SIGTERM if needed).

---

## 6. AGENTS.md

**Filepath:** `/Users/mdot/sillytavern-fork/AGENTS.md`
**Last modified:** 2026-05-21
**Document type:** Agent operating-instruction file — mandatory reading list + canonical store contracts for any future agent working in sillytavern-fork.

**What it specifies:** Names the three documents a future agent must read before proposing any UI change. Specifies canonical storage locations for all data types. Lists the two plugins and their principal contracts.

**Normative claims:**

> "Read this BEFORE proposing any change to the chat interface." (referring to `multi_user_agent_chat_interface_spec.md`)

> "Bios are personas. Personas are bios. Single store at `<dataRoot>/<user>/User Avatars/` + `settings.json → power_user.persona_descriptions`. No mirror writes. No parallel plugin store."

> "Agents persist as chara_card_v3 PNGs in `plugins/user-personas/agents/`. Axes as JSON cards in `plugins/user-personas/axes/`. Experiments as JSON cards in `plugins/user-personas/experiments/`."

Toolcards contract: "Visible-residue contract: every tool invocation leaves a card."

**Outstanding questions / gaps the doc flags:** None flagged. The doc is a pointer file; it defers normative depth to the three linked documents.

---

## Cross-doc synthesis

### Operational invariants

Extracted from all six documents, these are the project-wide rules repeated across multiple sources:

> **No empty state.** Every surface must have non-empty content on first paint. Forbidden: "No items yet, click X to begin." Required: pre-staged demos, already-streaming candidates, or default working artifacts.

> **Tests are not specs.** The principles are not discoverable from tests or code comments. Tests are downstream of the principles, not the source of them. Validation by passing end-to-end Playwright test is the only acceptance criterion — but the test must be written to fail against a broken implementation.

> **No fallbacks.** Full-list fallback in the drawer was explicitly deleted per design. The finite K-affinity picker (k_1 + k_2) is the only valid ontology. "never the full corpus, never empty."

> **Visible residue.** Every tool invocation leaves a card in the DOM. Silent deletion of tool output is forbidden.

> **One canonical store.** Any concept has exactly one store. Mirror writes between parallel stores are forbidden.

> **Ontological closure.** A bio without an agent is broken state. Selection IS design: bio selection auto-chains to agent design.

> **No post-hoc feature extraction.** Feature vectors must be persisted at creation time. Runtime recomputation is forbidden. "things that should be persisted not being persisted...which is the highest priority."

> **No DoS-equivalent patterns.** Abort-on-navigation and fire-and-retry loops are forbidden as if the bridge were a remote API.

> **Commit state is not a guard.** "Do not condition feature work on commit state. Commits are not synchronization points or design criteria."

### Conventions for future agents (from AGENTS.md)

Before proposing any change to the chat interface, an agent must read in order:
1. `../metal-microbench/docs/multi_user_agent_chat_interface_spec.md` — component-by-component contract + 11 principles + paired-agent acceptance protocol.
2. `../metal-microbench/docs/project_archeology.md` — indexed timeline of every interface version.
3. `../metal-microbench/docs/ux_debt_followup_tickets_2026_05_21.md` — ticket backlog + forbidden anti-patterns.

The archeology doc adds: consult it before designing; the project contains more accepted-to-spec features and design principles than regression cycles have preserved in working code.

### Open tickets / debt consolidated

| ID | Principle violated | One-line summary | Effort |
|----|-------------------|-----------------|--------|
| UX-T1 | P-EMPTY-FORM | Replace bare-JSON-field agent-creation form with synthesis-from-context affordance | 1–2 days |
| UX-T2 | P-NO-EMPTY-FIRST-PAINT | Suggester must first-paint with K=2 candidates already streaming; no button-click required | 1–2 days |
| UX-T3 | P-NO-FAKE-LOOKING-CODE | Corpus tab must demonstrate axis-split lineage + bio-from-coordinates widget on fresh install | 3–5 days |
| UX-T4 | (implicit) | Bootstrap seed must create a default welcome chat so suggester first-paint has content to score | 0.5 day |

Additionally, per the archeology's regression period log, three regressions were flagged as open at time of assembly (2026-05-21) but not yet assigned UX-T IDs: (a) resynth/suggest/poll affordances non-functional on click, (b) 22 axes in corpus vs spec-mandated minimal 4-axis set, (c) synthesized bios not visible in ST native persona UI.
