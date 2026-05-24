# UX debt: follow-up tickets accumulated through 2026-05-21

This is a debt register, not a fix plan. Each ticket below is a piece
of work that exists because the team shipped code that the user has
explicitly identified as violating a load-bearing UX principle (see
"Principles" at the bottom). Picking up any of these tickets means
fixing a regression against a principle the operator has restated
multiple times in the project transcript.

The principles are NOT discoverable from CLAUDE.md, code comments, or
test names. They are diegetic project rules that emerged through the
operator's UI/UX critique cycles. They must be obeyed even when no
test enforces them, because the tests are downstream of the
principles, not the source of them.

---

## Why this document exists

Stock SillyTavern ships with an Assistant character prompt baked in by
default. This is not an accident. Why?

Because a brand-new install with no chat history, no characters, no
personas, and no working examples would look — to a first-time
operator — like a broken application. They open the app, they see
empty drawers, blank textareas, "click here to begin" buttons with no
clue about what comes next, and they conclude that either the app is
broken or that they're missing some onboarding step they should
already know.

The Assistant character solves this with a single artifact: an
operator can immediately open a chat, send a message, and watch the
full pipeline run end-to-end with no setup. Every drawer, every
button, every panel is anchored to a working example. The empty-state
problem doesn't exist because there is no empty state.

This is the principle behind every ticket below. The operator phrased
it directly in the 2026-05-21 message that surfaced these:

> "without these traces/residues present in the user interface, the
>  code looks fake and underdeveloped or like it might not work or
>  never have worked in any useful capacity, parasitically stealing
>  attention and investment of time from the user of the client
>  interface. this is incorrect: We are never allowed to sap joy and
>  motivation from users of our clients, or present contorted
>  ambiguities at first startup."

---

## Tickets

### UX-T1 — replace "Add user agent: fill in JSON fields" form with a guided affordance

**Status**: accumulated debt
**Surfaced**: 2026-05-21
**Principle violated**: P-EMPTY-FORM (see below)

**Symptom (operator quote)**:
> "there is a diegetic 'add user agent: please fill out a json's
>  fields with a form?? by hand?? please??' user interface element.
>  this is incorrect: We are never allowed to 'ask a user for all of
>  the fields in a harness as strings without any contextual
>  suggestions'."

**What the broken UI does today**:
There is an agent-creation surface (likely in `designer.html` or the
agent-editor tab) that exposes the raw agent-card fields as bare
inputs: `id`, `name`, `injection_mode`, `system_prompt`,
`elicitation_overlay`, `target_axis_signature`, `derived_from`, etc.
The operator is expected to know what each field means and fill them
in by hand. There are no inline examples, no defaults pulled from the
existing corpus, no suggestion of "agent like these existing ones but
with X changed," no preview of what the resulting card will do.

**What the affordance must instead be**:
The interface starts from CONTEXT — the operator's current chat or
selected bio or selected target signature — and synthesizes a
candidate agent that they can then refine. The blank textarea is
replaced with:
  - A preview of the candidate agent's prose
  - The signature target it will be designed against (pre-filled from
    the current chat's behaviour signature, or from a selected
    reference)
  - "Variants" — a small set of K alternative candidates the operator
    can choose between (similar to the suggester's per-row Suggest)
  - An explicit "tweak this dimension" affordance (raise theft
    aggressiveness, lower star_sign, etc.) that mutates the candidate
    without ever asking the operator to type a JSON field
  - A "Save as agent" CTA that becomes available only after at least
    one candidate has been generated; until then the surface is in
    "explore" mode, not "fill out form" mode

**Acceptance**:
  - The agent-creation surface has zero bare `<input>` or `<textarea>`
    elements without a pre-filled value, an inline causal description,
    AND a worked example visible above/beside it.
  - First-paint of the surface ALREADY shows at least one candidate
    agent (drawn from the current chat context or a default). The
    operator is never asked to "type a name and see what happens."
  - An end-to-end playwright spec (write one) opens the surface in a
    fresh test instance with the canonical 3 personas, asserts the
    surface auto-populates with ≥1 candidate within bounded time, and
    fails if any bare-input pattern is present.

**Estimated effort**: 1-2 days. The synthesis pipeline already exists
(POST `/synthesize-agents-for-persona/<key>`). The work is FE-only:
restructure designer.html / the agent-editor tab to consume the same
endpoint and surface candidates instead of fields.

---

### UX-T2 — suggester must FIRST-PAINT with K=2 high-affinity candidates already streaming

**Status**: accumulated debt
**Surfaced**: 2026-05-21
**Principle violated**: P-NO-EMPTY-FIRST-PAINT

**Symptom (operator quote)**:
> "there are no k_1 and k_2 high affinity user agents for talking to
>  'the rock' with suggested user agent prose already streaming in
>  through the interface."

**What "talking to the rock" means**:
A reference to the existence of ANY in-progress chat the operator is
viewing — be it the Assistant default, a custom character chat, an
empty "Aria the bard" exchange, or whatever. The suggester's value
proposition is: given the chat-context-you-are-currently-viewing,
here are K user-personas + agent overlays that would produce
high-affinity continuations. The operator should NEVER have to do
anything to see this. It must already be on screen the first time
they open the suggester surface.

**What the broken UI does today**:
The suggester correctly opens, fetches `/yapper-seed`, and renders
ranked rows after the gemma extraction completes (~10-90s on first
call). But:
  - During the wait, the surface shows "POST /yapper-seed…" with no
    indication of what the operator will see when it lands.
  - The "candidates feed" panel shows "No suggestions yet — click
    Suggest on a ranked row" — a literal "click me to begin" empty
    state. Per the operator's principle: forbidden.
  - The per-row Suggest buttons work but the OPERATOR has to click
    them. The high-affinity (top-K) rows should be ALREADY POLLING in
    parallel so the operator's first view shows prose, not empty
    rows-with-buttons.

**What the affordance must instead be**:
On first paint of the suggester surface:
  1. The /yapper-seed call fires immediately (it does).
  2. The top-K (K=2 or K=3) rows are streaming their /poll suggestions
     in parallel, IMMEDIATELY visible as in-progress text under each
     row. No button-click required.
  3. The "candidates feed" panel either disappears entirely (the inline
     row-completion slots subsume it) OR shows a non-empty default
     state explaining what it accumulates.
  4. If yapper-seed itself is still in flight, the surface shows
     "Reading the chat ('the innkeeper drops the cup...' [+3 turns]).
      Suggesting K=2 user personas + agent overlays..." — a literal,
     readable explanation of what's happening, citing concrete content
     from the chat so the operator knows the data path works.

**Acceptance**:
  - Playwright spec: open the suggester with a non-empty chat present.
    Within 5 seconds, the surface contains either a streamed-prose
    candidate under at least one ranked row OR a concrete in-progress
    status that names a real chat turn the suggester is consuming.
  - The "click Suggest on a row" affordance still exists for tier-2+
    rows but is NOT the only path to seeing any suggestion content.
  - No "No suggestions yet" empty-state text appears on first paint of
    a populated chat.

**Estimated effort**: 1-2 days. Per-row caching + auto-fire-top-K logic
on initial render. The /poll endpoint already supports this.

---

### UX-T3 — interface must demonstrate feature-dimension splitting + bio-from-axes synthesis (residue/traces)

**Status**: accumulated debt
**Surfaced**: 2026-05-21
**Principle violated**: P-NO-FAKE-LOOKING-CODE / P-DEMONSTRATE-MECHANISMS

**Symptom (operator quote)**:
> "the interface doesn't *demonstrate* feature dimension splitting
>  and/or mechanisms to choose a new synthetic bio (ergo new synthetic
>  agents) motivated by an existing collection of feature axes of
>  variation, for example. without these traces/residues present in
>  the user interface, the code looks fake and underdeveloped or like
>  it might not work or never have worked in any useful capacity,
>  parasitically stealing attention and investment of time from the
>  user of the client interface."

**What the broken UI does today**:
  - The Corpus tab (corpus.html) shows the axes registry — three axes
    (rpg_class, star_sign, money_orientation) — with no historical
    context. A first-time operator sees three rows and has no idea
    why these axes, what they're for, what splitting would look like,
    or how they'd produce a new bio along one of them.
  - The plugin DOES support derived axes (`derived_from.parent`) and
    DOES have a working axis_splitter.mjs harness, and T6 confirms
    the API enforces correct genealogy. None of this is visible.
  - No surface shows "here's an axis that was split, and here are its
    children, with a worked example of how that split was motivated
    by what was happening in the corpus at the time."
  - No surface shows "here's a synthetic bio that was synthesized by
    picking a coordinate on these axes, and here's the chat trajectory
    that produced it."

**What the affordance must instead be**:
The Corpus tab (or a new sibling tab) shows DEMONSTRATIONS of every
mechanism the codebase implements:
  - **A pre-staged split demo**: at least one root axis with one
    derived child visible, with a "see why this split happened" link
    that opens an inline trace: the trajectory bucket where the parent
    axis showed a gap, the proposed children, the cohen's-d that
    justified the split, the operator's accept/reject record.
  - **A pre-staged "synthesize a bio from axis coordinates" demo**:
    an interactive widget where the operator picks (rpg_class=2,
    star_sign=4, money_orientation=3) and sees:
      - a synthesized candidate bio (already pre-computed for the demo,
        so it's instant)
      - a "regenerate" button that fires the live synthesis if they
        want a fresh sample
      - the resulting bio rendered as a normal user-persona row so
        they can pick it up and use it
  - **A "lineage" view per persona**: every bio in the corpus has a
    visible "derived_from" trail (or "root persona" badge if it's
    operator-authored). The operator should be able to point at any
    bio and answer "where did this come from?"

**Acceptance**:
  - Fresh-install state has a pre-staged derived axis (e.g.,
    `rpg_class` split into `rpg_class_combat_orientation` and
    `rpg_class_intellectual_intensity`, or whatever the canonical
    demo split is) visible in the Corpus tab on first open.
  - A "Synthesize bio from coordinates" widget is visible somewhere
    (Corpus tab or a new "Synthesis demo" tab) on first open and
    has a pre-staged demo synthesis ready to show without firing the
    live pipeline.
  - Every persona row in ST's native persona drawer (or the suggester's
    ranked rows) shows lineage: "Despotic Miscreant • root persona"
    vs "polite-courtier-with-restraint • derived from Despotic
    Miscreant via star_sign=2, money_orientation=4."
  - Playwright spec: open Corpus tab, assert that at least one axis
    row shows `derived_from` lineage; assert the synthesize-from-
    coordinates widget exists and is operable; assert at least one
    persona in the registry has a non-trivial lineage display.

**Estimated effort**: 3-5 days. Includes:
  - Pre-staging a demo split (1 day — run axis_splitter once, save
    the result as a fixture, ship it as part of the default install)
  - Pre-staging a demo synthesized bio with full lineage (0.5 day)
  - New UI: coordinate-picker widget for "synthesize bio from axes"
    (1-2 days)
  - Lineage badge component + wiring on persona drawer + suggester
    rows (1 day)

---

### UX-T4 — first-paint defaults: a working chat must exist immediately

**Status**: accumulated debt (implicit from the SillyTavern-ships-with-Assistant principle)
**Surfaced**: 2026-05-21 (via reasoning about why ST ships an Assistant)

**Principle**: stock SillyTavern ships with an Assistant character +
working chat by default precisely so a new install has zero empty
state. Our st-debug + plugin install should match this. Currently
st-debug's `_data/` seed includes a default-user persona but no
default chat content, so the suggester first-paint can be empty.

**Acceptance**:
  - The seed step in `scripts/bootstrap.sh` creates a default
    "Welcome — meet your user-personas suggester" chat with 2-3
    pre-written turns (e.g., "Hi, what should we build today?" /
    "Open the suggester panel — I've already drafted two candidate
    user-personas based on this conversation").
  - On first open, the suggester surface IMMEDIATELY has chat content
    to score against and renders candidates within 5s.
  - The "synthesize a bio from coordinates" widget (UX-T3) is reachable
    from the welcome chat (e.g., one of the pre-written turns is a
    link affordance).

**Estimated effort**: 0.5 day. Pure seed-data work.

---

## Principles (load-bearing, restated multiple times in transcript)

These are the project's actual UX rules. They override default
practice. They are evidenced by repeated operator statements; see the
haiku scan output above for verbatim quotes + turn references.

**P-EMPTY-FORM** — Never ask the operator to fill out a form-with-bare-fields
without contextual suggestions, pre-filled defaults pulled from the
existing corpus, or visible worked examples. JSON-fields-as-strings
is the canonical forbidden anti-pattern.

**P-NO-EMPTY-FIRST-PAINT** — Every interactive surface must have non-empty,
operator-meaningful content visible on first paint. "No items yet,
click X to begin" is forbidden. Pre-staged demos / defaults /
already-streaming candidates fill the space.

**P-NO-FAKE-LOOKING-CODE** — Every advertised mechanism (axis splitting,
bio synthesis, agent lineage, factorization, signature distance,
etc.) must be DEMONSTRATED by visible UI residue / traces on a fresh
install. Operators must never have to take it on faith that the
machinery exists; they must see it operating.

**P-NO-CHAT-DISPLACEMENT** — New UI elements must not hide, cover, or
replace the core chat interface. Drawers cover the chat ONLY when the
operator explicitly chose to view that drawer. Hamburger popovers,
menus, tooltips: all small, transient, chat-stays-visible.

**P-NO-DOS-CASCADE** — Client-side request patterns must not constitute
a DoS if the bridge were a remote API. Cache-hit on repeated context;
no abort-on-navigation; no fire-and-retry loops that starve the model.

**P-CANONICAL-NOT-MIRRORED** — Bios are personas. Personas are bios.
There is ONE canonical store for any concept. Mirror writes between
parallel stores are forbidden; if two surfaces show the same concept,
they read from the same underlying file.

**P-COMMITS-ARE-NOT-A-GUARD** — Do not condition feature work on commit
state. Commits are not synchronization points or design criteria.
Validation by passing end-to-end test is the only acceptance.

---

## How to consume this document

These tickets are claimable. Each one has acceptance criteria that
double as test specs. Picking one up means:
  1. Write the playwright spec that encodes the acceptance criteria.
     The spec should fail against the current code.
  2. Implement until the spec passes.
  3. The principle (P-...) the ticket maps to becomes a permanent
     invariant the spec enforces; future regressions will fail the
     same spec.

Do NOT skip step (1). Per the project's anti-vacuous-test-suite stance
(2026-05-20), "an e2e test through the GUI is equivalent to a curl
test against the public API; if your test would pass against a broken
implementation, it isn't a real e2e test."
