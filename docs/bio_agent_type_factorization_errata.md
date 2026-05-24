# Bio / Agent Type Factorization — Errata and Source Transcripts

**Date:** 2026-05-24  
**Purpose:** Canonical reference — verbatim operator statements establishing the
bio / agent type distinction, the postfix-template specification, the agents-as-separate-PNG
directory contract, and the settings.json deprecation mandate. Produced after a synthesis
pipeline committed a type error (overwriting a hand-written bio description with
LLM-generated agent content).

**Session file:** `/Users/mdot/.claude-personal/projects/-Users-mdot-metal-microbench/247a1b45-62c9-4cfe-a738-bb129a1145bd.jsonl`

---

## Section 1: Operator statements articulating the bio / agent type distinction

### 1.1 — The definitive type-error announcement (the trigger for this document)

**Source:** Session JSONL line 50218 | 2026-05-24T07:25:44.292Z

```quote
this is a conceptual error; user-agents have been specified. agian. and again. and again.
and again. and again. and again. as a completely different file and artifact and data type
from a 'user card' or 'user persona' or 'user bio' (bio==persona==card in this case btw.
they aren't 3 different or 2 different things.). there are to be any number of 'user agents'
for any one user bio, up to, say, 128,000 different agents, one might say, per 'user bio'.
how is this the case? it is the case because they are and remain literally different data
types hwich in the user agent chat harness code were repeatedly specified to be appended
(postfix) text templates *appended* to a context to articulate different behaviors that a
*user* can have which share invariants like the *user bio/history/definitions*, the *entire
chat context*, etc. if any of them ever ended up written to a 'user bio', this is a *type
error* in our *type system* of user bios, user agents, assistants, chat templates, assistant
cards, etc. it would be just as much a type error to overwrite an assistant card with a chat
turn or a base64 dump of a png. nevertheless, this type error happened; we need to trace its
conceptual origins and restore the feature coverage implied.
```

**Axis nailed down:** Establishes that bio and agent are fully separate data types and that
writing agent content into a bio slot is a *type error* equivalent to overwriting an
assistant card with a chat turn.

---

### 1.2 — Request to write documentation on the distinction (early articulation)

**Source:** Session JSONL line 21376 | 2026-05-13T07:03:05.733Z

```quote
okay time to stop playing around. 1: halt all running tasks or queries or behaviors in
themetal microbench sillytavenr fork server. 2: restart the metal microbench bridge server.
3: enumerate the feature spec of the user agent as a design and what the user actually asked
for in each part of the transcript related to this topic. 4: search for the string 'cowrite'
and identify how much of the scaffolding for all of these behaviors was already written and
even user interface tested into this program more than a day ago. 5: write a piece of project
documentation articulating exactly what the user agent is and how it is different from a user
bio or 'character description'.
```

**Axis nailed down:** Orders production of documentation specifically distinguishing user
agent from user bio — confirms the distinction was considered non-obvious and worth
permanently documenting as of 2026-05-13.

---

### 1.3 — "Saving agents as a different literal data structure" mandate

**Source:** Session JSONL line 30093 | 2026-05-15T23:27:20.138Z

```quote
the current focus for refinement now actually is to make usre we are saving user *agents*
as a different literal data structure and 'card' type than user *bios*. this means 2 things:
only author's-note kv-prefix-caching-compatible implementations of chat agents (they can't
be implemented by replacing a prefix at the start of a context; they must be implemented as
'authors note' type appended instructions), and saving a user agent artifact as a separate
factorized file from the biography card it is 'designed' for. this should dramatically reduce
the bloat of # of user biographies in our test/demo client, and should also necessitate some
clear refactoring intervention points in some related user interfaces to allow
agent-for-user-persona-prompt loading and running.
```

**Axis nailed down:** Explicitly mandates separate file storage for agents vs bios AND
mandates authors-note (postfix) injection as the only permitted implementation.

---

### 1.4 — Roster design: "modifiers upon different user cards"

**Source:** Session JSONL line 16613 | 2026-05-12T02:31:23.189Z

```quote
what user interface affordances (and so on) do we need to lay out in order to choose
specific rosters of (multi...?) fully/semi autonomous user agents in this interface pattern?
how would we handle rosters of 20+ user personas which themselves are tentative modifiers
upon different 'user cards'? how should the composition of these interface features be
handled, that of user characters vs user-agent-personality-motives?
```

**Axis nailed down:** Establishes the N-to-1 relationship: many agents per card, with agents
as "modifiers upon" a card — agents do not replace cards.

---

### 1.5 — Factorization: bio prose must not be overwritten by agent elicitation

**Source:** Session JSONL line 27809 | 2026-05-15T01:42:31.274Z

```quote
can we try to adjust our behavioral goals here s.t. a persona can be defined by a root
biography which has user-agent-elicitation implemented as an 'author's note' (see sillytavern
specification actually) for 3 primordial reasons: 1: we would like to be able to retain
extremely specific n-grams with distinct and precise meaning like a compact scrongle scrimble
description (the root scrongle scrimble spec we started from was like 24 tokens btw) without
changin the *biography* by overwriting user bios with a bio-for-elicitation
```

**Axis nailed down:** Changing how a user behaves (agent design) must never alter the bio
prose — the bio is a stable root that agents are built on top of, not overwritten into.

---

### 1.6 — Correction to agent designer prompt: agents "unique to a bio" but not context-pinned

**Source:** Session JSONL line 37663 | 2026-05-18T01:41:32.728Z

```quote
user agents are for specific bios. agents are unique to a bio but are supposed to maximize
certain behavioral cues (the metric space) in multiturn interactions over multiple different
'assistants' or 'counterparties' instead of being pinned to a specific context *and* a
specific user bio.
```

**Axis nailed down:** Agents are designed with a bio in mind (`designed_for_bio_id`) but are
not coupled to a single conversation context — they are reusable across counterparties.

---

### 1.7 — "Siloing of data flow and data types" as a regression

**Source:** Session JSONL line 45743 | 2026-05-20T17:22:13.398Z

```quote
synthesized bios aren't shown as user-personas in the stock baseline user interface. this is
incorrect and even a regression: every synthesized bio should be rendered as a user persona,
because they *are* user personas (and vice versa). therei s some kind of siloing of data flow
and data types imposed by one or more of the interfaces used for these features: those
siloizations are all incorrect.
```

**Axis nailed down:** Confirms that bio == user persona in every context; any data-flow path
that treats them as different objects is a bug, not a feature.

---

### 1.8 — Canonical vocabulary from design doc

**Source:** `/Users/mdot/metal-microbench/docs/user_agent_factorization_spec.md` (authored
spec, not agent-generated)

```quote
We deliberately avoid the word **persona**. It conflates two factorizable things:

- **biography** — system-prompt prefix information that lets a chat counterparty parse the
  semantic meaning of the user's first few messages in a fresh environment. Static identity
  flavor: voice, register, background. The thing a reader uses to make sense of the user's
  first turn before any history accumulates.
- **user-agent** — differences in *how* the user wants, strategizes, and picks among the
  courses of action the environment offers, including (in ambiguous or sparse environments)
  what the user is even aiming at. Dispositional engine. Move-set. Not specific to any single
  chat turn; consulted across many turns of many chats.

The MTG analogy: biography ≈ card art + flavor text + creature type line;
user-agent ≈ the mechanics box.
```

**Axis nailed down:** Canonical design vocabulary. Biography = static identity prefix;
agent = dispositional engine across turns. These are not synonyms and do not share a
storage location or schema.

---

## Section 2: Agent-as-appended-postfix-template specification

### 2.1 — Authors-note / postfix / KV-cache rationale (three reasons in one message)

**Source:** Session JSONL line 27809 | 2026-05-15T01:42:31.274Z

```quote
can we try to adjust our behavioral goals here s.t. a persona can be defined by a root
biography which has user-agent-elicitation implemented as an 'author's note' (see sillytavern
specification actually) for 3 primordial reasons: 1: we would like to be able to retain
extremely specific n-grams with distinct and precise meaning like a compact scrongle scrimble
description (the root scrongle scrimble spec we started from was like 24 tokens btw) without
changin the *biography* by overwriting user bios with a bio-for-elicitation 2: if we want to
change user agents midstream if we do this by changing the prefix for a context, we actually
lose kv caching and have to re-prefill the entire sequence in light of the changed elicitation
strategy 3: this lets us focus the user agent designer harness / tooling on the
*user-agent-behavior* in a more focused way with perhaps stronger results.
```

**What this nails down:** All three structural commitments in one message — agent text is
appended (not prefixed), bio prose is immutable, and KV-cache prefix sharing depends on
agents being at the END of context, not the beginning.

---

### 2.2 — Code enforcement: only `authors_note` injection mode permitted

**Source:** `/Users/mdot/sillytavern-fork/plugins/user-personas/index.mjs` lines 170–181

```quote
// Agents (= user-agent artifacts, the situational motivation layer
// designed to drive a bio toward a target Likert signature) live at
// agents/<id>.json. Schema: 'agent-v1'.
//
// Agents are injected at depth-N from the end of context as a
// role:system message ("author's-note style"). They MUST NOT replace
// the bio's leading system prompt — that would invalidate KV-cache
// sharing across agents and is forbidden by the AGENT_INJECTION_MODES
// allow-list.
const AGENT_INJECTION_MODES_ALLOWED = new Set(['authors_note']);
```

**What this nails down:** The enforcement mechanism — only `authors_note` mode is allowed,
and the comment states explicitly WHY: replacing the bio's leading system prompt would
invalidate KV-cache sharing across agents.

---

### 2.3 — Harness code: agent text assembled as postfix, `injection_depth: 1`

**Source:** `/Users/mdot/metal-microbench/tools/user-agent-harness/harness_lib.mjs` lines 205–206 (saveAgent) and line 263 (designAgent prompt):

```quote
        injection_mode: 'authors_note',
        injection_depth: 1,
```

```quote
        'You design a short user-agent overlay (author\'s-note style, '
```

**What this nails down:** The harness writes `injection_depth: 1` (appended 1 message from
end of context) and `injection_mode: 'authors_note'` on every synthesized agent.

---

### 2.4 — Linear algebra: bio feature space is strict superset of agent feature space

**Source:** `/Users/mdot/metal-microbench/docs/feature_factorization_design.md` (operator-directed spec)

```quote
Two action subspaces:

- **A ⊆ X** — agent-controllable. Axes that a depth-1 author's-note
  user-agent overlay can move from one bio. Dispositional / move-set:
  `theft_aggressiveness`, `romantic_advance`, `confrontation_style`,
  `risk_tolerance`, …
- **B ⊆ X** — bio-controllable. Axes that bio prose can move.
  Identity / register: `astrology_sagittarian`, `vocabulary_register`,
  `voice_warmth`, … and (typically) `B ⊇ A` since bios also condition
  the agent-controllable axes
```

**What this nails down:** A (agent space) ⊆ B (bio space) — agent axes are a strict subset
of bio axes. The bio provides the invariant scaffold; the agent is a restricted-scope overlay.

---

### 2.5 — Bio designer output IS the input for agent designer (nesting / shared invariants)

**Source:** Session JSONL line 38528 | 2026-05-18T03:43:27.278Z

```quote
the *user agent designer transitively wraps the intermediate results of any bio designer
turn*. the output, even the interim output, of the bio designer turn IS THE INPUT FOR A USER
AGENT DESIGNER INPUT. there is even literal semantically meaningful backpropagation of the
result vectors from user agent design rounds to the environment of the user bio designer, and
it is not necessary or sensible for the user bio designer feature vector space to be anything
but a strict superset of the feature vector space of the user agent designer *results* of each
*step* of the user bio designer.
```

**What this nails down:** The invariants the agent shares with the bio: the bio's feature
space, the bio prose, and the chat context are all inputs to agent design — the agent text
is built on top of, not instead of, these shared invariants.

---

### 2.6 — 128,000 agents per bio (N-to-1 scale statement)

**Source:** Session JSONL line 50218 | 2026-05-24T07:25:44.292Z

```quote
there are to be any number of 'user agents' for any one user bio, up to, say, 128,000
different agents, one might say, per 'user bio'.
```

**What this nails down:** The N-to-1 cardinality. One bio may have arbitrarily many agents
(the 128k figure is illustrative). This is the reason bio and agent cannot share a storage
scheme or naming space that could produce collisions.

---

### 2.7 — KV-prefix sharing depends on agents being at the tail

**Source:** Session JSONL line 27809 | 2026-05-15T01:42:31.274Z (reason 2, reproduced for emphasis)

```quote
if we want to change user agents midstream if we do this by changing the prefix for a context,
we actually lose kv caching and have to re-prefill the entire sequence in light of the changed
elicitation strategy
```

**What this nails down:** The prefix-cache implication of the design choice. The bio text is
the cache-stable prefix shared across all agents. If agents were prefixes, every agent swap
would bust the cache. Because agents are postfix, the prefix is shared and the cache hit
rate is maximized.

---

## Section 3: Agents written as PNG cards, stored in a separate directory

### 3.1 — "Separate factorized file from the biography card"

**Source:** Session JSONL line 30093 | 2026-05-15T23:27:20.138Z

```quote
saving a user agent artifact as a separate factorized file from the biography card it is
'designed' for. this should dramatically reduce the bloat of # of user biographies in our
test/demo client, and should also necessitate some clear refactoring intervention points in
some related user interfaces to allow agent-for-user-persona-prompt loading and running.
```

**What this nails down:** Agent artifacts live in a separate file, not embedded in or
co-located with the bio file.

---

### 3.2 — Storage table: agents in `plugins/user-personas/agents/`, bios in `User Avatars/`

**Source:** `/Users/mdot/metal-microbench/docs/multi_user_agent_chat_interface_spec.md`
lines 369–370 (storage canonical contracts table)

```quote
| Bios = ST personas | `<dataRoot>/<user>/User Avatars/<key>.png` (avatar) + `settings.json → power_user.persona_descriptions[<key>]` (bio text) | `/personas` GET, ST native persona drawer | `/personas` POST (writes both atomically), ST's own persona-create flow |
| Agents | `plugins/user-personas/agents/<id>.png` (chara_card_v3 PNG) | `/agents` GET | `/agents/:id` POST |
```

**What this nails down:** The exact directory paths. Bios live in `<dataRoot>/<user>/User Avatars/`. Agents live in `plugins/user-personas/agents/`. These are different directories, different read/write APIs, and different schemas — they can never be confused by a correctly-written storage call.

---

### 3.3 — Invariant: no mirror writes, `_mirrorPersonaToSettingsJson` permanently forbidden

**Source:** `/Users/mdot/metal-microbench/docs/multi_user_agent_chat_interface_spec.md`
lines 376–379 (Storage invariants section)

```quote
**Invariants:**
- One canonical store per concept. No mirror writes (the
  `_mirrorPersonaToSettingsJson` anti-pattern is permanently forbidden).
- Atomic writes to settings.json: tmp file + rename. Settings.json is
  shared with ST's own writes; partial writes corrupt the application.
```

**What this nails down:** Cross-directory writes from one type's store to the other's are
forbidden. The anti-pattern is named and banned by name in the spec.

---

### 3.4 — Plugin code: enforcement via `AGENT_INJECTION_MODES_ALLOWED` allow-list at boot

**Source:** `/Users/mdot/sillytavern-fork/plugins/user-personas/index.mjs` lines 954–955

```quote
            if (!AGENT_INJECTION_MODES_ALLOWED.has(a.injection_mode)) {
                console.warn(`[user-personas] agent ${a.id}: injection_mode=${JSON.stringify(a.injection_mode)} forbidden — only ${[...AGENT_INJECTION_MODES_ALLOWED].join(', ')} allowed (KV-prefix-share requirement)`);
```

**What this nails down:** The plugin actively rejects agent cards that specify a non-postfix
injection mode at load time, surfacing the KV-prefix-share requirement as an error.

---

## Section 4: The settings.json removal mandate

### 4.1 — The definitive deprecation statement (triggered this document's production)

**Source:** Session JSONL line 50218 | 2026-05-24T07:25:44.292Z

```quote
settings.json is deprecated due to concurrency and race condition issues; our sillytavern
fork was to remove absolutely all calls that used settings.json as a store of user bios and
replace them with a file based model of users equivalent to the assistant card file based
model sillytavern already uses. no fallbacks. no failovers! *no fallbacks*.
```

**What this nails down:** Unambiguous deprecation with no escape hatches. The mandate is
`no fallbacks` (repeated, starred for emphasis).

---

### 4.2 — "weren't you just saying settings.json had race conditions?"

**Source:** Session JSONL line 40126 | 2026-05-18T07:26:49.023Z

```quote
"persist to disk (settings.json for bios," why are you using settings.json instead of
something like expanding the cards interface to support user personas with bios, vectors,
metadata, whatever, for a totally uniform 'everything gets persisted in the right place'
iterface. weren't you just saying settings.json had race conditions? maybe oyu shouldn't be
doing things like that with it.
```

**What this nails down:** First explicit operator call-out that settings.json has race
conditions and should not be used for bios.

---

### 4.3 — "why does the plugin write to settings.json"

**Source:** Session JSONL line 40756 | 2026-05-19T04:44:23.919Z

```quote
why does the plugin write to settings.json. did anyone ask you to write to settings dot json
at any point? have you been told to do something strictly opposed to 'writing to settings dot
json' instead on many occasions? how many settings json writers remain in the codebase?
```

**What this nails down:** Establishes that settings.json writes were never authorized and
that a complete audit of remaining writers was demanded.

---

### 4.4 — "migrate bios out of a giant monofile and monorecord called settings.json which is not trustworthy"

**Source:** Session JSONL line 42627 | 2026-05-19T08:04:11.633Z

```quote
if one of the tasks here is to migrate bios out of a giant monofile and monorecord called
settings.json which is not trustworthy. why isn't all of the related sillytavern code being
ported and unit tested by a subagent or some other actual solution?
```

**What this nails down:** Frames settings.json as a "monofile... not trustworthy" and demands
full migration including porting of all related ST-side code, not just the plugin.

---

### 4.5 — Session summary confirming all settings.json writes were removed (with the quote "fuck the legacy installs if they ever existed")

**Source:** Session JSONL line 41043 | 2026-05-19T05:21:49.338Z (session continuation summary, summarizing prior operator-directed work)

```quote
All settings.json writes removed; legacy read also deleted ("fuck the legacy installs if they
ever existed")
```

**What this nails down:** At the time of this summary the migration was believed complete —
the regression diagnosed in line 50218 reveals it was not fully completed, because
`loadPlayers()` (lines 475–494 of `index.mjs`) still reads `power_user.persona_descriptions`
as a fallback source for bio text, and `_writePlayerCardPng` (lines 883–914) still writes
back to `settings.json` for every bio save.

---

### 4.6 — PERSONA_API.md: the contract for zero settings.json runtime reads/writes

**Source:** `/Users/mdot/sillytavern-fork/plugins/user-personas/PERSONA_API.md` lines 51–55

```quote
### runtime invariant

After this contract lands, NO code in the runtime reads or writes
`settings.json.power_user.{persona_descriptions, personas, default_persona,
character_persona_overrides}`. The keys exist in settings.json only because
upstream's schema declares them; they remain empty objects/null forever.
```

**What this nails down:** The final contract. After migration, the settings.json keys for
persona data must remain permanently empty. Any code that reads from or writes to those
keys is a violation.

---

## Remaining violations in plugin code (as of 2026-05-24)

The following live (non-comment) call sites in
`/Users/mdot/sillytavern-fork/plugins/user-personas/index.mjs`
still read or write `settings.json → power_user.persona_descriptions`.
These are the violations to be excised per the Section 4 mandates:

| Line(s) | Nature | Violation type |
|---------|--------|----------------|
| 475–478 | `loadPlayers()` reads `power_user.persona_descriptions` as bio text source | READ — fallback that must be removed |
| 888–899 | `_writePlayerCardPng()` writes `persona_descriptions[canonicalKey]` | WRITE — must be removed |
| 2701–2719 | `POST /personas/:id` reads `persona_descriptions` as `settingsBioFallback` | READ — fallback that must be removed |
| 2709 | `settings.power_user.persona_descriptions` checked for partial-update fallback | READ — same as above |

The migration script at
`plugins/user-personas/scripts/port_settings_personas_to_cards.mjs`
is the correct one-time migration tool. After it runs, the four call sites above
must be deleted (no fallback, no failover) per the mandate at Section 4.1.

---

## Summary of type constraints (normative, derived from above)

1. `bio` == `user persona` == `user card`. Three names, one type. PNG in
   `<dataRoot>/<user>/User Avatars/`. Never written to by agent synthesis code.

2. `user agent` is a distinct type. PNG in `plugins/user-personas/agents/`.
   Schema `agent-v1`. `injection_mode` must be `authors_note`. `injection_depth`
   defaults to 1 (appended near end of context, never at start).

3. There can be 0..N agents per bio (N arbitrarily large). A bio with 0 agents
   is unusable as a chat participant. The design loop's job is to produce agents
   for every bio; a bio lacking agents is a pending synthesis task, not an
   alternative valid state.

4. The bio is the KV-cache-stable prefix. The agent text is the postfix variant.
   Swapping agents does not require re-prefilling the bio or chat history.

5. Writing agent content into a bio record is a type error. The converse is also
   a type error. Shared directory paths or naming schemes that allow one type to
   overwrite the other are forbidden by design.

6. `settings.json → power_user.persona_descriptions` is deprecated. No reads.
   No writes. No fallbacks. No failovers. The card store (`User Avatars/` + PNG
   tEXt metadata) is the only legitimate bio store.
