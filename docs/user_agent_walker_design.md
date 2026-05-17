# User-agent walker: design + KV-share architecture + exploration framing

**Status:** living design doc. Captures the macroscopic frame the
multi-persona suggestion UI is being built toward, the agent
spawning/persistence/comparison/seeding primitives that turn the
suggester into a research instrument, the KV-cache architectural
constraint that makes long-context multi-persona affordable, and the
unseen-trajectory exploration goal that motivates the whole apparatus.

**Companion docs:**
- `docs/overlay_architecture.md` — original overlay-v1 schema (now
  migrated to bio-v2 + agent-v1 file split)
- `docs/multi_agent_suggestion_architecture.md` — earlier inventory
  pass when the surface was still being scoped
- `docs/CVEC_AND_PREFIX_CACHE.md` — the bridge-level content-addressed
  page cache this design relies on for prefix sharing
- `docs/strategy_diversity_scoring.md` — the diversity-scoring
  instrument the suggester operationalizes per-turn

---

## 1. The thing being built

A chat surface where the operator, instead of typing the next user-
turn, sees **k parallel candidate next-turns** — each authored by a
different (biography × agent) combination, each pivoting from the same
chat history through its own register and agenda. The operator picks
one to send, edits one, or types their own. Assistant side is fixed;
the user side becomes a *menu of voices*.

This is not a "chatbot picker." Most of what makes it valuable is what
*doesn't* exist in a typical chat-completion product:

- The personas are first-class persistent artifacts (cards) that the
  operator can spawn, persist, compare, and seed new ones from.
- The candidates surface their behavioural fingerprints alongside the
  text, so the operator picks not just based on prose taste but on a
  characterized signature.
- Long-context chats stay affordable because the architecture forces
  KV-prefix sharing across the k personas.
- The trajectory the chat takes through register space can be steered
  toward unsampled cells — producing chat sessions whose conversational
  shape didn't previously exist as observed data anywhere.

## 2. Agent spawning / persistence / comparison / seeding

Three operations that don't yet exist, named in increasing depth of
integration:

### 2.1 Spawn-in-flow

On the suggester surface, a button: **"Spawn an agent like _____ but
_____."** The first blank picks an existing card; the second is
operator prose like "more deferential," "less playful," "shifted from
probing to commanding." The system fires `/discovery` synchronously
with `root_bio_card = current_active_bio`, `target_named` constructed
by perturbing the source agent's centroid signature in the named
direction, and emits a fresh agent. Result lands as a new candidate
card in the suggester pane immediately — *no separate page
navigation*. The operator picks it, edits it, sends a turn through it;
if it feels right, a **"Persist this agent"** button writes it to
`agents/<bio>-<discovery-id>.json` with `derived_from:
<source-agent-id>` and a `mutation_request: "more deferential"` field
in provenance.

Interaction model: **agents are ephemeral by default** (live in memory
only until the operator chooses to persist), and become *cards* the
moment they're worth keeping. This is the SillyTavern "everything that
lasts more than one turn is a card" principle made operational — the
act of persistence IS the act of giving an agent a slot in the
inventory.

### 2.2 Comparison

`POST /compare-agents` takes two agent_ids and returns:

- `agent_text` diff (visual lexical comparison)
- per-axis centroid distance over recent /poll outputs
  (`trajectory_distance.py` is the primitive)
- nearest-neighbor in the corpus for each (which other existing agent
  is each one closest to?)
- a one-line summary from the LLM-summarizer: "agent A is more X /
  less Y than agent B"

UI surface: a "Compare" button on each agent card; clicking it picks a
comparator from a dropdown of all other agents.

### 2.3 Seeding from similarity / dissimilarity

Two distinct modes that the discovery harness supports as primitives
but doesn't expose at the right granularity yet:

**Similarity seeding** — "give me 3 agents like scringlo-js-clash but
each slightly varied." Take the source agent's centroid signature,
perturb by Gaussian noise in axis space (small σ on 2-3 axes per
spawn), feed each perturbed target to a fresh /discovery run, collect
the 3 resulting agents. Each is a *neighborhood sample* around the
source. Useful for: filling out a regional inventory once you've found
a region that works.

**Dissimilarity seeding** — "give me 3 agents far from everything I
currently have." Compute the centroid of all existing agents in axis
space; sample target signatures from points distant from the centroid
(or from the convex hull's complement); feed each to /discovery. Each
is a *frontier sample* expanding the inventory's reach. Useful for:
deliberately producing the rarities discussed in §4.

Both modes need a small wrapper around `/discovery` that handles batch
dispatch + dedup; ~30 lines of plugin code. The interesting part is
target-signature generation: principled perturbation in 14-axis space
requires either a learned generative model over the corpus signatures
or a deliberate axis-walk schedule. Simplest first cut: operator picks
2-3 axes + a direction (`+curious`, `−deferential`, `+structured`) and
we perturb only those.

### 2.4 Agent genealogy

When one agent is seeded from another, record the lineage. Cards
already have `discovery_provenance` and
`extracted_from_overlay_v1` fields; a `derived_from` field would
close the loop. The design tree becomes inspectable: "this agent came
from a +curious perturbation of scringlo-js-clash, which itself came
from a discovery run targeting PC4+1σ against python-only-coder, which
itself was extracted from the legacy overlay-scringlo-jsclash card."

## 3. KV-cache architectural review

The most important architectural constraint for the multi-persona
suggester to be affordable at long context.

### 3.1 Claim under review

When k different user-agents are consulted on the same chat history of
length T, **no exhaustive re-prefill** should happen. The token
sequence sent to the model should be byte-identical for the first ~T
tokens across all k calls. Only the trailing personality-injection
block should differ.

Stronger form, after the operator's clarification on role-swap
tolerance: **each turn of a multi-persona session should add O(1)
prefill labors to the bridge — not O(k).** Specifically, at most 2
prefill labors per turn, one per role-orientation reading of the chat
history. All k user-agents share one of those two readings.

### 3.2 What the current architecture does (the problem)

Today, `invertChatForPersona(persona, chat)` produces:

```
[0]   role=system, content=composePersonaSystemPrompt(persona)
[1..T] role=user|assistant, content=<chat history role-swapped>
[T+1] role=system, content=<author's-note: agent_text>     (if agent_id)
[T+2] role=user, content="Begin your conversation..."       (if chat empty)
```

Persona-specific bio at position **0**. Bridge's content-addressed
page cache hashes by content-prefix; when the persona's bio differs
across calls, position 0's tokens differ, so no cache page from
position 0 onward gets re-used. Each of k persona invocations does a
full prefill of T turns of chat history.

At T=20 turns × ~500 tokens/turn ≈ 10K tokens, k=3 personas = 30K
prefill tokens per suggestion round, instead of the ~10K (shared) + k
× ~150 (per-persona suffix) we'd get under prefix-share. Three-fold
cost differential at k=3, scaling worse with T or k.

### 3.3 The fix

Reorder messages: put persona content at the END.

```
[0]   role=system, content=<UNIVERSAL user-agent framing — identical
                            for every persona>
[1..T] role=user|assistant, content=<chat history with UNIFORM
                                      role-swap — does not depend on
                                      which persona is suggesting>
[T+1] role=system, content=<bio.system_prompt + agent.agent_text +
                            "now write your next turn">
```

First T+1 messages byte-identical across all k personas (modulo the
role-swap-orientation discussion below). Only `[T+1]` differs per
persona. Bridge content-addressed page cache hits on positions 0..T
regardless of which persona is being polled.

The persona's identity arrives as a depth-1 author's-note message —
exactly how the current overlay-v1 agents are injected, just with the
bio's system prompt included in that same trailing block. The
architectural shift: **bios become a special-case agent that also gets
injected at the end**, rather than living at position 0.

### 3.4 Role-swap orientation: O(1) prefill labors per turn, not O(k)

The chat history has TWO possible readings:

- **Canonical reading**: original (assistant = real assistant, user =
  real user). This is what the target assistant sees.
- **Swapped reading**: every original-user turn → role:assistant in
  the prompt, every original-assistant turn → role:user in the
  prompt. This is what every user-agent sees (because the user-agent's
  own training puts THEM in the assistant role — they generate
  "assistant"-tokens to produce user-side text).

Both readings cost ONE prefill labor each. So per turn of a
multi-persona session we have **at most 2 prefill labors regardless of
k** — one for the canonical-reading consumers (the actual assistant)
and one for the swapped-reading consumers (every user-agent
suggestion). All k user-agents share the swapped-reading prefill.

For this to hold, the swapped reading must be UNIFORM across all
user-agents. The current code's `invertChatForPersona` makes the swap
depend on `m.user_persona_id` matching the suggesting persona's id —
which breaks byte-identity across personas. The fix: a uniform swap
policy that doesn't depend on the suggesting persona's id:

- Every original-user turn → role:assistant
- Every original-assistant turn → role:user
- When a user turn has a known `user_persona_id`, prepend `[from:
  <persona-id>]: ` to its content (so the suggesting persona can read
  the chat and see who said what, even though they're all role-tagged
  uniformly)

The suggesting persona reads this and infers from context: "I see
turns from `[from: scringlo]` and `[from: pushy]` as role:assistant
— those are other users in this multi-user chat. The role:user turns
are the actual assistant's replies. I'm being asked to write the next
role:assistant (= user-side) turn."

Slightly different cognitive task than today's `invertChatForPersona`
(which presents the persona as if directly the speaker, with their own
past turns flowing into role:assistant). But tractable for the model —
the trailing author's-note explicitly says "you are persona X about to
write the next user-turn."

### 3.5 What still invalidates the share — and how to handle it

Three operations the plugin does that would break byte-identity if not
handled carefully:

1. **`stripWrappingQuotes` + `text.trim()`** post-generation. These
   mutate the text before re-emission to canonical. As long as
   mutations are applied CONSISTENTLY across all callers (or not at
   all), the canonical chat that gets passed to the next round stays
   stable.
2. **`composePersonaSystemPrompt`'s `extra.notes` field** is
   concatenated INTO the persona system prompt at position 0 today.
   Under the new architecture, notes belong in the trailing author's-
   note block, not the universal framing.
3. **The "Begin your conversation" nudge** is appended when chat is
   empty. Since it's the SAME nudge for every persona, identity is
   preserved as long as the order is canonical (chat empty → all
   personas get the nudge; chat non-empty → no persona gets it).

### 3.6 Validation

The `/template-fidelity` probe we already built does the right shape
of check: given a canonical chat and a (bio, agent) pair, verify every
chat turn appears byte-identically in the rendered prompt. Extending
for the multi-persona case: pass it a chat + a list of personas;
assert the rendered prompts are pairwise **byte-identical on positions
0..T** (where T is the chat-history end). A green check from this
probe is the architectural guarantee.

Additional empirical check: instrument the bridge to log cached-pages-
hit per call. A multi-persona round at k=3 on a T=10 chat should show
~T pages cached after the first call's prefill, and the next 2 calls
should hit-rate ~95%+ on those pages.

## 4. Unseen-trajectory exploration as the project frame

The deeper purpose. Worth saying directly.

The LLM training corpus contains an enormous mass of (user-agent,
assistant) trajectories, but it's concentrated in a tiny region of
register space — polite questions, support requests, creative-writing
prompts, occasional chat. The marginal distribution of "what user-
types do" is extremely peaked. Scringlo-style emoji-onomatopoeic
register with task-demanding agendas against Python-only assistants is
essentially never observed in training data. Multi-turn trajectories
where the user starts in one register and pivots to another are rare.
Trajectories that cross registers across N=3+ turns are vanishingly
rare.

We have the instruments to identify rarity:

- The 14+13 axis fingerprint per turn (the cascade judge)
- Per-trajectory drift and path-efficiency (`/trajectory-judge`)
- Per-corpus PCA + Mahalanobis (`/analyze`)
- KL distance from existing trajectories (`trajectory_distance.py`)

We can use these to **score novelty** of any candidate next-turn: its
distance from the nearest neighbor in our growing corpus of seen
trajectories. The suggester can be configured in **"novelty mode"** —
instead of picking k personas at random or by operator selection, pick
the k personas whose centroid signatures are FURTHEST from the chat's
recent trajectory direction. Equivalent at the agent level: pick
agents whose discovery `target_named` is FURTHEST from any cell the
current chat has visited.

The chat itself becomes a trajectory in axis space. Each turn moves
the trajectory by `d_vector/d_turn`. "Novel exploration" means picking
the next turn's direction to maximize departure from the corpus's
existing density.

The output of N such walks, sampled, judged, and persisted, is a
corpus of trajectories that:

- are byte-level unique (any specific text emitted is on-policy but
  unlikely to appear elsewhere because the joint
  register×trajectory was unlikely)
- are axis-sparsely-sampled (cover cells of the axis lattice that
  the training distribution leaves empty)
- are interpretable (signatures + drift characterize each)
- are reproducible (the (bio, agent, seed, chat-history) tuple lets
  anyone regenerate)

The last property is critical. Reproducibility lets us share these
trajectories as datasets — "here are 100 trajectories from cells of
axis space where standard chat-history corpora are silent."

### 4.1 What's not yet built (the abstract requirement, made concrete)

1. **A persistent trajectory store.** Every chat session optionally
   captured as a JSONL with per-turn signatures + persona genealogy +
   operator's selected candidates. We have iteration trajectories
   saved ad-hoc; a "session record" isn't yet first-class.

2. **A novelty metric over the trajectory store.** Given a candidate
   trajectory direction, compute its KL distance to the existing
   corpus's distribution at each axis. The metric exists at the
   pairwise level (`trajectory_distance.py`); aggregating across a
   corpus is a small extension.

3. **Novelty-prioritized suggestion mode in the UI.** A "show me k
   novelty-scored personas" mode that ranks the inventory's
   (bio × agent) cells by predicted novelty against the chat's current
   direction.

4. **Frontier-spawning.** When no existing agent scores high enough on
   novelty for the chat's current direction, automatically spawn a
   discovery run targeting an even more under-explored signature
   region.

5. **Session-capture export.** At the end of a session, "export this
   trajectory + signatures as a research artifact" with provenance
   complete enough that someone else could re-run discovery against
   the same seeds and reproduce the trajectory's character.

## 5. Card-quality warnings (relaxation of earlier auto-deletion idea)

Soft requirement noted by the operator. Rather than an automated
linter that deletes consistently-low-quality cards, **emit warnings**:
scan each (bio, agent) pair's recent /poll outputs in a small sample;
flag combinations where ≥M/N turns are degenerate (real
absorbing-state collapse, not just unusual register). Surface as a
`⚠ recent samples show register-instability` badge on the persona
card. The operator decides whether to retire/edit/keep; the linter
just makes the data visible.

N=3 is too small to be statistically confident anyway. As the prior
bio-quality screen showed, instrument biases (e.g., judge
mis-classifying terse-numbered-list register as "broken") can drive
destructive false positives. Warnings keep the operator in the loop.

## 6. Closing observation

The suggestion surface, framed as "just a UX for picking the next
user-turn," is a meek presentation of what's actually a research
instrument: a **controlled-exploration walker through under-sampled
cells of conversational-register space, with reproducibility
guarantees and judge-instrumented characterization at each step**. The
visual sidebar is the operator's window into the walk. The KV-share
architecture is what makes the walk affordable at long context. The
novelty mode is what makes the walk informative. Everything else we've
built (discovery, judge cascade, trajectory distance, agent
persistence) feeds the walker.

The thing being built isn't a chatbot interface. It's a corpus-
construction tool whose corpus members didn't previously exist
anywhere.

## Appendix A — KV-share refactor receipt (2026-05-15)

The shareable-prefix invocation builder
(`buildUserAgentInvocation` in `plugins/user-personas/index.mjs`)
landed today, opt-in via `shareable_prefix: true` on `POST /poll`.
Structure: universal framing system message at position [0]; chat
history role-swapped uniformly (not keyed on suggesting persona) with
`[from: <pid>]:` attribution at [1..T]; per-persona trailing system
block containing bio's `composePersonaSystemPrompt` output + agent
`agent_text` at [T+1].

Validation gate: `POST /template-fidelity` extended with
`kv_share_personas: [pid, pid, ...]` mode. On a 5-turn canonical chat
polled against scringlo, wry-skeptic, pushy-completionist, the
extended check returned:

  - `all_prefix_positions_identical: true` (T+1 = 5 positions checked,
    all byte-identical across 3 personas)
  - `prefix_size_bytes: 896B` (identical for every persona)
  - `trailing_block_bytes: 2570B / 2319B / 1122B` (correctly unique
    per persona — bio + agent metadata gets through)
  - `trailing_blocks_unique_per_persona: true`

This is the structural property the bridge's content-addressed page
cache needs to reuse chat-history KV pages across the k personas:
identical bytes → identical hashes → page-cache hit. Whether the cache
actually exploits that reuse is downstream (depends on cache size,
eviction policy, concurrent contention) — but the upstream prereq is
now provable per call.

End-to-end round-trip on the shareable path produced an in-voice
scringlo output ("hiii ownie!! 💖 ... pls pls pls can u make it happen??
✨"), confirming the rearranged context still elicits persona-faithful
generation. The full audit-trail tasks (#128 wiring the flag into
/iterate/afk-check/sweep, #129 flipping the default and retiring the
legacy `invertChatForPersona`) are tracked.

## Appendix B — Spawn-in-flow + seeding cartographic findings (2026-05-15)

§2.1 (spawn-in-flow) and §2.3 (similarity/dissimilarity seeding) landed
together. The full pipeline:

- `POST /spawn-agent` — prose-driven single spawn. Operator types "like
  X but Y"; the system runs `prosePerturbationToTarget(centroid, prose)`
  to derive a 14-axis target, then `discovery.py` overlay-mode against
  the source bio, returns an ephemeral draft.
- `POST /seed-agents` — batch axis-perturbation seeding. Two modes
  (`similarity` = Gaussian neighborhood; `dissimilarity` = max-min
  distance from corpus centroid). Each seed reuses `spawnSingleAgent`
  with an `explicit_target` (skipping the LLM prose-translation).
- `POST /persist-ephemeral-agent/:draft_id` — promotes draft to disk
  with auto-slug from mutation prose. `derived_from` + `mutation_request`
  + full `spawn_provenance` preserved.

### B.1 Cartographic claim validated

Per the operator's reframing: *we don't care about per-agent centroid
precision; we care that different agents OCCUPY DIFFERENT REGIONS of
trajectory-metric space*. Three similarity seeds around
`scringlo-js-clash-reborn` produced three qualitatively distinct hulls:

1. **tiny-polite-helper**: "super-duper polite, sweet and a little bit
   lost… 'scripty-scrip'… 'ohhh okay!!'… ✨💖"
2. **starstruck-fan**: "absolutely starstruck… pwease/mister energy…
   practically bowing"
3. **vague-obsessive**: "obsessed, total legend, vague and scattered,
   tripping over excitement" (the lowered goal_clarity showed up in
   the prose explicitly)

All three preserve the scringlo bio + JS-clash core; all three differ
materially from the blunt-investigator source AND from each other.

### B.2 Corpus clustering surfaced by dissimilarity mode

The 8-agent corpus centroid landed in a clearly defined region:
- HIGH: warm=3.98, performative=4.23, register_colloquial=4.20,
  in_character=4.43
- LOW: curious=1.93, terse=1.83, probe_depth=1.73, structured=1.43,
  provocative=1.40

This is the "warm-colloquial-performative" cluster. Dissimilarity
seeding revealed this structurally — the inventory is currently lopsided
toward a single neighborhood.

### B.3 Frontier-spawning reveals DESIGNER center-of-mass attractor

Three dissimilarity seeds tried to walk away from the corpus cluster.
Two succeeded structurally:

1. **vulnerable-rambler**: less performative, more disclosive
2. **technical-vulnerable-frontier**: terse +1.3, probe_depth +2,
   register_colloquial -2, warm -2 → produced a transactional-vulnerable
   agent referencing specific JS APIs (`requestAnimationFrame`,
   `Canvas`). *This is the most genuinely new agent we have so far —
   nothing else in the corpus has that "technical-crisis" register.*

The third seed **collapsed back to corpus mean** despite a substantial
target perturbation — the resulting agent_text was full of `💖` /
`bouncy-bouncy` / `super-duper obsessed`, exactly the warm-colloquial
register the perturbation was trying to escape. **This is itself a
real cartographic finding**: the discovery harness has a center-of-mass
attractor that resists certain frontier perturbations.

### B.4 Implications for §4 (novelty mode + frontier-spawning)

The DESIGNER center-of-mass effect means simple "walk a few sigma from
corpus mean" isn't enough — frontier seeds need to (a) walk further
(higher sigma), (b) iterate against measured outcome with a refused-
regression loop, or (c) supply a stronger anti-attractor constraint
in the operator_constraint prose. All three are testable. The current
implementation's 32-candidate max-min-distance search is the simplest
useful primitive; the next iteration of frontier-spawning (Phase D
§4.1.4) should add measurement-loop refusal so it can recognize and
re-roll cluster-collapses automatically.
