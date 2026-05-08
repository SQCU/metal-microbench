# Descendant-agent UX in chat-shaped interfaces

**Status**: Design spec — no implementation yet. Captures the design
constraints around presenting "the chat agent delegated work to a
forkagent that ran for several minutes in its own context, and here's
what the user should see." Followup to commits `cffca0b` (st-debug
harness) + `4a8f070` (no-tools-stream stripper) + the Phase-2
toolcards e2e tests added 2026-05-08.

## The problem

A user types "scringlo, draw me a fractal." Scringlo (a Gemma-4 chat
agent persona) decides to call a `query-to-svg__generate` tool. The
toolcard's Python service:

1. Spawns. Now there are TWO model contexts: the OUTER chat with
   scringlo, and the INNER multi-turn refinement loop in the toolcard's
   own conversation history.
2. Iterates 3-5 times. Each iter is an LLM call against the bridge,
   typically 5-30 seconds. Inner-context system prompt + initial query
   → SVG. Render to PNG. Show to model again. "Refine." Repeat.
3. Returns SVG + PNG.

Three UX truths in tension:

- **(a)** The user expects scringlo to "be doing something" with low-
  latency feedback. A 90-second silent gap is broken.
- **(b)** Dumping the entire 50,000-token inner refinement transcript
  into the chat is overwhelming, breaks the persona, and breaks the
  reading flow.
- **(c)** Hiding everything is dishonest — multi-minute background
  work running with no visible progress IS the bad-UX-design smell
  the user called out.

## What's already wired (st-debug e2e validates this)

ST's toolcards subsystem exposes three useful hooks:

- `progress` events emitted by the toolcard's `service.py` are
  rendered as a status bubble attached to the caller turn. By design:
  collapsed by default, shows "last 5 lines, rolling." Source:
  `public/scripts/extensions/toolcards/index.js:269` and the comment
  block at `:104` ("rendered inline by messageFormatting").
- `result` events with `embed: [text, image_url, ...]` parts get
  inlined into the caller turn via the `[[image:<id>]]` marker
  pipeline (`public/scripts/extensions/svg-rasterize/index.js`).
- The `tool_progress` extra field on the message has a stable contract
  for adding more affordances later.

These primitives are **enough** to build the UX described below
without forking ST's frontend further. The current implementation
already uses (collapsed status bubble + inline embed) — the gaps
below are about depth of presentation, not about whether the
plumbing exists.

## What's missing — the four design problems

### 1. "Where am I?" — depth indicator

A descendant agent can spawn its own descendant agents (a refinement
forkagent that uses its own tool calls inside the loop). The user
needs to know: "this status bubble represents work happening 2 levels
below my current chat" — not just "scringlo is busy."

Affordance: depth marker on the status bubble. e.g.,

```
🔧 scringlo → query-to-svg → iter 2/3
   refining the red circle with vision feedback
```

Implemented by extending `tool_progress[]` entries with a `depth: int`
field; UI renders `└─` glyphs for depth > 0.

### 2. "What's in the inner context?" — collapsed-by-default summary

The user shouldn't see the inner transcript by default, but should be
able to expand it on demand. Two surfaces:

- **Collapsed (default)**: just the last progress line + a "tap to
  expand" affordance.
- **Expanded**: full transcript view in a modal or side panel, with
  the inner system prompt + every llm_call/result pair, navigable.

The toolcard plugin already keeps the full session log
(`plugins/toolcards/index.mjs` — sessions are server-authoritative,
survive reconnect). It just needs an FE viewer.

### 3. **The summarization-forkagent affordance** (the user's idea)

For long-running descendant work, fork off a tiny **caption agent** in
parallel to the primary descendant. The caption agent's job:

> "Summarize the last 1000 tokens of this transcript in one sentence,
> live-updating as new tokens arrive."

Decode at low effort (small max_tokens, possibly draft model if we
have one), emit one summary line every ~5s. The status bubble shows
the LATEST caption — giving the user a "what is this thing currently
trying to do" affordance without exposing the raw transcript.

```
┌──────────────────────────────────────────────────────────┐
│ 🔧 scringlo → query-to-svg                            ▶  │
│ "now adjusting the gradient to be slightly less          │
│  saturated than the previous attempt"                    │
└──────────────────────────────────────────────────────────┘
```

Implementation sketch:
- Toolcard plugin spawns a SECOND inference channel for the caption
  agent, fed via a sliding-window pipe of the inner transcript.
- Caption agent's bridge call uses a tight max_tokens cap (~32) and
  a strong directive system prompt ("summarize in one sentence; no
  preamble").
- Captions get emitted as a NEW progress event type:
  `{type: "caption", text, depth}` — separate from `progress` so
  the UI can render them differently (italic, collapsed-by-default).
- At the bridge level, the caption agent's calls run concurrently
  with the primary descendant's calls; engine's M:K scheduler already
  handles 2 concurrent streams trivially (we have headroom for 64).

### 4. Cross-context-result handoff (the "different literal context" problem)

When the descendant agent finishes, the result needs to land back in
scringlo's outer context — but scringlo wasn't in the descendant's
loop and doesn't know what happened there. Two failure modes:

- **Hallucinated context**: scringlo says "the red circle I drew
  earlier..." referring to a circle she never saw, only the rasterized
  PNG of one. The descendant agent's REASONING about the SVG is in a
  context scringlo doesn't have.
- **Lost context**: scringlo says "here's your image!" with no actual
  knowledge of what's in it. The user asks "can you make it bluer?"
  and scringlo doesn't know the current colour because it never reasoned
  about the SVG content.

Affordance: when the descendant returns, inject a structured handoff
into the outer context that's MORE than just the embed:

```
[descendant query-to-svg result]
  query: "a fractal"
  iters: 3 (last 2 mse-improving)
  final SVG: 4127 chars, depicts a circular paisley pattern in
    purple/cyan with 3 levels of recursive ornamentation
  rendered: [[image:abc123]]
```

The "depicts ..." line is itself a descendant-agent output (a single
captioning call against the final PNG). Cheap, gives scringlo enough
to answer follow-up questions coherently.

Implementation sketch:
- Toolcard's `result.embed` gets a NEW preceding text part:
  `descendant_summary` — generated by a final captioning call before
  the toolcard returns.
- ST's marker pipeline already handles multipart content; this is
  just a new content shape inside the same pipe.

## Open questions

1. **Should the user be able to interrupt a descendant?** A "cancel"
   affordance on long-running toolcards. Plugin already has
   `/api/plugins/toolcards/cancel/:session_id`; FE just needs a button.
2. **What about deeply-nested descendants?** A toolcard that spawns
   a toolcard that spawns a toolcard. The depth indicator design
   above generalizes, but the chat layout needs to handle vertical
   nesting visually.
3. **Caption agent budget?** Hot-loop capped at ~32 tokens × every
   5s × N descendants × M depth = nontrivial bridge load. Likely
   fine on M5 at current throughput (~135 tok/s), but worth
   measuring once we have N>2 concurrent active caption agents.
4. **Persistence across sessions**: a toolcard runs for 5 minutes,
   the user closes their browser, comes back. The plugin keeps the
   session running server-side; the result lands when they
   reconnect. UI affordance: "scringlo finished her drawing 12
   minutes ago — here it is" entry rather than a real-time bubble.

## Connection to the qualitative regression observed today

The Phase-2c st-debug test (`04_toolcards_direct_invoke.spec.js`)
PASSES — the SVG-render pipeline works, descendant agents run, the
PNG comes back inline. The Phase-2b test (`03_toolcards_svg_query`)
FAILS at the **outer model declining to call the tool** — the model
hallucinates "OK done!" without emitting a tool_call.

That's a model-prompting issue, not a pipeline issue. The fixes
above (especially #3, the caption-forkagent) become more compelling
once we ALSO fix the outer-prompting:

- Stronger persona/system-prompt scaffolding that primes scringlo
  to use tools (currently the Default character is "concise,
  factual" — bad fit for "wiggles fingers and draws").
- Bridge-side `tool_choice: required` honouring (currently
  advisory-only). When ST wants to force a tool call, we should
  honour it; right now `04` had to hit the plugin endpoint
  directly to avoid the model's decision step.

Both of these are tractable next-session work and would close the
loop on the user's report.

## Cross-references

- `tools/st-debug/tests/02_toolcards_hello_world.spec.js` — minimal
  tool-call e2e; passes.
- `tools/st-debug/tests/03_toolcards_svg_query.spec.js` — full SVG
  workflow via the chat; currently fails by model hallucination.
- `tools/st-debug/tests/04_toolcards_direct_invoke.spec.js` —
  rendering-pipeline validation; passes in 7s.
- `~/sillytavern-fork/data/toolcards/installed/query-to-svg/service.py`
  — the multi-turn refinement loop the harness exercises.
- `~/sillytavern-fork/public/scripts/extensions/toolcards/index.js`
  — the FE that plumbs progress/result events into chat DOM.
- `tools/quant_search/svg_canonical.py` — async port of the same
  refinement methodology, useful for distributional studies of
  refinement quality vs # iters / temp / etc.
