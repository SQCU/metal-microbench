# Forked-agent toolcard patterns

**Status**: Design spec. Captures the taxonomy of descendant-agent
shapes the metal-microbench / SillyTavern-fork harness should support,
plus example cards that demonstrate each pattern. Followup to
`descendant_agent_ux_spec.md` (the UX/affordances side) and
`tool_elicitation_findings.md` (the elicitation-rate side).

## The four shapes of forked agent

Every toolcard so far (hello-world, image-to-svg, query-to-svg,
query-to-svg-captioned) is shape **A** — fresh-context. The user
(2026-05-08) called out three more shapes worth supporting. Together
they form a 2×2:

```
                    fresh context        copy parent context
                    ────────────────     ─────────────────────
synchronous (call,  A: fresh-call        B: context-copying
wait, return):         (all current        (parent's full
                        toolcards)          history goes with
                                            the call; descendant
                                            sees what scringlo
                                            saw)

asynchronous (fire, C: fresh-async        D: context-copying-
return later via       (no current         async (deferred
interrupt):            example, but        result interrupts
                       cheap to add)       the conversation
                                            whenever it lands)
```

### A — fresh-context synchronous

**What**: descendant agent's invocation gets only the tool args
(`{query: "a sunset"}`). No knowledge of the parent conversation.
Returns a result; parent receives it via the marker pipeline.

**Where it fits**: stateless tool calls. The descendant doesn't need
context — it's doing a localized computation (rendering an SVG from a
description, executing a Python script with given inputs, looking up
a fact).

**Example**: `query-to-svg__generate` today. The user says "draw a
sunset", scringlo emits a tool call with just `{query: "a sunset"}`,
the descendant agent (whose system prompt is "render this query into
an SVG") iterates, returns the rendered image. The descendant has no
idea who scringlo is or what came before.

**Cost**: cheap — minimal prompt overhead per descendant call.

**Persona-violation risk**: low at descendant level (descendant has its
own task-shaped prompt, not the parent's persona). Some risk at the
parent's level if the descendant's response shape doesn't match what
the parent expects to see (e.g., descendant returns prose instead of
just inlining an image — see also: scaffolding-bleed bugs).

### B — context-copying synchronous

**What**: descendant inherits the parent's full conversation history
(or a configurable suffix thereof), THEN gets its tool-specific
framing tacked on. Returns a result that the parent treats as part of
the same conversation thread.

**Where it fits**: tasks that genuinely need the conversation context
to do their work — e.g., "summarize what we just discussed,"
"continue this story," "given the user's stated preferences, choose
between these options."

**Example to design**: `extended_thinking__deliberate`. User asks
scringlo a hard question. scringlo emits a tool call to delegate a
deeper-reasoning pass; descendant inherits the conversation, has its
system prompt augmented with "you have unlimited tokens to reason
out loud; produce a chain-of-thought then a 1-2 sentence summary";
returns the summary. scringlo presents the summary as her own answer.

**Cost**: prompt cost grows with context length. For long chats,
this is non-trivial — but the descendant's max_tokens for its
deliberation can be larger than scringlo's would otherwise be, so
net cost may be amortized.

**Persona-violation risk**: HIGH if not handled carefully. The
descendant has scringlo's history in its prompt; if its system
prompt is "deliberate carefully," the descendant may *break
character* and respond in a senior-engineer voice that scringlo then
echoes verbatim. Mitigations:
- Descendant's system prompt explicitly says "your job is to produce
  a private deliberation that the parent will summarize in their own
  voice; do NOT respond in the parent's persona"
- OR descendant returns a structured output (key facts, options,
  reasoning steps) that the parent rephrases.

### C — fresh-context asynchronous (interrupt-on-ready)

**What**: descendant fires off, parent continues talking, descendant's
result lands later as an interrupt — either to scringlo (who weaves
it in mid-sentence: "oh wait! the answer just came back: ...") or to
the user (a notification surfaced in the chat UI).

**Where it fits**: long-running queries where the parent shouldn't
block. E.g., "look up the weather in 5 cities" — fire 5 fresh-async
calls, scringlo says "looking those up now!" and continues talking
about something else; results filter in over the next ~10s.

**Example to design**: `web_search__async`. User asks scringlo about
some current event. scringlo fires the query async, says "okie *taps
the screen*" and changes topic; when the search returns, the chat UI
inserts a "result arrived" affordance.

**Cost**: same as A per call. Plus UI complexity (interrupts, ordering).

**Persona-violation risk**: low at descendant. At parent level,
mid-sentence interrupt requires scringlo to integrate gracefully —
some prompting needed.

**Phase-3 handoff**: the `async-lookup` prototype demonstrates that
the plugin/protocol layer is already capable of fresh-context async
work: `/start_invoke` can return a `session_id` immediately, the
descendant can continue running in the background, and the result can
land later through `/poll` plus `pushChatResult`. Phase-3 work is
wiring that capability to UX-level fire-and-forget behavior: mid-
conversation interrupt affordances, partial-result rendering, and
reconciliation states in the frontend. Those need FE changes; the
toolcards plugin API itself does not need a new blocking/non-blocking
mode to expose this pattern.

### D — context-copying asynchronous (deferred interrupt)

**What**: combines B + C. Descendant inherits context, runs in
background, drops a result whenever ready. Used for genuinely
heavyweight reasoning that you want to defer.

**Where it fits**: agentic patterns where multiple background
investigations spawn, each with the parent's context, each producing
an answer that arrives whenever it's done.

**Example to design**: `tree_of_thoughts__deep`. User asks scringlo
something open-ended. scringlo fires a context-copying-async tool
that spawns N descendant agents, each pursuing a different branch
of the question with full chat context. Results arrive over minutes.
UI shows a "deep thinking..." affordance per branch with captions.

**Cost**: highest. Worth it for the right tasks; over-applying makes
the chat unmanageable.

**Persona-violation risk**: highest. The descendants have the parent's
full context AND are running with substantial autonomy. Tight system-
prompt scoping is essential.

## Programmatic / non-LLM tools (orthogonal axis)

The user (2026-05-08) flagged a separate concern: *language models
can't necessarily uniformly draw from a list of tarot cards* — minor
arcana are "harder to choose" because of training distribution skew.
A non-LLM tool that does **programmatic random sampling** sidesteps
this entirely.

The 2×2 above is about CONTEXT/TIMING. Orthogonal to that: whether
the descendant is itself an LLM call or pure code (Python script,
deterministic algorithm, RNG).

For programmatic tools (Python script, random sampler, etc.),
mechanism shape A (fresh-call sync) usually fits best — they don't
need conversation context.

## Three example cards to prototype

These are concrete realizations of the patterns above. Each is a real
toolcard manifest + service.py implementation, designed to be small
enough to code in one pair-programming session.

### 1. `random_choice` — programmatic sampling (shape A, non-LLM)

**Manifest sketch**:
```json
{
    "id": "random-choice",
    "tools": [{
        "name": "uniform",
        "description": "Uniformly randomly select N items from a list. Use when you need a fair sample (e.g. tarot reading, dice roll, lottery) — language models tend to bias toward training-distribution-favored items, so for genuinely random selection delegate to this tool.",
        "parameters": {
            "type": "object",
            "properties": {
                "items":  {"type": "array", "items": {"type": "string"}},
                "n":      {"type": "integer", "default": 1},
                "with_replacement": {"type": "boolean", "default": false}
            },
            "required": ["items"]
        }
    }],
    "runtime": { "kind": "python", "deps": [], "entrypoint": "service.py" }
}
```

**Service**: a stdio-loop service that reads `{args:{items:[...], n, with_replacement}}` and returns `{ok:true, result: random.sample(items, n)}`. Trivial Python.

**Why it's worth having**:
- Demonstrates programmatic > LLM-sampling for fairness
- Trivial to validate
- Useful for actual product (tarot, dice, choice paralysis)

### 2. `python_exec` — descendant agent runs code (shape A, LLM-augmented)

**Manifest sketch**:
```json
{
    "id": "python-exec",
    "tools": [{
        "name": "run",
        "description": "Run a Python script in a sandboxed subprocess and return stdout. The descendant agent receives the user's high-level request, writes a script, executes it, returns the script + stdout.\n\nWhen to call: a user describes a computation that's tedious for an LLM but trivial for code (numeric work, list manipulation, simulation, data parsing).\n\nHow to call: pass the user's request in natural language as `task`; the tool decides on its own implementation. Example: task='sample 10 unique words from a Wikipedia random-words list with no repeats and sort by length'.",
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string"}
            },
            "required": ["task"]
        }
    }],
    "runtime": { "kind": "python", "deps": [], "entrypoint": "service.py" }
}
```

**Service**: 
1. Receives `{args:{task:"..."}}`
2. Fires an `llm_call` to write a Python script: system prompt = "You produce only Python code in a code block, no explanation. The code's stdout will be returned to the user." User: the task.
3. Extracts the code block from the response.
4. Runs it in a subprocess with `subprocess.run(["python3", "-c", code], capture_output=True, timeout=30)`.
5. Returns `{ok:true, result: {script: code, stdout: out, stderr: err}}`.

**Persona safety**: descendant's LLM-call is fresh-context (shape A) — it sees only the task string, not the parent's conversation. So even if the parent persona is silly/improv, the descendant produces serious Python code without contamination.

**Risk**: arbitrary code execution. Mitigation: subprocess timeout, no network access, no filesystem writes outside `/tmp` — but for THIS prototype, target a debug-instance audience, document the surface.

### 3. `extended_thinking` — context-copying deliberation (shape B)

**Manifest sketch**:
```json
{
    "id": "extended-thinking",
    "tools": [{
        "name": "deliberate",
        "description": "Delegate a hard question to a deeper-reasoning descendant agent. The descendant inherits the recent conversation and produces a chain-of-thought followed by a 1-2 sentence summary. Use when the user asks something that requires careful step-by-step reasoning.\n\nWhen to call: the user asks 'is this safe?', 'which option is better given X, Y, Z?', or anything where rushing produces a worse answer than thinking carefully.\n\nHow to call: pass the question in `question`. The tool will inherit the parent conversation context automatically. Example: question='given the constraints we just discussed, which database should we pick?'.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string"}
            },
            "required": ["question"]
        }
    }],
    "runtime": { "kind": "python", "deps": [], "entrypoint": "service.py" }
}
```

**Service**:
1. Receives the args. Also needs the parent conversation — for this we'd extend the toolcards plugin's `start_invoke` to optionally pass `caller_messages: [...]` (currently it captures `chat_id` and `caller_message_id` but not the messages themselves).
2. Fires an `llm_call` with messages = `[parent_messages..., {role:"system", content:"You have unlimited tokens to reason carefully. Produce a chain-of-thought then a 1-2 sentence summary. The parent agent will rephrase your summary in their own voice — do not assume they will quote you verbatim."}, {role:"user", content: question}]`
3. Extracts the summary from the response.
4. Returns `{ok:true, result: {cot: full_text, summary: extracted_summary}}`.

**Persona safety**: the system message in step 2 explicitly says "the parent agent will rephrase your summary." This is the **key elicitation discipline** for shape B: the descendant's role-instructions must explicitly NOT assume parent character continuity. Otherwise the parent (scringlo) gets back a senior-engineer-voiced answer and either (a) parrots it (breaking character) or (b) fights it (jarring transition).

**Plugin extension required**: the plugin's `start_invoke` currently captures `chat_id` and `caller_message_id` but not the actual messages array. For shape B we need either the FE to pass `caller_messages` explicitly, OR a server-side lookup keyed by `chat_id + caller_message_id`. The latter requires the plugin to know about ST's chat storage — tightly coupled. The former is an FE-side change but simpler.

## Integration plan

For this session — pair with codex on **#1 (`random_choice`)** because:
- Smallest scope (no LLM call inside the descendant)
- Most concretely demonstrates "non-diegetic forked agent doing something the parent can't"
- Tarot example is the canonical test
- No plugin extensions required

After #1 lands cleanly, #2 (`python_exec`) is the natural next prototype
in another session — it adds the descendant-LLM-call dimension.

#3 (`extended_thinking`) requires the plugin extension (`caller_messages`)
which is a bigger change; defer until #1 + #2 establish the toolcard
authoring pattern in our debug environment.

## Cross-references

- `docs/descendant_agent_ux_spec.md` — UX affordances for the
  descendant work (caption agents, depth indicators, etc.)
- `docs/tool_elicitation_findings.md` — what makes the parent
  reliably emit a tool call in the first place (round-2 findings:
  qualitative + syntax examples on non-overlapping topics)
- `docs/toolcards_fifo_session_finding.md` — why the plugin can now
  handle multiple concurrent forked agents (the FIFO fix from
  commit `54a64bb`)
- `tools/st-debug/scripts/install_captioned_toolcard.sh` — pattern
  for installing a custom toolcard in the debug data root
- `tools/st-debug/scripts/spur_caption_subagent.py` — the caption
  primitive that already implements shape A for live progress
  summarization
