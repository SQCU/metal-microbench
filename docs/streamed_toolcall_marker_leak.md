# Bug: streamed tool_call markers leak into ST chat content

**Discovered 2026-05-10. Fixed same day** in `sillytavern-fork/public/scripts/openai.js`
via a small FE-side strip in the streaming-delta accumulator. The repro
at `tools/st-debug/tests/23_streamed_toolcall_marker_leak.spec.js` now
gates on the absence of marker text and passes.

Before/after artifacts (in `docs/media/`):

- `2026-05-10_test23_streamed_toolcall_marker_leak_PRE-FIX_<sha>.{webm,png,json}`
- `2026-05-10_test23_streamed_toolcall_marker_leak_FIXED_<sha>.{webm,png,json}`

The remainder of this doc is the original analysis preserved as a
record of the reasoning that led to the fix shape.

## Symptom

When `stream: true` is set on `/v1/chat/completions` AND the model
emits `<|tool_call>call:NAME{args}<tool_call|>` markers in its
generated text, SillyTavern's chat history ends up containing the raw
marker bytes as the `.mes` field of an assistant turn — markdown-
rendered, with characters like the `#` from a Python comment
promoted into HTML headings. The structured tool_call is *also*
present (in `.extra.tool_progress` / `.extra.tool_invocations` on the
same or the next assistant turn), so the user sees both: a wall of
raw engine scaffolding rendered as text, and the proper collapsible
tool-progress card right after it.

## Repro (captured artifact)

```
turn [0]: dicemother first_mes ("the iron-banded door is half-rotted...")
turn [1]: user ("i kick the door open... use the python tool to do the rolling")
turn [2]: assistant (1262 chars)
    .mes head: '<|tool_call>call:python-exec__run{task:<|"|>import random
                 # Encounter Table ...'
    .extra.tool_progress: [{label: "Python Exec • Run Python task", status: "done"}]
    🩸 BUG: turn .mes contains literal <|tool_call> and <|"|> markers
turn [3]: assistant (571 chars)
    .mes head: "the door slams against the stone wall. you step into a chamber..."
    (wrap-up narration after the tool result lands; renders fine)
```

Artifacts:

- `docs/media/2026-05-10_test23_streamed_toolcall_marker_leak_<sha>.json`
  — full chat snapshot + network records.
- `docs/media/2026-05-10_test23_streamed_toolcall_marker_leak_<sha>.png`
  — full-page screenshot.
- `docs/media/2026-05-10_test23_streamed_toolcall_marker_leak_<sha>.webm`
  — video of the failing run.

## Cause

`server/bridge.py`'s SSE streaming path (the `gen()` coroutine) was
simplified earlier today by the inline-HTML / direct-bridge-probe
cleanup. The previous version had a rolling-buffer detector that
watched the streamed tokens for a `<|tool_call` prefix and switched
into a "buffer until end-of-call" mode, then emitted a structured
`tool_calls` delta and *no* content delta for the marker bytes. That
detector was deleted in the cleanup because the user observed that it
"buffered ALL tool-enabled chats" (the marker-detector path forced
near-100% of toolcards-enabled prose responses to wait until end-of-
generation before any token surfaced to the client).

The replacement path streams content live for ALL responses,
including tool-enabled ones. At end-of-generation it runs
`_extract_tool_calls(full_text)` and emits a `delta.tool_calls`
chunk if any markers parsed. That's structurally OAI-correct for
clients that strip marker text from `content` when `tool_calls`
arrives in a later delta — but ST's frontend doesn't do that
mid-stream-strip pass. It accumulates whatever lands in
`delta.content` into `chat[i].mes` and treats the later
`delta.tool_calls` as a separate signal that gets stored under
`extra.tool_invocations` without touching the already-accumulated
`mes`.

## Fix that landed

Option B from the original analysis — strip in `sillytavern-fork/
public/scripts/openai.js`. Two pieces:

1. `MODEL_TOOL_CALL_SENTINELS` — a module-level table of regex
   patterns, one per open-weights model family that emits a known
   tool-call sentinel pair in its chat-template token stream. Today
   it has one entry, `<\|tool_call>[\s\S]*?<tool_call\|>` for gemma-4.
   New families (llama-3.1, mistral, qwen) get one-line additions.
2. `stripModelToolCallSentinels(text)` — idempotent string transform
   called from the streaming-delta accumulator after each delta is
   appended to the accumulated `text` (or to the per-swipe accumulator
   under multi-swipe). The strip preserves the structured
   `delta.tool_calls` path the upstream bridge ALSO emits, which
   `ToolManager.parseToolCalls` consumes into `extra.tool_invocations`
   on the assistant turn — that's where the tool-card UI renders from.

The strip is unconditional because the sentinel pair is unambiguous
inside the model's chat-template byte alphabet: a normal token stream
cannot contain `<|tool_call>` without it being a tool-call marker
(it's an atomic tokenizer-vocab token, not a constructible byte
sequence). False-positive risk is zero for compliant model output.

Original option-A (bridge-side lookahead) was not pursued — it would
have walked back the streaming simplification we did the same morning
and added bridge complexity that's only needed for clients that don't
implement the FE strip. With the open-weights / chat-template-aware
client framing (see `docs/cvec_activations_validation.md` lineage on
"engine is a tensor service; client handles model-native bytes") the
FE strip is the structurally-correct home and Option A would be
mimicking corporate-API ergonomics in our local bridge.

## Reproducing

The bug is gated on the dicemother persona (added to
`bootstrap.sh` 2026-05-10) plus a prompt that diegetically calls for a
non-trivial python-exec invocation:

```bash
cd tools/st-debug/tests
npx playwright test 23_streamed_toolcall_marker_leak.spec.js
```

The persona was chosen because TTRPG GMs have a natural reason to call
`python-exec` for encounter tables, loot rolls, gacha rarities — the
combination produces a long enough `task=` argument that the rendered
markdown form of the leak is unmistakably visible in screenshots.
Other personas with structured-randomness prompts (combat resolution,
weather simulators, randomized creative briefs) should reproduce the
same pattern; this one is just the canonical repro.
