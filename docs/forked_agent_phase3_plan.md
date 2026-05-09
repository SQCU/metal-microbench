# Forked-agent toolcards: Phase 3 (FE wiring for async + streaming + cancel)

**Status**: Plan + execution checklist. Builds on `forked_agent_patterns.md`
(taxonomy + working example cards for shapes A/B/C/D) and the prior
plugin/protocol work that already makes async + concurrent llm_calls
+ caller_messages plumbing all server-authoritative.

The cards landed in commits `8d5aa18` (A/B/C examples) and `a0438f7`
(shape D `tree-of-thoughts`) prove the protocol layer is ready. What's
left is the FE wiring that exposes those capabilities to the user
during normal conversation.

## Current FE state (audit, 2026-05-08)

Already present in `public/scripts/extensions/toolcards/index.js`:
- ✅ Long-poll `/poll/<session_id>` event loop
- ✅ Reconcile-on-visibility-change via `/sessions?chat_id=X`
- ✅ `tool_progress[]` entry attached to caller assistant message
- ✅ `streamText` / `streamDone` for live raw-LLM transcripts
- ✅ `cancelToolSession(session_id)` global function (calls `/cancel`)

Already present in `public/script.js#messageFormatting` (~line 1790):
- ✅ Inline rendering of `tool_progress[]` as `<details>` collapsibles
- ✅ Status lines + transcript scopes nested per-entry

What's **missing**:

| # | Feature | Symptom today |
|---|---------|---------------|
| 3a | Cancel UI button | `cancelToolSession()` exists but no button; user can only cancel via console |
| 3b | Per-branch streaming for shape D | `tree-of-thoughts` shows one progress line; branches arrive all at once in the final result |
| 3c | True fire-and-forget for async cards | Action callback `await`s the result; even shape-C `async-lookup` blocks the parent's chat completion |

## Phase 3a — Cancel UI button

### Scope

Smallest piece. The plumbing is all there; just need a UI affordance.

### Implementation

1. **FE state** (`extensions/toolcards/index.js`):
   - Pass `sessionId` into `attachToCallerMessage` (already on the
     placeholder; just needs to be persisted on `entry.session_id`)
   - On `finalize`/`fail`, clear `entry.session_id` (so the cancel
     button disappears from terminal entries)

2. **Render** (`script.js#messageFormatting`):
   - When rendering a `tool_progress` entry with `status === 'running'`
     and `entry.session_id`, emit a small `✕ cancel` button
   - Button has `onclick="window.toolcardsCancelSession('<session_id>')"`
     (escaped session_id) — uses the existing global function

### Test (Playwright)

`tests/12_cancel_button.spec.js`:
- Use a slow card (the existing `async-lookup` with 6s sleep is perfect)
- Drive a real chat through the browser: ask scringlo to lookup X
- Wait for the tool_progress collapsible to render with status=running
- Click the cancel button
- Assert: the entry transitions to status=failed within ~2s
- Assert: a subsequent /sessions reconcile doesn't resurrect a stray
  result (i.e., cancel actually killed the descendant)

### Estimated complexity: small

## Phase 3b — Per-branch streaming for shape D

### Scope

`tree-of-thoughts` currently emits one `progress` line per phase
("dispatching N branches in parallel" → "synthesizing"). The branches
themselves arrive all at once in the result. For UX legibility we
want each branch to render as its own progress sub-entry, updating
as it lands.

### New event type

Extend the protocol with an optional `branch_progress` event from
service → plugin → FE:

```jsonc
{
    "type": "branch_progress",
    "branch_index": 0,
    "branch_label": "practical",
    "status": "started" | "complete",
    "summary": "<short one-liner, only on status=complete>",
    "reasoning": "<full text, only on status=complete>"
}
```

### Implementation

1. **Plugin** (`/start_invoke` event passthrough):
   - Add `branch_progress` to the recognized event types in
     `_handleProtoMessage` (currently handles `llm_call`, `progress`,
     `result`)
   - Just route it through to the FE poll queue, same as `progress`

2. **Service** (`tree-of-thoughts/service.py`):
   - When `parallel_llm_call` reads each `llm_response`, immediately
     emit a `{type:"branch_progress", branch_index, branch_label,
     status:"complete", summary, reasoning}` event before continuing
     to read the next response
   - Optionally emit `status:"started"` events upfront for all branches
     so the FE can render N empty placeholder slots immediately

3. **FE driver** (`extensions/toolcards/index.js`):
   - Extend the poll-loop event handler to recognize `branch_progress`
   - Add an `entry.branches[]` array on the `tool_progress` entry
   - Each `branch_progress` event upserts the matching slot by
     `branch_index`

4. **Render** (`script.js#messageFormatting`):
   - When `tool_progress` entry has `branches[]`, render them as
     a nested grid: each branch is a small panel with label, status
     icon (spinner / checkmark), and (when complete) a one-line
     summary expandable to full reasoning

### Test (Playwright)

`tests/13_branch_streaming.spec.js`:
- Direct-invoke `tree-of-thoughts` via browser-driven chat
- Set up SSE/poll inspector to watch events arrive
- Assert: 3 `branch_progress` events arrive BEFORE the final result
- Assert: each event has unique `branch_index`
- Assert: rendering shows 3 branch panels with summaries

### Estimated complexity: medium

## Phase 3c — True fire-and-forget for async cards

### Scope

The biggest behavioral change. Currently the action callback `await`s
the tool result, which blocks the parent's chat completion. For shape
C (and shape D when not needed inline) we want:

1. User asks scringlo a question that triggers a tool call
2. Action returns IMMEDIATELY with a placeholder ("looking that up;
   I'll let you know when it lands")
3. Model continues conversation; user can ask follow-up questions
4. When the real result lands later (seconds → minutes), it gets
   injected into chat as a new system message + the existing
   `tool_progress` entry on the caller bubble updates
5. On next user turn, the model sees the result in its prompt context
   and can reference it naturally

### Implementation

1. **Manifest extension**: add optional `async: true` flag at the tool
   level (per-tool granularity, since some cards have both sync and
   async tools)

2. **FE driver** (`extensions/toolcards/index.js`):
   - In the action callback, check `tool.async`
   - If async: kick off the session; return a placeholder string
     immediately like:
     `"Background ${tool.name} started for {args}; result will arrive separately."`
   - The poll loop continues running (no `await` on it from the
     action); when the result lands, the existing `finalize` path
     updates the `tool_progress` entry
   - **New**: on result land, ALSO push a new `is_system` chat message
     with the summary, marked with `extra.tool_async_result = true`
     and `extra.summary` for the prompt builder to pick up
   - Save chat so the new message persists

3. **Optional: auto-regenerate on result land**:
   - Behind a card-level `auto_continue: true` flag (off by default)
   - When the result lands, trigger a fresh assistant generation
     so the model proactively weaves in the answer rather than
     waiting for the user's next turn
   - Defer this; the manual-next-turn path is sufficient for the
     first cut

### Test (Playwright)

`tests/14_async_tool.spec.js`:
- Mark `async-lookup` as `async: true` (modify its manifest in the
  installer; or add a new `async-lookup-fire` variant)
- Drive a chat: user asks scringlo to lookup X
- Assert: model's response arrives within ~3s (NOT blocked on the
  6s descendant sleep)
- Assert: model's response contains a placeholder phrase
- Wait ~10s
- Assert: a new `is_system` message has appeared with the result
- User asks a followup
- Assert: model's response references the result (the model saw
  it in context on this turn)

### Estimated complexity: large

## Execution order

1. **3a first** — smallest, validates the FE-modification + Playwright-
   browser-test pattern end to end. Ground truth for whether codex can
   reliably modify this file set with passing tests.
2. **3b next** — medium scope. Builds on 3a's testing harness.
3. **3c last** — biggest behavioral change. By the time we get here
   we'll have validated the toolchain works.

Each phase as a separate commit so it can be reverted if it breaks
something user-facing. After 3c, the patterns doc gets a "Phase 3
complete" note removing the deferred-FE-work caveats.

## Out of scope (deferred to Phase 4+)

- **Reconciling the OpenAI tool_call / tool_result handshake under
  async**: when action returns a placeholder and the real result
  lands as a system message, the original tool_call has its tool_result
  set to the placeholder string. The model can still reason about
  the system message but the tool_call/tool_result pairing in the
  prompt is "lying" (the result was a placeholder, not the real
  answer). Acceptable for v1 but worth revisiting.
- **Persistent task queue / cross-chat results**: if the user closes
  the chat and reopens later, the result reconciles via the
  existing `/sessions` path. But what if the user navigates to a
  DIFFERENT chat — should the result follow? Probably no, but the
  question deserves an explicit answer.
- **Branch-level cancellation**: cancelling a single branch in
  shape-D mid-flight (vs. cancelling the whole session). The plugin
  protocol doesn't currently support per-llm_call cancel. Defer.
- **Parallel `tool_progress` rendering for shape D**: visually showing
  branches as a grid that reflows responsive to width. Phase 3b
  ships a baseline render; visual polish defers.
