# Scalable oversight via n-of-k tool-call summaries

**Status**: Design spec + implementation plan. Companion to
`docs/forked_agent_patterns.md` and `docs/forked_agent_phase3_plan.md`.

## The pattern

Every tool call produces output of length k (tokens, lines, branches,
intermediate state, recursive sub-calls). The user-facing surface
must show n << k, where n is small enough that a human reader can
scan it in seconds. The compression must:

1. **Faithfully describe** what the tool did, in the parent persona's
   voice, so the parent agent can also use it as context on its
   next turn.
2. **Recurse** — when a tool's descendant agent invokes its own
   tools, those sub-tool's k-line activity gets compressed to its
   own n_sub lines. The parent tool sees `n_sub` lines per
   descendant call, not the descendant's raw k.
3. **Cost less than the work being summarized.** Summarization
   calls go through the same chat-completions backend as the tool
   itself; with prefix-cache-aware batching (already in the bridge),
   the per-summary marginal cost is ~50 max_tokens at decode time
   over a near-zero-cost prefill (shared persona+history prefix).

## The math

Let:
- `D` = recursion depth of tool calls (parent → descendant → grand-descendant → ...)
- `K` = raw output lines at each level
- `N` = compression ratio per level (e.g. N=10 means we summarize 10
  raw lines into 1 summary line)

Raw work across all levels: `sum_total = K * (1 + B + B² + ... + B^D)` where B is the branching factor (descendants per node).

Visible at parent level after summarization: `n_visible = K/N + (descendants' n) / N`. Solving recursively:

    n_visible = K/N + B * (K/N) / N + B * B * (K/N) / N²  + ...
              = (K/N) * sum_{i=0..D} (B/N)^i

When `B < N` (compression beats branching), this geometric sum
converges, and the parent sees `O(log_N(sum_total))` lines for
work that grows as `O(B^D)`. Concretely: with N=10 and B=3, total
work K^D=10000 lines compresses to ~5 visible summary lines.

The user said it as "log(sum_of_all_k_of_all_recursions)". Same
result; the constant factor is bounded by how aggressively we set
`N`, which is just a max_tokens cap on the summary llm_call.

## Cost model (cheaper than naive estimate)

The summary llm_call's cost looks like:
- **Prefill**: parent persona system prompt + recent chat history +
  the raw lines being summarized. At least the first two are
  identical to what the parent's own next response would prefill —
  so the bridge's content-hash KV page cache (`page_manager` in
  `lm_engine.swift`) hits and the prefill is ~free.
- **Decode**: ~50 tokens at temp 0.4 = ~1s wall time on the
  metal-microbench setup. Empirically (see `parallel_llm_call`
  in `tree-of-thoughts/service.py`): 3 concurrent llm_calls with
  shared-prefix run at ~2.64x speedup, so 3 summaries amortize to
  ~1.1s wall.

The total cost to produce `n_visible` summary lines is:
    `n_visible * decode_per_summary` (prefill is amortized)
≈ `O(log K_total)` decode chunks, each ~1s wall.

For comparison, the descendant's actual work is `O(K_total)` decode
chunks. Summarization adds a `log/linear` overhead — vanishing as
the workload grows.

## Why this matters

Multi-agent eval / harness / RL / supervision setups need to surface
descendant activity to a supervisor. Naive solutions either:
- Dump raw descendant chat logs to the supervisor (blows context),
  OR
- Hide descendant activity entirely (no oversight).

The n-of-k summary pattern is the third option: faithful, bounded,
auditable. The same primitive serves:
- User-facing UX (the human reader scans n lines instead of k)
- Supervisor agents (the parent persona reads n lines on its next
  turn and decides whether to intervene)
- Training data (n-line summaries are themselves a high-quality
  training signal — they're parent-voice descriptions of descendant
  actions, which is exactly the schema for an RL reward model that
  scores "did this descendant do something the parent endorses?")

The compute cost is dominated by the descendant's own work; the
oversight overhead is logarithmic. **If you can afford the
descendant work, you can afford the oversight, because the bridge
already has the prefix-cache-aware batching that makes it amortize.**

## Protocol extension: `summary_progress`

```jsonc
{
    "type": "summary_progress",
    "scope": "main" | "branch:0" | "iteration:3" | etc,
    "summary": "<1-line annotation in parent persona's voice>",
    "compressed_lines": K_segment,
    "cumulative_summary_count": N_so_far
}
```

Plugin pass-through is the same as `progress` and `branch_progress`
(forward to the FE poll queue).

FE driver: route to `entry.summary_trace[]` on the tool_progress
entry. Each summary line appears in the inline tool_progress UI as a
compact serial list, formatted distinctly from raw status_lines so
the user can read "what scringlo says is happening" separately from
"what the descendant emitted as raw status."

The summary_trace is also surfaced as a concatenated string in the
tool's `result.summary` field, so the parent's next-turn prompt
context includes the n-line trace as part of the role:"tool"
message content.

## Recursive composition

When a descendant tool itself emits `summary_progress` events, those
are forwarded up through the same plugin path to the FE. The PARENT
tool's service can either:
- Subscribe to its own descendants' summaries (via plugin extension)
  and roll them up into its own summary_trace, OR
- Let the FE flatten descendant trace into a nested view.

For v1 we go with the FE-flatten approach: the FE driver tracks
which descendant called which parent, and renders a nested
collapsible: parent's summary_trace contains links to descendants'
summary_traces.

Mathematically equivalent to the direct roll-up (each level
contributes its n lines), but visually the user sees a tree.

## Iterative SVG refinement as the demo workload

The existing `scripts/archival/svg_refinement_loop.py` is a
canonical high-k workload:

| Per-iteration output | Approximate lines |
|----------------------|------------------|
| Generated SVG code   | 30–60            |
| Render + MSE compute | 5                |
| Self-critique prose  | 10–30            |
| Plan for next iter   | 5                |
| **Total per iter**   | **~50–100**      |

For a 5-iteration run: K_total ≈ 250–500 lines.

The summary trace should be ~5–10 lines, in scringlo's voice:

> 💭 starting iter 0; the target's a green donut shape. drew the outer ring first, MSE landed at 0.14
>
> 💭 iter 1 added the inner ring. MSE got slightly worse (0.145) — i think the proportions drifted
>
> 💭 iter 2 went back to iter 0's outer ring + a smaller inner. MSE 0.12, better
>
> ...

That's ~10:1 compression at one level, no recursion. With recursion
(if each iteration's "self-critique" was a sub-tool that spawned
its own sub-eval), we'd get ~100:1 compression total.

## Implementation plan

### Phase 1: protocol + FE

- Plugin: pass through `summary_progress` events (one-line edit
  in `_handleProtoMessage`, same shape as `branch_progress`).
- FE driver: `ph.summaryAppend(event)` helper; appends to
  `entry.summary_trace[]`; rerenders.
- Render in `script.js#messageFormatting`: a new
  `tool-progress-summary-trace` block, distinct from status_lines
  and branches, formatted as a compact serial list with the
  parent-persona voice marker (💭 prefix or similar).

### Phase 2: extend tree-of-thoughts to emit summary_progress

The existing `tree-of-thoughts` card can immediately demo the
pattern:
- Each branch completion emits a `summary_progress` event in
  scringlo's voice that summarizes the branch's reasoning in 1
  sentence (different from the existing `branch_progress` which is
  the descendant's own SUMMARY: line — a different role).
- The synthesis emits one final summary_progress.

Net: 4 summary lines for ~2K chars of branch reasoning + synthesis.

### Phase 3: iter-svg-refine card

New shape-A LLM-augmented card that:
- Takes a target image (data URL or path) + max_iterations
- Wraps `svg_refinement_loop.py` logic in a service.py
- Per iteration: generates SVG, renders, computes MSE
- Per iteration: emits `summary_progress` with a 1-line
  scringlo-voice description of what just happened
- Final result: best SVG image (returned as image_url for inline
  display) + the full summary_trace concatenated

### Phase 4: vertical slice demo with video

Drive the test-16 pattern: Scringlo, fresh chat, user asks for an
SVG, tool fires, summary trace lands line-by-line as iterations
run, final image renders. Capture video, frame-validate, curate.

The video should clearly show:
- Tool collapsible expanding
- Summary trace appearing line-by-line as iterations complete
- Each summary in scringlo's voice (lowercase, playful, etc)
- Final SVG image embedded in the chat
- Total wall time: minutes, but the user can read what happened
  from ~5–10 summary lines

## What this proves (and what it doesn't)

**Proves**: The n-of-k summary pattern is buildable today on the
existing toolcards substrate, with logarithmic oversight overhead,
and is visually verifiable end-to-end via video evidence.

**Doesn't prove**: That the summaries are GOOD enough for high-stakes
oversight. Quality-of-summary is a separate axis: the n-line
compression's faithfulness depends on the summarizer model's
behavior, the parent persona's voice consistency, and prompt design.
We measure compression ratio and wall-time cost; we don't yet measure
"does a human supervisor catch the same problems by reading n lines
that they'd catch by reading k?" That study comes after we can
generate the n-line summaries reliably.

**Acknowledged but deferred**: The "scaleable oversight, not selected
for anthropic fellows cohort 2026 may" framing — multi-agent eval
ensembles using this primitive as the supervisor's input channel —
is a follow-on application. The plumbing here is the precondition.
