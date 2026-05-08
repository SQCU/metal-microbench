# Kernel throughput ceiling: where B=8 came from, what would lift it

**Status**: Reference notes for future kernel-shape work. Captures the
empirical findings from the 2026-04-28 MAX_B sweep, the 2026-05-07 M:K
saturation probe, and the architectural reasoning around why "raise B"
is NOT obviously the way to lift the ~135 tok/s aggregate AR ceiling.

**Audience**: future kernel-author session that wants to attack the AR
throughput ceiling without re-deriving the "why is B=8" history.

## TL;DR

- `let B = 8` in `bootstrap.swift:414` is the kernel batch width.
- The v4 GEMV kernels (10 sites in `kernels.swift`) hardcode
  `constexpr uint MAX_B = 8;` to match.
- B=8 was chosen empirically. The 2026-04-28 sweep
  (`kernels.swift:510-512`) found:

  | MAX_B | wall ms | vs V5 baseline (150 ms) |
  |---|---|---|
  | 8  | 106 | **−29%** (the win) |
  | 16 | 152 | +1% |
  | 32 | 251 | +67% |

  Above 8, register-pressure / occupancy loss erases the bandwidth-
  amortization win. This is hardware-specific (Apple M5 Max, currently)
  but has held across kernel refactors.

- After the M:K + Session-deletion + atomic-construction work landed,
  the engine reaches **100% kernel-position fill at M=8** (one busy
  resident per kernel position every CB). Aggregate AR throughput
  plateaus at ~135 tok/s at M≥8 (`probe_oversubscription.py`).

- The plateau **is the kernel ceiling**, not a scheduler ceiling. To go
  past 135 tok/s the kernel itself has to change shape — not its B.

## Where the B=8 wiring lives

CPU side:
- `bootstrap.swift:414` — `let B = 8` (canonical)
- `bootstrap.swift:416` — `let TOTAL_SLOTS = B * TOPK` (downstream sizing)
- All allocations sized as `B * HIDDEN`, `B * SLIDE_H * SLIDE_HD`, etc.
  start around `bootstrap.swift:450` and continue through ~line 500.

GPU side (kernels.swift, 10 sites):
| Line | Kernel | Notes |
|---|---|---|
| 474 | (q4_K MoE up GEMV variant) | accs[MAX_B] |
| 513 | (q5_1 MoE down v4) | The MAX_B sweep notes live here |
| 549 | (q8_0 attn-Q v4) | |
| 1180 | (q4_K dense FFN-up v4) | |
| 1261 | (q8_0 dense FFN-down v4) | |
| ... | (5 more) | All identical shape: `constexpr uint MAX_B = 8;` |

Each kernel's body uses `accs[MAX_B]` register-allocated accumulators
and tile-walks 32 output cols per dispatch. The `MAX_B` const is
template-friendly — could be parameterized via constant function args
if a kernel-shape variant study calls for it.

## Why "just bump B to 16 or 32" doesn't work

The naive intuition is: bigger kernel batch → more concurrency →
higher throughput. The empirical reality on Apple M5 Max:

**Register pressure**. Each MAX_B accumulator is a per-thread float.
At MAX_B=8, the kernel uses ~8 float regs per thread for accs plus
working set for K-tile reads, weight unpacks, etc. — comfortably
within the per-thread register budget that lets the SIMD-group
scheduler dispatch the maximum number of warps to the SM.

At MAX_B=16, register count climbs above the threshold where the
GPU can keep enough warps in flight to hide memory latency. Memory
stalls cease to overlap with compute on neighboring warps, and
effective bandwidth drops. The +1% measurement at MAX_B=16 is
consistent with "exactly enough register-spill that occupancy halves
but compute doesn't fully recover."

At MAX_B=32, occupancy collapses entirely. The 251 ms wall time is
nearly 2× MAX_B=16 even though each kernel call now does 2× the
nominal work — the scheduler is effectively running serialized.

**This is an Apple-Silicon-specific profile.** Different kernel
shapes on different hardware (e.g., Hopper or Blackwell with 256 KB
register files per SM) might have different sweet spots. Re-running
the sweep on a new GPU is a half-day's work, but B=8 should not
be assumed optimal there.

## What WOULD lift the 135 tok/s ceiling

Three classes of kernel work, in increasing order of effort:

### 1. Different MAX_B variants per kernel (low-medium)

Not all 10 GEMV kernels are equally register-pressured. The MoE
expert-down (q5_1) and the dense FFN-up (q4_K) have different
weight-unpack cost profiles. A targeted study of "is MAX_B=12 a
sweet spot for kernel X but MAX_B=8 for kernel Y" might extract
5-10% more throughput at zero structural cost.

The hard part: each kernel rewrite needs an empirical sweep on the
target hardware (~2 hours per kernel including kernel rebuild
cadence) to find its individual sweet spot. With 10 kernels that's
2 days of focused sweep time, plus the per-kernel constants
introduce shape-mismatch risk if not all kernels agree on B at the
batch-axis boundary. Worth doing if the goal is "extract every drop
on existing kernel design," skip if a deeper refactor is on the
table.

### 2. Split-K MoE (medium-high)

The MoE down-projection is the worst register-pressure case because
each token contributes to 8 expert outputs (TOPK=8) that are then
combined. The current kernel walks experts serially per token. A
split-K variant (one threadgroup per expert, atomic-add into combined
output) cuts per-thread register count enough to potentially run with
larger MAX_B without occupancy loss.

This is a **substantial** kernel rewrite: Metal atomic-add on fp16/bf16
isn't free, threadgroup memory layout has to change, the combine pass
becomes a separate kernel. Estimate: 1 week. Potential win: 2× MoE
throughput at scale; AR-step time would drop significantly because
MoE down dominates the per-step wall.

### 3. Tensor cores / SIMD-group matrix ops (high, currently impossible on M5)

Apple Silicon's `simdgroup_matrix` types (Metal 3.x) provide
hardware FMA on small matrix tiles. The current GEMV kernels don't
use them — the design predates `simdgroup_multiply_accumulate` being
broadly tuned. A kernel rewrite that uses these primitives could
deliver 2-4× per-kernel-invocation throughput on weight-bandwidth-
limited shapes like Q4_K dense FFN.

This is a **complete kernel-shape redesign**: the `MAX_B`
accumulator-array pattern goes away entirely, replaced by tile-by-
tile matrix-multiply-accumulate. Estimate: 2-3 weeks for one kernel
family (all attn variants, or all FFN variants). Potential win:
2-4× per-kernel; likely brings aggregate AR past 250 tok/s before
hitting the next ceiling (likely memory bandwidth on Q-decode for
the next-token feedback path).

## What the throughput probes can and can't measure

`server/probe_oversubscription.py` (the M:K saturation probe):
- ✓ Detects scheduler / pool / admission regressions
- ✓ Detects per-stream rate decline (1/M behavior)
- ✓ Confirms 100% kernel-slot fill at M=B
- ✗ Cannot distinguish "kernel ceiling" from "memory ceiling"

`server/probe_sustained_eval.py` (the rolling 5-minute eval shape):
- ✓ Detects accumulating-state slowdowns (polynomial in request count)
- ✓ Catches per-request latency long-tail growth
- ✗ Doesn't isolate kernel time from CPU-side overhead

`server/bench_b_sweep.py` (the AR / prefill split):
- ✓ Reports avg engine step time in ms — directly measures per-CB
  kernel wall
- ✗ Doesn't decompose by kernel (would need MTLCaptureScope
  instrumentation)

For kernel-ceiling diagnosis: run `bench_b_sweep` and watch `avg_step_ms`.
Today (2026-05-07) this should be ~80 ms for the AR-dominated path
at B=8. If a kernel rewrite takes that to 60 ms, aggregate tok/s
should rise from ~135 to ~180 in lockstep.

## Other levers (not kernel-shape)

The 135 tok/s number is dominated by AR step time. Things outside the
kernel that ALSO move that number:

**Scheduler-side bubble reduction**. The current per-CB picker
(2026-05-07) makes kernel-position fill 100% at M≥B. Bubbles only
appear at the prefill→AR transition (one CB of single-slot prefill
silences 7/8 positions). At sustained M=64 with continuous arrivals,
bubble fraction is <5% of total CBs.

**KV-write fusion**. Currently the AR step does two CBs: one for the
forward pass (slide+full attn → MoE → output projection), one for
KV-write of the new K/V. Fusing them halves the GPU-side wall
overhead. Substantial Metal CB-graph refactor. Out of scope unless
the kernel ceiling itself stops moving.

**Speculative decoding**. Run a smaller-cheaper draft model in
parallel with the verifier. AR steps that accept K draft tokens
deliver K tokens per step instead of 1. Implementation: would need a
second resident model (Gemma-4-2b or similar at the same quantization),
draft kernel path, verification logic. Estimate: 2 weeks. Potential:
2-4× tok/s if draft acceptance rate stays >50%. Wins are
*per-request*, not per-batch — so they compound differently with
oversubscription than batch-width changes.

## Open empirical questions worth a half-day each

1. **Does `simdgroup_matrix` actually FMA at full precision on M5?**
   30-min microbench will tell. If yes, makes the lever-3 rewrite
   far more attractive.

2. **Is the MAX_B=16 +1% slowdown caused by register spill, or by
   threadgroup occupancy halving?** A profile under MTLCaptureScope
   would distinguish these. If occupancy, threadgroup-size adjustment
   might reclaim it without dropping MAX_B.

3. **What's the actual memory-bandwidth utilization of a steady-state
   AR step?** If we're already at 80% of M5 bandwidth (≈1 TB/s), the
   ceiling is hardware not kernel and only quant-side moves help
   (Q4_K_M → Q3_K_M is the next quant step that meaningfully reduces
   weight bytes per step).

## When to revisit

Kernel-shape work is a focused multi-day push. Don't half-start it.
Revisit when:

- A new GPU generation arrives (M6, etc.) — re-run the MAX_B sweep
  before assuming the same B is optimal.
- Server load grows past what 135 tok/s aggregate can serve (i.e.,
  >100 concurrent requests sustained) — at that point split-K MoE
  and `simdgroup_matrix` become economically obvious.
- The MoE-down kernel comes up in a profile as the dominant cost —
  that's the highest-yield single-kernel target.

Until then, the engine is operating at its kernel ceiling and the
configuration audit (commit 2f730db) confirms there's no slumps left
between the API surface and the kernel ceiling.

## Cross-references

- `docs/kv_pool_split_spec.md` — the 8192-page Metal-buffer cliff
  (multi-session-at-64k blocker; orthogonal to the kernel ceiling).
- `kernels.swift:510-512` — the original MAX_B sweep note.
- `bootstrap.swift:414-416` — the canonical B definition.
- `server/probe_oversubscription.py` / `server/probe_sustained_eval.py`
  — the throughput-probe entry points referenced above.
- Recent commits that build the slot-utilization story:
  - `c446ad0` — M:K permutation layer
  - `f6cfedd` — Session deletion → stateless RequestRun shape
  - `2f730db` — long-context default audit
