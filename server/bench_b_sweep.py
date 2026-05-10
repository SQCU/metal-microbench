#!/usr/bin/env python3
"""B-sweep bandwidth benchmark.

Two scenarios per B:
  (a) AR-dominated: B streams, short prompts (32 tok), long completions
      (256 tok). Most wall time is in AR steps. → measures AR tok/s ceiling.
  (b) Prefill-dominated: B streams with DISTINCT 1024-token prompts (no
      §2 dedup), 1 completion token each. Most wall time is in prefill.
      → measures prefill tok/s ceiling.

Each measurement reports:
  - aggregate tok/s (B × per-stream rate)
  - per-stream tok/s
  - avg engine step time (ms)
  - peak active_streams seen
  - resulting bandwidth utilization estimate

Run AT a specific B by editing bootstrap.swift's `let B = ` constant and
`MAX_B` in the 6 v4 GEMV kernels in kernels.swift, rebuild the dylib,
then run this script. Records results to bench_b_sweep_results.jsonl.
"""
from __future__ import annotations
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gemma_ffi as g  # noqa: E402

GGUF = os.environ.get(
    "GGUF_PATH",
    "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
)
RESULTS = os.environ.get(
    "BENCH_RESULTS",
    "/Users/mdot/metal-microbench/server/bench_b_sweep_results.jsonl")


def collect_all(stream_ids: list[int], deadline_s: float = 300.0):
    pending = set(stream_ids)
    finals: dict[int, g.StreamUpdate] = {}
    deadline = time.time() + deadline_s
    while pending and time.time() < deadline:
        for u in g.poll(timeout_ms=50):
            if u.stream_id in pending and u.state == 2:
                finals[u.stream_id] = u
                pending.discard(u.stream_id)
    return finals


def get_engine_b() -> int:
    """The engine's B is fixed at compile time; we infer it from the
    saturation behavior of an oversupply test or from a sentinel
    configured at startup. For now, accept it via env."""
    return int(os.environ.get("EXPECTED_B", "4"))


def scenario_ar(B: int):
    """AR-dominated: each stream has short distinct prompt and long
    completion. Measures AR throughput at full slot occupancy."""
    print(f"\n  [AR-dominated] B={B}, prompt=32 tok, completion=256 tok")
    sampling = g.SamplingParams(
        temperature=0.7, max_new_tokens=256, eos_token_id=106)
    streams = []
    base = 9000
    for i, sid in enumerate(range(base, base + B)):
        # Distinct prompts (different first-page hashes, no §2 dedup
        # masking the AR phase) so all B streams are independent AR
        # workers in steady state.
        prompt = [2] + [(100 + sid * 19 + j * 7) % 32000 for j in range(31)]
        streams.append(g.StreamSpec(
            stream_id=sid, action=0, tokens=prompt, sampling=sampling))

    pre_status = g.status()
    pre_steps = pre_status.total_steps
    pre_tokens = pre_status.total_tokens_emitted
    t0 = time.time()
    rc = g.submit(streams)
    if rc != 0:
        print(f"    FAIL submit rc={rc}"); return None
    finals = collect_all(list(range(base, base + B)), deadline_s=600.0)
    wall = time.time() - t0
    if len(finals) != B:
        print(f"    FAIL only {len(finals)}/{B} completed"); return None
    post_status = g.status()
    delta_steps = post_status.total_steps - pre_steps
    delta_tokens = post_status.total_tokens_emitted - pre_tokens

    completions = [u.completion_tokens_emitted for u in finals.values()]
    total_completion = sum(completions)
    aggregate_toks = total_completion / wall
    per_stream_toks = (total_completion / B) / wall
    avg_step_ms = (wall * 1000) / max(delta_steps, 1)

    return {
        "scenario": "ar_dominated", "B": B,
        "wall_s": wall,
        "total_completion_tokens": total_completion,
        "engine_steps": delta_steps,
        "avg_step_ms": avg_step_ms,
        "aggregate_tok_per_s": aggregate_toks,
        "per_stream_tok_per_s": per_stream_toks,
        "completions_per_stream": completions,
    }


def scenario_prefill(B: int):
    """Prefill-dominated: each stream has DISTINCT 1024-tok prompt, 1-tok
    completion. Most wall time is prefill of distinct content. No §2
    dedup since prompts diverge at first token."""
    print(f"\n  [Prefill-dominated] B={B}, prompt=1024 tok, completion=1 tok")
    sampling = g.SamplingParams(
        temperature=0.7, max_new_tokens=1, eos_token_id=106)
    streams = []
    base = 9100
    for i, sid in enumerate(range(base, base + B)):
        # 1024 distinct tokens per stream
        prompt = [2] + [(100 + sid * 23 + j * 11) % 32000 for j in range(1023)]
        streams.append(g.StreamSpec(
            stream_id=sid, action=0, tokens=prompt, sampling=sampling))

    pre_status = g.status()
    pre_steps = pre_status.total_steps
    pre_tokens = pre_status.total_tokens_emitted
    t0 = time.time()
    rc = g.submit(streams)
    if rc != 0:
        print(f"    FAIL submit rc={rc}"); return None
    finals = collect_all(list(range(base, base + B)), deadline_s=600.0)
    wall = time.time() - t0
    if len(finals) != B:
        print(f"    FAIL only {len(finals)}/{B} completed"); return None
    post_status = g.status()
    delta_steps = post_status.total_steps - pre_steps

    # Prefill tokens = sum of prompt lengths for all streams
    total_prefill = 1024 * B
    aggregate_prefill_toks = total_prefill / wall
    per_stream_prefill_toks = 1024 / wall
    avg_step_ms = (wall * 1000) / max(delta_steps, 1)

    return {
        "scenario": "prefill_dominated", "B": B,
        "wall_s": wall,
        "total_prefill_tokens": total_prefill,
        "engine_steps": delta_steps,
        "avg_step_ms": avg_step_ms,
        "aggregate_prefill_tok_per_s": aggregate_prefill_toks,
        "per_stream_prefill_tok_per_s": per_stream_prefill_toks,
    }


def main() -> int:
    print(f"=== B-sweep bandwidth benchmark ===")
    rc = g.init(GGUF)
    if rc != 0:
        print(f"  init failed: rc={rc}"); return 1

    B = get_engine_b()
    print(f"  expected engine B = {B}")

    # Run scenarios
    ar_result = scenario_ar(B)
    if ar_result:
        print(f"    aggregate AR throughput: {ar_result['aggregate_tok_per_s']:.1f} tok/s")
        print(f"    per-stream:              {ar_result['per_stream_tok_per_s']:.1f} tok/s")
        print(f"    avg step:                {ar_result['avg_step_ms']:.1f} ms")

    prefill_result = scenario_prefill(B)
    if prefill_result:
        print(f"    aggregate prefill throughput: {prefill_result['aggregate_prefill_tok_per_s']:.1f} tok/s")
        print(f"    per-stream:                   {prefill_result['per_stream_prefill_tok_per_s']:.1f} tok/s")
        print(f"    avg step:                     {prefill_result['avg_step_ms']:.1f} ms")

    # Append to results file for cross-B comparison
    with open(RESULTS, "a") as f:
        if ar_result:
            f.write(json.dumps({**ar_result, "ts": time.time()}) + "\n")
        if prefill_result:
            f.write(json.dumps({**prefill_result, "ts": time.time()}) + "\n")
    print(f"\n  results appended to {RESULTS}")
    g.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
