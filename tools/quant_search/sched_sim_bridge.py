#!/usr/bin/env python3
"""Empirical simulation: characterize the bridge as a queueing problem.

The engine's kernel-ceiling aggregate is ~92 tok/s at full B=8 saturation.
The current bridge mediates this at ~57 tok/s. This simulator models the
bridge as a series of queues with parametric per-token / per-CB overheads,
to identify which bridge-side patterns block tokens vs which are noise.

Pipeline modeled:
    HTTP request submit
        → _submit_q.put         (asyncio queue ~50µs)
        → _submit_pump          (drains, calls to_thread(g.submit))
        → asyncio.to_thread     (T_threadpool ~5-10ms one-shot per dispatch)
        → engine intake queue   (lock-free push)
        → engine drains intake at top of poll loop  (microsecond)
        → engine runs prefill+AR CBs (~86ms each)
        → engine emits updates after each productive CB
        → gemma_poll returns to Python (T_threadpool ~5-10ms again)
        → _poll_pump fans updates to per-stream response_qs (asyncio queue ~50µs each)
        → handler awaits queue.get (per-token asyncio scheduling)
        → handler builds response (jsonification, dict construction)
        → HTTP response sent

Patterns to evaluate as INTERVENTIONS (vs current behavior):

  P1: poll deadline (100ms) — too short forces frequent to_thread round-trips.
      Sweep: 50ms, 100ms (current), 250ms, 1000ms, ∞ (only return on idle).
      Tradeoff: latency for non-stream SSE = poll_deadline.

  P2: per-token asyncio overhead — fanning each token to its response_q
      requires an event loop schedule. Current: ~50µs per put + get.
      Intervention: batch tokens per stream, fan a list-of-tokens once per
      few CBs. Tradeoff: latency seen by stream consumer.

  P3: stream=False handler holds connection — for the bench's 25-sec
      requests, this is fine for throughput (handler is just async-awaiting
      a queue), but request body parsing + JSON response build happen at
      the request-end boundary. Intervention: pre-build response shape;
      stream incrementally.

  P4: parse_rollout judge call competes — bench's _fire_one_rollout makes
      an eval bridge call, then a judge bridge call. The eval/judge calls
      are in series within one ticket but interleave across tickets; engine
      sees a steady flow. Should NOT be a bridge-side bottleneck — but
      handler overhead per request adds up if request count is high.

  P5: HTTP (httpx) keepalive / connection setup — bench config has
      max_keepalive_connections=0 (forced no-keepalive). Each request is
      a fresh TCP+TLS handshake. Intervention: keepalive=8.

  P6: asyncio.to_thread overhead per poll — each poll roundtrip does Python
      → C → Swift → C → Python. Sweep different poll cadences.

  P7: GIL contention from request handlers awaiting response_qs — on each
      poll, _poll_pump puts to N queues, then handlers race to get. With
      8 active streams, that's 8 puts and 8 awakes per poll.

This is queueing/bin-packing, not GPU work. The simulator runs in Python
in <1 sec for any param combo.
"""
from __future__ import annotations
import argparse
import dataclasses
from dataclasses import dataclass, field
import math
import random
import sys

# ─── Engine / bridge constants ────────────────────────────────────────────────
B = 8
T_AR_MS = 86.0          # measured AR-tick wall under prefill-priority fix
T_PREFILL_MS = 86.0
Q_TILE = 128
SIM_WALL_MS = 60_000.0


# ─── Bridge configurable knobs ────────────────────────────────────────────────
@dataclass
class BridgeKnobs:
    # asyncio.to_thread one-shot overhead per dispatch
    t_threadpool_dispatch_ms: float = 5.0
    # asyncio queue.put / queue.get pair scheduling cost per token
    t_per_token_asyncio_us: float = 50.0
    # poll deadline that gates how long gemma_poll drives before returning
    poll_deadline_ms: float = 100.0
    # poll_pump → response_q put: per-update overhead
    t_per_update_async_put_us: float = 30.0
    # handler-side per-token overhead (queue.get + append)
    t_per_token_handler_us: float = 20.0
    # JSON response build at end (per-request, fixed cost)
    t_response_build_ms: float = 1.0
    # HTTP keepalive enabled? 0 = new connection per request
    keepalive: bool = False
    # HTTP per-request handshake cost when keepalive=False
    t_handshake_ms: float = 5.0
    # batch-fan-tokens: if True, _poll_pump puts ONE update-batch per stream
    # per poll instead of per-CB. Reduces asyncio overhead 8× at cost of
    # streaming responsiveness.
    batch_fan_tokens: bool = False


# ─── Simulation ───────────────────────────────────────────────────────────────
@dataclass
class BridgeResult:
    tokens_emitted: int = 0
    sim_wall_ms: float = 0.0
    ar_cbs_run: int = 0
    poll_calls: int = 0
    handler_calls: int = 0
    avg_request_latency_ms: float = 0.0
    bridge_overhead_ms_per_cb: float = 0.0

    @property
    def aggregate_tps(self) -> float:
        return 1000.0 * self.tokens_emitted / max(self.sim_wall_ms, 1.0)


def simulate(arrivals_per_sec: float,
              prompt_len_mean: int,
              completion_len_mean: int,
              knobs: BridgeKnobs,
              seed: int = 42,
              sim_wall_ms: float = SIM_WALL_MS) -> BridgeResult:
    """Run the bridge queueing simulation.

    The engine itself is modeled as: at each poll iteration, it can run up
    to floor(poll_deadline / T_AR_MS) AR ticks back-to-back without Python
    intervention. After each poll returns, Python overhead applies:
      - to_thread dispatch overhead (T_threadpool_dispatch_ms)
      - per-update async puts to response_qs (T_per_update_async_put_us per update)
      - poll_pump itself awaits queue (negligible)

    Active sessions are modeled as B slots; each slot in `.generating`
    state contributes 1 token per AR tick.
    """
    rng = random.Random(seed)
    t_ms = 0.0
    next_arrival_ms = 0.0
    handler_count = 0
    poll_count = 0
    tokens_total = 0
    cbs_total = 0
    handler_latencies_ms: list[float] = []

    # B engine slots: each is None or a dict {prompt_left, completion_left, arrival_t}
    slots: list[dict | None] = [None] * B
    pending: list[dict] = []   # request queue at the bridge (HTTP submitted, not yet in slot)

    while t_ms < sim_wall_ms:
        # Generate any arrivals up to now
        while next_arrival_ms <= t_ms:
            req = {
                "prompt_left": max(1, int(rng.expovariate(1.0 / prompt_len_mean))),
                "completion_left": max(1, int(rng.expovariate(1.0 / completion_len_mean))),
                "arrival_t": next_arrival_ms,
                "tokens_emitted": 0,
            }
            handshake = 0.0 if knobs.keepalive else knobs.t_handshake_ms
            req["effective_arrival_t"] = next_arrival_ms + handshake
            pending.append(req)
            handler_count += 1
            if arrivals_per_sec <= 0:
                next_arrival_ms = sim_wall_ms + 1e9
            else:
                next_arrival_ms += rng.expovariate(arrivals_per_sec / 1000.0)

        # Admit pending into free slots
        for i in range(B):
            if slots[i] is None and pending and pending[0]["effective_arrival_t"] <= t_ms:
                slots[i] = pending.pop(0)

        # Decide path: if any slot has prompt_left > 0, run prefill (multi
        # if 2+, else single). Else run AR. Mirrors the lm_engine
        # prefill-priority rule we just landed.
        n_priming = sum(1 for s in slots if s is not None and s["prompt_left"] > 0)
        n_generating = sum(1 for s in slots if s is not None and s["prompt_left"] == 0
                            and s["completion_left"] > 0)
        if n_priming + n_generating == 0:
            # No work — advance to next arrival
            t_ms = max(t_ms + 1.0, next_arrival_ms)
            continue

        # Drive 1 poll's worth of CBs back-to-back (engine's drive loop)
        poll_drive_start = t_ms
        deadline = t_ms + knobs.poll_deadline_ms
        cbs_this_poll = 0
        tokens_this_poll = 0
        # In this poll, alternate: any prefill needed → prefill CB; else AR
        while t_ms < deadline:
            n_priming = sum(1 for s in slots if s is not None and s["prompt_left"] > 0)
            n_generating = sum(1 for s in slots if s is not None
                                and s["prompt_left"] == 0 and s["completion_left"] > 0)
            if n_priming == 0 and n_generating == 0:
                break
            if n_priming > 0:
                # Multi-prefill if 2+, else single — mirror engine
                multi = n_priming >= 2
                for i, s in enumerate(slots):
                    if s is None: continue
                    if s["prompt_left"] > 0 and (multi or sum(1 for ss in slots[:i+1]
                                                                if ss and ss["prompt_left"] > 0) == 1):
                        chunk = min(Q_TILE, s["prompt_left"])
                        s["prompt_left"] -= chunk
                t_ms += T_PREFILL_MS
                cbs_this_poll += 1
            else:
                # AR step: each generating slot emits 1 token
                for i, s in enumerate(slots):
                    if s is None: continue
                    if s["completion_left"] > 0 and s["prompt_left"] == 0:
                        s["completion_left"] -= 1
                        s["tokens_emitted"] += 1
                        tokens_total += 1
                        tokens_this_poll += 1
                        if s["completion_left"] == 0:
                            # Finished
                            handler_latencies_ms.append(t_ms - s["arrival_t"] + knobs.t_response_build_ms)
                            slots[i] = None
                t_ms += T_AR_MS
                cbs_this_poll += 1
        cbs_total += cbs_this_poll

        # End of poll: pay Python overhead
        # - to_thread dispatch (one-shot per poll)
        # - asyncio queue.put per active stream this poll (or once if batched)
        n_active_streams = sum(1 for s in slots if s is not None)
        if knobs.batch_fan_tokens:
            n_puts = n_active_streams  # one batched update per stream per poll
        else:
            n_puts = n_active_streams * cbs_this_poll  # one update per stream per CB
        bridge_overhead_ms = (knobs.t_threadpool_dispatch_ms
                               + n_puts * (knobs.t_per_update_async_put_us / 1000.0)
                               + tokens_this_poll * (knobs.t_per_token_handler_us / 1000.0))
        t_ms += bridge_overhead_ms
        poll_count += 1

    avg_latency = sum(handler_latencies_ms) / max(len(handler_latencies_ms), 1)
    return BridgeResult(
        tokens_emitted=tokens_total,
        sim_wall_ms=t_ms,
        ar_cbs_run=cbs_total,
        poll_calls=poll_count,
        handler_calls=handler_count,
        avg_request_latency_ms=avg_latency,
        bridge_overhead_ms_per_cb=(t_ms - cbs_total * T_AR_MS) / max(cbs_total, 1),
    )


# ─── Sweeps ───────────────────────────────────────────────────────────────────
def main():
    print(f"# Bridge queueing simulation, sim_wall={SIM_WALL_MS/1000:.0f}s, "
          f"B={B}, T_AR={T_AR_MS}ms")
    print(f"# Engine ceiling (full saturation): {1000*B/T_AR_MS:.0f} tok/s")
    print()

    base = BridgeKnobs()
    fixed_workload = dict(arrivals_per_sec=2.0, prompt_len_mean=128,
                           completion_len_mean=200)

    print("## P1: poll_deadline_ms (longer = fewer to_thread round-trips)")
    print(f"  fixed: arrival=2/s, prompt=128, comp=200, batch_fan={base.batch_fan_tokens}")
    print()
    print(f"  {'deadline':>10}  {'tps':>7}  {'cbs':>6}  {'polls':>6}  "
          f"{'overhead/CB':>12}  {'avg lat (ms)':>14}")
    print(f"  {'-'*10}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*12}  {'-'*14}")
    for dl in (50, 100, 250, 500, 1000, 5000):
        k = dataclasses.replace(base, poll_deadline_ms=float(dl))
        r = simulate(knobs=k, **fixed_workload)
        print(f"  {dl:>10}  {r.aggregate_tps:>7.1f}  {r.ar_cbs_run:>6}  "
              f"{r.poll_calls:>6}  {r.bridge_overhead_ms_per_cb:>11.2f}ms  "
              f"{r.avg_request_latency_ms:>13.0f}ms")
    print()

    print("## P2: per-token asyncio overhead (handler queue.get cost)")
    print(f"  fixed: poll=100ms, arrival=2/s, prompt=128, comp=200")
    print()
    print(f"  {'per-token µs':>14}  {'tps':>7}  {'overhead/CB':>12}")
    print(f"  {'-'*14}  {'-'*7}  {'-'*12}")
    for usec in (10, 20, 50, 100, 200, 500):
        k = dataclasses.replace(base, t_per_token_handler_us=float(usec))
        r = simulate(knobs=k, **fixed_workload)
        print(f"  {usec:>14}  {r.aggregate_tps:>7.1f}  {r.bridge_overhead_ms_per_cb:>11.2f}ms")
    print()

    print("## P5: keepalive (TCP/TLS handshake cost per request)")
    print(f"  fixed: poll=100ms, arrival=2/s, prompt=128, comp=200")
    print()
    print(f"  {'config':>20}  {'tps':>7}  {'avg lat (ms)':>14}")
    print(f"  {'-'*20}  {'-'*7}  {'-'*14}")
    for label, ka, hs in [("no-keepalive (current)", False, 5.0),
                            ("no-keepalive (10ms)", False, 10.0),
                            ("keepalive", True, 0.0),
                            ("keepalive + 0ms HS", True, 0.0)]:
        k = dataclasses.replace(base, keepalive=ka, t_handshake_ms=hs)
        r = simulate(knobs=k, **fixed_workload)
        print(f"  {label:>20}  {r.aggregate_tps:>7.1f}  {r.avg_request_latency_ms:>13.0f}ms")
    print()

    print("## P6: t_threadpool_dispatch_ms (asyncio.to_thread overhead)")
    print(f"  fixed: poll=100ms, arrival=2/s, prompt=128, comp=200")
    print()
    print(f"  {'tt_ms':>7}  {'tps':>7}  {'overhead/CB':>12}")
    print(f"  {'-'*7}  {'-'*7}  {'-'*12}")
    for tt in (1, 5, 10, 25, 50):
        k = dataclasses.replace(base, t_threadpool_dispatch_ms=float(tt))
        r = simulate(knobs=k, **fixed_workload)
        print(f"  {tt:>7}  {r.aggregate_tps:>7.1f}  {r.bridge_overhead_ms_per_cb:>11.2f}ms")
    print()

    print("## P7: ARCHITECTURAL — single native FFI owner thread vs asyncio.to_thread")
    print("   Models codex findings (1)+(2)+(3) collapsed: one dedicated thread")
    print("   owns submit+poll, calls FFI directly (no to_thread hop), uses")
    print("   loop.call_soon_threadsafe to push to async queues.")
    print()
    print(f"  {'config':>40}  {'tps':>7}  {'overhead/CB':>12}")
    print(f"  {'-'*40}  {'-'*7}  {'-'*12}")
    # Current: asyncio.to_thread overhead = 5ms/poll, plus per-stream put per CB
    cur = BridgeKnobs(poll_deadline_ms=100, t_threadpool_dispatch_ms=5.0,
                       batch_fan_tokens=False)
    r = simulate(knobs=cur, **fixed_workload)
    print(f"  {'CURRENT (asyncio.to_thread × 2 tasks)':>40}  "
          f"{r.aggregate_tps:>7.1f}  {r.bridge_overhead_ms_per_cb:>11.2f}ms")
    # Native thread: dispatch overhead → ~0.5ms (loop.call_soon_threadsafe), longer poll
    nat = BridgeKnobs(poll_deadline_ms=1000, t_threadpool_dispatch_ms=0.5,
                       t_per_update_async_put_us=5.0, batch_fan_tokens=False)
    r = simulate(knobs=nat, **fixed_workload)
    print(f"  {'NATIVE owner thread, poll=1000ms':>40}  "
          f"{r.aggregate_tps:>7.1f}  {r.bridge_overhead_ms_per_cb:>11.2f}ms")
    # Plus batch fan (codex #2): one put per stream per poll, not per CB
    nat_b = BridgeKnobs(poll_deadline_ms=1000, t_threadpool_dispatch_ms=0.5,
                        t_per_update_async_put_us=5.0, batch_fan_tokens=True)
    r = simulate(knobs=nat_b, **fixed_workload)
    print(f"  {'+ batch fan-out (one put/stream/poll)':>40}  "
          f"{r.aggregate_tps:>7.1f}  {r.bridge_overhead_ms_per_cb:>11.2f}ms")
    print()
    # Sensitivity: what if real overhead is much higher (GIL contention,
    # httpx connection setup serialized via GIL, etc.)?
    print("## P8: sensitivity — if production overhead is HIGHER than sim modeled")
    print("   (real bridge measures ~57 tok/s vs sim CURRENT ~78 — 21 tok/s gap")
    print("   is unmodeled GIL/httpx/json/print/asyncio cost. Show effect at")
    print("   higher overhead values.)")
    print()
    print(f"  {'overhead/CB scenario':>40}  {'tps':>7}")
    print(f"  {'-'*40}  {'-'*7}")
    for label, tt, pt in [
        ("optimistic (sim baseline)", 5.0, 50.0),
        ("realistic (GIL+IO)", 25.0, 200.0),
        ("pessimistic (TCP per req)", 50.0, 500.0),
        ("native thread, low overhead", 0.5, 10.0),
        ("native thread, optimal", 0.2, 5.0),
    ]:
        k = BridgeKnobs(poll_deadline_ms=100, t_threadpool_dispatch_ms=tt,
                        t_per_token_handler_us=pt)
        r = simulate(knobs=k, **fixed_workload)
        print(f"  {label:>40}  {r.aggregate_tps:>7.1f}")
    print()

    print("## P-combined: best of all interventions vs current")
    print(f"  fixed workload: arrival=2/s, prompt=128, comp=200")
    print()
    configs = [
        ("CURRENT (poll=100, no-KA, fan-per-CB)",
         BridgeKnobs(poll_deadline_ms=100, keepalive=False, batch_fan_tokens=False)),
        ("poll=500 only",
         BridgeKnobs(poll_deadline_ms=500, keepalive=False, batch_fan_tokens=False)),
        ("poll=500 + keepalive",
         BridgeKnobs(poll_deadline_ms=500, keepalive=True, batch_fan_tokens=False,
                      t_handshake_ms=0.0)),
        ("poll=500 + KA + batch_fan",
         BridgeKnobs(poll_deadline_ms=500, keepalive=True, batch_fan_tokens=True,
                      t_handshake_ms=0.0)),
        ("poll=1000 + KA + batch_fan + low-overhead",
         BridgeKnobs(poll_deadline_ms=1000, keepalive=True, batch_fan_tokens=True,
                      t_handshake_ms=0.0, t_threadpool_dispatch_ms=2.0,
                      t_per_token_handler_us=10.0,
                      t_per_update_async_put_us=10.0)),
    ]
    print(f"  {'label':>45}  {'tps':>7}  {'CBs':>6}  {'polls':>6}  {'overhead/CB':>12}")
    print(f"  {'-'*45}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*12}")
    for label, k in configs:
        r = simulate(knobs=k, **fixed_workload)
        print(f"  {label:>45}  {r.aggregate_tps:>7.1f}  {r.ar_cbs_run:>6}  "
              f"{r.poll_calls:>6}  {r.bridge_overhead_ms_per_cb:>11.2f}ms")
    print()


if __name__ == "__main__":
    main()
