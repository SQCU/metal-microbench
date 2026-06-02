"""
KV retention ⊥ connection lifecycle — the time-series invariant.

THE CONTRACT (operator, 2026-06-02):
  Pinned (un-evictable) KV pages are a function of the ACTIVE WORKING SET,
  never of the number of connections/turns. Every turn in a multi-turn session
  is a *different connection*; the old impl created a pin event per connection
  ("for no reason"), so pinned grew ~linearly with turn/request count, exhausted
  the pool in <100 requests, and throughput cliffed ~100x and NEVER recovered.
  The correct impl frees used pages *eventually under page pressure* (recency
  eviction over a content-addressed cache), so pinned stays bounded and cold
  pages are reclaimed.

WHY NO TEST CAUGHT THIS (operator point #2): the existing playwright/pytest
suites assert per-request CONTENT correctness; none sample the engine's
page/session state ACROSS many connections to see the monotonic pin growth.
This file is that detector. It needs no new instrumentation — it reads the
existing /health telemetry:

    pinned_pages   = total_pages - free_pages - cached_pages      # refcount>0
    resident_sessions  (engine FFI outResidentCount, surfaced in /health)

DISCRIMINATORS (each independently fails the OLD impl, passes the NEW):
  T1  pinned_pages does NOT trend upward with request index (slope≈0); its max
      is bounded by the working set, NOT by M.                   [the leak]
  T2  resident_sessions stays bounded (≈ concurrent live gens), not climbing to
      MAX_RESIDENT_SESSIONS with M.                              [the leak, direct]
  T3  per-request throughput (tok/s) of request M ≈ request 1 (no >Nx cliff);
      and no outOfPages / 5xx.                          [the 100x-never-recovers]
  T4  MULTI-TURN (the named scenario): T turns in ONE growing conversation, each
      a fresh connection. Later turns are cache HITS (usage.cache_hits>0 →
      warmth survives without pinning), AND pinned after turn k scales with
      CONTEXT LENGTH, not with k (turn count).   [per-turn-pin-for-no-reason]

RUN AGAINST A THROWAWAY BRIDGE. On the OLD impl this test deliberately drives the
pool into the permanent slow state, which requires a bridge restart — point it at
a dedicated `serve.py` instance via BRIDGE_URL, never the precious live bridge.
Skips cleanly if BRIDGE_URL is unreachable so CI without an engine is unaffected.
"""
import json
import os
import time
import urllib.error
import urllib.request

import pytest

BRIDGE = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("KV_PIN_TEST_MODEL", "gemma-4-a4b")
M = int(os.environ.get("KV_PIN_TEST_M", "150"))          # > the <100 leak threshold
TURNS = int(os.environ.get("KV_PIN_TEST_TURNS", "40"))   # multi-turn depth
CLIFF_FACTOR = float(os.environ.get("KV_PIN_TEST_CLIFF", "5.0"))  # generous: bug is ~100x


def _get(path, timeout=5):
    with urllib.request.urlopen(f"{BRIDGE}{path}", timeout=timeout) as r:
        return json.load(r)


def _health():
    # /health is the engine-wide snapshot; tolerate field-name drift.
    h = _get("/health")
    fp = h.get("free_pages")
    cp = h.get("cached_pages")
    tp = h.get("total_pages")
    if tp is None and fp is not None and cp is not None:
        # derived below from a clean baseline (pinned≈0 at rest)
        tp = None
    resident = h.get("resident_sessions", h.get("resident_count",
               h.get("active_stream_count")))
    return {"free": fp, "cached": cp, "total": tp, "resident": resident, "raw": h}


def _complete(messages, max_tokens=48, seed=1):
    body = json.dumps({
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.7, "seed": seed, "stream": False,
    }).encode()
    req = urllib.request.Request(f"{BRIDGE}/v1/chat/completions", data=body,
                                 headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    dt = time.time() - t0
    usage = d.get("usage", {}) or {}
    n = usage.get("completion_tokens") or max_tokens
    text = (d.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
    return {"latency": dt, "tok_s": n / max(dt, 1e-6), "usage": usage,
            "text": text, "n": n}


@pytest.fixture(scope="module")
def bridge_up():
    try:
        h = _get("/health")
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"no bridge at {BRIDGE} ({e}); set BRIDGE_URL to a throwaway serve.py")
    if h.get("free_pages") is None and h.get("cached_pages") is None:
        pytest.skip("bridge /health lacks page telemetry; cannot observe pinning")
    return h


def _pinned(snap, total_baseline):
    if snap["free"] is None or snap["cached"] is None:
        return None
    total = snap["total"] or total_baseline
    return total - snap["free"] - snap["cached"]


def _slope(ys):
    # least-squares slope of y over index — positive = monotonic growth (leak)
    n = len(ys)
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    return num / den


def test_pinned_pages_bounded_not_proportional_to_connection_count(bridge_up):
    """T1+T2+T3: M independent connections must not grow pinned/resident or cliff throughput."""
    base = _health()
    # At rest, pinned≈0 → total ≈ free+cached. Capture once.
    total_baseline = (base["total"]
                      or ((base["free"] or 0) + (base["cached"] or 0)))
    pinned_series, resident_series, tok_s_series, errors = [], [], [], 0

    for i in range(M):
        try:
            r = _complete([{"role": "user",
                            "content": f"In one sentence, describe object number {i}."}],
                          seed=1000 + i)  # distinct seeds → IID, distinct tails
            tok_s_series.append(r["tok_s"])
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            errors += 1
        snap = _health()
        p = _pinned(snap, total_baseline)
        if p is not None:
            pinned_series.append(p)
        if snap["resident"] is not None:
            resident_series.append(snap["resident"])

    # T3: no hard failures (outOfPages surfaces as 5xx).
    assert errors == 0, f"{errors}/{M} completions errored (pool exhaustion / outOfPages)"

    # T3: throughput must not cliff. Compare last decile to first decile.
    k = max(1, M // 10)
    first = sorted(tok_s_series[:k])[len(tok_s_series[:k]) // 2]
    last = sorted(tok_s_series[-k:])[len(tok_s_series[-k:]) // 2]
    assert last >= first / CLIFF_FACTOR, (
        f"throughput cliff: first-decile median {first:.1f} tok/s -> "
        f"last-decile median {last:.1f} tok/s (>{CLIFF_FACTOR}x degradation = never-recovers)")

    # T1: pinned must not trend upward with connection count.
    if len(pinned_series) >= 20:
        slope = _slope(pinned_series)
        span = max(pinned_series) - min(pinned_series)
        # A leak shows a clear positive slope and a span that scales with M.
        # Bound: net pinned growth over the whole run < 10% of the pool.
        assert pinned_series[-1] - pinned_series[0] < 0.10 * total_baseline, (
            f"pinned grew {pinned_series[0]} -> {pinned_series[-1]} over {M} connections "
            f"(pool={total_baseline}); slope={slope:.2f}/req, span={span} "
            f"— pins accumulate with connection count (the leak)")

    # T2: resident sessions must not climb toward the cap with M.
    if len(resident_series) >= 20:
        assert resident_series[-1] - resident_series[0] <= 4, (
            f"resident_sessions climbed {resident_series[0]} -> {resident_series[-1]} "
            f"over {M} connections — sessions accumulate per connection (the leak)")


def test_multiturn_reuse_pins_by_context_not_by_turn(bridge_up):
    """T4: the named scenario — a growing multi-turn convo, each turn a new connection.

    Warmth must survive WITHOUT per-turn pinning: later turns hit the prefix cache,
    and pinned scales with context length, not with turn index.
    """
    base = _health()
    total_baseline = (base["total"] or ((base["free"] or 0) + (base["cached"] or 0)))

    convo = [{"role": "system", "content": "You are a terse assistant."}]
    pinned_after_turn, cache_hits_after_warm = [], []

    for t in range(TURNS):
        convo.append({"role": "user", "content": f"Turn {t}: give me one short fact."})
        r = _complete(convo, max_tokens=32, seed=7)
        convo.append({"role": "assistant", "content": r["text"][:200] or f"fact {t}"})
        ch = (r["usage"] or {}).get("cache_hits")
        if t >= 2 and ch is not None:
            cache_hits_after_warm.append(ch)
        snap = _health()
        p = _pinned(snap, total_baseline)
        if p is not None:
            pinned_after_turn.append(p)

    # Warmth: once the convo has history, later turns must adopt the prior prefix.
    if cache_hits_after_warm:
        assert max(cache_hits_after_warm) > 0, (
            "no cache_hits on later turns — prefix warmth lost "
            "(the trie must retain the growing prefix without a pinned session)")

    # No per-turn pin: pinned at the LAST turn must not be ~TURNS× the first turn's.
    # Context grows ~linearly per turn, so SOME growth is expected; a per-turn-pin
    # leak grows super-linearly (each turn re-pins the whole shared prefix).
    if len(pinned_after_turn) >= TURNS // 2 and pinned_after_turn[0] > 0:
        growth_ratio = pinned_after_turn[-1] / max(pinned_after_turn[0], 1)
        assert growth_ratio < 0.5 * TURNS, (
            f"pinned grew {growth_ratio:.1f}x over {TURNS} turns — scaling with TURN COUNT, "
            "not context (each turn re-pinned the shared prefix = the per-connection pin bug)")
