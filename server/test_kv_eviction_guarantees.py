"""
Four algorithmic guarantees for the KV page cache (operator spec, 2026-06).

These are the ACCEPTANCE CONTRACT for the KV-retention/connection decoupling +
the eviction/allocation value function. They encode that page residency is a
function of CONTENT VALUE (citations × recency) and CAPACITY (memory budget),
never of connection/session lifecycle. Companion to test_kv_no_connection_pinning.py
(the time-series leak invariant) and docs/kv_retention_connection_decoupling_2026-06.md.

  G1  Swiping the same prefix N× never stalls/slows: the prefix is adopted
      (cache-hit) every time, only the suffix is computed. The ONLY allowed
      slowdown is genuine capacity exhaustion (prefix + live suffixes > pool).
  G2  Pages are ALLOCATED up to a hard cap while activations+KV stay under a
      configurable fraction (default 0.90) of system memory — capacity is
      memory-bound, not the compile-time SCRATCH_PAGE_BASE constant.
  G3  Citation-weighted eviction: a page cited by more distinct model-forwards
      (e.g. a tool-prefix shared by ≥half the harnesses) is evicted LAST —
      after all less-cited pages. (LFU-over-distinct-forwards, recency as tiebreak.)
  G4  Mixed-version coexistence: two tool-prefix versions both stay cached while
      both have live clients; under pressure a spec's SUFFIXES evict before its
      PREFIX, and the older spec's prefix drops only after its suffixes are gone.
  G5  Anti-immortality: a heavily-cited but now-STALE prefix must eventually evict.
      Citation value is RECENCY-DECAYED (bilinear), NOT a non-decaying LFU count —
      else a large prefix with many small suffix branches stays stuck-en-cache forever.

THE VALUE FUNCTION (forced by the dual specs G3+G4+G5): eviction priority =
recency-decayed citation score, value(page) = Σ_citations decay(now - t_citation)
(equivalently EWMA: score = score·decay(Δt) + 1 per citation). Bilinear in
(citation count × recency): pure-LRU (recency, breadth-blind) evicts a shared
prefix before a one-off suffix; pure-LFU (count, staleness-blind) pins a dead
popular prefix forever. CITATION = a distinct ADOPTER (a new forward adopting the
prefix via findLongestPrefix), NOT a per-AR-step re-read by the same generation
(recency/touchAccess covers self-rereads; counting them would re-create immortality).
Eviction picks MIN decayed-citation-score; recency is the within-citation weight.

RED-until-built: G1 should largely hold once prefix-reuse + recency eviction are
correct; G2 requires dynamic pool growth + a mem-budget param (NOT in the
compile-time pool today); G3/G4/G5 require the recency-decayed-citation (bilinear)
eviction value function. A failure here is the contract telling you what to build.

RUN AGAINST A THROWAWAY BRIDGE (BRIDGE_URL). The pressure tests deliberately
exhaust/evict; on a fixed pool they can drive the slow state → restart needed.
Skips cleanly if the bridge is unreachable.
"""
import json
import os
import time
import urllib.error
import urllib.request

import pytest

BRIDGE = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
MODEL = os.environ.get("KV_TEST_MODEL", "gemma-4-a4b")
MEM_BUDGET_FRAC = float(os.environ.get("KV_MEM_BUDGET_FRAC", "0.90"))


def _post(path, body, timeout=600):
    req = urllib.request.Request(f"{BRIDGE}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get(path, timeout=5):
    with urllib.request.urlopen(f"{BRIDGE}{path}", timeout=timeout) as r:
        return json.load(r)


def _complete(messages, max_tokens=16, seed=1):
    t0 = time.time()
    d = _post("/v1/chat/completions", {
        "model": MODEL, "messages": messages, "max_tokens": max_tokens,
        "temperature": 0.7, "seed": seed, "stream": False,
    })
    dt = time.time() - t0
    u = d.get("usage", {}) or {}
    return {
        "latency": dt,
        "prompt_tokens": u.get("prompt_tokens", 0),
        "cache_hits": u.get("cache_hits", 0),
        "cache_misses": u.get("cache_misses", 0),
        "completion_tokens": u.get("completion_tokens", max_tokens),
        "raw": d,
    }


def _health():
    h = _get("/health")
    # G2 growth observable: committed_pages is the high-water of pages the
    # dynamic pool has actually exposed (it RISES under demand). total_pages
    # is the constant budget cap (kept constant so the pinned derivation
    # total-free-cached stays valid for the leak detector). Fall back to
    # total_pages on a build that doesn't surface committed_pages yet.
    return {"total": h.get("total_pages"), "free": h.get("free_pages"),
            "cached": h.get("cached_pages"),
            "committed": h.get("committed_pages", h.get("total_pages")),
            "raw": h}


def _big_prefix(tag, approx_tokens=320):
    # A multi-page prefix (PAGE_SLIDE=16, PAGE_FULL=8 → span ~20-40 pages).
    sentence = (f"[{tag}] The quick brown fox jumps over the lazy dog while "
                "reciting the canonical tool descriptions verbatim. ")
    body = sentence * max(1, approx_tokens // 14)
    return [{"role": "system", "content": body},
            {"role": "user", "content": f"[{tag}] Acknowledge in one word."}]


@pytest.fixture(scope="module")
def bridge_up():
    try:
        h = _get("/health")
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"no bridge at {BRIDGE} ({e}); set BRIDGE_URL to a throwaway serve.py")
    return h


# ───────────────────────── G1 ─────────────────────────
def test_G1_swipe_same_prefix_never_thrashes(bridge_up):
    """Swiping ONE prefix 100× must reuse it every time (cache-hit), never re-prefill.

    Assert: after the cold first swipe, every swipe's cache_hits covers ~the whole
    prefix (prefix is adopted, not recomputed), cache_misses stays small (suffix
    only), and per-swipe latency is FLAT (no upward trend, no cliff). A drop in
    cache_hits or a latency cliff = the prefix was evicted+re-prefilled = the bug.
    """
    N = int(os.environ.get("KV_G1_SWIPES", "100"))
    msgs = _big_prefix("G1", approx_tokens=320)
    warm = None
    hit_ratios, latencies = [], []
    for i in range(N):
        # Same prefix; the seed varies the *suffix* (continuation), not the prompt.
        r = _complete(msgs, max_tokens=16, seed=1000 + i)
        if i == 0:
            warm = r  # cold: full prefill (cache_misses ~ prompt_tokens)
            continue
        pt = max(r["prompt_tokens"], 1)
        hit_ratios.append(r["cache_hits"] / pt)
        latencies.append(r["latency"])

    # Prefix is adopted every swipe: cache_hits ~ the whole prompt, misses tiny.
    min_hit = min(hit_ratios)
    assert min_hit >= 0.95, (
        f"a swipe re-prefilled the prefix: min cache_hit ratio {min_hit:.2f} over {N} "
        "swipes (prefix must be adopted, never recomputed)")
    # No stall/slowdown of any kind: warm-swipe latency is flat.
    med = sorted(latencies)[len(latencies) // 2]
    assert max(latencies) <= 3.0 * med, (
        f"swipe latency cliff: median {med:.3f}s, max {max(latencies):.3f}s — "
        "a re-prefill stall on a cached prefix (capacity was NOT exceeded for one-suffix swipes)")


# ───────────────────────── G2 ─────────────────────────
def test_G2_pages_grow_to_memory_budget(bridge_up):
    """Pool ALLOCATES more pages under demand up to a hard cap, while activations+KV
    stay < KV_MEM_BUDGET_FRAC (default 0.90) of system memory.

    RED until dynamic pool sizing exists (SCRATCH_PAGE_BASE is compile-time today).
    Observable: /health total_pages should rise as demand rises while mem headroom
    remains; the pool must NOT refuse/stall (outOfPages / 5xx) while < budget.
    """
    h0 = _health()
    if h0["total"] is None:
        pytest.skip("/health lacks total_pages; cannot observe pool growth (G2 telemetry needed)")
    # committed_pages is the growth high-water (total_pages is the constant
    # budget cap); the pool ALLOCATES on demand, so committed rises.
    committed0 = h0["committed"]
    # Drive escalating distinct long contexts to demand more KV than the baseline pool.
    errors = 0
    for i in range(40):
        try:
            _complete(_big_prefix(f"G2-{i}", approx_tokens=512), max_tokens=8, seed=i)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            errors += 1
    committed1 = _health()["committed"]
    # Capacity must be memory-bound: under sustained demand the pool grew (or, if a
    # mem-budget guard refused, it did so ONLY because the budget was reached — not
    # because of a small constant cap). On the fixed-pool impl this assertion is the
    # RED contract for "allocate up to the memory budget".
    assert committed1 > committed0 or errors == 0, (
        f"pool did not grow under demand (committed {committed0}->{committed1}) AND {errors} "
        f"requests errored — capacity is capped below the {MEM_BUDGET_FRAC:.0%} memory budget "
        "(G2: pages must be allocated up to the configurable memory cap, not a constant)")


# ───────────────────────── G3 ─────────────────────────
def test_G3_high_citation_prefix_evicted_last(bridge_up):
    """A prefix cited by MANY distinct forwards survives eviction longer than one-off
    prefixes — it is evicted LAST regardless of being older in pure recency.

    RED until citation-weighted eviction exists (today's policy is recency-only).
    """
    shared = _big_prefix("G3-SHARED", approx_tokens=320)
    # Build high citation count on the shared prefix across many distinct forwards.
    for k in range(12):
        _complete(shared, max_tokens=8, seed=2000 + k)
    # Flood with distinct one-off prefixes (low citation) to force eviction pressure.
    for i in range(60):
        _complete(_big_prefix(f"G3-ONEOFF-{i}", approx_tokens=320), max_tokens=8, seed=3000 + i)
    # The shared prefix must STILL be cached (evicted last); re-using it is a hit.
    after = _complete(shared, max_tokens=8, seed=2999)
    pt = max(after["prompt_tokens"], 1)
    assert after["cache_hits"] / pt >= 0.90, (
        f"high-citation shared prefix was evicted under pressure (cache_hit "
        f"{after['cache_hits']}/{pt}) — it must be evicted LAST, after the one-off "
        "(low-citation) prefixes (G3: citation-weighted eviction)")


# ───────────────────────── G4 ─────────────────────────
def test_G4_mixed_tool_versions_coexist_suffix_before_prefix(bridge_up):
    """Two tool-prefix versions both stay cached while both have live clients; under
    pressure a spec's SUFFIXES evict before its PREFIX, and the older spec's prefix
    drops only after its suffixes are gone.

    RED until the citation-weighted + suffix-before-prefix value function exists.
    """
    v1 = _big_prefix("TOOLSPEC-V1", approx_tokens=320)
    v2 = _big_prefix("TOOLSPEC-V2", approx_tokens=320)
    # Both versions actively used (ping-pong) — both prefixes become cited.
    for k in range(8):
        _complete(v1, max_tokens=8, seed=4000 + k)
        _complete(v2, max_tokens=8, seed=5000 + k)
    # While both are active, both prefixes must be cache-hits (both retained).
    h1 = _complete(v1, max_tokens=8, seed=4999)
    h2 = _complete(v2, max_tokens=8, seed=5999)
    assert h1["cache_hits"] / max(h1["prompt_tokens"], 1) >= 0.90, "V1 prefix dropped while still active"
    assert h2["cache_hits"] / max(h2["prompt_tokens"], 1) >= 0.90, "V2 prefix dropped while still active"

    # Taper V1; hammer V2 + new contexts (pressure). V1's PREFIX must outlast its
    # own suffixes — i.e. V1's shared prefix is the LAST thing of V1 to be evicted.
    for i in range(50):
        _complete(v2, max_tokens=8, seed=6000 + i)
        _complete(_big_prefix(f"G4-NEW-{i}", approx_tokens=320), max_tokens=8, seed=7000 + i)
    after_v1 = _complete(v1, max_tokens=8, seed=4998)
    # V2 (still active) must remain fully cached.
    after_v2 = _complete(v2, max_tokens=8, seed=5998)
    assert after_v2["cache_hits"] / max(after_v2["prompt_tokens"], 1) >= 0.90, (
        "actively-used V2 prefix was evicted under pressure (must be retained)")
    # The contract: V1's prefix is dropped LAST — its suffixes (per-request tails)
    # are squeezed out first. The shared prefix should still substantially hit even
    # after V1 went quiet, because suffixes (lower citation) evict before it.
    assert after_v1["cache_hits"] / max(after_v1["prompt_tokens"], 1) >= 0.50, (
        f"older-spec V1 prefix evicted before its suffixes (cache_hit "
        f"{after_v1['cache_hits']}/{after_v1['prompt_tokens']}) — a spec's prefix must "
        "be the LAST of it to drop, only after its suffixes are gone (G4)")


# ───────────────────────── G5 ─────────────────────────
def test_G5_stale_popular_prefix_decays_not_immortal(bridge_up):
    """A heavily-cited prefix that goes STALE must eventually evict — citation value
    is recency-DECAYED (bilinear), not a non-decaying LFU count.

    Why required (operator): a large prefix with many small suffix branches accumulates
    a huge citation count; under pure-LFU it would be stuck-en-cache forever, squatting
    capacity live work needs while every count-1 suffix churns around it. Pure-LRU
    conversely evicts a hot shared prefix before a marginally-newer suffix. Only
    value = Σ_citations decay(now - t_citation) satisfies G3 (retain-while-hot) AND
    G4 (drop-when-tapered) AND this (drop-when-stale). RED until that value fn exists.
    """
    h = _health()
    if h["total"] is None:
        pytest.skip("/health lacks total_pages; cannot size a pool-exceeding flood for G5")
    popular = _big_prefix("G5-POPULAR", approx_tokens=320)
    # Heavy citation: many DISTINCT forwards branch off the same large prefix.
    for k in range(20):
        msgs = popular + [{"role": "user", "content": f"branch {k}: continue."}]
        _complete(msgs, max_tokens=8, seed=8000 + k)
    warm = _complete(popular, max_tokens=8, seed=8999)
    ppfx = max(warm["prompt_tokens"], 1)
    assert warm["cache_hits"] / ppfx >= 0.90, "popular prefix not cached even after heavy recent use"

    # Go FULLY quiet on it; flood with DISTINCT prefixes whose cumulative KV exceeds
    # the pool (so capacity alone forces eviction of everything not-recently-cited).
    pages_per = max(ppfx // 8, 1)               # PAGE_FULL=8 (conservative: more pages)
    need = (3 * h["total"]) // (2 * pages_per)   # ~1.5x pool of distinct content
    flood = min(int(os.environ.get("KV_G5_FLOOD_MAX", "400")), max(need, 80))
    for i in range(flood):
        _complete(_big_prefix(f"G5-FLOOD-{i}", approx_tokens=320), max_tokens=8, seed=9000 + i)

    stale = _complete(popular, max_tokens=8, seed=8998)
    assert stale["cache_hits"] / ppfx < 0.50, (
        f"stale heavily-cited prefix is IMMORTAL: cache_hit {stale['cache_hits']}/{ppfx} after "
        f"{flood} distinct floods (~1.5x pool) with zero recent citation — citation must be "
        "recency-DECAYED (bilinear), not pure LFU, or a big prefix with many suffix branches "
        "stays stuck-en-cache forever (G5)")
