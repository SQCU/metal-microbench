# Engine telemetry endpoint (`/v1/engine/state`)

The OpenAI-compatible chat-completions surface carries everything a
generic chat client needs — content stream, finish reason, token
counts, and (as extension fields the official spec doesn't define but
that don't break compliant parsers) the engine's per-completion
billing telemetry: `usage.cache_hits`, `usage.cache_misses`,
`usage.vision_cache_hits`. Streaming deltas carry the same in the
terminal usage chunk.

That covers the **per-completion** axis — "how did this one request
behave on the engine."

It does NOT cover the **engine-wide instantaneous** axis — "what is
the page pool doing right now, across all live streams." A chat
client doesn't care; a dashboard that visualizes KV cache page
tenancy, vision-cache size, active-stream registry across multiple
concurrent completions, or the moment-to-moment pulse of the
content-hash prefix cache absolutely does.

`/v1/engine/state` is the single endpoint that fills that gap. The
bespoke engine-internals visualizers at `server/static/` poll it at
1–2 Hz alongside their per-completion fetches.

## Why this is decoupled from a single completion

A completion's `usage` block is a property of one request's
trajectory. The engine-state snapshot is global: at the instant the
request was made, what was in the page cache? How many other streams
were live? How many phys pages are content-indexed for the next
sibling fork to adopt?

These two views answer different questions, and merging them onto the
chat-completions response would either (a) inflate every response
with state the typical client doesn't want, or (b) tie the
visualizer's update cadence to its own completion stream — exactly
the wrong cadence for a global dashboard that should keep ticking
while the user is idle between requests.

## Payload shape (current)

```json
{
  "kv_cache": {
    "total_pages": 8192,
    "free_pages": 8185,
    "cached_pages": 3,
    "pages_in_use": 7,
    "pages": [
      {
        "phys": 0,
        "refcount": 1,
        "promoted": true,
        "hash": "00000000e6597571",
        "pair_mate": 8191
      }
    ]
  },
  "vision_cache": {
    "entries": 0,
    "hits": 0,
    "misses": 0
  },
  "active_streams": [
    {
      "stream_id": 1,
      "state": "generating",
      "position": 15,
      "prompt_tokens_seen": 15,
      "completion_tokens_emitted": 4,
      "pages_owned": 7,
      "cache_hit_tokens": 0,
      "cache_miss_tokens": 15
    }
  ],
  "engine": {
    "total_steps": 14,
    "total_tokens": 4,
    "last_step_ms": 31.2,
    "max_b": 8
  }
}
```

Field notes:

- **`kv_cache.pages`** lists only pages with `refcount > 0` (currently
  held by some stream) OR a non-null `hash` (free but cached, eligible
  for adoption by the next stream with a matching prefix). Pages that
  have never been touched are omitted to keep the payload bounded; at
  full pool occupancy you still get ≤ 4096 records (a hard cap
  enforced in `PageManager.livePageSnapshot()`).
- **`promoted`** = `true` IFF this page has a content hash in the
  content-index. A page with `refcount > 0 && promoted == true` is
  shareable across streams; `refcount > 0 && promoted == false` is
  in-use but private (typically a fresh page mid-prefill, not yet
  hashed).
- **`hash`** is the FNV-1a digest of the page's tokens (16-token slide
  units), formatted as a lowercase 16-character hex string. Identical
  hashes ↔ identical tokens, so the visualizer can group siblings by
  hash to color "this page is shared across these three streams."
- **`pair_mate`** is the companion phys page in the
  slide-page/full-attention-pair structure (Gemma-4 PAGE_SLIDE=16,
  PAGE_FULL=8 share one block_table). Non-visualizer clients can
  ignore it.
- **`active_streams[].stream_id`** is the same internal ID the bridge
  uses for routing SSE updates; the visualizer can correlate stream
  state with the page-set indirectly via aggregate counts, but
  per-page owner tracking is not exposed (it doesn't exist in the
  anonymous-pool refactor; pages have refcount only, not an owner
  list).

## Polling cost characteristics

The underlying FFI call (`gemma_engine_state`) takes `gEngineLock`,
walks `requestForStream` (typically ≤ 8 entries) and the page array
(8192 slots, but the body of the loop only allocates JSON for the
non-idle subset), then releases. On M5 Max with the default 8192-page
pool the payload is < 256 KB worst case and the walk completes in
single-digit milliseconds. A 1–2 Hz poll from a browser dashboard is
well below the noise floor of the AR loop.

Do not poll from a hot path (e.g., from inside an SSE generator loop).
Use the same cadence a UI uses for animation: 250–500 ms is plenty.

## Why no per-page websocket / per-tick stream

The visualizers explicitly use polling rather than streaming. Two
reasons:

1. **Polling is simpler.** Re-rendering a tenancy strip every 500 ms
   from a single GET is enough latency-fidelity for "watch the cache
   adoption pulse" — page residency turns over on the second scale,
   not per-token. The complexity of an event channel doesn't earn its
   keep here.
2. **The completion stream already carries per-token deltas.** Token-
   wise side-channel data (e.g., per-token cvec activations for the
   steering visualizer) belongs on `choices[0].delta.*` in the SSE
   completion stream, not on a parallel channel. The completion
   stream is the natural place for anything that follows the token
   emission rate; engine-state is the natural place for anything that
   follows the engine's slower aggregate state.

## Extending the shape

When a visualizer needs a field genuinely outside completion
semantics that this endpoint doesn't already carry, add it here
rather than introducing a parallel endpoint. The payload shape is
free to grow — it's read by visualizers we own, not by external
clients. Per-token data goes on `delta`, never here.

## Source files

- **Swift exporter:** `ffi_batch.swift::gemma_engine_state` —
  serializes JSON under `gEngineLock`.
- **Page snapshot:** `page_manager.swift::PageManager.livePageSnapshot`
  — enumerates in-use + cached pages.
- **Python wrapper:** `server/gemma_ffi.py::engine_state` — re-parses
  the JSON bytes into a dict (the buffer doubles on `-ENOSPC`).
- **HTTP route:** `server/bridge.py::engine_state` — returns the dict
  as JSONResponse, mounted at `/v1/engine/state`.
- **Consumers:** `server/static/index.html`,
  `server/static/labeler.html`, `server/static/loom.html`,
  `server/static/steering.html` — each polls at 500 ms.
