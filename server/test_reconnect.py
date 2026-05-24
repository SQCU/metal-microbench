#!/usr/bin/env python3
"""End-to-end test for the reconnect endpoint
(GET /v1/streams/{stream_id}/sse).

Verifies the disconnect-tolerance + reconnect machinery added to the
bridge in 2026-05-23:

  1. POST /v1/chat/completions with stream=true and a long enough
     max_tokens that we can plausibly abort mid-generation.
  2. Capture the engine-side stream_id from the X-Stream-Id response
     header on the initial POST.
  3. Read a few SSE chunks, then abort the connection (close the
     socket while the engine is still producing tokens).
  4. Verify the engine session continues (background drain takes over).
  5. GET /v1/streams/{stream_id}/sse to attach as the new consumer.
  6. Consume the remainder of the stream; assert we see a final chunk
     with finish_reason and a `usage` block carrying non-zero
     prompt_tokens + completion_tokens.

This test requires the bridge to be running (default
http://127.0.0.1:8001; override with BRIDGE_URL env). It does NOT
restart anything. Skip-with-explanatory-message if /health is not
reachable so CI doesn't false-fail when the bridge isn't up.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request

BRIDGE_URL = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")


def _parse_url(u: str) -> tuple[str, int, str]:
    """Return (host, port, path-prefix). Accepts http://host:port."""
    # urllib's URL parser is fine, but we want raw socket-friendly bits.
    from urllib.parse import urlparse
    p = urlparse(u)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    return host, port, p.path or ""


def _bridge_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{BRIDGE_URL}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _post_chat_stream_and_abort(prompt: str, max_tokens: int,
                                  abort_after_bytes: int = 256
                                  ) -> tuple[int, list[str]]:
    """POST /v1/chat/completions stream=true via a raw socket, read the
    response headers + a few SSE chunks, then close the socket.

    Returns: (stream_id, sse_chunks_seen_before_abort).
    Uses raw sockets (not urllib/httpx) so we can deterministically
    abort the connection mid-stream without surfacing the abort to the
    server via a graceful HTTP shutdown — the goal is to simulate a
    flaky network drop, which a graceful httpx-level disconnect doesn't
    fully reproduce.
    """
    host, port, _ = _parse_url(BRIDGE_URL)
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }).encode()
    req_lines = [
        f"POST /v1/chat/completions HTTP/1.1",
        f"Host: {host}:{port}",
        f"Content-Type: application/json",
        f"Content-Length: {len(body)}",
        f"Connection: close",
        f"Accept: text/event-stream",
        "",
        "",
    ]
    raw_req = ("\r\n".join(req_lines)).encode() + body

    s = socket.create_connection((host, port), timeout=30)
    try:
        s.sendall(raw_req)
        # Read headers (up to the blank line).
        buf = b""
        while b"\r\n\r\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError("server closed before headers")
            buf += chunk
        header_blob, _, body_start = buf.partition(b"\r\n\r\n")
        header_text = header_blob.decode("latin-1", "replace")
        # Parse status + X-Stream-Id.
        status_line, _, rest = header_text.partition("\r\n")
        stream_id_hdr = None
        for line in rest.split("\r\n"):
            if not line:
                continue
            k, _, v = line.partition(":")
            if k.strip().lower() == "x-stream-id":
                stream_id_hdr = v.strip()
                break
        if not stream_id_hdr:
            raise AssertionError(
                f"X-Stream-Id header missing from initial response. "
                f"Status: {status_line!r}. Headers:\n{rest}")
        stream_id = int(stream_id_hdr)
        # Read a few SSE bytes (whatever has arrived) so we can be sure
        # the engine actually started producing.
        seen = body_start
        s.settimeout(2.0)
        deadline = time.time() + 2.0
        while len(seen) < abort_after_bytes and time.time() < deadline:
            try:
                chunk = s.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            seen += chunk
        # Split into SSE event blocks (separator = "\n\n").
        chunks = [c for c in seen.decode("utf-8", "replace").split("\n\n") if c]
        return stream_id, chunks
    finally:
        # Force-close (RST-ish) by setting SO_LINGER 0. Goal: server
        # observes a TCP disconnect, not an HTTP keep-alive shutdown.
        try:
            import struct
            s.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                         struct.pack("ii", 1, 0))
        except Exception:
            pass
        try:
            s.close()
        except Exception:
            pass


def _reconnect_and_consume(stream_id: int, timeout_s: float = 60.0
                              ) -> tuple[list[dict], dict | None]:
    """GET /v1/streams/{stream_id}/sse, consume to [DONE].

    Returns (parsed_data_chunks, final_usage_dict_or_None). Each entry
    in parsed_data_chunks is the JSON object from a `data: {...}` line
    (excluding the trailing `[DONE]` marker).
    """
    url = f"{BRIDGE_URL}/v1/streams/{stream_id}/sse"
    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    chunks: list[dict] = []
    usage: dict | None = None
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        assert resp.status == 200, f"reconnect expected 200, got {resp.status}"
        assert resp.headers.get("X-Stream-Id") == str(stream_id), (
            f"reconnect response X-Stream-Id mismatch: "
            f"{resp.headers.get('X-Stream-Id')!r} vs {stream_id}")
        buf = b""
        deadline = time.time() + timeout_s
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"reconnect stream did not produce [DONE] within "
                    f"{timeout_s}s")
            chunk = resp.read(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n\n" in buf:
                evt, _, buf = buf.partition(b"\n\n")
                evt_text = evt.decode("utf-8", "replace")
                # SSE event = one or more lines. We care about the
                # `data:` line. 2026-05-23 append-log refactor: events
                # are now prefixed with a `: offset=N` comment line
                # for replay cursor tracking; skip non-data lines.
                payload = None
                for ln in evt_text.splitlines():
                    ln = ln.strip()
                    if ln.startswith("data:"):
                        payload = ln[len("data:"):].strip()
                        break
                if payload is None:
                    continue
                if payload == "[DONE]":
                    return chunks, usage
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                chunks.append(obj)
                # The OpenAI-compat final chunk carries `usage`.
                if "usage" in obj and obj["usage"]:
                    usage = obj["usage"]
    return chunks, usage


def main() -> int:
    print(f"=== reconnect endpoint smoke test ({BRIDGE_URL}) ===")
    if not _bridge_alive():
        print(f"  bridge not reachable at {BRIDGE_URL}/health — skipping")
        print(f"  (start the bridge with ./server/serve.py and re-run)")
        return 0

    # Step 1: open a stream + abort mid-flight.
    prompt = ("Tell me a long story about a brave knight, "
              "a clever dragon, and the talking sword they share. "
              "Take your time and use vivid descriptions.")
    print("\n  [1] POST /v1/chat/completions stream=true; abort after ~256B")
    stream_id, pre_chunks = _post_chat_stream_and_abort(
        prompt, max_tokens=400, abort_after_bytes=256)
    print(f"      stream_id={stream_id} (from X-Stream-Id header)")
    print(f"      sse chunks seen before abort: {len(pre_chunks)}")
    if len(pre_chunks) == 0:
        print(f"  WARN: abort fired before any SSE chunks — engine may "
              f"not have started producing yet. Test still valid; the "
              f"reconnect should still see content from this point.")

    # Step 2: tiny wait so the bridge's background-drain task spins up
    # (the disconnect handler runs in `_consume_engine_stream`'s finally
    # block on the next disconnect-poll tick; default cadence is 0.25s).
    time.sleep(0.5)

    # Step 3: reconnect.
    print(f"\n  [2] GET /v1/streams/{stream_id}/sse — reconnect attach")
    try:
        chunks, usage = _reconnect_and_consume(stream_id, timeout_s=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        print(f"  ✗ reconnect failed: HTTP {e.code} — {body[:200]}")
        return 1

    print(f"      received {len(chunks)} SSE chunks during reconnect")
    if usage is None:
        print(f"  ✗ no usage block in reconnect stream — "
              f"engine did not signal natural state==2")
        return 1
    print(f"      final usage: prompt_tokens={usage.get('prompt_tokens')}, "
          f"completion_tokens={usage.get('completion_tokens')}, "
          f"cache_hits={usage.get('cache_hits')}")

    # Step 4: assertions.
    assert usage.get("prompt_tokens", 0) > 0, (
        f"usage.prompt_tokens={usage.get('prompt_tokens')!r} expected > 0")
    assert usage.get("completion_tokens", 0) > 0, (
        f"usage.completion_tokens={usage.get('completion_tokens')!r} expected > 0")
    # We expect the final chunk to have a finish_reason set.
    final_chunk_with_finish = None
    for c in reversed(chunks):
        ch = (c.get("choices") or [{}])[0]
        if ch.get("finish_reason"):
            final_chunk_with_finish = c
            break
    assert final_chunk_with_finish is not None, (
        f"reconnect stream missing terminal chunk with finish_reason; "
        f"saw {len(chunks)} chunks")
    finish = final_chunk_with_finish["choices"][0]["finish_reason"]
    print(f"      finish_reason={finish!r}")
    assert finish in ("stop", "length", "tool_calls", "error"), (
        f"unexpected finish_reason={finish!r}")

    # Step 5: a second reconnect within the retention window must
    # SUCCEED with a full replay. 2026-05-23 APPEND-LOG REFACTOR:
    # the bridge keeps the per-stream log alive for
    # BRIDGE_STREAM_LOG_RETENTION_S seconds (default 300s) after the
    # engine emits state==2; a reconnect arriving in that window gets
    # 200 + full replay from offset 0 (matches real OpenAI/Anthropic
    # "addressable stream id within a few minutes post-completion"
    # semantics). Previous behavior was to pop the queue on the first
    # consumer's clean exit → second reconnect always 404'd. The new
    # behavior is strictly stronger.
    print(f"\n  [3] GET /v1/streams/{stream_id}/sse — second reconnect must succeed (retention window)")
    try:
        chunks2, usage2 = _reconnect_and_consume(stream_id, timeout_s=60)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        print(f"  ✗ second reconnect failed: HTTP {e.code} — {body[:200]}")
        return 1
    if usage2 is None:
        print(f"  ✗ second reconnect did not deliver usage")
        return 1
    print(f"      second reconnect delivered {len(chunks2)} chunks; "
          f"usage prompt_tokens={usage2.get('prompt_tokens')}, "
          f"completion_tokens={usage2.get('completion_tokens')}")
    # Replay equivalence: the completion_tokens count must match the
    # first reconnect (same underlying log).
    assert usage2.get("completion_tokens") == usage.get("completion_tokens"), (
        f"replay-divergence: first reconnect saw "
        f"completion_tokens={usage.get('completion_tokens')}, second saw "
        f"{usage2.get('completion_tokens')}")

    print(f"\n  PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
