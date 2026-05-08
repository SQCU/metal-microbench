#!/usr/bin/env python3
"""Replay a captured bridge request through the bridge.

Pair with BRIDGE_LOG_REQUESTS=/path/to/file.jsonl env var on the
bridge process. Each captured request is one JSON line containing
{"ts": ..., "path": "...", "body": <chat_completions request body>}.

Usage:
    # 1. Start bridge with capture on:
    BRIDGE_LOG_REQUESTS=/tmp/bridge_reqs.jsonl ./server/serve.py

    # 2. Fire the failing request from your client (SillyTavern, etc.).

    # 3. Replay the LAST captured request:
    scripts/replay_bridge_request.py --last /tmp/bridge_reqs.jsonl

    # Or replay request at index N (0 = first):
    scripts/replay_bridge_request.py /tmp/bridge_reqs.jsonl --index 0

    # Override fields on the way through (handy for bisecting):
    scripts/replay_bridge_request.py --last /tmp/bridge_reqs.jsonl \\
        --override 'temperature=0.0' --override 'max_tokens=512'

    # Disable streaming for cleaner output:
    scripts/replay_bridge_request.py --last /tmp/bridge_reqs.jsonl \\
        --override 'stream=false'

Output: assertions against the response (finish_reason, tool_calls
count, content presence, scaffolding-bleed checks) plus the full
response body. If the original request was streaming, replay also
streams and aggregates the SSE deltas.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "server"))
from bridge_config import chat_completions_url


def parse_override(s: str):
    """e.g. 'temperature=0.0' → ('temperature', 0.0)
           'stream=true'      → ('stream', True)
           'max_tokens=512'   → ('max_tokens', 512)"""
    k, _, v = s.partition('=')
    k = k.strip()
    v = v.strip()
    # type-coerce
    if v.lower() == "true":  return k, True
    if v.lower() == "false": return k, False
    if v.lower() == "null":  return k, None
    try: return k, int(v)
    except ValueError: pass
    try: return k, float(v)
    except ValueError: pass
    return k, v


def load_record(path: pathlib.Path, *, last: bool, index: int | None) -> dict:
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise SystemExit(f"  no captured requests in {path}")
    if last:
        return json.loads(lines[-1])
    if index is not None:
        return json.loads(lines[index])
    raise SystemExit("specify --last or --index N")


def fire_streaming(body: dict) -> dict:
    req = urllib.request.Request(
        chat_completions_url(),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    chunks = []
    text = []
    tool_calls = []
    finish = None
    with urllib.request.urlopen(req, timeout=600) as r:
        for line_bytes in r:
            line = line_bytes.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunks.append(obj)
            for choice in obj.get("choices", []):
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                d = choice.get("delta", {})
                if isinstance(d.get("content"), str) and d["content"]:
                    text.append(d["content"])
                if isinstance(d.get("tool_calls"), list):
                    tool_calls.extend(d["tool_calls"])
    return {
        "n_chunks": len(chunks),
        "finish_reason": finish,
        "content": "".join(text),
        "tool_calls": tool_calls,
    }


def fire_aggregate(body: dict) -> dict:
    req = urllib.request.Request(
        chat_completions_url(),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logfile", type=pathlib.Path,
                    help="Path to captured-requests jsonl.")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--last", action="store_true",
                     help="Replay the LAST captured request.")
    sel.add_argument("--index", type=int,
                     help="Replay the request at this 0-based index.")
    ap.add_argument("--override", action="append", default=[],
                    help='Override body field: --override "temperature=0.0"')
    args = ap.parse_args()

    record = load_record(args.logfile, last=args.last, index=args.index)
    body = dict(record["body"])

    for override in args.override:
        k, v = parse_override(override)
        body[k] = v
        print(f"  [override] {k} = {v!r}", file=sys.stderr)

    print(f"  bridge URL:  {chat_completions_url()}", file=sys.stderr)
    print(f"  request ts:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(record.get('ts', 0)))}", file=sys.stderr)
    print(f"  messages:    {len(body.get('messages', []))}", file=sys.stderr)
    print(f"  tools:       {len(body.get('tools') or [])}", file=sys.stderr)
    print(f"  tool_choice: {body.get('tool_choice')!r}", file=sys.stderr)
    print(f"  stream:      {body.get('stream', False)}", file=sys.stderr)
    print(f"  max_tokens:  {body.get('max_tokens')}", file=sys.stderr)
    print(f"  temperature: {body.get('temperature')}", file=sys.stderr)
    print(file=sys.stderr)

    t0 = time.time()
    if body.get("stream"):
        result = fire_streaming(body)
    else:
        agg = fire_aggregate(body)
        choice = agg["choices"][0]
        msg = choice["message"]
        result = {
            "n_chunks": 1,
            "finish_reason": choice.get("finish_reason"),
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls") or [],
        }
    elapsed = time.time() - t0

    # Diagnostics
    content = result["content"]
    print(f"=== response (wall {elapsed:.2f}s) ===", file=sys.stderr)
    print(f"  finish_reason: {result['finish_reason']!r}", file=sys.stderr)
    print(f"  n SSE chunks:  {result['n_chunks']}", file=sys.stderr)
    print(f"  tool_calls n:  {len(result['tool_calls'])}", file=sys.stderr)
    for i, tc in enumerate(result["tool_calls"]):
        fn = tc.get("function", {})
        args_preview = (fn.get("arguments") or "")[:200]
        print(f"    [{i}] {fn.get('name')!r}  args[:200]={args_preview!r}", file=sys.stderr)
    print(f"  content len:   {len(content)}", file=sys.stderr)
    print(f"    has <|channel marker bleed:  {'<|channel' in content}", file=sys.stderr)
    print(f"    has <turn|> trailer:         {'<turn|>' in content}", file=sys.stderr)
    print(f"    has <|tool_call> raw:        {'<|tool_call' in content}", file=sys.stderr)
    print(file=sys.stderr)
    print(f"--- response content ---", file=sys.stderr)
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
