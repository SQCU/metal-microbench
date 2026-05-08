#!/usr/bin/env python3
"""Curl-style probe of the ST debug instance's chat-completions backend.

Hits ST at http://127.0.0.1:8002/api/backends/chat-completions/generate
(the same path the ST frontend hits internally). ST forwards the body
to our bridge at :8001. This lets us validate the round trip:

    Claude → ST backend → bridge → engine → bridge → ST backend → Claude

WITHOUT involving a browser. Pair with the playwright e2e tests which
add the ST frontend + DOM-rendering layer on top of this.

Usage:
    ./scripts/api_probe.py                         # default: simple chat
    ./scripts/api_probe.py --tools                 # request includes tools[]
    ./scripts/api_probe.py --stream                # SSE streaming
    ./scripts/api_probe.py --message "your text"   # custom user message
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
import urllib.error


ST_URL = "http://127.0.0.1:8002"
ST_GENERATE = f"{ST_URL}/api/backends/chat-completions/generate"

# The ST frontend posts a request body to its own backend; the backend
# then fans out to the configured chat_completion_source. For our bridge
# (chat_completion_source="custom"), the backend forwards to whatever
# `custom_url` was set to (we set it to http://127.0.0.1:8001 in
# bootstrap.sh).
#
# Body shape mirrors what public/scripts/openai.js's createGenerationParameters
# emits — see SillyTavern source for the full schema.
DEFAULT_TOOLS = [{
    "type": "function",
    "function": {
        "name": "render_svg",
        "description": "render an SVG markup string and display it",
        "parameters": {
            "type": "object",
            "properties": {
                "svg": {"type": "string"},
            },
            "required": ["svg"],
        },
    },
}]


def build_body(*, message: str, tools: bool, stream: bool) -> dict:
    body = {
        "messages": [
            {"role": "user", "content": message},
        ],
        "model": "gemma-4-a4b",
        "chat_completion_source": "custom",
        "custom_url": "http://127.0.0.1:8001",
        "max_tokens": 1024,
        "temperature": 1.0,
        "stream": stream,
    }
    if tools:
        body["tools"] = DEFAULT_TOOLS
        body["tool_choice"] = "auto"
    return body


def fire_aggregate(body: dict) -> dict:
    req = urllib.request.Request(
        ST_GENERATE,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": str(e), "body": e.read().decode("utf-8", "replace")}


def fire_streaming(body: dict) -> dict:
    """Drain the SSE stream from ST's backend; collect content + tool_calls."""
    req = urllib.request.Request(
        ST_GENERATE,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    text_pieces = []
    tool_call_deltas = []
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
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
                        text_pieces.append(d["content"])
                    if isinstance(d.get("tool_calls"), list):
                        tool_call_deltas.extend(d["tool_calls"])
        return {
            "n_chunks": len(chunks),
            "finish_reason": finish,
            "content": "".join(text_pieces),
            "tool_call_deltas": tool_call_deltas,
        }
    except urllib.error.HTTPError as e:
        return {"error": str(e), "body": e.read().decode("utf-8", "replace")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--message", default="say hi briefly")
    ap.add_argument("--tools", action="store_true",
                    help="include render_svg tool in request")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    body = build_body(message=args.message, tools=args.tools, stream=args.stream)
    print(f"[probe] POST {ST_GENERATE}")
    print(f"[probe]   message: {args.message!r}")
    print(f"[probe]   tools:   {args.tools}, stream: {args.stream}")

    t0 = time.time()
    result = fire_streaming(body) if args.stream else fire_aggregate(body)
    elapsed = time.time() - t0

    print(f"[probe] response in {elapsed:.2f}s")
    if "error" in result:
        print(f"[probe] ERROR: {result['error']}")
        print(f"[probe]   body[:500]: {result.get('body', '')[:500]!r}")
        return 1

    if args.stream:
        print(f"[probe] SSE chunks: {result['n_chunks']}")
        print(f"[probe] finish:     {result['finish_reason']!r}")
        print(f"[probe] tool_calls: {len(result['tool_call_deltas'])}")
        if result["tool_call_deltas"]:
            for tc in result["tool_call_deltas"]:
                fn = tc.get("function") or {}
                args_preview = (fn.get("arguments") or "")[:160]
                print(f"            → {fn.get('name')!r}  args[:160]={args_preview!r}")
        print(f"[probe] content ({len(result['content'])} chars):")
        print(result["content"][:600])
    else:
        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        print(f"[probe] finish:    {choice.get('finish_reason')!r}")
        print(f"[probe] content:   {(msg.get('content') or '')[:400]!r}")
        if msg.get("tool_calls"):
            print(f"[probe] tool_calls: {len(msg['tool_calls'])}")
            for tc in msg["tool_calls"]:
                fn = tc.get("function") or {}
                print(f"            → {fn.get('name')!r}  args={fn.get('arguments')[:160]!r}")
        usage = result.get("usage") or {}
        print(f"[probe] usage:     prompt={usage.get('prompt_tokens')}, "
              f"completion={usage.get('completion_tokens')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
