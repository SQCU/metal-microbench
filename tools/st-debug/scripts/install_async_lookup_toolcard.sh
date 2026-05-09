#!/usr/bin/env bash
# Install the async-lookup toolcard into the debug data root.
#
# Shape-C true fire-and-forget async prototype: FE action returns immediately
# immediately, while a deliberately slow descendant process completes later
# through the existing /poll and pushChatResult paths.
# Idempotent. Lives under tools/st-debug/_data/toolcards/, NEVER touches
# the user's main install at ~/sillytavern-fork/data/toolcards/.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
DST_INSTALLED="$TOOLCARDS_DST/installed/async-lookup"
DST_CARD="$TOOLCARDS_DST/cards/async-lookup.toolcard.json"

export DST_CARD DST_INSTALLED

# The cards/ and installed/ symlinks the bootstrap created point at the
# user's main install. Replace each symlink with a real directory that
# keeps the original entries as symlinks, then add our debug-only card as
# real files under _data/toolcards/.
mkdir -p "$TOOLCARDS_DST"
if [[ -L "$TOOLCARDS_DST/cards" ]]; then
    SRC_CARDS_DIR="$(readlink "$TOOLCARDS_DST/cards")"
    rm "$TOOLCARDS_DST/cards"
    mkdir -p "$TOOLCARDS_DST/cards"
    for f in "$SRC_CARDS_DIR"/*.toolcard.json; do
        [[ -f "$f" ]] || continue
        ln -sf "$f" "$TOOLCARDS_DST/cards/$(basename "$f")"
    done
    echo "[install-async-lookup] cards/ -> real dir + symlinks to $SRC_CARDS_DIR"
fi
if [[ -L "$TOOLCARDS_DST/installed" ]]; then
    SRC_INSTALLED_DIR="$(readlink "$TOOLCARDS_DST/installed")"
    rm "$TOOLCARDS_DST/installed"
    mkdir -p "$TOOLCARDS_DST/installed"
    for d in "$SRC_INSTALLED_DIR"/*/; do
        [[ -d "$d" ]] || continue
        ln -sf "$d" "$TOOLCARDS_DST/installed/$(basename "$d")"
    done
    echo "[install-async-lookup] installed/ -> real dir + symlinks to $SRC_INSTALLED_DIR"
fi

mkdir -p "$(dirname "$DST_CARD")" "$DST_INSTALLED"

python3 <<'PY'
import json
import os
import pathlib

dst_card = pathlib.Path(os.environ["DST_CARD"])
dst_installed = pathlib.Path(os.environ["DST_INSTALLED"])

service_src = '''"""async-lookup tool service.

Shape-C true fire-and-forget async prototype.

The service deliberately takes about six seconds before returning a result.
That bounded delay is the point: start_invoke should still return a session_id
quickly, while the result arrives later through the normal poll/chat-result
paths.
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any


SYSTEM_PROMPT = (
    "You are a brief lookup assistant. Given a topic, produce a single "
    "1-2 sentence synthetic answer. No preamble, no caveats."
)
SIMULATED_LOOKUP_DELAY_S = 6

_NEXT_CALL_ID = 0


def progress(text: str) -> None:
    print(json.dumps({"type": "progress", "text": text}), flush=True)


def llm_call(messages: list[dict[str, str]], max_tokens: int = 64) -> dict[str, Any]:
    global _NEXT_CALL_ID
    _NEXT_CALL_ID += 1
    cid = _NEXT_CALL_ID
    print(json.dumps({
        "type": "llm_call",
        "id": cid,
        "messages": messages,
        "max_tokens": max_tokens,
    }), flush=True)
    t0 = time.time()
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed during llm_call")
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            print(f"[async-lookup] non-JSON stdin: {line[:120]!r}", file=sys.stderr)
            continue
        if msg.get("type") == "llm_response" and msg.get("id") == cid:
            if msg.get("ok"):
                return {"text": msg.get("data", ""), "elapsed_s": time.time() - t0}
            raise RuntimeError(f"llm_call failed: {msg.get('error', 'unknown')}")


def handle(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    topic = args.get("topic")
    if not isinstance(topic, str) or not topic.strip():
        raise ValueError("topic is required and must be a non-empty string")
    topic = topic.strip()

    t0 = time.time()
    progress("preparing lookup")
    time.sleep(3)
    progress("querying source")
    time.sleep(2)
    resp = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": topic},
    ], max_tokens=64)
    time.sleep(1)

    return {
        "topic": topic,
        "answer": str(resp.get("text") or "").strip(),
        "elapsed_s": time.time() - t0,
        "simulated_lookup_delay_s": SIMULATED_LOOKUP_DELAY_S,
    }


def emit_result(rid: Any, ok: bool, result: Any = None, error: str | None = None) -> None:
    msg = {"type": "result", "id": rid, "ok": ok}
    if ok:
        msg["result"] = result
    else:
        msg["error"] = error or "unknown error"
    print(json.dumps(msg), flush=True)


def main() -> None:
    print(json.dumps({"type": "ready"}), flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            emit_result(0, False, error=f"bad request json: {exc}")
            continue
        if req.get("type") != "invoke":
            continue
        rid = req.get("id", 0)
        try:
            result = handle(req.get("args") or {})
            emit_result(rid, True, result=result)
        except ValueError as exc:
            emit_result(rid, False, error=str(exc))
        except Exception as exc:
            emit_result(rid, False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
'''

description = (
    "Runs a fresh-context background lookup that can continue while the "
    "conversation moves on. Use for slow work such as fetching a city's "
    "current weather, summarizing a long document, or doing a slow database "
    "query where the caller should not wait inline. How to call: pass a "
    "single topic string. Example call: topic='current bird species spotted "
    "in Hyde Park, London this week'. This debug card simulates the slow "
    "lookup with a bounded six-second delay, then asks a small descendant "
    "LLM call for a brief synthetic answer."
)

card = {
    "card_format_version": "1",
    "id": "async-lookup",
    "display_name": "Async Lookup",
    "description": description,
    "version": "0.1.1",
    "author": "fork",
    "runtime": {
        "kind": "python",
        "deps": [],
        "entrypoint": "service.py",
        "idle_timeout_s": 300,
    },
    "tools": [
        {
            "name": "lookup",
            "display_name": "Background lookup",
            "async": True,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Lookup topic to answer in the background.",
                    },
                },
                "required": ["topic"],
            },
        }
    ],
    "files": {"service.py": service_src},
}

dst_card.write_text(json.dumps(card, indent=2) + "\n")
dst_installed.mkdir(parents=True, exist_ok=True)
(dst_installed / "service.py").write_text(service_src)

print(f"  wrote {dst_card}")
print(f"  wrote {dst_installed / 'service.py'}")
print(f"  embedded service.py: {len(service_src)} chars")
PY

echo "[install-async-lookup] done."
echo "  Card: $DST_CARD"
echo "  Service: $DST_INSTALLED/service.py (mirrored from manifest's files.service.py)"
