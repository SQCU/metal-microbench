#!/usr/bin/env bash
# Install the extended-thinking toolcard into the debug data root.
#
# Shape-B context-copying forked agent: receives the current parent chat
# context once via caller_messages, delegates the next question to a descendant
# llm_call, and returns a summary separately from the full descendant text.
# Idempotent. Lives under tools/st-debug/_data/toolcards/, NEVER touches
# the user's main install at ~/sillytavern-fork/data/toolcards/.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
DST_INSTALLED="$TOOLCARDS_DST/installed/extended-thinking"
DST_CARD="$TOOLCARDS_DST/cards/extended-thinking.toolcard.json"

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
    echo "[install-extended-thinking] cards/ -> real dir + symlinks to $SRC_CARDS_DIR"
fi
if [[ -L "$TOOLCARDS_DST/installed" ]]; then
    SRC_INSTALLED_DIR="$(readlink "$TOOLCARDS_DST/installed")"
    rm "$TOOLCARDS_DST/installed"
    mkdir -p "$TOOLCARDS_DST/installed"
    for d in "$SRC_INSTALLED_DIR"/*/; do
        [[ -d "$d" ]] || continue
        ln -sf "$d" "$TOOLCARDS_DST/installed/$(basename "$d")"
    done
    echo "[install-extended-thinking] installed/ -> real dir + symlinks to $SRC_INSTALLED_DIR"
fi

mkdir -p "$(dirname "$DST_CARD")" "$DST_INSTALLED"

python3 <<'PY'
import json
import os
import pathlib

dst_card = pathlib.Path(os.environ["DST_CARD"])
dst_installed = pathlib.Path(os.environ["DST_INSTALLED"])

service_src = '''"""extended-thinking tool service.

Shape-B context-copying forked agent prototype.

The plugin may pass caller_messages once in the invoke event. This service
copies up to 32K characters of that parent conversation into the descendant
llm_call, then appends a system prompt that explicitly prevents persona
continuity: the descendant must not respond as the parent character, must not
use the parent's voice, and must not roleplay.

The result separates `summary` from `reasoning_full`. The parent can present
or paraphrase the summary in its own voice without dumping the descendant's
full reasoning into chat, which is the persona-safe delegation pattern being
tested here.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any


CALLER_CONTEXT_CHAR_CAP = 32_000
SUMMARY_RE = re.compile(r"(?im)^SUMMARY:\\s*(.+?)\\s*$")
SYSTEM_PROMPT = (
    "You are a deliberation subroutine. The parent agent has delegated the "
    "next question to you with the conversation above as context. Respond "
    "with EXACTLY this two-section format and nothing else:\\n\\n"
    "SUMMARY: <one-line answer to the question>\\n\\n"
    "REASONING:\\n<chain-of-thought, multi-paragraph if useful>\\n\\n"
    "The SUMMARY line must come FIRST so the parent can read it even if the "
    "REASONING is long. The parent agent will rephrase your summary in their "
    "own voice — do NOT respond as the parent character, do NOT use the "
    "parent's voice, do NOT roleplay. Just summarize and reason."
)


_NEXT_CALL_ID = 0


def progress(text: str) -> None:
    print(json.dumps({"type": "progress", "text": text}), flush=True)


def llm_call(messages: list[dict[str, str]], max_tokens: int = 1024) -> dict[str, Any]:
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
            print(f"[extended-thinking] non-JSON stdin: {line[:120]!r}", file=sys.stderr)
            continue
        if msg.get("type") == "llm_response" and msg.get("id") == cid:
            if msg.get("ok"):
                return {"text": msg.get("data", ""), "elapsed_s": time.time() - t0}
            raise RuntimeError(f"llm_call failed: {msg.get('error', 'unknown')}")


def capped_caller_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        return []

    capped: list[dict[str, str]] = []
    used = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        remaining = CALLER_CONTEXT_CHAR_CAP - used
        if remaining <= 0:
            break
        clipped = content[:remaining]
        capped.append({"role": role, "content": clipped})
        used += len(clipped)
        if len(content) > len(clipped):
            break
    return capped


def extract_summary(text: str) -> str:
    match = SUMMARY_RE.search(text or "")
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else ""


def handle(args: dict[str, Any], caller_messages: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required and must be a non-empty string")

    progress("preparing deliberation context")
    parent_context = capped_caller_messages(caller_messages)
    messages = [
        *parent_context,
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    progress("deliberating")
    resp = llm_call(messages, max_tokens=2048)
    reasoning_full = str(resp.get("text") or "")
    return {
        "summary": extract_summary(reasoning_full),
        "reasoning_full": reasoning_full,
        "used_caller_messages": bool(parent_context),
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
            result = handle(req.get("args") or {}, req.get("caller_messages"))
            emit_result(rid, True, result=result)
        except Exception as exc:
            emit_result(rid, False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
'''

description = (
    "Delegates a question to a context-copying deliberation subroutine when "
    "the user asks something that benefits from careful step-by-step "
    "reasoning rather than fast intuition. How to call: pass a question "
    "string. Example call: question='given everything we just discussed "
    "about the trip budget, which city should we visit first?'."
)

card = {
    "card_format_version": "1",
    "id": "extended-thinking",
    "display_name": "Extended Thinking",
    "description": description,
    "version": "0.1.0",
    "author": "fork",
    "runtime": {
        "kind": "python",
        "deps": [],
        "entrypoint": "service.py",
        "idle_timeout_s": 300,
    },
    "tools": [
        {
            "name": "deliberate",
            "display_name": "Deliberate",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to deliberate on using copied parent context when available.",
                    },
                },
                "required": ["question"],
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

echo "[install-extended-thinking] done."
echo "  Card: $DST_CARD"
echo "  Service: $DST_INSTALLED/service.py (mirrored from manifest's files.service.py)"
