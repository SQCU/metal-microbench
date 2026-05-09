#!/usr/bin/env bash
# Install the python-exec toolcard into the debug data root.
#
# LLM-augmented forked agent: receive a natural-language task, ask a fresh
# descendant LLM call for Python source, execute it, and return stdout/stderr.
# Idempotent. Lives under tools/st-debug/_data/toolcards/, NEVER touches
# the user's main install at ~/sillytavern-fork/data/toolcards/.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
DST_INSTALLED="$TOOLCARDS_DST/installed/python-exec"
DST_CARD="$TOOLCARDS_DST/cards/python-exec.toolcard.json"

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
    echo "[install-python-exec] cards/ -> real dir + symlinks to $SRC_CARDS_DIR"
fi
if [[ -L "$TOOLCARDS_DST/installed" ]]; then
    SRC_INSTALLED_DIR="$(readlink "$TOOLCARDS_DST/installed")"
    rm "$TOOLCARDS_DST/installed"
    mkdir -p "$TOOLCARDS_DST/installed"
    for d in "$SRC_INSTALLED_DIR"/*/; do
        [[ -d "$d" ]] || continue
        ln -sf "$d" "$TOOLCARDS_DST/installed/$(basename "$d")"
    done
    echo "[install-python-exec] installed/ -> real dir + symlinks to $SRC_INSTALLED_DIR"
fi

mkdir -p "$(dirname "$DST_CARD")" "$DST_INSTALLED"

python3 <<'PY'
import json
import os
import pathlib

dst_card = pathlib.Path(os.environ["DST_CARD"])
dst_installed = pathlib.Path(os.environ["DST_INSTALLED"])

service_src = '''"""python-exec tool service.

Shape-A LLM-augmented forked agent. The service receives only the tool
invocation args, makes a fresh-context llm_call to draft Python source, then
executes that source in a short-lived subprocess and returns stdout/stderr.

This debug card relies on the low-trust dev boundary plus a 30s timeout. The
prompt forbids network and filesystem side effects, and the subprocess gets a
minimal environment; future production work could add a stricter sandbox such
as firejail, bubblewrap, seccomp, or a dedicated container jail.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from typing import Any


CODE_BLOCK_RE = re.compile(r"```(?:python)?\\s*([\\s\\S]*?)```", re.IGNORECASE)
SYSTEM_PROMPT = (
    "You produce only Python code in a markdown code block, no "
    "explanation, no preamble. The code's stdout will be returned "
    "to the user. Standard library only — no pip-installable "
    "imports. Keep the script under 80 lines. The script must "
    "print() its final result to stdout. Do not write files or "
    "access the network."
)


# Runtime RPC: ask the host to make an LLM call.

_NEXT_CALL_ID = 0


def progress(text: str) -> None:
    """Emit a non-final progress note to the host runtime."""
    print(json.dumps({"type": "progress", "text": text}), flush=True)


def llm_call(messages: list, max_tokens: int = 2048) -> dict:
    """Request the host runtime to dispatch a chat completion via the
    user's currently selected ST connection profile. Blocks until the
    runtime feeds back the response. Returns a dict with `text` and
    `elapsed_s` (no token counts since the runtime doesn't expose them).
    """
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
            print(f"[python-exec] non-JSON stdin: {line[:120]!r}", file=sys.stderr)
            continue
        if msg.get("type") == "llm_response" and msg.get("id") == cid:
            if msg.get("ok"):
                return {"text": msg.get("data", ""), "elapsed_s": time.time() - t0}
            raise RuntimeError(f"llm_call failed: {msg.get('error', 'unknown')}")


def extract_code(text: str) -> str:
    m = CODE_BLOCK_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return (text or "").strip()


def handle(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    task = args.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task is required and must be a non-empty string")

    progress("drafting script")
    resp = llm_call([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ], max_tokens=2048)
    code = extract_code(resp["text"])
    if not code:
        raise RuntimeError("llm_call returned empty Python source")

    progress("executing")
    completed = subprocess.run(
        ["python3", "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PYTHONUNBUFFERED": "1"},
    )
    return {
        "script": code,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
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
        except Exception as exc:
            emit_result(rid, False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
'''

description = (
    "Writes and runs a short Python script for computation that is tedious "
    "for an LLM but trivial for code: numeric work, list manipulation, "
    "small simulations, or exact transformations. "
    "How to call: pass a natural-language task string. Example call: "
    "task='compute the SHA-256 of the string \"hello world\" and print the "
    "hex digest'. The generated script is executed with standard library "
    "only, a minimal environment, and a 30s timeout; future work could add "
    "a stricter sandbox for filesystem and network isolation."
)

card = {
    "card_format_version": "1",
    "id": "python-exec",
    "display_name": "Python Exec",
    "description": description,
    "version": "0.1.0",
    "author": "fork",
    "runtime": {
        "kind": "python",
        "deps": [],
        "entrypoint": "service.py",
        "idle_timeout_s": 120,
    },
    "tools": [
        {
            "name": "run",
            "display_name": "Run Python task",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural-language computation task to solve with a generated Python script.",
                    },
                },
                "required": ["task"],
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

echo "[install-python-exec] done."
echo "  Card: $DST_CARD"
echo "  Service: $DST_INSTALLED/service.py (mirrored from manifest's files.service.py)"
