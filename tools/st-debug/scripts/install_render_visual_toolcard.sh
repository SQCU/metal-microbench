#!/usr/bin/env bash
# Install render-visual: 2-fork + 2-spawn demonstration of recursive
# decomposition with parent-voice n-of-k summaries.
#
# Topology (see docs/scalable_oversight_recursive_decomposition.md):
#   F1 (fork)  : scringlo voicing the technique she'll use
#   S1 (spawn) : "rendering engine" prefix → write Python that
#                takes a description and returns SVG
#   S2 (spawn) : SAME PREFIX as S1 → validate invariants on the
#                function's output (prefix-cache shares)
#   F2 (fork)  : scringlo wrapping the result for the user
#
# Each step targets 256–2k tokens of decode by skillful task
# assignment, not max_tokens caps.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/_data}"
CARDS_DIR="$DATA_ROOT/toolcards/cards"
INSTALLED_DIR="$DATA_ROOT/toolcards/installed/render-visual"

mkdir -p "$CARDS_DIR" "$INSTALLED_DIR"

SERVICE_PY="$INSTALLED_DIR/service.py"

cat > "$SERVICE_PY" << 'PY'
"""render-visual: 2-fork + 2-spawn recursive decomposition demo.

User asks for a visual that needs computed control points (e.g.,
a Voronoi diagram, Lissajous curve). Scringlo (the parent agent)
invokes this card via toolcards. The card decomposes:

    F1 (fork): scringlo voices what she's about to draw + briefly
        explains the technique. ~256-400 tokens.
    S1 (spawn): "rendering engine" prefix → writes a Python function
        with type contract: (description: str) -> svg_string. The
        function does its own math + SVG composition. ~800-1800 tokens.
    S2 (spawn, SHARES PREFIX with S1): validates invariants of S1's
        function (well-formed SVG, stays in-canvas, etc) without
        re-running it. ~256-600 tokens.
    Host: subprocess-exec S1's Python with the description; capture
        SVG bytes.
    F2 (fork): scringlo wraps for the user, mentions any validation
        findings, presents the SVG via embed[image_url].

Each LLM step's max_tokens is generous (4096) so the model has room
to demonstrate capability; budget is achieved by problem shape.

Failure surfacing: if S1's code subprocess-execs to an error or
S2 reports invariants broken, that surfaces as a 💭 line in the
parent's summary_trace and F2's prompt sees it. We DON'T auto-retry
within the card; failures are honest debugging affordances.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


CALLER_CONTEXT_CHAR_CAP = 32_000
SVG_RE = re.compile(r"<svg\b[^>]*>.*?</svg>", re.IGNORECASE | re.DOTALL)
PYTHON_BLOCK_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)

# Spawn prefix shared between S1 (write) and S2 (validate). The
# bridge's content-hash KV page cache will hit on this when the
# two calls run in proximity — only the per-call tail decodes.
RENDERER_SPAWN_PREFIX = (
    "You write small, self-contained Python rendering functions. "
    "Each function takes a structured data input (e.g., a description "
    "string or a data dict) and returns an SVG string. The functions "
    "use only the Python standard library — math, random, etc. — no "
    "external packages. They produce well-formed SVG that fits in a "
    "viewBox you choose. Output ONLY the function in a ```python ... ``` "
    "code block, with NO explanation text, NO preamble, NO postamble."
)


_NEXT_CALL_ID = 0


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def progress(text: str) -> None:
    emit({"type": "progress", "text": text})


def summary_progress(scope: str, summary: str,
                     compressed_lines: int | None = None) -> None:
    """Parent-persona-voiced n-of-k line. See
    docs/scalable_oversight_n_of_k.md."""
    event: dict[str, Any] = {
        "type": "summary_progress",
        "scope": scope,
        "summary": summary.strip()[:300],
    }
    if compressed_lines is not None:
        event["compressed_lines"] = compressed_lines
    emit(event)


def next_call_id() -> int:
    global _NEXT_CALL_ID
    _NEXT_CALL_ID += 1
    return _NEXT_CALL_ID


def parse_stdin_json() -> dict[str, Any]:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed during llm_call")
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            print(f"[render-visual] non-JSON stdin: {line[:120]!r}",
                  file=sys.stderr)


def llm_call(messages: list[dict[str, str]],
             max_tokens: int = 4096) -> dict[str, Any]:
    cid = next_call_id()
    emit({"type": "llm_call", "id": cid,
          "messages": messages, "max_tokens": max_tokens})
    t0 = time.time()
    while True:
        msg = parse_stdin_json()
        if msg.get("type") == "llm_response" and msg.get("id") == cid:
            if msg.get("ok"):
                return {"text": str(msg.get("data", "") or ""),
                        "elapsed_s": time.time() - t0}
            raise RuntimeError(f"llm_call {cid} failed: {msg.get('error', 'unknown')}")


def parallel_llm_call(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Emit all llm_calls first, then collect responses. Sibling
    spawns benefit from prefix-cache hits when their messages share
    a leading system block — that's the optimization the renderer
    spawn prefix is designed for."""
    if not calls:
        return []

    cids: list[int] = []
    starts: dict[int, float] = {}
    pending: set[int] = set()
    responses: dict[int, dict[str, Any]] = {}

    for call in calls:
        cid = next_call_id()
        cids.append(cid)
        pending.add(cid)
        starts[cid] = time.time()
        emit({"type": "llm_call", "id": cid,
              "messages": call["messages"],
              "max_tokens": call.get("max_tokens", 4096)})

    while pending:
        msg = parse_stdin_json()
        if msg.get("type") != "llm_response":
            continue
        cid = msg.get("id")
        if cid not in pending:
            continue
        pending.remove(cid)
        if msg.get("ok"):
            responses[cid] = {
                "text": str(msg.get("data", "") or ""),
                "elapsed_s": time.time() - starts[cid],
            }
        else:
            raise RuntimeError(f"llm_call {cid} failed: {msg.get('error', 'unknown')}")

    return [responses[cid] for cid in cids]


def cap_caller_messages(value: Any) -> list[dict[str, str]]:
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


def extract_python_block(text: str) -> str | None:
    m = PYTHON_BLOCK_RE.search(text or "")
    if m:
        return m.group(1).strip()
    return None


def extract_svg(text: str) -> str | None:
    m = SVG_RE.search(text or "")
    if m:
        return m.group(0)
    return None


def run_python_rendering(code: str, description: str,
                         timeout_s: int = 8) -> dict[str, Any]:
    """Subprocess-exec the rendering Python and capture its output.

    Calling convention: the LLM-written code defines a function
    `render(description: str) -> str`. Our wrapper imports the
    code, calls render(description), and prints the SVG. Stdout
    is the SVG bytes; we extract via SVG_RE.
    """
    wrapper = (
        f"{code}\n\n"
        f"if __name__ == '__main__':\n"
        f"    import sys\n"
        f"    out = render({description!r})\n"
        f"    if not isinstance(out, str):\n"
        f"        sys.stderr.write('render returned non-str\\n')\n"
        f"        sys.exit(2)\n"
        f"    sys.stdout.write(out)\n"
    )
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w",
                                      delete=False) as f:
        f.write(wrapper)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        return {
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"timeout after {timeout_s}s", "returncode": -1}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def handle(args: dict[str, Any], caller_messages: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    description = args.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    description = description.strip()

    parent_context = cap_caller_messages(caller_messages)

    # ─── F1 (FORK): scringlo voices the plan ───
    progress("F1: planning the render in parent voice")
    f1_resp = llm_call(
        parent_context + [{
            "role": "user",
            "content": (
                f"i'm about to delegate to a small python helper to render "
                f"this for me: {description!r}. in YOUR own voice (matching "
                f"the persona above), in 2-3 sentences, tell the user what "
                f"i'm about to draw and the technique i'll be using. be "
                f"playful but specific about the technique — name it (e.g. "
                f"voronoi tessellation, parametric curve, etc.) and gesture "
                f"at why it needs computed control points. don't quote, "
                f"don't preamble; speak as if directly to the user."
            ),
        }],
        max_tokens=4096,
    )
    f1_text = f1_resp["text"].strip()
    summary_progress("F1:plan", f1_text.splitlines()[0][:300] if f1_text else "(blank)")

    # ─── S1 + S2 (SPAWNS, shared prefix): write + validate ───
    # Two siblings; bridge's content-hash KV cache hits on the
    # shared RENDERER_SPAWN_PREFIX system message between them.
    # NOTE: S2 needs S1's output to validate, so we run S1 alone
    # first, then S2. That sacrifices the in-batch parallelism for
    # this pair, but the prefix-cache hit still benefits S2's
    # prefill (the system message is identical).
    progress("S1: writing the rendering function")
    s1_messages = [
        {"role": "system", "content": RENDERER_SPAWN_PREFIX},
        {"role": "user", "content": (
            f"Write a function `render(description: str) -> str` that "
            f"interprets the description and returns SVG. For THIS call, "
            f"the description will be {description!r}. Use the math you "
            f"need (sin/cos for parametric curves, distance comparisons "
            f"for voronoi tessellation, etc.). Choose a viewBox that fits "
            f"the rendered shape. Use the standard library only.\n\n"
            f"Output ONLY the python code block."
        )},
    ]
    s1_resp = llm_call(s1_messages, max_tokens=4096)
    s1_text = s1_resp["text"]
    s1_code = extract_python_block(s1_text) or s1_text.strip()
    summary_progress("S1:write",
                     f"wrote a {len(s1_code.splitlines())}-line render() function")

    progress("S2: validating the rendering function")
    s2_messages = [
        {"role": "system", "content": RENDERER_SPAWN_PREFIX},
        {"role": "user", "content": (
            f"Below is a Python `render(description)` function. WITHOUT "
            f"executing it, review it for these invariants and report "
            f"pass/fail per invariant in a structured list:\n"
            f"  1. defines a function named `render` taking one str arg\n"
            f"  2. returns a string starting with `<svg` and ending with `</svg>`\n"
            f"  3. uses only standard-library imports (no scipy, numpy, etc.)\n"
            f"  4. the math/coordinates are bounded (no obvious overflow)\n"
            f"  5. the output SVG would render visually meaningful content "
            f"for a description like {description!r}, not blank\n\n"
            f"```python\n{s1_code}\n```\n\n"
            f"Output: just the 5 lines, one per invariant, format "
            f"`[N] PASS|FAIL: <reason>`. No preamble."
        )},
    ]
    s2_resp = llm_call(s2_messages, max_tokens=4096)
    s2_report = s2_resp["text"].strip()
    pass_count = len(re.findall(r"PASS", s2_report))
    fail_count = len(re.findall(r"FAIL", s2_report))
    summary_progress(
        "S2:validate",
        f"validation: {pass_count} PASS / {fail_count} FAIL out of 5 invariants",
    )

    # ─── HOST: actually run the python ───
    progress("host: executing the rendering function")
    run = run_python_rendering(s1_code, description)
    svg = extract_svg(run["stdout"]) or extract_svg(run["stderr"])
    run_ok = run["returncode"] == 0 and bool(svg)
    summary_progress(
        "host:run",
        f"run: rc={run['returncode']}, svg_chars={len(svg) if svg else 0}",
    )

    # ─── F2 (FORK): scringlo wraps up ───
    progress("F2: wrapping up in parent voice")
    wrap_user = (
        f"i just finished the render task. report:\n"
        f"  technique-plan: {f1_text[:600]}\n"
        f"  python-validation: {s2_report[:400]}\n"
        f"  python-execution: rc={run['returncode']}, "
        f"stderr={run['stderr'][:200]!r}, svg-chars={len(svg) if svg else 0}\n\n"
        f"in YOUR own voice (matching the persona above), in 1-2 sentences, "
        f"present the result to the user. mention any notable validation "
        f"or execution issues honestly. if the python ran fine and svg "
        f"looks good, just present it cheerfully. don't quote, don't preamble."
    )
    f2_resp = llm_call(
        parent_context + [{"role": "user", "content": wrap_user}],
        max_tokens=4096,
    )
    f2_text = f2_resp["text"].strip()
    summary_progress("F2:wrap", f2_text.splitlines()[0][:300] if f2_text else "(blank)")

    # ─── Result envelope ───
    summary_lines = [
        f"[F1:plan]      {f1_text.splitlines()[0][:200] if f1_text else '(empty)'}",
        f"[S1:write]     wrote a {len(s1_code.splitlines())}-line render() function",
        f"[S2:validate]  {pass_count} PASS / {fail_count} FAIL of 5 invariants",
        f"[host:run]     rc={run['returncode']}, svg_chars={len(svg) if svg else 0}",
        f"[F2:wrap]      {f2_text.splitlines()[0][:200] if f2_text else '(empty)'}",
    ]
    embed = []
    if svg:
        # Embed as a data: URL so the FE inline-media pipeline picks
        # it up via marker substitution + extra.media (see
        # script.js#messageFormatting).
        b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
        embed.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/svg+xml;base64,{b64}"},
        })
    return {
        "summary": "\n".join(summary_lines),
        "embed": embed,
        "f1_plan": f1_text,
        "s1_code": s1_code,
        "s2_report": s2_report,
        "host_run": {
            "returncode": run["returncode"],
            "stdout_chars": len(run["stdout"]),
            "stderr": run["stderr"][:500],
            "svg_chars": len(svg) if svg else 0,
        },
        "f2_wrap": f2_text,
    }


def emit_result(rid: Any, ok: bool, result: Any = None,
                 error: str | None = None) -> None:
    msg: dict[str, Any] = {"type": "result", "id": rid, "ok": ok}
    if ok:
        msg["result"] = result
    else:
        msg["error"] = str(error or "unknown")
    emit(msg)


def main() -> None:
    emit({"type": "ready"})
    while True:
        try:
            msg = parse_stdin_json()
        except EOFError:
            return
        if msg.get("type") != "invoke":
            continue
        rid = msg.get("id")
        try:
            result = handle(msg.get("args") or {},
                            msg.get("caller_messages"))
            emit_result(rid, True, result)
        except Exception as e:
            emit_result(rid, False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
PY

# Embed service.py into the manifest's files field so the plugin's
# install path mirrors a single source of truth.
SERVICE_CONTENT=$(python3 -c "import json,sys; print(json.dumps(open('$SERVICE_PY').read()))")

cat > "$CARDS_DIR/render-visual.toolcard.json" << JSON
{
  "card_format_version": "1",
  "id": "render-visual",
  "version": "0.1.0",
  "display_name": "Render Visual",
  "description": "Generate an SVG figure from a natural-language description by delegating to a small chain of focused subagents (2 forked + 2 spawned). The forked subagents produce parent-voice planning + wrap-up; the spawned subagents share a 'rendering engine' prefix and write/validate Python that emits SVG. See docs/scalable_oversight_recursive_decomposition.md for the architecture.",
  "tools": [
    {
      "name": "render",
      "display_name": "Render visual",
      "description": "Use when the user asks for a visual that needs computed geometry — voronoi diagrams, lissajous curves, parametric shapes, fractal patterns, anything where the model alone would fake the math but a small validated python helper would get it right. The output is an inline SVG image. Example trigger: user says 'draw me a voronoi diagram with 12 seeds' → call render(description='voronoi diagram with 12 seeds').",
      "parameters": {
        "type": "object",
        "properties": {
          "description": {
            "type": "string",
            "description": "Natural-language description of the visual. Specific enough that a python helper can map it to control points (e.g. 'voronoi diagram, 12 seeds randomly placed, viewBox 0 0 400 400, distinct fill colors per cell')."
          }
        },
        "required": ["description"]
      },
      "async": false
    }
  ],
  "runtime": {
    "kind": "python",
    "deps": [],
    "entrypoint": "service.py",
    "idle_timeout_s": 300
  },
  "files": {
    "service.py": $SERVICE_CONTENT
  }
}
JSON

# Mirror service.py from manifest's files field (this is what the
# plugin does on every reload — verify our local copy matches).
python3 -c "
import json, pathlib, sys
manifest = json.load(open('$CARDS_DIR/render-visual.toolcard.json'))
content = manifest['files']['service.py']
target = pathlib.Path('$SERVICE_PY')
target.write_text(content)
print(f'  embedded service.py: {len(content)} chars')
"

echo "[install-render-visual] done."
echo "  Card: $CARDS_DIR/render-visual.toolcard.json"
echo "  Service: $SERVICE_PY (mirrored from manifest's files.service.py)"
