#!/usr/bin/env bash
# Install the tree-of-thoughts toolcard into the debug data root.
#
# Shape-D context-copying async prototype: receives caller_messages, fans out
# multiple descendant llm_call branches concurrently, then synthesizes the
# results. Idempotent. Lives under tools/st-debug/_data/toolcards/, NEVER
# touches the user's main install at ~/sillytavern-fork/data/toolcards/.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
DST_INSTALLED="$TOOLCARDS_DST/installed/tree-of-thoughts"
DST_CARD="$TOOLCARDS_DST/cards/tree-of-thoughts.toolcard.json"
SERVICE_SRC="$DST_INSTALLED/service.py"

export DST_CARD DST_INSTALLED SERVICE_SRC

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
    echo "[install-tree-of-thoughts] cards/ -> real dir + symlinks to $SRC_CARDS_DIR"
fi
if [[ -L "$TOOLCARDS_DST/installed" ]]; then
    SRC_INSTALLED_DIR="$(readlink "$TOOLCARDS_DST/installed")"
    rm "$TOOLCARDS_DST/installed"
    mkdir -p "$TOOLCARDS_DST/installed"
    for d in "$SRC_INSTALLED_DIR"/*/; do
        [[ -d "$d" ]] || continue
        ln -sf "$d" "$TOOLCARDS_DST/installed/$(basename "$d")"
    done
    echo "[install-tree-of-thoughts] installed/ -> real dir + symlinks to $SRC_INSTALLED_DIR"
fi

mkdir -p "$(dirname "$DST_CARD")" "$DST_INSTALLED"

cat > "$SERVICE_SRC" <<'PY_SERVICE'
"""tree-of-thoughts tool service.

Shape-D context-copying async prototype with concurrent branch reasoning.

Why a parallel_llm_call: the toolcards plugin's server-side llm_call
dispatcher is async and fire-and-forget. Emitting N llm_calls back-to-back
gets us real concurrent upstream calls without any plugin changes.

Persona-violation safety: every branch prompt and the synthesis prompt tells
the descendant not to respond as the parent character, not to use the parent's
voice, and not to roleplay. The parent should rephrase the synthesis or choose
from the branches in its own voice, never dump the full branch reasoning into
chat verbatim.
"""
from __future__ import annotations

import json
import re
import sys
import time
from typing import Any


CALLER_CONTEXT_CHAR_CAP = 32_000
DEFAULT_BRANCHES = ["practical", "creative", "skeptical"]
SUMMARY_RE = re.compile(r"(?im)^SUMMARY:\s*(.+?)\s*$")
PERSONA_GUARD = (
    "do NOT respond as the parent character, do NOT use the parent's voice, "
    "do NOT roleplay. Just analyze."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a synthesis subroutine. The parent agent has delegated a "
    "multi-branch analysis to you with the conversation above as context. "
    "Compare the branch analyses, resolve tensions, and produce a concise "
    "recommendation grounded in the provided context. "
    f"{PERSONA_GUARD}"
)

_NEXT_CALL_ID = 0


def progress(text: str) -> None:
    print(json.dumps({"type": "progress", "text": text}), flush=True)


def branch_progress(
    branch_index: int,
    branch_label: str,
    status: str,
    summary: str | None = None,
    reasoning: str | None = None,
) -> None:
    event: dict[str, Any] = {
        "type": "branch_progress",
        "branch_index": branch_index,
        "branch_label": branch_label,
        "status": status,
    }
    if summary is not None:
        event["summary"] = summary
    if reasoning is not None:
        event["reasoning"] = reasoning
    print(json.dumps(event), flush=True)


def summary_progress(
    scope: str,
    summary: str,
    compressed_lines: int | None = None,
) -> None:
    """Emit a parent-persona-voiced n-of-k compression of recent
    descendant work. See docs/scalable_oversight_n_of_k.md."""
    event: dict[str, Any] = {
        "type": "summary_progress",
        "scope": scope,
        "summary": summary.strip()[:300],
    }
    if compressed_lines is not None:
        event["compressed_lines"] = compressed_lines
    print(json.dumps(event), flush=True)


def next_call_id() -> int:
    global _NEXT_CALL_ID
    _NEXT_CALL_ID += 1
    return _NEXT_CALL_ID


def parse_stdin_json(service_name: str) -> dict[str, Any]:
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
            print(f"[{service_name}] non-JSON stdin: {line[:120]!r}", file=sys.stderr)


def parallel_llm_call(
    calls: list[dict[str, Any]],
    branch_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Emit all llm_call events first, then collect matching responses."""
    if not calls:
        return []

    ordered_ids: list[int] = []
    branch_by_id: dict[int, tuple[int, str]] = {}
    start_times: dict[int, float] = {}
    pending: set[int] = set()
    responses: dict[int, dict[str, Any]] = {}

    for index, call in enumerate(calls):
        cid = next_call_id()
        ordered_ids.append(cid)
        pending.add(cid)
        start_times[cid] = time.time()
        if branch_labels is not None and index < len(branch_labels):
            branch_by_id[cid] = (index, branch_labels[index])
            branch_progress(index, branch_labels[index], "started")
        print(json.dumps({
            "type": "llm_call",
            "id": cid,
            "messages": call["messages"],
            "max_tokens": call.get("max_tokens", 512),
        }), flush=True)

    while pending:
        msg = parse_stdin_json("tree-of-thoughts")
        if msg.get("type") != "llm_response":
            continue
        cid = msg.get("id")
        if cid not in pending:
            continue
        pending.remove(cid)
        if msg.get("ok"):
            text = str(msg.get("data", "") or "").strip()
            responses[cid] = {
                "text": text,
                "elapsed_s": time.time() - start_times[cid],
            }
            branch = branch_by_id.get(cid)
            if branch is not None:
                index, label = branch
                branch_progress(
                    index,
                    label,
                    "complete",
                    summary=extract_summary(text),
                    reasoning=text,
                )
        else:
            raise RuntimeError(f"llm_call {cid} failed: {msg.get('error', 'unknown')}")

    return [responses[cid] for cid in ordered_ids]


def llm_call(messages: list[dict[str, str]], max_tokens: int = 512) -> dict[str, Any]:
    return parallel_llm_call([{"messages": messages, "max_tokens": max_tokens}])[0]


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


def validate_branches(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_BRANCHES)
    if not isinstance(value, list):
        raise ValueError("branches must be an array of strings when provided")

    branches: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("branches must contain only non-empty strings")
        branches.append(item.strip())
    if not branches:
        raise ValueError("branches must contain at least one label when provided")
    return branches


def branch_system_prompt(label: str) -> str:
    return (
        "You are one branch of a tree-of-thoughts analysis. Consider the "
        f"question from a strictly {label} angle. Stay scoped to that angle, "
        "use the parent conversation above only as context, and respond with "
        "EXACTLY this two-section format and nothing else:\n\n"
        "SUMMARY: <one-line branch conclusion>\n\n"
        "REASONING:\n<concise rationale for this branch>\n\n"
        f"{PERSONA_GUARD}"
    )


def extract_summary(text: str) -> str:
    match = SUMMARY_RE.search(text or "")
    if match:
        return match.group(1).strip()
    return (text or "").strip()[:200]


def build_synthesis_user_message(
    question: str,
    branch_results: list[dict[str, str]],
) -> str:
    chunks = [f"Original question:\n{question.strip()}\n", "Branch analyses:"]
    for branch in branch_results:
        chunks.append(
            f"\n[{branch['label']}]\n"
            f"Summary: {branch['summary']}\n"
            f"Reasoning:\n{branch['reasoning']}"
        )
    chunks.append("\nSynthesize these branches into the best final answer.")
    return "\n".join(chunks)


def handle(args: dict[str, Any], caller_messages: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    question = args.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question is required and must be a non-empty string")

    branches = validate_branches(args.get("branches"))
    parent_context = capped_caller_messages(caller_messages)

    progress(f"dispatching {len(branches)} branches in parallel")
    branch_calls = [
        {
            "messages": [
                *parent_context,
                {"role": "system", "content": branch_system_prompt(label)},
                {"role": "user", "content": question.strip()},
            ],
            "max_tokens": 512,
        }
        for label in branches
    ]
    branch_responses = parallel_llm_call(branch_calls, branch_labels=branches)
    branch_results = []
    for label, response in zip(branches, branch_responses):
        reasoning = str(response.get("text") or "").strip()
        branch_results.append({
            "label": label,
            "reasoning": reasoning,
            "summary": extract_summary(reasoning),
        })

    # n-of-k: fire one parent-voiced summary per branch IN PARALLEL.
    # Each summary is ~50 tokens of decode atop a shared-prefix
    # prefill (parent_context + summary instructions), so the bridge's
    # in-batch share gives near-free cost. Compresses each branch's
    # full reasoning (~30-60 lines) to one line.
    if parent_context:
        progress("summarizing branches in parent voice")
        summary_calls = [
            {
                "messages": [
                    *parent_context,
                    {
                        "role": "user",
                        "content": (
                            f"i just finished thinking through this question from the "
                            f"\"{branch['label']}\" angle. in ONE short sentence in my OWN voice "
                            f"(matching the persona established above; first-person, "
                            f"casual if the persona is casual), summarize what i concluded. "
                            f"i'm narrating to my user — describe what i thought, briefly.\n\n"
                            f"---\n{branch['reasoning'][:1500]}\n---\n\n"
                            f"output: just the one sentence. no preamble. no quotes."
                        ),
                    },
                ],
                "max_tokens": 80,
            }
            for branch in branch_results
        ]
        summary_responses = parallel_llm_call(summary_calls)
        for branch, summary_resp in zip(branch_results, summary_responses):
            line = str(summary_resp.get("text", "") or "").strip()
            line = line.split("\n")[0][:200] if line else f"explored the {branch['label']} angle"
            summary_progress(
                f"branch:{branch['label']}",
                line,
                compressed_lines=len(branch["reasoning"].splitlines()),
            )

    progress("synthesizing")
    synthesis_resp = llm_call([
        *parent_context,
        {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_synthesis_user_message(question, branch_results),
        },
    ], max_tokens=512)
    synthesis_text = str(synthesis_resp.get("text") or "").strip()

    # Final n-of-k summary: parent voice, the synthesis distilled to
    # one sentence the user (and the parent agent on next turn) can
    # use as the durable handle on this whole tool call.
    if parent_context:
        synth_summary_resp = llm_call([
            *parent_context,
            {
                "role": "user",
                "content": (
                    "i just synthesized those branches into a final answer. "
                    "in ONE short sentence in MY voice (matching the persona "
                    "above), tell my user what my conclusion was. casual if "
                    "the persona is casual. don't quote. don't preamble.\n\n"
                    f"---\n{synthesis_text[:2000]}\n---\n\n"
                    "output: just the one sentence."
                ),
            },
        ], max_tokens=80)
        synth_line = str(synth_summary_resp.get("text", "") or "").strip()
        synth_line = synth_line.split("\n")[0][:200] if synth_line else "synthesized the branches into a final answer"
        summary_progress(
            "synthesis",
            synth_line,
            compressed_lines=len(synthesis_text.splitlines()),
        )

    return {
        "branches": branch_results,
        "synthesis": synthesis_text,
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
        except ValueError as exc:
            emit_result(rid, False, error=str(exc))
        except Exception as exc:
            emit_result(rid, False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
PY_SERVICE

python3 <<'PY'
import json
import os
import pathlib

dst_card = pathlib.Path(os.environ["DST_CARD"])
dst_installed = pathlib.Path(os.environ["DST_INSTALLED"])
service_src_path = pathlib.Path(os.environ["SERVICE_SRC"])
service_src = service_src_path.read_text()

description = (
    "Explores a hard question through multiple context-copying background "
    "branches and synthesizes them when a question genuinely benefits from "
    "exploring multiple framings concurrently and you want a synthesis across "
    "them. How to call: pass question plus optional branches. Example call: "
    "question='which open-source license fits this hobby project best', "
    "branches=['legal', 'pragmatic', 'community']."
)

card = {
    "card_format_version": "1",
    "id": "tree-of-thoughts",
    "display_name": "Tree of Thoughts",
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
            "name": "explore",
            "display_name": "Explore Branches",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Question to explore using copied parent context when available.",
                    },
                    "branches": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional branch labels. Defaults to practical, "
                            "creative, and skeptical."
                        ),
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

echo "[install-tree-of-thoughts] done."
echo "  Card: $DST_CARD"
echo "  Service: $DST_INSTALLED/service.py (mirrored from manifest's files.service.py)"
