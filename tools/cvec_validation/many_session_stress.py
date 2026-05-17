#!/usr/bin/env python3
"""Stress repro: run N independent chat-completion sessions back-to-back
against the bridge, each shaped so the model reliably emits a tool_call.
Watch for the symptom the user reported on 2026-05-10: "repeated empty
responses after ~20-30 different tool-call-using sessions on the same
persistent bridge process."

Not a SillyTavern simulation — this is an engine-stress benchmark. We
hit /v1/chat/completions directly with well-formed requests; each
session is independent (fresh bridge stream_id, no shared chat history).
What carries across sessions is whatever the bridge / engine accumulates
internally (page cache, residency, cvec registry growth, etc.).

Per-session metrics captured:
    - response content length
    - finish_reason
    - tool_calls extracted (count, names)
    - usage.cache_hits / cache_misses (to see when the page cache starts
      thrashing or producing wrong hits)
    - latency

A session that produces empty content with finish_reason in
{"stop","length"} is the smoking gun. Sessions are saved to a JSONL so a
post-hoc analysis can chart cache-hit ratio vs session index, content
length vs session index, etc.
"""
import argparse, json, time, urllib.request
from pathlib import Path

BRIDGE = "http://127.0.0.1:8001"
OUT = Path(__file__).parent / "many_session_stress.jsonl"

# Prompts each shape a TOOL-CALL-PROVOKING request. The persona is
# inlined as a system message so we don't depend on ST or any cards.
SYSTEM = (
    "You are a tabletop RPG GM. Whenever the user asks for randomness "
    "(an encounter roll, a loot drop, a dice check), you MUST call the "
    "python-exec__run tool with a short Python script that computes the "
    "outcome. Do not fake the dice in your head — invoke the tool. "
    "After the tool returns, narrate the outcome in 1-2 sentences. "
    "If the user prompts you for narration only, narrate without the tool."
)
PROMPTS = [
    "roll d20 + 3 for a stealth check",
    "roll an encounter from the c/u/r tier table for level 3",
    "drop a random magic item from the uncommon table",
    "what's the weather outside the dungeon today? roll d4 over [sun,rain,fog,storm]",
    "roll initiative for 4 combatants — 1d20+2 each",
    "i open the next chest, roll for trap (DC 15)",
    "roll a random NPC reaction (hostile/neutral/friendly)",
    "what time of day did we exit the dungeon? roll d6",
]

# A pseudo-toolcard description that maps to our installed python-exec
# card. Not registered through the toolcards plugin (this is a direct
# bridge stress test) — we just declare the tool and let the model emit
# a tool_call; the bridge extracts and returns structured tool_calls.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "python-exec__run",
        "description": (
            "Writes and runs a short Python script for computation. "
            "Pass a natural-language task string. Use for any tedious "
            "randomness or numeric work."),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {"type": "string",
                         "description": "Natural-language computation task."},
            },
            "required": ["task"],
        },
    },
}]


def call(session_idx, prompt):
    body = {
        "model": "gemma-4-a4b",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "max_tokens": 400,
        "temperature": 0.7,
        "seed": session_idx,    # deterministic per-session
        "tools": TOOLS,
        "tool_choice": "auto",
    }
    t0 = time.time()
    req = urllib.request.Request(
        f"{BRIDGE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
    except Exception as e:
        return {"i": session_idx, "error": str(e)[:200],
                "elapsed_s": round(time.time() - t0, 2)}
    elapsed = round(time.time() - t0, 2)
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    usage = resp.get("usage") or {}
    return {
        "i": session_idx,
        "prompt_head": prompt[:60],
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "content_len": len(content),
        "content_head": content[:120],
        "tool_calls": [{"name": tc.get("function", {}).get("name"),
                        "args_len": len((tc.get("function", {}).get("arguments")) or "")}
                       for tc in tool_calls],
        "n_tool_calls": len(tool_calls),
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cache_hits": usage.get("cache_hits"),
            "cache_misses": usage.get("cache_misses"),
        },
    }


def run(n: int, bail_on_empties: int):
    print(f"=== many-session stress: n={n} sessions, bail_on_empties={bail_on_empties} ===")
    fh = OUT.open("w")
    empties = 0
    consecutive_empties = 0
    for i in range(1, n + 1):
        prompt = PROMPTS[(i - 1) % len(PROMPTS)]
        r = call(i, prompt)
        fh.write(json.dumps(r) + "\n")
        fh.flush()
        is_empty = (r.get("content_len", 0) == 0
                    and r.get("n_tool_calls", 0) == 0
                    and not r.get("error"))
        if is_empty:
            empties += 1
            consecutive_empties += 1
        else:
            consecutive_empties = 0
        flag = ('🩸' if is_empty
                else '🔧' if r.get('n_tool_calls', 0) > 0
                else '·')
        print(f"  [{i:3d}] {flag} {r.get('elapsed_s')}s "
              f"content={r.get('content_len')}ch "
              f"tool_calls={r.get('n_tool_calls')} "
              f"finish={r.get('finish_reason')} "
              f"cache=hit{r.get('usage', {}).get('cache_hits')}/"
              f"miss{r.get('usage', {}).get('cache_misses')}")
        if bail_on_empties and consecutive_empties >= bail_on_empties:
            print(f">>> {consecutive_empties} consecutive empties at session {i}; bailing")
            break
    fh.close()
    print(f"\nrecorded {OUT}")
    print(f"empties: {empties}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--bail-on-empties", type=int, default=3)
    args = p.parse_args()
    run(args.n, args.bail_on_empties)
