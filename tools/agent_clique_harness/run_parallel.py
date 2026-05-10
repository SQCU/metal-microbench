#!/usr/bin/env python3
"""Agent clique harness — parallel variant.

Submits all agents to the bridge concurrently (each via its own
threadpool worker) so the engine's textMultiPrefill batches their
priming work, AR runs at B=N concurrent slots, and the bridge's new
work-conserving submit_pump/poll_pump architecture gets exercised.

Compare against run.py (sequential) to see where the parallel-submit
architecture pays off.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

# Reuse the corpus + agent definitions from the sequential harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import AGENTS, build_system_prompt, build_user_message, load_corpus  # noqa: E402


def post_chat(bridge_url: str, messages, max_tokens: int):
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{bridge_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def fire_agent(bridge_url, agent, sys_prompt, max_tokens, t0):
    """Run one agent. Returns dict for results table."""
    t_start = time.time()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": build_user_message(agent)},
    ]
    resp = post_chat(bridge_url, messages, max_tokens)
    elapsed = time.time() - t_start
    u = resp["usage"]
    return {
        "agent": agent.name,
        "t_submit": t_start - t0,
        "elapsed_s": elapsed,
        "prompt_tokens": u["prompt_tokens"],
        "completion_tokens": u["completion_tokens"],
        "cache_hits": u["cache_hits"],
        "cache_misses": u["cache_misses"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=os.environ.get(
        "BRIDGE_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--max-tokens", type=int, default=int(
        os.environ.get("MAX_TOKENS", "200")))
    args = ap.parse_args()

    corpus = load_corpus()
    sys_prompt = build_system_prompt(corpus)

    print("=== agent clique harness (PARALLEL) ===")
    print(f"bridge:    {args.bridge}")
    print(f"corpus:    {len(corpus)} docs, {sum(len(c) for _,c in corpus):,} chars")
    print(f"agents:    {[a.name for a in AGENTS]} (concurrent)")
    print(f"max_toks:  {args.max_tokens}\n")

    # Sanity: bridge up.
    h = json.loads(urllib.request.urlopen(
        f"{args.bridge}/health", timeout=5).read())
    print(f"bridge: {h['model']} caps={h.get('capabilities')}\n")

    t0 = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(AGENTS)) as ex:
        futures = [
            ex.submit(fire_agent, args.bridge, agent,
                       sys_prompt, args.max_tokens, t0)
            for agent in AGENTS
        ]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    t_total = time.time() - t0

    # Sort by submit-time so the print order matches the launch order.
    results.sort(key=lambda r: r["t_submit"])

    print("                  submit    wall   prompt  comp   hits  miss")
    for r in results:
        print(f"  {r['agent']:<13} "
              f"{r['t_submit']:6.2f}s "
              f"{r['elapsed_s']:6.1f}s "
              f"{r['prompt_tokens']:6d} "
              f"{r['completion_tokens']:5d} "
              f"{r['cache_hits']:6d} "
              f"{r['cache_misses']:5d}")
    print()
    print(f"--- summary ---")
    print(f"total wall (parallel):     {t_total:5.1f}s")
    print(f"total wall would-be-seq:   "
          f"{sum(r['elapsed_s'] for r in results):5.1f}s")
    speedup = sum(r['elapsed_s'] for r in results) / max(t_total, 1e-6)
    print(f"speedup vs serial:         {speedup:.2f}×")
    return 0


if __name__ == "__main__":
    sys.exit(main())
