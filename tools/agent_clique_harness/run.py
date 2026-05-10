#!/usr/bin/env python3
"""Agent clique harness — drives the Gemma bridge with a shared
document corpus as the cacheable system prefix and a per-agent role +
task as the per-call suffix. Reports cache_hits per agent so the
prefix-cache amplification across the clique is observable.

Architecture follows project_prompt_prefix_suffix_rule:
  * System message = full document corpus (~few thousand tokens, byte-
    identical across all agents). This is the cacheable prefix; the
    engine's content-hash page cache adopts it from the first agent's
    prefill onward, so subsequent agents pay only their suffix cost.
  * User message = "[role] task framing — N words." Tiny, varies per
    agent. Suffix-only.
  * No tools, no images, no role:tool — keeps the warm path eligible
    for full segment-replay. (Even if it weren't, the engine-side
    page cache adopts shared prefix bytes regardless.)

Usage:
    python tools/agent_clique_harness/run.py [--bridge URL] [--max-tokens N]

Env:
    BRIDGE_URL   — defaults to http://127.0.0.1:8001
    MAX_TOKENS   — defaults to 200
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "agent_clique_corpus"


@dataclass
class Agent:
    name: str
    role_blurb: str       # short role description put in system block tail
    task: str             # the per-agent task framing in the user message


# ── The clique ─────────────────────────────────────────────────────────
# Each agent reads the same corpus. The roles differ in interpretive
# stance: critic finds weak claims, synthesist threads connections,
# implementor extracts next steps, translator recasts for a different
# audience. Same documents, four orthogonal lenses.

AGENTS: list[Agent] = [
    Agent(
        name="critic",
        role_blurb="critical reviewer who specializes in identifying "
                   "the weakest claim in technical writing and naming "
                   "the evidence that would falsify it",
        task="For each of the three documents, identify ONE weakest "
             "claim and state the smallest experiment that would "
             "falsify it. Be concrete. 200 words total.",
    ),
    Agent(
        name="synthesist",
        role_blurb="cross-document synthesist who reads several pieces "
                   "of writing in parallel and identifies the latent "
                   "thread that connects them",
        task="What is the single thread connecting these three "
             "documents that none of them states explicitly? Then: "
             "what new question emerges from reading them together "
             "that none asks individually? 200 words total.",
    ),
    Agent(
        name="implementor",
        role_blurb="implementor focused on extracting concrete next "
                   "steps and mapping their dependencies",
        task="List the top 3 concrete next steps across these "
             "documents, ranked by impact. For each: name the "
             "blocking dependency and the wall-time estimate. "
             "200 words total.",
    ),
    Agent(
        name="translator",
        role_blurb="audience translator who recasts dense technical "
                   "writing for a non-technical reader without losing "
                   "the essential point",
        task="Pick the most technical of the three documents. Recast "
             "its core claim in 100 words for a non-technical reader. "
             "Then in 100 more words, explain what readers *of the "
             "original* might miss that your recast preserves. "
             "200 words total.",
    ),
]


def load_corpus() -> list[tuple[str, str]]:
    """Return (filename, contents) for each .md in the corpus dir,
    sorted by filename so the prefix is deterministic across runs."""
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files in {CORPUS_DIR}")
    return [(f.name, f.read_text()) for f in files]


def build_system_prompt(corpus: list[tuple[str, str]]) -> str:
    """The system prompt is the cacheable PREFIX: corpus delimiters +
    document contents, byte-identical across all agents. Per-agent role
    blurb is NOT included here — that's part of the suffix to keep the
    prefix maximally shareable.

    Rationale: causal attention + content-hash page cache means
    identical leading bytes → cache adoption from first call onward.
    Putting the per-agent role at the top of the system prompt would
    diverge the bytes at position ~200 and lose ~95% of the cache
    benefit."""
    parts = [
        "You will read a small corpus of documents and respond to a "
        "task framed in the user's next message. The corpus follows. "
        "Read carefully, then wait for the user's instruction.\n",
    ]
    for fname, contents in corpus:
        parts.append(f"\n---\n## {fname}\n\n{contents}\n")
    parts.append("\n---\nEnd of corpus. The user will now state your "
                  "specific role and task.\n")
    return "".join(parts)


def build_user_message(agent: Agent) -> str:
    """The user message is the per-call SUFFIX. Role blurb + task. This
    is what differs across agents, so it goes last. The suffix is short
    (~80 tokens) so it adds minimal cold prefill on top of the cached
    system prefix."""
    return (
        f"You are a {agent.role_blurb}. "
        f"\n\nTask: {agent.task}"
    )


def post_chat(bridge_url: str, messages: list[dict],
              max_tokens: int) -> dict:
    body = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,    # deterministic so cache-hit signal is clean
        "stream": False,
    }
    req = urllib.request.Request(
        f"{bridge_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default=os.environ.get(
        "BRIDGE_URL", "http://127.0.0.1:8001"))
    ap.add_argument("--max-tokens", type=int, default=int(
        os.environ.get("MAX_TOKENS", "200")))
    ap.add_argument("--out", default=None,
                    help="optional path to write per-agent responses as JSONL")
    args = ap.parse_args()

    corpus = load_corpus()
    sys_prompt = build_system_prompt(corpus)

    print("=== agent clique harness ===")
    print(f"bridge:      {args.bridge}")
    print(f"corpus:      {len(corpus)} docs, {sum(len(c) for _,c in corpus):,} chars")
    print(f"system_len:  {len(sys_prompt):,} chars")
    print(f"agents:      {[a.name for a in AGENTS]}")
    print(f"max_tokens:  {args.max_tokens}\n")

    # Sanity: bridge is up.
    try:
        with urllib.request.urlopen(f"{args.bridge}/health", timeout=5) as r:
            health = json.loads(r.read())
    except Exception as e:
        print(f"FAIL: bridge not reachable at {args.bridge}: {e}")
        return 1
    print(f"bridge ready: {health['model']} "
          f"caps={health.get('capabilities')}\n")

    results: list[dict] = []
    cumulative_prompt_tokens = 0
    cumulative_cache_hits = 0
    t_start = time.time()

    for i, agent in enumerate(AGENTS):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": build_user_message(agent)},
        ]
        t0 = time.time()
        try:
            resp = post_chat(args.bridge, messages, args.max_tokens)
        except Exception as e:
            print(f"  [{agent.name}] FAIL: {e}")
            continue
        elapsed = time.time() - t0
        u = resp["usage"]
        text = resp["choices"][0]["message"]["content"]
        cumulative_prompt_tokens += u["prompt_tokens"]
        cumulative_cache_hits += u["cache_hits"]
        hit_pct = 100.0 * u["cache_hits"] / max(u["prompt_tokens"], 1)
        print(f"[{i+1}/{len(AGENTS)}] {agent.name:<13} "
              f"prompt={u['prompt_tokens']:5d} "
              f"completion={u['completion_tokens']:4d} "
              f"hits={u['cache_hits']:5d} ({hit_pct:5.1f}%) "
              f"misses={u['cache_misses']:4d} "
              f"wall={elapsed:5.1f}s")
        results.append({
            "agent": agent.name,
            "usage": u,
            "elapsed_s": elapsed,
            "response": text,
        })

    t_total = time.time() - t_start
    print()
    print(f"--- summary ---")
    print(f"total wall:                 {t_total:5.1f}s")
    print(f"total prompt tokens:        {cumulative_prompt_tokens:,}")
    print(f"total cache_hits:           {cumulative_cache_hits:,}")
    print(f"aggregate hit rate:         "
          f"{100.0 * cumulative_cache_hits / max(cumulative_prompt_tokens,1):5.1f}%")
    print()
    print("Expected pattern: agent #1 cold (0 hits), agents #2..N hit "
          "the system-prompt-sized cached prefix.")
    print("If agents #2..N have hits ≪ system_prompt_tokens, the "
          "prefix-cache adoption isn't working as designed; "
          "investigate page-hash divergence at the system→user "
          "boundary.")

    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nresponses written to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
