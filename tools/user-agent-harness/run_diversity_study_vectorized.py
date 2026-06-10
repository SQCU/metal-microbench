#!/usr/bin/env python3
"""Vectorized version of run_diversity_study.py.

Same outputs (markdown report). Different scheduling:

  - Old: K workers; each worker runs one full conversation as ~15
    sequential LLM calls. Per-conversation latency dominates wall
    time; bridge parallelism capped at K (worker count).
  - New: lockstep rounds. All K conversations advance one turn at a
    time, all issuing their LLM calls in parallel within a round.
    K simultaneous calls → bridge saturates its multi-stream queue
    (8-way batched AR decode).

For K=12 conversations and ≤8 concurrent calls the bridge can batch,
expected wall time ≈ (n_calls_per_conv / 1) × (avg_call_time / min(K, 8))
vs the old (n_calls_per_conv / 1) × (avg_call_time / 4). About 2x
speedup if K=12 with K_bridge=8.

Per-turn summaries still happen, but in a final all-at-once batch
after the conversations conclude (one parallel pass over every turn).
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "elicitation"))
from llm_client import llm_call, plugin_get, st_character_card

MODEL = os.environ.get("USER_PERSONAS_MODEL", "gemma-4-a4b")


def scringlo_system_prompt():
    card = st_character_card("scringlo_scrambler.png", timeout=10)
    parts = []
    for k in ("description", "personality", "scenario"):
        v = card.get(k) or ""
        if v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def load_personas():
    out = []
    for data in plugin_get("/personas").get("personas", []):
        name = data.get("name") or data.get("id")
        bio = data.get("bio") or data.get("description") or ""
        if not bio.strip():
            continue
        out.append({
            "id": data.get("id"),
            "name": name,
            "bio": bio,
            "system_prompt": data.get("system_prompt") or f"You are {name}. {bio}",
        })
    return out


def bridge_call(messages, *, seed=None):
    return llm_call(messages, seed=seed)


# ────────────────────────────────────────────────────────────────────
# State: each conversation tracks its persona, seed, and two parallel
# histories (user-agent's view + scringlo's view). The scheduler
# walks all conversations in lockstep.
# ────────────────────────────────────────────────────────────────────

def make_conversation_state(persona, scringlo_sys, conv_i, base_seed):
    return {
        "persona_id": persona["id"],
        "conv_i": conv_i,
        "seed": base_seed + conv_i * 1000,
        "user_view": [{"role": "system", "content": persona["system_prompt"]}],
        "scringlo_view": [{"role": "system", "content": scringlo_sys}],
        "transcript": [],
    }


def open_turn_call(state):
    """Generate the user-agent's opening message."""
    messages = state["user_view"] + [{
        "role": "user",
        "content": "Begin your conversation with the character. Write only what you'd type as your first message — no quotes, no narration. Just your opening message.",
    }]
    return bridge_call(messages, seed=state["seed"])


def user_reply_call(state, turn_i):
    """User-agent replies to scringlo's most recent message."""
    return bridge_call(state["user_view"], seed=state["seed"] + 2000 + turn_i)


def scringlo_reply_call(state, turn_i):
    return bridge_call(state["scringlo_view"], seed=state["seed"] + 1000 + turn_i)


def summary_call(text, role_label, seed):
    return bridge_call([
        {"role": "system", "content": "Summarize the following chat turn in ONE concise sentence (under 18 words). Capture what the speaker DID communicatively (asked / claimed / reacted / changed-subject / etc.) and the substance, not just the topic. No preamble; just the sentence."},
        {"role": "user", "content": f"Speaker: {role_label}\nTurn:\n{text}"},
    ], seed=seed)


def run_vectorized(personas, scringlo_sys, *, n_conversations, n_turns, base_seed, max_workers=12):
    """Lockstep rounds. All conversations advance one step at a time, in parallel."""
    states = []
    for p_idx, persona in enumerate(personas):
        for c in range(n_conversations):
            states.append(make_conversation_state(persona, scringlo_sys, c,
                                                  base_seed + p_idx * 1_000_000))

    print(f"[vec] {len(states)} conversations, {n_turns} turns each, max_workers={max_workers}")

    # Round 0: all user-agents open simultaneously.
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        opens = list(pool.map(open_turn_call, states))
    for state, opening in zip(states, opens):
        opening = opening.strip().strip('"').strip("'")
        state["user_view"].append({"role": "assistant", "content": opening})
        state["scringlo_view"].append({"role": "user", "content": opening})
        state["transcript"].append({"role": "user", "text": opening})
    print(f"[vec] round 0 (openers): {time.monotonic() - t0:.1f}s")

    for turn_i in range(n_turns):
        # Scringlo replies (parallel across all conversations)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            replies = list(pool.map(lambda s: scringlo_reply_call(s, turn_i), states))
        for state, sc in zip(states, replies):
            state["scringlo_view"].append({"role": "assistant", "content": sc})
            state["user_view"].append({"role": "user", "content": sc})
            state["transcript"].append({"role": "scringlo", "text": sc})
        print(f"[vec] turn {turn_i} scringlo-reply: {time.monotonic() - t0:.1f}s")

        if turn_i == n_turns - 1:
            break

        # User-agent replies (parallel across all conversations)
        t0 = time.monotonic()
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            user_replies = list(pool.map(lambda s: user_reply_call(s, turn_i), states))
        for state, u in zip(states, user_replies):
            u = u.strip().strip('"').strip("'")
            state["user_view"].append({"role": "assistant", "content": u})
            state["scringlo_view"].append({"role": "user", "content": u})
            state["transcript"].append({"role": "user", "text": u})
        print(f"[vec] turn {turn_i} user-reply: {time.monotonic() - t0:.1f}s")

    # Per-turn summaries — flatten all turns from all conversations and parallel-call.
    all_turns_flat = []
    for state in states:
        for t in state["transcript"]:
            all_turns_flat.append((state, t))

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        summaries = list(pool.map(
            lambda pair: summary_call(
                pair[1]["text"],
                "the user" if pair[1]["role"] == "user" else "scringlo",
                pair[0]["seed"] + 5000,
            ), all_turns_flat))
    for (state, t), summary in zip(all_turns_flat, summaries):
        t["summary"] = summary.strip().strip('"').strip()
    print(f"[vec] {len(all_turns_flat)} summaries: {time.monotonic() - t0:.1f}s")

    # Reshape into the legacy {(persona_id, conv_i): transcript} dict
    results = {}
    for state in states:
        results[(state["persona_id"], state["conv_i"])] = state["transcript"]
    return results


def render_markdown(personas, results, out_path, n_conversations):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# User-agent x scringlo diversity study (vectorized)\n\n")
        f.write(f"- Model: `{MODEL}`\n")
        f.write("- Provider: SillyTavern user-personas /llm-call\n")
        f.write(f"- Personas: {len(personas)}\n")
        f.write(f"- Conversations per persona: {n_conversations}\n")
        f.write(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary grid\n\n")
        f.write("Each row = persona. Each column = one independent conversation.\n\n")
        for persona in personas:
            f.write(f"### {persona['name']} (`{persona['id']}`)\n\n")
            f.write(f"_{persona['bio']}_\n\n")
            for conv_i in range(n_conversations):
                transcript = results.get((persona["id"], conv_i), [])
                f.write(f"**Conversation {conv_i+1}:**\n\n")
                for t in transcript:
                    icon = "🙋" if t["role"] == "user" else "💬"
                    f.write(f"- {icon} {t.get('summary', '(no summary)')}\n")
                f.write("\n")
            f.write("---\n\n")
        f.write("## Full transcripts\n\n")
        for persona in personas:
            f.write(f"### {persona['name']}\n\n")
            for conv_i in range(n_conversations):
                transcript = results.get((persona["id"], conv_i), [])
                f.write(f"#### Conversation {conv_i+1}\n\n")
                for t in transcript:
                    role = "user-agent" if t["role"] == "user" else "scringlo"
                    f.write(f"**{role}:** {t['text']}\n\n")
            f.write("---\n\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-conversations", type=int, default=3)
    p.add_argument("--n-turns", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-workers", type=int, default=12,
                   help="parallelism per lockstep round")
    p.add_argument("--out", default="output/diversity_report_vectorized.md")
    args = p.parse_args()

    personas = load_personas()
    print(f"Loaded {len(personas)} personas: {[p['id'] for p in personas]}")
    scringlo_sys = scringlo_system_prompt()

    wall_t0 = time.monotonic()
    results = run_vectorized(personas, scringlo_sys,
                             n_conversations=args.n_conversations,
                             n_turns=args.n_turns,
                             base_seed=args.seed,
                             max_workers=args.max_workers)
    wall = time.monotonic() - wall_t0
    print(f"[vec] TOTAL WALL: {wall:.1f}s")
    render_markdown(personas, results, args.out, args.n_conversations)
    print(f"Report → {args.out}")


if __name__ == "__main__":
    main()
