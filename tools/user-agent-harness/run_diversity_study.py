#!/usr/bin/env python3
"""User-agent x scringlo diversity study.

For each user-persona in the plugin corpus, run N independent
M-turn conversations with scringlo (using the canonical scringlo
system prompt). For each turn, generate a 1-sentence summary via a
separate LLM call. Output a markdown grid showing the K × N
conversations side-by-side so we can eyeball whether persona-rows
are distinguishable from each other.

If two rows blur together → the persona space is collapsing those
two into the same behavioral output; iterate the persona definitions.

If rows are visibly distinct → the architecture works; we can scale
up the persona count and conversation count, and graduate the
summarizer from this inline LLM call to a real toolcard.

Generation goes through the user-personas plugin /llm-call shim, which
uses SillyTavern's configured provider profile.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def bridge_call(messages, *, seed=None, reasoning_effort=None):
    del reasoning_effort
    return llm_call(messages, seed=seed)


def run_conversation(persona, scringlo_sys, *, n_turns, seed):
    """Drive N turns of (user-agent, scringlo) back-and-forth.

    Returns a list of turn dicts:
      {"role": "user"|"assistant", "text": str, "summary": str}

    The user-agent is the persona; scringlo is the assistant. Both
    sides are LLM-driven against the same bridge, just with different
    system prompts. Per-turn summary is a separate small LLM call.
    """
    # Two parallel histories, one from each side's perspective.
    # From the user-agent's view: their assigned persona is "system",
    #   scringlo's turns are "assistant", their own turns are "user".
    # From scringlo's view: scringlo's persona is "system", user's
    #   turns are "user", scringlo's own turns are "assistant".
    user_view = [{"role": "system", "content": persona["system_prompt"]}]
    scringlo_view = [{"role": "system", "content": scringlo_sys}]

    transcript = []

    # User-agent opens (no prior assistant turn yet; they have to start fresh).
    open_messages = user_view + [{
        "role": "user",
        "content": "Begin your conversation with the character. Write only what you'd type as your first message — no quotes, no narration, no stage directions about yourself. Just your opening message.",
    }]
    user_text = bridge_call(open_messages, seed=seed)
    user_text = user_text.strip().strip('"').strip("'")
    user_view.append({"role": "assistant", "content": user_text})  # the agent's own emission
    scringlo_view.append({"role": "user", "content": user_text})
    transcript.append({"role": "user", "text": user_text})

    for turn_i in range(n_turns):
        # Scringlo replies.
        sc_text = bridge_call(scringlo_view, seed=seed + 1000 + turn_i)
        scringlo_view.append({"role": "assistant", "content": sc_text})
        user_view.append({"role": "user", "content": sc_text})  # from agent's view, sc is the chat counterparty
        transcript.append({"role": "assistant", "text": sc_text})

        if turn_i == n_turns - 1:
            break

        # User-agent responds.
        u_text = bridge_call(user_view, seed=seed + 2000 + turn_i)
        u_text = u_text.strip().strip('"').strip("'")
        user_view.append({"role": "assistant", "content": u_text})
        scringlo_view.append({"role": "user", "content": u_text})
        transcript.append({"role": "user", "text": u_text})

    # Per-turn summarization. One LLM call per turn, ~1 sentence each.
    for t in transcript:
        speaker = "the user" if t["role"] == "user" else "scringlo"
        summary = bridge_call([
            {"role": "system", "content": "Summarize the following chat turn in ONE concise sentence (under 18 words). Capture what the speaker DID communicatively (asked / claimed / reacted / changed-subject / etc.) and the substance, not just the topic. No preamble; just the sentence."},
            {"role": "user", "content": f"Speaker: {speaker}\nTurn:\n{t['text']}"},
        ], seed=seed + 5000)
        t["summary"] = summary.strip().strip('"').strip()

    return transcript


def run_all(personas, scringlo_sys, *, n_conversations, n_turns, base_seed):
    """For each persona × n_conversations independent runs, in parallel."""
    futures = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for p_idx, persona in enumerate(personas):
            for conv_i in range(n_conversations):
                seed = base_seed + p_idx * 1_000_000 + conv_i * 1000
                fut = pool.submit(run_conversation, persona, scringlo_sys, n_turns=n_turns, seed=seed)
                futures[fut] = (persona["id"], conv_i)
        results = {}
        for fut in as_completed(futures):
            pid, conv_i = futures[fut]
            try:
                results[(pid, conv_i)] = fut.result()
                print(f"[done] {pid} conv {conv_i}")
            except Exception as e:
                print(f"[fail] {pid} conv {conv_i}: {e}", file=sys.stderr)
                results[(pid, conv_i)] = [{"role": "user", "text": f"(failed: {e})", "summary": "(error)"}]
    return results


def render_markdown(personas, results, out_path, n_conversations):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# User-agent x scringlo diversity study\n\n")
        f.write(f"- Model: `{MODEL}`\n")
        f.write("- Provider: SillyTavern user-personas /llm-call\n")
        f.write(f"- Personas: {len(personas)}\n")
        f.write(f"- Conversations per persona: {n_conversations}\n")
        f.write(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Summary grid\n\n")
        f.write("Each row = persona. Each column = one independent conversation. Cells show per-turn summaries (~1 sentence each, alternating user/scringlo).\n\n")
        for persona in personas:
            f.write(f"### {persona['name']} (`{persona['id']}`)\n\n")
            f.write(f"_{persona['bio']}_\n\n")
            for conv_i in range(n_conversations):
                transcript = results.get((persona["id"], conv_i), [])
                f.write(f"**Conversation {conv_i+1}:**\n\n")
                for t in transcript:
                    icon = "🙋" if t["role"] == "user" else "💬"
                    f.write(f"- {icon} {t['summary']}\n")
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
    p.add_argument("--n-conversations", type=int, default=5)
    p.add_argument("--n-turns", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="output/diversity_report.md")
    args = p.parse_args()

    personas = load_personas()
    print(f"Loaded {len(personas)} personas: {[p['id'] for p in personas]}")
    scringlo_sys = scringlo_system_prompt()
    print(f"Scringlo system prompt: {len(scringlo_sys)} chars")

    results = run_all(personas, scringlo_sys,
                      n_conversations=args.n_conversations,
                      n_turns=args.n_turns,
                      base_seed=args.seed)
    render_markdown(personas, results, args.out, args.n_conversations)
    print(f"Report → {args.out}")


if __name__ == "__main__":
    main()
