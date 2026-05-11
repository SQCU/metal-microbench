#!/usr/bin/env python3
"""User-agent × scringlo diversity study.

For each user-persona in plugins/user-personas/players/, run N independent
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

Talks to the bridge directly at $BRIDGE_URL (default http://127.0.0.1:8001).
No SillyTavern in the loop — we're measuring user-agent diversity,
not ST integration. (ST e2e gets its own Playwright test once the
diversity invariant holds.)
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BRIDGE = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL = "gemma-4-a4b"

PLAYERS_DIR = Path("/Users/mdot/sillytavern-fork/plugins/user-personas/players")
SCRINGLO_PNG = Path("/Users/mdot/sillytavern-fork/default/content/scringlo_scrambler.png")


def read_chara_from_png(path):
    """Pull the embedded chara_card_v3 JSON from a SillyTavern character PNG."""
    with path.open("rb") as f:
        data = f.read()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{path} not a PNG")
    i = 8
    while i < len(data):
        n = struct.unpack(">I", data[i:i+4])[0]
        typ = data[i+4:i+8]
        payload = data[i+8:i+8+n]
        if typ == b"tEXt":
            sep = payload.find(b"\x00")
            if sep >= 0 and payload[:sep] == b"chara":
                return json.loads(base64.b64decode(payload[sep+1:]))
        if typ == b"IEND":
            break
        i += 8 + n + 4
    raise ValueError(f"no chara tEXt chunk in {path}")


def scringlo_system_prompt():
    card = read_chara_from_png(SCRINGLO_PNG)
    parts = []
    for k in ("description", "personality", "scenario"):
        v = card.get(k) or ""
        if v.strip():
            parts.append(v.strip())
    return "\n\n".join(parts)


def load_personas():
    out = []
    for d in sorted(PLAYERS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "manifest.json"
        if not manifest.exists():
            continue
        out.append(json.load(manifest.open()))
    return out


def bridge_call(messages, *, temperature=0.9, max_tokens=400, seed=None, reasoning_effort=None):
    """One non-streaming chat/completions call. Returns assistant text or None."""
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        body["seed"] = seed
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    req = urllib.request.Request(
        f"{BRIDGE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    choice = (resp.get("choices") or [{}])[0]
    return (choice.get("message") or {}).get("content") or ""


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
    user_text = bridge_call(open_messages, temperature=0.95, max_tokens=200, seed=seed)
    user_text = user_text.strip().strip('"').strip("'")
    user_view.append({"role": "assistant", "content": user_text})  # the agent's own emission
    scringlo_view.append({"role": "user", "content": user_text})
    transcript.append({"role": "user", "text": user_text})

    for turn_i in range(n_turns):
        # Scringlo replies.
        sc_text = bridge_call(scringlo_view, temperature=0.85, max_tokens=400, seed=seed + 1000 + turn_i)
        scringlo_view.append({"role": "assistant", "content": sc_text})
        user_view.append({"role": "user", "content": sc_text})  # from agent's view, sc is the chat counterparty
        transcript.append({"role": "assistant", "text": sc_text})

        if turn_i == n_turns - 1:
            break

        # User-agent responds.
        u_text = bridge_call(user_view, temperature=0.95, max_tokens=300, seed=seed + 2000 + turn_i)
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
        ], temperature=0.3, max_tokens=80, seed=seed + 5000)
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
    with open(out_path, "w") as f:
        f.write("# User-agent × scringlo diversity study\n\n")
        f.write(f"- Model: `{MODEL}`\n")
        f.write(f"- Bridge: `{BRIDGE}`\n")
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
    p.add_argument("--out", default="diversity_report.md")
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
