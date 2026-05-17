#!/usr/bin/env python3
"""Drive Zed's agent panel through Hammerspoon to run the agent-clique
test inside a real Zed thread.

Flow per agent message:
  1. Build message text (msg 1 carries the corpus; msgs 2..N just
     carry "now respond as <role>: <task>").
  2. Stage on clipboard via Hammerspoon.
  3. Atomic Lua: focus Zed, paste, submit. Single event-loop tick so
     focus can't be raced by terminal refocus.
  4. Poll the bridge's /health.active_streams until it returns to 0.
  5. Snapshot bridge cache_hits delta from log; report.

Usage:
    python tools/agent_clique_harness/zed_drive.py [--max-tokens N]

This script assumes:
  - Hammerspoon control surface up at localhost:27843
  - Bridge up at localhost:8001
  - Zed running with the agent panel open and our model selected
  - The user has signed off on this scope (see hammerspoon_signoffs.md)
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HS = "http://localhost:27843"
BRIDGE = "http://127.0.0.1:8001"
BRIDGE_LOG = "/tmp/bridge_phase05.log"
CORPUS_DIR = Path(__file__).resolve().parent.parent.parent / "docs" / "agent_clique_corpus"


# Agent definitions match the parallel Python harness. For Zed-driven
# multi-turn use, msg 1 carries the corpus + first agent's role+task;
# subsequent messages are just "now respond as <role>: <task>".
AGENTS = [
    ("critic",
     "critical reviewer who specializes in identifying the weakest "
     "claim in technical writing and naming the evidence that would "
     "falsify it",
     "For each of the three documents, identify ONE weakest claim "
     "and state the smallest experiment that would falsify it. Be "
     "concrete. 200 words total."),
    ("synthesist",
     "cross-document synthesist who reads several pieces of writing "
     "in parallel and identifies the latent thread that connects them",
     "What is the single thread connecting these three documents that "
     "none of them states explicitly? Then: what new question emerges "
     "from reading them together that none asks individually? "
     "200 words total."),
    ("implementor",
     "implementor focused on extracting concrete next steps and "
     "mapping their dependencies",
     "List the top 3 concrete next steps across these documents, "
     "ranked by impact. For each: name the blocking dependency and "
     "the wall-time estimate. 200 words total."),
    ("translator",
     "audience translator who recasts dense technical writing for a "
     "non-technical reader without losing the essential point",
     "Pick the most technical of the three documents. Recast its "
     "core claim in 100 words for a non-technical reader. Then in "
     "100 more words, explain what readers of the original might "
     "miss that your recast preserves. 200 words total."),
]


def hs_post(path, body):
    req = urllib.request.Request(
        f"{HS}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=20).read())


def bridge_health():
    return json.loads(urllib.request.urlopen(
        f"{BRIDGE}/health", timeout=5).read())


def wait_for_idle(deadline_s=600.0):
    """Wait until bridge active_streams == 0. Returns elapsed seconds."""
    t0 = time.time()
    while time.time() - t0 < deadline_s:
        h = bridge_health()
        if h["active_streams"] == 0 and time.time() - t0 > 0.5:
            return time.time() - t0
        time.sleep(0.5)
    raise TimeoutError(f"bridge stayed active >{deadline_s}s")


def bridge_usage_lines_after(timestamp_marker):
    """Return new `usage:` lines from the bridge log appended after
    `timestamp_marker` (a string we wrote to the log via a probe to
    delineate). Falls back to tailing the last few usage entries."""
    try:
        with open(BRIDGE_LOG) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return []
    return [l.strip() for l in lines if "usage:" in l]


def load_corpus_text():
    files = sorted(CORPUS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no .md files in {CORPUS_DIR}")
    parts = ["You will read a small corpus of documents and respond "
             "to a series of role-shifted tasks.\n\nThe corpus:\n"]
    for f in files:
        parts.append(f"\n---\n## {f.name}\n\n{f.read_text()}")
    parts.append("\n---\nEnd of corpus.\n")
    return "".join(parts)


def build_message(agent_idx, corpus_text):
    name, role_blurb, task = AGENTS[agent_idx]
    if agent_idx == 0:
        return (corpus_text + f"\nYou are a {role_blurb}.\n"
                f"\nTask: {task}")
    return (f"Now switch your role: you are a {role_blurb}.\n"
            f"\nTask: {task}")


def send_via_zed(msg_text):
    """Stage clipboard, then run atomic focus+paste+submit Lua."""
    # 1. clipboard
    hs_post("/clipboard", {"contents": msg_text})
    # 2. atomic drive — paste with cmd-v, submit with return.
    #    Some Zed versions take Enter as submit; if it inserts a
    #    newline instead, escalate to cmd-Enter manually.
    lua = """
local zed = hs.application.find('Zed')
if not zed then return 'zed not running' end
zed:activate(true)
hs.timer.usleep(180 * 1000)
hs.eventtap.keyStroke({'cmd'}, 'v', 0)
hs.timer.usleep(250 * 1000)
hs.eventtap.keyStroke({}, 'return', 0)
return 'submitted'
"""
    return hs_post("/eval", {"code": lua})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None,
                    help="run only one agent index (0=critic, 1=synthesist, "
                         "2=implementor, 3=translator)")
    args = ap.parse_args()

    print("=== zed-driven agent clique ===")
    h = bridge_health()
    print(f"bridge: model={h['model']} steps={h['total_steps']} "
          f"cached_pages={h['cached_pages']} active={h['active_streams']}")
    if h['active_streams'] != 0:
        print("WARN: bridge has in-flight streams; waiting for idle")
        wait_for_idle()

    hs_health = json.loads(urllib.request.urlopen(
        f"{HS}/health", timeout=5).read())
    print(f"hammerspoon: accessibility={hs_health['accessibility']}")
    if not hs_health['accessibility']:
        print("FAIL: accessibility not granted; can't drive Zed")
        return 1

    corpus_text = load_corpus_text()
    print(f"corpus: {len(corpus_text):,} chars")

    indices = [args.only] if args.only is not None else list(range(len(AGENTS)))
    pre_usage_count = len([l for l in open(BRIDGE_LOG)
                           if "usage:" in l]) if Path(BRIDGE_LOG).exists() else 0

    for i in indices:
        name = AGENTS[i][0]
        msg = build_message(i, corpus_text)
        print(f"\n[{i+1}/{len(indices)}] agent={name}  msg_len={len(msg):,} chars")

        send_result = send_via_zed(msg)
        print(f"  send: {send_result}")

        # Wait for the bridge to receive + process the request.
        # Streaming starts within ~1s of submit.
        time.sleep(2.0)
        try:
            elapsed = wait_for_idle(deadline_s=300.0)
            print(f"  bridge idle after {elapsed:.1f}s")
        except TimeoutError as e:
            print(f"  ! {e}")
            return 2

        # Read the latest usage line.
        usage_lines = [l.strip() for l in open(BRIDGE_LOG) if "usage:" in l]
        if len(usage_lines) > pre_usage_count:
            new_lines = usage_lines[pre_usage_count:]
            for l in new_lines:
                print(f"  {l}")
            pre_usage_count = len(usage_lines)
        else:
            print(f"  ! no new usage line (was the request submitted?)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
