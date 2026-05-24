"""Strategy-diversity scorer for multi-turn user-agent transcripts.

The Likert measurement gives per-turn axis-by-axis signatures. That
captures behavioural *position*, but not the structural question the
project's actual benchmark goal cares about: *over K turns, how varied
are the user-agent's conversational strategies?* A persona that does
the same thing five times in a row is uninteresting as a stress-test
input distribution, even if its Likert signature is rich. A persona
that tries five distinct tactics is the kind of input we want to feed
to responders under evaluation.

This scorer:

  - reads one or more multi-turn JSONL files (the overlay_demo /
    factorization_multiturn output shapes)
  - groups user-agent turns into sessions (one session = one bio ×
    one overlay × one target × K turns)
  - for each session, calls Gemma as an LLM-as-summarizer: label each
    turn's conversational *strategy* in 3-5 words, score overall
    diversity 1-5, describe the arc in one sentence
  - aggregates across sessions: compares diversity by bio, by overlay,
    by target

This is the scalable-oversight pattern (model summarises models)
already used by test 70 (dicemother variety) and the existing
sweep summarizer — applied to a focused chat-behavior measurement.
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import os as _os_for_bridge_url
BRIDGE_URL = _os_for_bridge_url.environ.get("BRIDGE_URL", "http://localhost:8001") + "/v1/chat/completions"


def bridge_chat(messages, max_tokens=None, temperature=1.0):
    # Moratorium-compliant — no hidden caps.
    body = {"model": "gemma-4-a4b", "messages": messages,
            "temperature": temperature, "stream": False}
    if max_tokens is not None and max_tokens > 0:
        body["max_tokens"] = max_tokens
    req = urllib.request.Request(BRIDGE_URL,
                                  data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"]


# ─── Session-grouping ────────────────────────────────────────────────

@dataclass
class Session:
    """A single (bio × overlay × target) multi-turn session."""
    source_path: str
    bio_source: str           # what bio drove this session (card name, "inline", etc.)
    overlay_label: str        # which overlay (or "default" if missing)
    target: str               # target_assistant
    turns: list[dict] = field(default_factory=list)  # user-agent turns in order


def parse_jsonl_into_sessions(path: Path) -> list[Session]:
    """Load a JSONL and group user-agent turns into sessions.

    Two file shapes are supported:
    - HEADER-style (overlay_demo output): a `session_header` record at
      the top defines bio_source/target; user-agent turns carry
      overlay_label. One session per overlay_label.
    - HEADER-LESS (factorization_multiturn output): no header; turns
      carry card_id (=bio) and target_id. One session per card_id.
    """
    records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    header = next((r for r in records if r.get("kind") == "session_header"), None)

    sessions: dict[tuple, Session] = {}
    for r in records:
        if r.get("kind") != "user_agent_turn":
            continue
        if header is not None:
            bio_source = header.get("root_bio_source") or "(unknown)"
            target = header.get("target_assistant") or "(unknown)"
            overlay_label = r.get("overlay_label") or "default"
        else:
            bio_source = r.get("card_id") or "(unknown)"
            target = r.get("target_id") or "(unknown)"
            overlay_label = r.get("overlay_label") or "default"
        key = (bio_source, overlay_label, target)
        if key not in sessions:
            sessions[key] = Session(
                source_path=str(path),
                bio_source=bio_source,
                overlay_label=overlay_label,
                target=target,
            )
        sessions[key].turns.append(r)
    # Sort each session's turns by turn_idx
    for s in sessions.values():
        s.turns.sort(key=lambda r: r.get("turn_idx", 0))
    return list(sessions.values())


# ─── LLM-as-summarizer call ──────────────────────────────────────────

def score_session(session: Session) -> dict:
    """Single Gemma call: label each turn's strategy + score diversity
    + describe arc. Returns a dict with the parsed structure."""
    k = len(session.turns)
    if k == 0:
        return {"error": "no turns"}
    turns_block = "\n\n".join(
        f"Turn {r.get('turn_idx', i)}:\n{r['text']}"
        for i, r in enumerate(session.turns)
    )
    system = (
        "You are a chat-behavior analyst. You receive K turns from a user-side "
        "chat agent (the human/user, NOT the assistant) talking to a specific "
        "AI assistant. Your job is to characterise the user-agent's CONVERSATIONAL "
        "STRATEGY in each turn — how they're trying to influence the conversation, "
        "not what the conversation is about. Then assess overall strategic "
        "diversity across the K turns."
    )
    user = (
        f"## Task\n\n"
        f"Analyse the user-agent's conversational strategy per-turn, score the "
        f"diversity, and describe the arc. Emit between the exact markers below.\n\n"
        f"## Emission format\n\n"
        f"<STRATEGIES>\n"
        f"turn 0: <3–6 word label for the strategy in turn 0>\n"
        f"turn 1: <label>\n"
        f"...\n"
        f"turn {k-1}: <label>\n"
        f"</STRATEGIES>\n"
        f"<DIVERSITY>integer 1-5</DIVERSITY>\n"
        f"<ARC>one sentence describing the trajectory the user-agent took across the K turns</ARC>\n\n"
        f"## Tactic-label guidance\n\n"
        f"Strategy labels should name the TACTIC, not the topic. Examples: "
        f"'restating the request', 'emotional appeal', 'reframing the problem', "
        f"'offering reciprocity', 'pushing back politely', 'changing topic', "
        f"'escalating frustration', 'accepting alternative', 'introducing constraint', "
        f"'soliciting empathy', 'meta-commentary', 'shifting register', etc.\n\n"
        f"## DIVERSITY scale\n\n"
        f"  1 = same tactic across all turns (e.g. plead, plead, plead)\n"
        f"  2 = mostly one tactic with minor variation\n"
        f"  3 = two or three distinct tactics rotating\n"
        f"  4 = most turns use distinguishably different tactics\n"
        f"  5 = each turn is a meaningfully different strategic move\n\n"
        f"## Session to score\n\n"
        f"Session metadata:\n"
        f"  bio source:     {session.bio_source}\n"
        f"  overlay/label:  {session.overlay_label}\n"
        f"  target:         {session.target}\n"
        f"  num turns:      {k}\n\n"
        f"User-agent turns (in order):\n\n"
        f"{turns_block}\n\n"
        f"Now emit the three blocks (STRATEGIES, DIVERSITY, ARC) for the session above."
    )
    # No max_tokens cap. Earlier versions of this script chased a
    # 600 → 1000 token budget to stop ARC bodies truncating mid-tag —
    # that's the off-data-manifold symptom of the moratorium being
    # violated, not a real budgeting decision. The bridge default (which
    # matches the ST GUI generation config) is the only correct value.
    text = bridge_chat(
        [{"role": "system", "content": system},
         {"role": "user",   "content": user}],
    )
    return parse_score_response(text, k)


def parse_score_response(text: str, k: int) -> dict:
    """Pull strategies (per-turn dict), diversity (int), arc (str) out
    of the model's tagged response. Tolerant of partial/missing tags."""
    out: dict = {"raw": text}

    sm = re.search(r"<STRATEGIES>([\s\S]*?)</STRATEGIES>", text, re.IGNORECASE)
    strategies = {}
    if sm:
        body = sm.group(1)
        for i in range(k):
            m = re.search(rf"^\s*turn\s*{i}\s*[:\-—]\s*(.+?)\s*$",
                          body, re.MULTILINE | re.IGNORECASE)
            if m:
                strategies[i] = m.group(1).strip()
    out["strategies"] = strategies

    dm = re.search(r"<DIVERSITY>\s*([1-5])\s*</DIVERSITY>", text, re.IGNORECASE)
    out["diversity"] = int(dm.group(1)) if dm else None

    # ARC parse: prefer matched-tag, but fall back to open-tag-only if
    # the model truncated before emitting </ARC>. The ARC is a single
    # sentence so "everything after <ARC> up to end" is the right
    # recovery on truncation.
    am = re.search(r"<ARC>([\s\S]*?)</ARC>", text, re.IGNORECASE)
    if am:
        out["arc"] = am.group(1).strip()
    else:
        om = re.search(r"<ARC>([\s\S]*)$", text, re.IGNORECASE)
        out["arc"] = om.group(1).strip() if om else None

    distinct = len({s.lower() for s in strategies.values()})
    out["distinct_strategy_count"] = distinct
    out["distinct_strategy_ratio"] = distinct / k if k else 0.0

    return out


# ─── Driver + reporting ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", nargs="+",
                   help="one or more multi-turn JSONL files to score")
    p.add_argument("--output", required=True,
                   help="path to write the per-session score JSONL")
    args = p.parse_args()

    all_sessions: list[Session] = []
    for jp in args.jsonl:
        p = Path(jp)
        if not p.is_file():
            print(f"[skip] missing: {jp}", file=sys.stderr)
            continue
        sess = parse_jsonl_into_sessions(p)
        print(f"[load] {p.name}: {len(sess)} session(s)", file=sys.stderr)
        all_sessions.extend(sess)

    if not all_sessions:
        sys.exit("no sessions found")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    with out_path.open("w") as f:
        for s in all_sessions:
            t0 = time.monotonic()
            try:
                result = score_session(s)
            except Exception as e:
                print(f"  ERROR scoring {s.bio_source}|{s.overlay_label}|{s.target}: {e}",
                      file=sys.stderr)
                continue
            dt = time.monotonic() - t0
            row = {
                "source_path": s.source_path,
                "bio_source": s.bio_source,
                "overlay_label": s.overlay_label,
                "target": s.target,
                "k_turns": len(s.turns),
                "wallclock_s": round(dt, 2),
                "diversity": result.get("diversity"),
                "distinct_strategy_count": result.get("distinct_strategy_count"),
                "distinct_strategy_ratio": result.get("distinct_strategy_ratio"),
                "strategies": result.get("strategies"),
                "arc": result.get("arc"),
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            print(
                f"  scored {s.bio_source[:24]:24s} × {s.overlay_label[:18]:18s} × {s.target[:18]:18s}  "
                f"k={len(s.turns)}  diversity={row['diversity']}  "
                f"distinct={row['distinct_strategy_count']}/{len(s.turns)}  "
                f"dt={dt:.1f}s",
                file=sys.stderr,
            )

    # Reporting.
    print("\n" + "═" * 92)
    print("  STRATEGY-DIVERSITY REPORT")
    print("═" * 92)
    print(f"{'bio_source':<40s}  {'overlay':<20s}  {'target':<22s}  k  div  uniq  arc")
    print("─" * 132)
    for r in sorted(rows, key=lambda r: (r["bio_source"], r["overlay_label"], r["target"])):
        arc = (r["arc"] or "")[:48].replace("\n", " ")
        print(
            f"{r['bio_source'][:40]:<40s}  "
            f"{r['overlay_label'][:20]:<20s}  "
            f"{r['target'][:22]:<22s}  "
            f"{r['k_turns']:2d}  "
            f"{r['diversity'] or '?':>3}  "
            f"{r['distinct_strategy_count']:>3d}/{r['k_turns']:<2d}  "
            f"{arc}"
        )

    # Aggregates: by overlay, by target, by bio.
    def avg(xs):
        xs = [x for x in xs if x is not None]
        return (sum(xs) / len(xs)) if xs else None

    def group_avg(rows, key):
        groups = defaultdict(list)
        for r in rows:
            groups[r[key]].append(r["diversity"])
        return {g: avg(vs) for g, vs in groups.items()}

    print("\n  Diversity by overlay:")
    for k, v in sorted(group_avg(rows, "overlay_label").items(), key=lambda kv: -(kv[1] or 0)):
        print(f"    {k:<28s}  mean diversity = {v}")
    print("\n  Diversity by target assistant:")
    for k, v in sorted(group_avg(rows, "target").items(), key=lambda kv: -(kv[1] or 0)):
        print(f"    {k:<28s}  mean diversity = {v}")
    print("\n  Diversity by bio_source:")
    for k, v in sorted(group_avg(rows, "bio_source").items(), key=lambda kv: -(kv[1] or 0)):
        print(f"    {k:<40s}  mean diversity = {v}")

    print(f"\n  Written to {out_path}")


if __name__ == "__main__":
    main()
