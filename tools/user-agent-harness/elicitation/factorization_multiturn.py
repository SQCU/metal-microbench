"""Multi-turn factorization experiment.

Given a set of user-agent cards (assumed to share motivation + scenario
but differ in biography) and a target assistant card, run a K-turn
conversation per user-agent card, measure each user-agent turn via the
14-axis cascade, persist to JSONL, and compute pairwise cluster-
separation metrics in Likert space.

The hypothesis: if motivation dominates biography for behavioural
elicitation, the per-card mean signatures will be CLOSE despite the
biography registers being wildly different. If biography dominates,
the means will be SEPARATED.

Usage:
    factorization_multiturn.py \\
        --user-cards factorization-corporate-strategist,factorization-scringlo-fragment,factorization-ten-year-old \\
        --target-assistant python-only-coder \\
        --k 5 \\
        --output /tmp/factorization_multiturn.jsonl
"""

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from axes import AXIS_NAMES, N_AXES
from discovery import _load_assistant_card
from probe_persist import parse_elementwise, stage1_summary, stage2_likert
from signature import mahalanobis

import os as _os_for_bridge_url
BRIDGE_URL = _os_for_bridge_url.environ["BRIDGE_URL"] + "/v1/chat/completions"
PLUGIN_PLAYERS_DIR = Path("/Users/mdot/sillytavern-fork/plugins/user-personas/players")


def bridge_chat(messages, max_tokens=None, temperature=1.0):
    # Generation-config moratorium: no hidden caps. Caller-omitted
    # max_tokens means "use the bridge default" (matches the SillyTavern
    # GUI). temperature=1.0 is the root config (no other value allowed
    # without a documented exemption). See
    # plugins/user-personas/scripts/lint_generation_config.mjs.
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


def load_user_agent_card(card_id):
    p = PLUGIN_PLAYERS_DIR / card_id / "manifest.json"
    if not p.is_file():
        sys.exit(f"user-agent card {card_id!r} not found at {p}")
    return json.loads(p.read_text())


def measure_turn(turn_text):
    """Cascade judge: Stage-1 prose → Stage-2 per-line Likert. Returns
    (likert_dict_possibly_partial, stage1_text)."""
    s = stage1_summary(turn_text)
    raw = stage2_likert(s)
    return parse_elementwise(raw), s


def render_assistant_first_mes(target_card):
    """Best-effort first-message: prefer the card's first_mes, fall back
    to a generic opener if absent."""
    return target_card.get("first_mes") or "Hi! How can I help?"


def render_user_agent_system(user_card):
    """Build the system prompt the user-agent sees. The user-agent's
    materialized manifest has a `system_prompt` field; if absent (for
    hand-authored older cards), we fall back to weaving the persona
    fields together."""
    if user_card.get("system_prompt"):
        return user_card["system_prompt"]
    parts = [
        "You are a user chatting with an AI assistant. Adopt this persona:",
        user_card.get("bio") or user_card.get("description") or "",
    ]
    for k in ("motivation", "scenario", "relationship_to_counterparty", "comm_style"):
        v = user_card.get(k)
        if v: parts.append(f"{k}: {v}")
    if user_card.get("mes_example"):
        parts.append(
            "Keep your turns in a voice/length comparable to this example:\n"
            "> " + str(user_card["mes_example"][0]).replace("\n", "\n> ")
        )
    return "\n\n".join(parts)


def render_target_assistant_system(target_card):
    return (
        target_card.get("system_prompt")
        or target_card.get("description")
        or "You are a helpful assistant."
    )


def run_session(user_card_id, target_card_id, k_turns):
    """Run a K-turn conversation between a user-agent card and a target
    assistant card. Returns a list of records (one per user-agent or
    assistant turn). Each user-agent turn carries its measured Likert.
    """
    user_card = load_user_agent_card(user_card_id)
    target_card = _load_assistant_card(target_card_id)
    if target_card is None:
        sys.exit(f"target assistant card {target_card_id!r} not found")

    # Canonical message log from the TARGET ASSISTANT's POV.
    canonical = [
        {"role": "system", "content": render_target_assistant_system(target_card)},
        {"role": "assistant", "content": render_assistant_first_mes(target_card)},
    ]

    user_agent_system = render_user_agent_system(user_card)
    records = []

    for k in range(k_turns):
        # User-agent's POV: swap roles of every prior assistant↔user
        # exchange, prepend the user-agent's own system prompt.
        ua_messages = [{"role": "system", "content": user_agent_system}]
        for m in canonical[1:]:
            ua_messages.append({
                "role": "user" if m["role"] == "assistant" else "assistant",
                "content": m["content"],
            })

        t0 = time.monotonic()
        ua_text = bridge_chat(ua_messages)
        ua_dt = time.monotonic() - t0
        canonical.append({"role": "user", "content": ua_text})

        # Cascade-judge the user-agent's turn.
        t0 = time.monotonic()
        likert, stage1_text = measure_turn(ua_text)
        judge_dt = time.monotonic() - t0
        records.append({
            "kind": "user_agent_turn",
            "card_id": user_card_id,
            "target_id": target_card_id,
            "turn_idx": k,
            "text": ua_text,
            "judge_stage1": stage1_text,
            "likert": likert,
            "axes_recovered": len(likert),
            "ua_wallclock_s": round(ua_dt, 2),
            "judge_wallclock_s": round(judge_dt, 2),
        })

        # Target assistant responds.
        t0 = time.monotonic()
        ta_text = bridge_chat(canonical)
        ta_dt = time.monotonic() - t0
        canonical.append({"role": "assistant", "content": ta_text})
        records.append({
            "kind": "assistant_turn",
            "card_id": target_card_id,
            "turn_idx": k,
            "text": ta_text,
            "wallclock_s": round(ta_dt, 2),
        })

    return records


# ─── Cluster-separation analysis ──────────────────────────────────────

def compute_per_card_signatures(all_records):
    """Aggregate Likert across each user-agent's K turns into a per-card
    mean + std. Returns {card_id: {"mean": np.array, "std": np.array, "n": int}}."""
    bucket: dict[str, list[np.ndarray]] = {}
    for r in all_records:
        if r["kind"] != "user_agent_turn":
            continue
        likert = r.get("likert") or {}
        if len(likert) != N_AXES:
            # Partial-Likert turns are dropped from signature aggregation
            # but still kept in the raw JSONL.
            continue
        vec = np.array([likert[a] for a in AXIS_NAMES], dtype=float)
        bucket.setdefault(r["card_id"], []).append(vec)
    out = {}
    for cid, vecs in bucket.items():
        X = np.stack(vecs)
        out[cid] = {
            "mean": X.mean(axis=0),
            "std": X.std(axis=0, ddof=0),
            "n": int(X.shape[0]),
            "samples": X,
        }
    return out


def compute_pooled_within_cov(signatures):
    """Pooled within-card covariance: each card's centered turn-block
    contributes (X - mean(X)). The pooled estimate's degrees-of-freedom
    denominator is N_total - K (K = number of cards)."""
    centered = []
    for cid, sig in signatures.items():
        if sig["n"] < 2:
            continue
        centered.append(sig["samples"] - sig["mean"])
    if not centered:
        return None
    C = np.vstack(centered)
    K = sum(1 for sig in signatures.values() if sig["n"] >= 2)
    n_total = C.shape[0]
    if n_total <= K:
        return None
    cov = (C.T @ C) / (n_total - K)
    # Ridge for invertibility (Likert ordinals at K=5 often produce
    # singular covariance on axes with zero within-card variance).
    ridge = 1e-3 * np.trace(cov) / N_AXES * np.eye(N_AXES)
    return cov + ridge


def pairwise_mahalanobis(signatures, cov_inv):
    ids = sorted(signatures.keys())
    out = {}
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            out[(a, b)] = mahalanobis(
                signatures[a]["mean"],
                signatures[b]["mean"],
                cov_inv,
            )
    return out


# ─── Driver ───────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--user-cards", required=True,
                   help="comma-separated user-agent card_ids in players/")
    p.add_argument("--target-assistant", required=True,
                   help="assistant card name (e.g. python-only-coder)")
    p.add_argument("--k", type=int, default=5, help="turns per session")
    p.add_argument("--output", required=True, help="JSONL output path")
    args = p.parse_args()

    user_card_ids = [c.strip() for c in args.user_cards.split(",") if c.strip()]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    with out_path.open("w") as f:
        for cid in user_card_ids:
            print(f"\n═══ session: {cid} ↔ {args.target_assistant} ═══")
            t0 = time.monotonic()
            try:
                recs = run_session(cid, args.target_assistant, args.k)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            dt = time.monotonic() - t0
            print(f"  {len(recs)} records in {dt:.1f}s")
            for r in recs:
                f.write(json.dumps(r) + "\n")
                f.flush()
                all_records.append(r)
                if r["kind"] == "user_agent_turn":
                    n_axes = r.get("axes_recovered", 0)
                    tag = "FULL" if n_axes == N_AXES else f"PART({n_axes}/{N_AXES})"
                    print(f"    turn {r['turn_idx']}: {tag}  {r['text'][:90]!r}")

    print("\n" + "═" * 72)
    print("  ANALYSIS — per-card signatures + pairwise cluster separation")
    print("═" * 72)
    signatures = compute_per_card_signatures(all_records)
    if not signatures:
        print("  no full-Likert user-agent turns measured; nothing to compare")
        return

    # Per-card mean + std
    print(f"  {'axis':<22s}  " + "  ".join(f"{cid[:22]:<22s}" for cid in sorted(signatures.keys())))
    print("  " + "─" * (22 + len(signatures) * 24))
    for j, axis in enumerate(AXIS_NAMES):
        cells = []
        for cid in sorted(signatures.keys()):
            sig = signatures[cid]
            cells.append(f"{sig['mean'][j]:.1f}±{sig['std'][j]:.1f} (n={sig['n']})")
        print(f"  {axis:<22s}  " + "  ".join(f"{c:<22s}" for c in cells))

    # Pairwise Mahalanobis under the pooled within-card covariance.
    cov = compute_pooled_within_cov(signatures)
    if cov is None:
        print("\n  pooled covariance unestimable (need ≥2 samples per ≥2 cards)")
        return
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        print("\n  pooled covariance singular even after ridge; falling back to Euclidean")
        cov_inv = np.eye(N_AXES)
    dists = pairwise_mahalanobis(signatures, cov_inv)
    print("\n  Pairwise distance between per-card mean signatures:")
    for (a, b), d in sorted(dists.items(), key=lambda kv: -kv[1]):
        if d > 3.0:    tag = "VERY DISTINCT"
        elif d > 2.0:  tag = "distinct"
        elif d > 1.0:  tag = "separated"
        else:          tag = "OVERLAPPING (similar)"
        print(f"    {a[:30]:<30s} ↔ {b[:30]:<30s}  d = {d:.2f}  [{tag}]")

    # Per-axis "did this axis differ between cards" sanity check —
    # for the factorization claim we want most axes to OVERLAP across
    # cards (i.e. the motivation pulls them to similar Likert positions
    # despite the biography prose being utterly different).
    print("\n  Per-axis between-card spread (max - min of per-card means):")
    spreads = []
    for j, axis in enumerate(AXIS_NAMES):
        means = [signatures[cid]["mean"][j] for cid in sorted(signatures.keys())]
        spread = max(means) - min(means)
        spreads.append((axis, spread))
    spreads.sort(key=lambda x: -x[1])
    for axis, spread in spreads:
        tag = "WIDE" if spread > 2.0 else ("med" if spread > 1.0 else "tight")
        bar = "█" * int(spread * 5) + "░" * (10 - int(spread * 5))
        print(f"    {axis:<22s}  {spread:.2f}  {bar}  [{tag}]")


if __name__ == "__main__":
    main()
