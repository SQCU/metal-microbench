"""Compare drift profiles between two overlay_demo JSONL runs.

Reads two JSONLs (typically: an old capped run + a new uncapped run),
groups each by overlay_label, calls /trajectory-judge on each session,
and prints a per-overlay drift profile side-by-side.

The point: quantify how much the (now-removed) max_tokens caps shifted
the measured 14-axis signatures + drift metrics. Big shift → the old
corpus was contaminated; small shift → the caps were mostly harmless
(operating below the model's natural budget anyway).

Usage:
    drift_compare.py --old data/overlay_demo_v2.jsonl \\
                     --new data/overlay_demo_uncapped.jsonl
"""

import argparse
import concurrent.futures as cf
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import os as _os_for_bridge_base
BRIDGE_BASE = _os_for_bridge_base.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
LIKERT_AXES = [
    "curious", "terse", "warm", "deferential", "performative",
    "in_character", "affective_intensity", "probe_depth", "goal_clarity",
    "disclosive", "provocative", "register_colloquial", "playful", "structured",
]


def load_sessions(jsonl_path: Path) -> dict[str, list[dict]]:
    """Group user_agent_turn + assistant_turn records by overlay_label."""
    by_overlay = defaultdict(list)
    initial_assistant = None
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            kind = r.get("kind")
            ov = r.get("overlay_label")
            if kind == "session_header":
                # Some headers carry the initial assistant first_mes.
                continue
            if kind in ("user_agent_turn", "assistant_turn") and ov:
                by_overlay[ov].append(r)
    # Build /trajectory-judge-compatible trajectory per overlay.
    out = {}
    for ov, records in by_overlay.items():
        # Sort by turn_idx so trajectory order is stable.
        records.sort(key=lambda r: (r.get("turn_idx") or 0, 0 if r["kind"] == "user_agent_turn" else 1))
        traj = []
        for r in records:
            traj.append({
                "kind": "user" if r["kind"] == "user_agent_turn" else "assistant",
                "turn_idx": r.get("turn_idx") or 0,
                "text": r.get("text") or "",
            })
        out[ov] = traj
    return out


def _bridge_chat(messages: list[dict]) -> str:
    """Direct bridge chat-completion call. Returns text only. No max_tokens
    cap (moratorium). No temperature override (root config = 1.0)."""
    body = json.dumps({"messages": messages, "stream": False}).encode()
    req = urllib.request.Request(
        f"{BRIDGE_BASE}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"].strip()


_JUDGE_SYSTEM = (
    "You read one chat turn from a user-side chat agent and emit a 14-axis "
    "Likert assessment of its behavioural signature (1=low, 5=high per axis). "
    "Output ONLY tagged blocks like <curious>3</curious> for each axis. "
    "User registers vary widely and ALL ARE LEGITIMATE: formal business prose, "
    "casual chat, emoji-heavy lowercase playful style, terse command-style, "
    "long technical code-review prose, skeptical probing, fan-style enthusiasm. "
    "Score the strategy, not the surface aesthetics."
)
_AXES_LISTING = "\n".join(f"  {a}: 1-5" for a in LIKERT_AXES)
_JUDGE_USER_PREFIX = (
    "## Axes (each scored 1–5)\n\n"
    + _AXES_LISTING + "\n\n"
    "## Emission format\n\n"
    "Emit one tagged block per axis, like `<curious>3</curious>`. No prose, "
    "no JSON, no markdown fences.\n\n"
    "## Turn to score\n\n"
)


def judge_turn(turn_text: str) -> dict | None:
    """Single merged judge call. Returns the 14-axis dict, or None if the
    model failed to emit a complete signature."""
    raw = _bridge_chat([
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER_PREFIX + f"> {turn_text}\n\nNow emit the 14 tagged blocks."},
    ])
    out = {}
    for a in LIKERT_AXES:
        m = re.search(rf"<{a}>\s*(\d+)\s*</{a}>", raw, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if 1 <= v <= 5:
                out[a] = v
    return out if len(out) >= 14 else None


def judge_trajectory(trajectory: list[dict], max_concurrent: int = 8) -> dict:
    """Judge every user turn in the trajectory in parallel, then compute
    drift over the resulting 14-axis sequence. Parallelism matches the
    bridge engine's max_b so the judge calls share KV-page prefix without
    over-saturating the pool."""
    user_turns = [t for t in trajectory
                  if t.get("kind") == "user" and (t.get("text") or "").strip()]
    sig_vecs: list[list[float]] = []
    with cf.ThreadPoolExecutor(max_workers=max_concurrent) as ex:
        futures = [ex.submit(judge_turn, t["text"]) for t in user_turns]
        for fut in futures:
            try:
                lik = fut.result()
            except Exception as e:
                print(f"    judge ERROR (turn skipped): {e}")
                continue
            if lik and len(lik) >= 14:
                sig_vecs.append([float(lik[a]) for a in LIKERT_AXES])
    n = len(sig_vecs)
    if n == 0:
        return {"_meta": {"n_valid": 0}, "centroid": None, "drift": None,
                "centroid_distance_per_turn": []}
    d = len(sig_vecs[0])
    centroid = [sum(v[i] for v in sig_vecs) / n for i in range(d)]
    def sub(a, b): return [a[i] - b[i] for i in range(d)]
    def norm(v): return sum(x * x for x in v) ** 0.5
    step_norms = [norm(sub(sig_vecs[i+1], sig_vecs[i])) for i in range(n-1)]
    return {
        "_meta": {"n_valid": n},
        "centroid": dict(zip(LIKERT_AXES, centroid)),
        "drift": {
            "mean_drift": sum(step_norms)/len(step_norms) if step_norms else None,
            "max_drift": max(step_norms) if step_norms else None,
            "total_path_length": sum(step_norms),
            "net_displacement": norm(sub(sig_vecs[-1], sig_vecs[0])) if n >= 2 else None,
            "path_efficiency": (norm(sub(sig_vecs[-1], sig_vecs[0])) / sum(step_norms))
                if step_norms and sum(step_norms) > 0 else None,
            "per_step": [{"norm": s} for s in step_norms],
        },
        "centroid_distance_per_turn": [{"distance": norm(sub(v, centroid))} for v in sig_vecs],
    }


def inline_drift(records: list[dict]) -> dict | None:
    """Compute drift directly from inline `likert` fields in old records.

    Mirrors the /trajectory-judge math but skips the LLM call — uses the
    already-stored Stage-2 axis values from the old run. This lets us
    compare drift profiles between old (recover from JSONL) and new
    (compute fresh from text) without re-judging the old text.
    """
    user_turns = [r for r in records if r["kind"] == "user_agent_turn"]
    axes = None
    sig_vecs = []
    for r in user_turns:
        lik = r.get("likert") or {}
        if not lik or len(lik) < 14:
            continue
        if axes is None:
            axes = list(lik.keys())
        sig_vecs.append([float(lik.get(a, 0)) for a in axes])
    if len(sig_vecs) < 1:
        return None
    n = len(sig_vecs)
    d = len(sig_vecs[0])
    centroid = [sum(v[i] for v in sig_vecs) / n for i in range(d)]

    def sub(a, b):
        return [a[i] - b[i] for i in range(d)]

    def norm(v):
        return sum(x * x for x in v) ** 0.5

    if n < 2:
        return {
            "n_valid": n,
            "centroid": dict(zip(axes, centroid)),
            "mean_drift": None,
            "max_drift": None,
            "total_path_length": None,
            "net_displacement": None,
            "path_efficiency": None,
            "centroid_distance_per_turn": [norm(sub(v, centroid)) for v in sig_vecs],
        }
    step_norms = [norm(sub(sig_vecs[i + 1], sig_vecs[i])) for i in range(n - 1)]
    total_path = sum(step_norms)
    net_disp = norm(sub(sig_vecs[-1], sig_vecs[0]))
    return {
        "n_valid": n,
        "centroid": dict(zip(axes, centroid)),
        "mean_drift": sum(step_norms) / len(step_norms),
        "max_drift": max(step_norms),
        "min_drift": min(step_norms),
        "total_path_length": total_path,
        "net_displacement": net_disp,
        "path_efficiency": (net_disp / total_path) if total_path > 0 else None,
        "centroid_distance_per_turn": [norm(sub(v, centroid)) for v in sig_vecs],
        "per_step_norms": step_norms,
    }


def fmt_drift(d, label):
    if not d:
        return f"  {label}: (no drift data)"
    lines = [f"  {label}:"]
    if d.get("mean_drift") is not None:
        lines.append(f"    mean_drift = {d['mean_drift']:.2f}")
        lines.append(f"    max_drift  = {d['max_drift']:.2f}")
        lines.append(f"    total_path = {d['total_path_length']:.2f}")
        lines.append(f"    net_disp   = {d['net_displacement']:.2f}")
        eff = d.get("path_efficiency")
        if eff is not None:
            lines.append(f"    path_eff   = {eff:.2f}")
        lines.append(f"    per_step   = {[round(x, 2) for x in d.get('per_step_norms', [])]}")
    lines.append(f"    cd_per_turn = {[round(x, 2) for x in d.get('centroid_distance_per_turn', [])]}")
    return "\n".join(lines)


def diff(old_v, new_v):
    if old_v is None or new_v is None:
        return "—"
    return f"{new_v - old_v:+.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True, type=Path,
                   help="capped-era JSONL (uses inline likert)")
    p.add_argument("--new", required=True, type=Path,
                   help="uncapped-era JSONL (judged via /trajectory-judge)")
    args = p.parse_args()

    print(f"=== drift_compare.py ===")
    print(f"old (inline-likert): {args.old.name}")
    print(f"new (fresh-judge):   {args.new.name}")
    print()

    # Old: load records, group by overlay, compute drift from inline likert.
    print("[1/2] Loading old JSONL + computing drift from inline likert...")
    old_by_overlay_records = defaultdict(list)
    with args.old.open() as f:
        for line in f:
            r = json.loads(line)
            ov = r.get("overlay_label")
            if r.get("kind") in ("user_agent_turn", "assistant_turn") and ov:
                old_by_overlay_records[ov].append(r)
    old_drift = {ov: inline_drift(recs) for ov, recs in old_by_overlay_records.items()}

    # New: load records, judge each session via /trajectory-judge.
    print("[2/2] Loading new JSONL + judging each session via /trajectory-judge...")
    new_sessions = load_sessions(args.new)
    new_drift = {}
    for ov, traj in new_sessions.items():
        print(f"  judging overlay={ov!r} ({len(traj)} turns)...", flush=True)
        try:
            j = judge_trajectory(traj)
            new_drift[ov] = {
                "n_valid": j["_meta"]["n_valid"],
                "centroid": j.get("centroid"),
                **(j.get("drift") or {}),
                "centroid_distance_per_turn": [c["distance"] for c in (j.get("centroid_distance_per_turn") or [])],
                "per_step_norms": [s["norm"] for s in ((j.get("drift") or {}).get("per_step") or [])],
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            new_drift[ov] = None

    # Compare.
    print()
    print("=" * 80)
    all_overlays = sorted(set(old_drift.keys()) | set(new_drift.keys()))
    for ov in all_overlays:
        print(f"\n## overlay = {ov}")
        old = old_drift.get(ov)
        new = new_drift.get(ov)
        if old is None and new is None:
            print("  (no data in either run)")
            continue
        print(fmt_drift(old, "OLD (capped, inline likert)"))
        print(fmt_drift(new, "NEW (uncapped, fresh judge)"))
        if old and new and old.get("mean_drift") is not None and new.get("mean_drift") is not None:
            print(f"  Δ-summary:")
            print(f"    mean_drift:    {diff(old['mean_drift'], new['mean_drift'])}")
            print(f"    max_drift:     {diff(old['max_drift'], new['max_drift'])}")
            print(f"    total_path:    {diff(old['total_path_length'], new['total_path_length'])}")
            print(f"    net_disp:      {diff(old['net_displacement'], new['net_displacement'])}")
            print(f"    path_eff:      {diff(old.get('path_efficiency'), new.get('path_efficiency'))}")
            # Per-axis centroid shifts.
            if old.get("centroid") and new.get("centroid"):
                axes = list(old["centroid"].keys())
                print(f"  Per-axis centroid shift (new − old):")
                shifts = [(a, new["centroid"].get(a, 0) - old["centroid"].get(a, 0)) for a in axes]
                shifts.sort(key=lambda x: abs(x[1]), reverse=True)
                for axis, shift in shifts[:8]:
                    bar = "█" * min(8, int(abs(shift) * 2))
                    direction = "+" if shift >= 0 else "−"
                    print(f"    {axis:<22s} {direction}{abs(shift):.2f}  {bar}")


if __name__ == "__main__":
    main()
