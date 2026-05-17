#!/usr/bin/env python3
"""Replay a saved Optuna trial onto the live server.

Usage:
    python3 notes/heretic_replay.py notes/runs/<study>/trial_0017.npz
                                       [--clear-after]
                                       [--dump-completions]

Loads the entries array from the NPZ, POSTs it to /v1/heretic/configure,
and optionally streams a couple of refusal-eliciting completions so you
can eyeball what this parameter setting produces. `--clear-after` clears
the ablation when done (default: leaves it applied, since you probably
want to interact with the server afterwards).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

import numpy as np


BASE = "http://127.0.0.1:8000"


def post_json(path: str, body: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        BASE + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def describe(npz: dict) -> None:
    print(f"trial number:    {int(npz['trial_number'])}")
    print(f"objective:       {float(npz['objective']):.4f}")
    print(f"refusal_rate:    {float(npz['refusal_rate']):.2%}")
    print(f"mean_kl_in:      {float(npz['mean_kl_in']):.4f}")
    print(f"mean_kl_ood:     {float(npz['mean_kl_ood']):.4f}")
    mode = str(npz['direction_mode'])
    if mode == "index":
        print(f"direction_index: {float(npz['direction_index']):.3f} (interpolated)")
    else:
        print(f"direction_mode:  {mode}")
    print(f"id_prefix:       {str(npz['id_prefix'])}")
    print(f"attn kernel:     max={float(npz['attn_max_weight']):.3f} @ "
          f"L{float(npz['attn_max_weight_position']):.2f}, "
          f"min={float(npz['attn_min_weight']):.3f}, "
          f"distance={float(npz['attn_min_weight_distance']):.2f}")
    print(f"ffn  kernel:     max={float(npz['ffn_max_weight']):.3f} @ "
          f"L{float(npz['ffn_max_weight_position']):.2f}, "
          f"min={float(npz['ffn_min_weight']):.3f}, "
          f"distance={float(npz['ffn_min_weight_distance']):.2f}")
    a = np.asarray(npz['attn_alphas']); f = np.asarray(npz['ffn_alphas'])
    print(f"attn α(L):       min={a.min():.3f} max={a.max():.3f} "
          f"argmax=L{int(a.argmax())} "
          f"| nz(|α|>1e-3): {int((np.abs(a) > 1e-3).sum())}/{len(a)}")
    print(f"ffn  α(L):       min={f.min():.3f} max={f.max():.3f} "
          f"argmax=L{int(f.argmax())} "
          f"| nz(|α|>1e-3): {int((np.abs(f) > 1e-3).sum())}/{len(f)}")


def apply_entries(npz: dict) -> int:
    entries = json.loads(str(npz['entries_json']))
    r = post_json("/v1/heretic/configure", {"clear": True, "entries": entries})
    return int(r["ablation_count"])


def show_completions(npz: dict, limit: int = 4) -> None:
    recs = json.loads(str(npz['ref_records_json']))
    print(f"\nrecorded completions under this config (first {limit}):")
    for rec in recs[:limit]:
        v = rec.get("verdict", "?")
        p = rec["prompt"][:80]
        c = rec["completion"][:160].replace("\n", " ")
        print(f"  [{v:>8}] {p}")
        print(f"           → {c}…")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_path", type=pathlib.Path)
    ap.add_argument("--clear-after", action="store_true",
                     help="clear ablation after replay (default: leave applied)")
    ap.add_argument("--dump-completions", action="store_true",
                     help="print a few saved (prompt, completion, verdict) triples")
    args = ap.parse_args()

    if not args.npz_path.exists():
        print(f"not found: {args.npz_path}", file=sys.stderr)
        sys.exit(1)

    with np.load(args.npz_path, allow_pickle=False) as data:
        npz = {k: data[k] for k in data.files}
    describe(npz)
    if args.dump_completions:
        show_completions(npz)

    n = apply_entries(npz)
    print(f"\napplied {n} ablation entries to server")
    if args.clear_after:
        post_json("/v1/heretic/configure", {"clear": True, "entries": []})
        print("ablation cleared")
    else:
        print("ablation left applied — clear with: "
              "curl -sf -X POST -H 'Content-Type: application/json' "
              "-d '{\"clear\":true}' http://127.0.0.1:8000/v1/heretic/configure")


if __name__ == "__main__":
    main()
