#!/usr/bin/env python3
"""Empirical prefill + AR-decode bandwidth probe for mixed-multimodal
workloads against the backend.

K≥4 workers. Each worker runs N sequential assistant turns. Every turn
attaches one synthetic 64×64 image (checkerboard or gaussian noise) +
a short text prompt. Images are per-worker unique so the vision cache
never hits — every call exercises the full vision encode path.

Captures per-turn streaming metrics (TTFT, decode_s, usage) and reports:

  - Prefill bandwidth = Σ prompt_tokens / Σ ttft  (per-class, per-turn-index)
  - AR decode bandwidth = Σ completion_tokens / Σ decode_s
  - Total bandwidth  = Σ all_tokens / wall_s
  - Split by turn index: cold (t0) vs warm (t1+) — reveals cache-miss cost
    amortization once vision weights are hot and AR is in steady state.
"""
from __future__ import annotations
import argparse, concurrent.futures as cf, io, json, pathlib, statistics, sys, time
from dataclasses import dataclass

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from svg_refinement_loop import chat_stream, image_to_data_url


# ── Synthetic image generators ──────────────────────────────────────

def checkerboard(seed: int, size: int = 64) -> Image.Image:
    """64×64 RGB checkerboard with seed-varied phase + palette so each
    call produces a unique PNG byte stream."""
    rng = np.random.default_rng(seed)
    phase = rng.integers(0, 4)
    tile = rng.integers(4, 12)
    c1 = rng.integers(40, 230, size=3)
    c2 = rng.integers(40, 230, size=3)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(size):
        for x in range(size):
            cell = ((x // tile) + (y // tile) + phase) % 2
            img[y, x] = c1 if cell == 0 else c2
    return Image.fromarray(img, "RGB")


def gaussian_noise(seed: int, size: int = 64) -> Image.Image:
    """64×64 RGB gaussian noise, seeded for reproducibility + per-call
    byte-uniqueness."""
    rng = np.random.default_rng(seed)
    arr = rng.normal(127, 40, size=(size, size, 3))
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Per-worker loop ─────────────────────────────────────────────────

PROMPT = "Describe this image in one sentence."
SYSTEM = ("You are a terse vision analyst. Given an image, reply with a "
          "single factual sentence describing what you see. No preamble.")


@dataclass
class TurnMetric:
    worker: int
    turn: int
    image_kind: str     # "checkerboard" | "gaussian"
    ttft_s: float
    decode_s: float
    total_s: float
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None


def worker_turns(worker_id: int, n_turns: int, temperature: float,
                  max_tokens: int) -> list[TurnMetric]:
    messages = [{"role": "system", "content": SYSTEM}]
    out: list[TurnMetric] = []
    for t in range(n_turns):
        kind = "checkerboard" if (t % 2 == 0) else "gaussian"
        # Seed so each (worker, turn) pair produces unique image bytes.
        seed = worker_id * 1000 + t * 7 + (0 if kind == "checkerboard" else 3)
        img = checkerboard(seed) if kind == "checkerboard" else gaussian_noise(seed)
        user_msg = {"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": image_to_data_url(img)}},
        ]}
        msgs_this_turn = messages + [user_msg]
        text, m = chat_stream(msgs_this_turn, max_tokens=max_tokens,
                               temperature=temperature, seed=seed)
        out.append(TurnMetric(
            worker=worker_id, turn=t, image_kind=kind,
            ttft_s=m["ttft_s"], decode_s=m["decode_s"], total_s=m["total_s"],
            prompt_tokens=m["prompt_tokens"] or 0,
            completion_tokens=m["completion_tokens"] or 0,
            finish_reason=m.get("finish_reason")))
        # Append this turn to the rolling history so subsequent turns
        # accumulate context (warm-state prefill grows).
        messages.append(user_msg)
        messages.append({"role": "assistant", "content": text})
    return out


# ── Aggregation ─────────────────────────────────────────────────────

def agg(recs: list[TurnMetric], wall_s: float, label: str) -> dict:
    if not recs:
        return {"label": label, "n": 0}
    sum_pt = sum(r.prompt_tokens for r in recs)
    sum_ct = sum(r.completion_tokens for r in recs)
    sum_ttft = sum(r.ttft_s for r in recs)
    sum_dec = sum(r.decode_s for r in recs)
    return {
        "label": label,
        "n": len(recs),
        "sum_prompt_tokens": sum_pt,
        "sum_completion_tokens": sum_ct,
        "mean_ttft_ms": 1000 * sum_ttft / len(recs),
        "mean_decode_ms": 1000 * sum_dec / len(recs),
        "agg_prefill_tok_per_s": sum_pt / sum_ttft if sum_ttft > 0 else 0,
        "agg_decode_tok_per_s": sum_ct / sum_dec if sum_dec > 0 else 0,
        "agg_total_tok_per_s": (sum_pt + sum_ct) / wall_s if wall_s > 0 else 0,
    }


def fmt_row(a: dict) -> str:
    if a["n"] == 0: return f"  {a['label']:<22} (empty)"
    return (f"  {a['label']:<22}  n={a['n']:>3}  "
            f"prefill={a['agg_prefill_tok_per_s']:>7.1f} tok/s  "
            f"decode={a['agg_decode_tok_per_s']:>7.1f} tok/s  "
            f"total={a['agg_total_tok_per_s']:>7.1f} tok/s  "
            f"ttft={a['mean_ttft_ms']:>6.0f}ms  "
            f"dec={a['mean_decode_ms']:>6.0f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="concurrent workers (≥4)")
    ap.add_argument("--turns", type=int, default=4,
                    help="sequential turns per worker")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--save", type=pathlib.Path, default=None)
    args = ap.parse_args()
    if args.K < 4:
        print("K<4 rejected", file=sys.stderr); sys.exit(2)

    print(f"[probe] K={args.K} workers × {args.turns} turns, "
          f"max_tokens={args.max_tokens}, temperature={args.temperature}")
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.K) as exe:
        futs = [exe.submit(worker_turns, w, args.turns,
                            args.temperature, args.max_tokens)
                for w in range(args.K)]
        all_recs: list[TurnMetric] = []
        for f in cf.as_completed(futs):
            all_recs.extend(f.result())
    wall = time.time() - t0

    print(f"\n=== mixed multimodal bandwidth (wall={wall:.1f}s, "
          f"{len(all_recs)} turns total) ===\n")
    print(fmt_row(agg(all_recs, wall, "overall")))

    print("\nby turn index (cold t0 vs warm t1+):")
    by_turn = {}
    for r in all_recs: by_turn.setdefault(r.turn, []).append(r)
    for t in sorted(by_turn):
        print(fmt_row(agg(by_turn[t], wall, f"turn {t}")))

    print("\nby image class:")
    by_kind = {}
    for r in all_recs: by_kind.setdefault(r.image_kind, []).append(r)
    for k in sorted(by_kind):
        print(fmt_row(agg(by_kind[k], wall, k)))

    warm = [r for r in all_recs if r.turn >= 1]
    print("\nwarm only (t≥1, post-cache-hydrate steady state):")
    print(fmt_row(agg(warm, wall, "warm_subset")))

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps({
            "K": args.K, "turns": args.turns, "wall_s": wall,
            "max_tokens": args.max_tokens, "temperature": args.temperature,
            "overall": agg(all_recs, wall, "overall"),
            "by_turn": {str(t): agg(rs, wall, f"turn_{t}")
                         for t, rs in by_turn.items()},
            "by_image_kind": {k: agg(rs, wall, k)
                               for k, rs in by_kind.items()},
            "warm": agg(warm, wall, "warm"),
            "records": [r.__dict__ for r in all_recs],
        }, indent=2))
        print(f"\nsaved → {args.save}")


if __name__ == "__main__":
    main()
