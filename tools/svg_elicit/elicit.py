#!/usr/bin/env python3
"""Upgraded SVG elicitation harness — stronger elicitation + multi-measure.

See docs/causal_ophthalmology_harness_design.md. This pass demonstrates the
*capability* upgrades (no lesion study):

  - Full SVG feature set (gradients/filters/patterns/text/transforms) — the
    archived harness banned exactly the primitives real images need.
  - Programmatic / scripted emission as a first-class, swept mode
    (--mode svg|python|both): the model writes code that LOOPS to build
    grids/lattices/gradient-stops/path-following motifs.
  - Multi-measure, reported side by side: MSE (pixel, kept as a feedback
    channel + covariate, NOT the selector), SSIM (perceptual, model-indep),
    and an on-device VLM-judge faithfulness score + structured critique.
  - Multi-channel feedback: MSE + SSIM + render + diff-heatmap + judge critique.
  - Non-MSE selector (--primary-metric, default SSIM).
  - Calibrated generation config + a finish_reason=='length' TRUNCATION GUARD
    (retry once at 2x budget, else mark the sample INVALID — never silently
    average a truncated rollout in as a missing None).

NOTE: the VLM judge here is gemma-grading-gemma (circular). That is fine for
measuring capability but MUST be replaced by a fixed external reference for any
differential/lesion study (see the design note, §5).

Usage:
  GEMMA_BASE=http://127.0.0.1:8001 \\
    uv run --with numpy --with pillow --with playwright --with scikit-image \\
      python tools/svg_elicit/elicit.py \\
      --frame test_data/frames_v2/KCrfDHS_YUw/frame_0000.png \\
      --mode both --max-iters 3 --rollouts 1
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, statistics, sys, time
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
# Reuse the working rasterizer + extractors from the (archived) original.
sys.path.insert(0, str(REPO / "scripts" / "archival"))
from svg_refinement_loop import (  # noqa: E402
    load_target_from_path, render_svg, extract_svg_directly,
    extract_python_and_run, mse_images, diff_heatmap, image_to_data_url,
)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

BASE = os.environ.get("GEMMA_BASE", "http://127.0.0.1:8001")

# ── Elicitation: full-featureset prompts ─────────────────────────────────
SYSTEM_SVG_RICH = (
    "You are an expert at reproducing a reference raster image as a single SVG "
    "document. Use the FULL SVG feature set wherever it improves faithfulness: "
    "linearGradient/radialGradient (for metallic, shaded, or smoothly varying "
    "fills), filters such as feGaussianBlur (glow, soft edges), <pattern> "
    "(repeated texture, grids, lattices), <text>, clipPath, opacity, blend "
    "modes, and transform. Choose the primitive that matches the IMAGE "
    "STRUCTURE: a chrome/metallic look is a gradient with a bright highlight "
    "band, NOT a flat color; a perspective grid is many lines under a transform "
    "or patternTransform; fine texture is a <pattern> tile. Declare "
    'viewBox="0 0 {W} {H}" so it rasterizes to {W}x{H}. Wrap the SVG in a '
    "```svg code fence. Reasoning is optional; the SVG is what matters."
)
SYSTEM_PYTHON_RICH = (
    "You are an expert at writing a Python program that GENERATES an SVG "
    "reproduction of a reference raster image. Define `def make_svg() -> str:` "
    "that returns the SVG document as a string. Because you are writing CODE, "
    "USE LOOPS AND COMPUTATION to express repetitive or parametric structure "
    "that is tedious to place by hand: perspective grids/lattices (a loop "
    "emitting <line>/<path> under a computed transform), gradient stop arrays, "
    "hatching or stippling, motifs repeated along a path, procedurally "
    "positioned shapes. The generated SVG may use the full feature set "
    "(gradients, filters, patterns, text, transforms, clipPath). Declare "
    'viewBox="0 0 {W} {H}". Use only the Python standard library. Wrap the '
    "function in a ```python code fence."
)
FIRST_USER_TEXT = (
    "Reproduce this image as faithfully as you can. Match the composition, the "
    "dominant colors AND their material/shading (metallic vs flat), the spatial "
    "layout, any text, and any repeated structure like grids or textures. Use "
    "whichever SVG features fit best."
)


def first_user_turn(target_img: Image.Image) -> dict:
    return {"role": "user", "content": [
        {"type": "text", "text": FIRST_USER_TEXT},
        {"type": "image_url", "image_url": {"url": image_to_data_url(target_img)}},
    ]}


# ── Measures ─────────────────────────────────────────────────────────────
def ssim_score(a: Image.Image, b: Image.Image) -> float:
    """Structural similarity in [−1,1] (1 = identical). Perceptual; correlates
    with perceived fidelity far better than MSE."""
    from skimage.metrics import structural_similarity as ssim
    aa = np.asarray(a.convert("RGB"))
    bb = np.asarray(b.convert("RGB"))
    return float(ssim(aa, bb, channel_axis=2, data_range=255))


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


import judge as _judge  # same-dir calibrated judge (rubric + on-policy k-shot)  # noqa: E402
JUDGE_EXEMPLARS: list[dict] = []   # loaded in main() from --judge-exemplars


def judge_render(target: Image.Image, render: Image.Image, seed: int) -> dict | None:
    """Calibrated VLM-judge: anchored rubric + on-policy k-shot exemplars
    (judge_calibrate.py showed this removes floor-bias and lifts rank-corr to
    ~0.87). Returns {faithfulness:1-5, missing:[...], biggest_fix:str} or None.
    Still circular (gemma judging gemma) — fine for driving the loop; a lesion
    study needs an external referee (design note §5)."""
    def _chat(msgs):
        return call_lm(msgs, max_tokens=400, temperature=0.2, seed=seed)[0]
    return _judge.score(_chat, target, render, JUDGE_EXEMPLARS)


# ── LM call with truncation guard ────────────────────────────────────────
def call_lm(messages: list[dict], max_tokens: int, temperature: float,
            seed: int, timeout: float = 3600.0) -> tuple[str, str | None, dict]:
    """Stream a chat completion; return (text, finish_reason, usage)."""
    body = {"messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "seed": int(seed)}
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Accept": "text/event-stream"}, method="POST")
    chunks, usage, finish = [], None, None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choice = (obj.get("choices") or [{}])[0]
            d = choice.get("delta", {})
            if d.get("content"):
                chunks.append(d["content"])
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            if obj.get("usage"):
                usage = obj["usage"]
    return "".join(chunks), finish, (usage or {})


def call_lm_guarded(messages, max_tokens, temperature, seed,
                    max_cap: int = 12288) -> tuple[str, str | None, dict, bool]:
    """Truncation guard: if finish_reason=='length', retry once at 2x budget.
    Returns (text, finish_reason, usage, retried)."""
    text, fr, usage = call_lm(messages, max_tokens, temperature, seed)
    if fr == "length" and max_tokens < max_cap:
        bigger = min(max_tokens * 2, max_cap)
        text, fr, usage = call_lm(messages, bigger, temperature, seed)
        return text, fr, usage, True
    return text, fr, usage, False


# ── Multi-channel feedback ───────────────────────────────────────────────
def feedback_turn(prev_render: Image.Image, mse: float, ssim: float,
                  heatmap: Image.Image, critique: dict | None) -> dict:
    text = (f"Scores for your previous attempt — SSIM={ssim:.3f} "
            f"(1.0 = perfect structural match), MSE={mse:.4f} (0 = perfect "
            f"pixels). Below is your previous render, then an amplified "
            f"pixel-difference heatmap (bright = where you diverge from target).")
    content = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_to_data_url(prev_render)}},
        {"type": "text", "text": "Pixel-difference heatmap (amplified 3x):"},
        {"type": "image_url", "image_url": {"url": image_to_data_url(heatmap)}},
    ]
    if critique:
        miss = "; ".join(str(x) for x in (critique.get("missing") or [])[:5])
        content.append({"type": "text", "text":
            f"A visual critic scored faithfulness {critique.get('faithfulness')}/5. "
            f"Missing/wrong: {miss}. Biggest fix: {critique.get('biggest_fix')}"})
    content.append({"type": "text", "text":
        "Produce a refined version. Prioritize the biggest fix. You may "
        "restructure completely and use any SVG features that would help."})
    return {"role": "user", "content": content}


# ── Selector ─────────────────────────────────────────────────────────────
def is_better(new: float | None, cur: float | None, metric: str) -> bool:
    if new is None:
        return False
    if cur is None:
        return True
    return new < cur if metric == "mse" else new > cur  # mse lower; ssim/judge higher


def primary_value(measures: dict, metric: str) -> float | None:
    if metric == "mse":
        return measures.get("mse")
    if metric == "ssim":
        return measures.get("ssim")
    if metric == "judge":
        c = measures.get("judge")
        return c.get("faithfulness") if c else None
    return None


# ── One rollout ──────────────────────────────────────────────────────────
def one_rollout(target: Image.Image, mode: str, w: int, h: int, max_iters: int,
                max_tokens: int, temperature: float, base_seed: int,
                primary: str, use_judge: bool, out_dir: pathlib.Path,
                prefix: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    target.save(out_dir / f"{prefix}_target.png")
    system = (SYSTEM_SVG_RICH if mode == "svg" else SYSTEM_PYTHON_RICH).format(W=w, H=h)
    messages = [{"role": "system", "content": system}, first_user_turn(target)]
    history: list[dict] = []
    best = {"value": None, "iter": -1, "svg": None}

    for it in range(max_iters):
        t0 = time.time()
        seed = base_seed + it
        resp, fr, usage, retried = call_lm_guarded(messages, max_tokens, temperature, seed)
        (out_dir / f"{prefix}_iter_{it:02d}_raw.txt").write_text(resp)
        ct = usage.get("completion_tokens")
        svg = extract_svg_directly(resp) if mode == "svg" else extract_python_and_run(resp)

        # Truncation guard: a still-truncated or unparseable turn is INVALID,
        # recorded explicitly (never folded in as a silent success).
        if fr == "length" or svg is None:
            err = "truncated (finish_reason=length)" if fr == "length" else "no SVG parsed"
            history.append({"iter": it, "valid": False, "error": err,
                            "finish_reason": fr, "retried": retried,
                            "completion_tokens": ct, "elapsed_s": time.time() - t0})
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                f"That response was unusable ({err}). Emit a single complete "
                f"{'```python make_svg()' if mode=='python' else '```svg'} block."})
            continue

        try:
            rendered = render_svg(svg, width=w, height=h)
        except Exception as e:
            history.append({"iter": it, "valid": False, "error": f"render: {e}",
                            "finish_reason": fr, "completion_tokens": ct,
                            "elapsed_s": time.time() - t0})
            messages.append({"role": "assistant", "content": resp})
            messages.append({"role": "user", "content":
                f"SVG failed to render: {e}. Try again."})
            continue

        rendered.save(out_dir / f"{prefix}_iter_{it:02d}_rendered.png")
        (out_dir / f"{prefix}_iter_{it:02d}.svg").write_text(svg)
        mse = mse_images(target, rendered)
        ssim = ssim_score(target, rendered)
        critique = judge_render(target, rendered, seed=seed + 7777) if use_judge else None
        measures = {"mse": mse, "ssim": ssim, "judge": critique}
        history.append({"iter": it, "valid": True, "finish_reason": fr,
                        "retried": retried, "completion_tokens": ct,
                        "svg_chars": len(svg), "elapsed_s": time.time() - t0,
                        **{"mse": mse, "ssim": ssim,
                           "judge_faithfulness": (critique or {}).get("faithfulness"),
                           "judge_missing": (critique or {}).get("missing"),
                           "judge_biggest_fix": (critique or {}).get("biggest_fix")}})
        pv = primary_value(measures, primary)
        if is_better(pv, best["value"], primary):
            best = {"value": pv, "iter": it, "svg": svg}
        # Multi-channel feedback for the next turn.
        messages.append({"role": "assistant", "content": resp})
        messages.append(feedback_turn(rendered, mse, ssim,
                                      diff_heatmap(target, rendered), critique))

    if best["svg"]:
        (out_dir / f"{prefix}_best.svg").write_text(best["svg"])
        render_svg(best["svg"], width=w, height=h).save(out_dir / f"{prefix}_best_rendered.png")
    valid = [h for h in history if h.get("valid")]
    rep = {"prefix": prefix, "mode": mode, "primary_metric": primary,
           "best_iter": best["iter"], "best_primary_value": best["value"],
           "n_valid": len(valid), "n_invalid": len(history) - len(valid),
           "history": history}
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, required=True)
    ap.add_argument("--mode", choices=["svg", "python", "both"], default="both")
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=6144,
                    help="set above the measured output tail; programmatic + "
                         "full-featureset SVG is longer than shape-only output")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--primary-metric", choices=["ssim", "mse", "judge"], default="ssim")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--judge-exemplars", type=pathlib.Path,
                    default=REPO / "tools" / "svg_elicit" / "amongus_onpolicy_exemplars.json",
                    help="on-policy k-shot exemplars for the calibrated judge "
                         "(empty/missing → 0-shot rubric judge)")
    ap.add_argument("--out-root", type=pathlib.Path,
                    default=REPO / "output_data" / "svg_runs" / f"elicit_{int(time.time())}")
    args = ap.parse_args()

    global JUDGE_EXEMPLARS
    JUDGE_EXEMPLARS = _judge.load_exemplars(REPO, args.judge_exemplars)

    args.out_root.mkdir(parents=True, exist_ok=True)
    target = load_target_from_path(str(args.frame))
    w, h = target.size
    frame_id = f"{args.frame.parent.name}_{args.frame.stem}"
    modes = ["svg", "python"] if args.mode == "both" else [args.mode]
    use_judge = not args.no_judge
    print(f"[elicit] frame={frame_id} {w}x{h} modes={modes} primary={args.primary_metric} "
          f"judge={use_judge} ({len(JUDGE_EXEMPLARS)}-shot calibrated) → {args.out_root}")

    summary = {}
    for mode in modes:
        for r in range(args.rollouts):
            prefix = f"{frame_id}_{mode}_r{r:02d}"
            seed = args.base_seed + r * 1000
            t0 = time.time()
            rep = one_rollout(target, mode, w, h, args.max_iters, args.max_tokens,
                              args.temperature, seed, args.primary_metric,
                              use_judge, args.out_root, prefix)
            wall = time.time() - t0
            iters = [h for h in rep["history"] if h.get("valid")]
            ssim_traj = " -> ".join(f"{h['iter']}:{h['ssim']:.3f}" for h in iters)
            mse_traj = " -> ".join(f"{h['iter']}:{h['mse']:.4f}" for h in iters)
            jf = [h.get("judge_faithfulness") for h in iters]
            print(f"\n[{prefix}] wall={wall:.0f}s valid={rep['n_valid']} "
                  f"invalid={rep['n_invalid']} best={args.primary_metric}="
                  f"{rep['best_primary_value']!r}@iter{rep['best_iter']}")
            print(f"   SSIM  {ssim_traj}")
            print(f"   MSE   {mse_traj}")
            print(f"   judge {jf}")
            summary[prefix] = {"best_iter": rep["best_iter"],
                               "best_primary_value": rep["best_primary_value"],
                               "n_valid": rep["n_valid"], "n_invalid": rep["n_invalid"],
                               "wall_s": wall}
    (args.out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[elicit] done → {args.out_root}/summary.json")


if __name__ == "__main__":
    main()
