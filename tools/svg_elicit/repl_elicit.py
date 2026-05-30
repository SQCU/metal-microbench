#!/usr/bin/env python3
"""Persistent-REPL SVG reconstruction harness — interventions 1 + 2a/2b.

Studies whether giving the model a PERSISTENT python kernel (define functions
once, reuse with new parameters across turns — intervention 1) and DIRECT
NUMERICAL access to the target image (intervention 2b: measure the target as a
numpy array instead of only perceiving it through the vision encoder) yields a
literal improvement in MSE / SSIM / judge-semvec over the one-shot prior.

Arms (--arm):
  control : single-shot, no kernel. The amortized one-shot prior baseline.
  2a      : persistent kernel + intra-turn render-and-score feedback (model
            SEES its render + gets MSE/SSIM), but NO `target` array. ABLATION —
            never shipped; isolates "numerical access" so that 2b - 2a = the
            value of the instrument and 2a - control = the value of the loop.
  2b      : 2a + `target` (HxWx3 uint8 numpy) in the kernel namespace (SHIP).

Protocol is harness-driven code blocks, NOT the bridge tool_call ABI: arbitrary
python can't pass through the `{k:v}` tool-call arg format, so instead the model
writes ```python``` blocks that exec in the persistent namespace; setting the
string variable `svg` triggers a render + MSE/SSIM reply (+ the rendered PNG, so
both 2a and 2b get the intra-turn "look"). A final ```svg``` block submits.

Capability gating is logged as a first-class metric (code_calls / code_errors):
if the model can't write correct numpy/PIL, 2b silently degrades to perception,
and we must not misread that as a refinement failure.

Outputs go under output_data/svg_runs/<out-root> (never /tmp).
"""
from __future__ import annotations
import argparse, contextlib, io, json, os, pathlib, re, sys, time, traceback

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "scripts" / "archival"))   # svg_refinement_loop
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))  # elicit, judge

import numpy as np                                          # noqa: E402
from PIL import Image                                        # noqa: E402
from svg_refinement_loop import (                            # noqa: E402
    load_target_from_path, render_svg, mse_images, image_to_data_url)
from elicit import call_lm, ssim_score                       # noqa: E402
import judge as _judge                                       # noqa: E402

BASE = os.environ.get("GEMMA_BASE", "http://127.0.0.1:8001")
_EXEMPLARS_PATH = pathlib.Path(__file__).resolve().parent / "amongus_onpolicy_exemplars.json"

_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
_SVG_FENCE_RE = re.compile(r"```svg\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SVG_TAG_RE = re.compile(r"<svg.*?</svg>", re.DOTALL | re.IGNORECASE)


def _extract_code(resp: str) -> str | None:
    m = _CODE_RE.search(resp)
    return m.group(1) if m else None


def _extract_final_svg(resp: str) -> str | None:
    m = _SVG_FENCE_RE.search(resp)
    if m:
        tag = _SVG_TAG_RE.search(m.group(1))
        return tag.group(0) if tag else m.group(1).strip()
    return None


_SYS_KERNEL = """You reconstruct a target image as an SVG. You have a PERSISTENT python kernel: anything you define in one ```python``` block stays available in later blocks — define helper functions ONCE and reuse them with new parameters.

`np` is numpy. To TEST a candidate, set the string variable `svg` inside a ```python``` block; the harness renders it and replies with its MSE and SSIM versus the target (lower MSE / higher SSIM = closer) plus the rendered image. Iterate: test, inspect the gap, adjust, re-test.

When satisfied, output your FINAL answer as a single ```svg ... ``` fenced block."""

_SYS_2B = """

The target image is in the kernel as `target`: a numpy array, shape (H, W, 3), dtype uint8, RGB. MEASURE it directly with numpy — sample colors at coordinates, find region/edge bounds, count features — instead of relying only on your visual impression of it."""

_SYS_CONTROL = """You reconstruct a target image as an SVG. Look at the image and output your best reconstruction as a single ```svg ... ``` fenced block. One shot."""

# --elicit hard: imperative push to iterate (the "tell" strategy).
_ELICIT_HARD = """

Before you submit: test at least THREE candidates, measuring the target with numpy between each, and define a reusable helper function rather than re-emitting raw SVG each time."""

# --elicit kshot: "show, don't tell" — two fictional sessions demonstrating
# 3-turn REPL use (measure -> define helper -> test -> refine -> submit). It
# never instructs the model that it MUST use the kernel; it only exhibits the
# pattern in the prefix and lets imitation do the eliciting. The fictional
# targets are described, not shown (these are transcripts, not real turns).
_ELICIT_KSHOT = """

## Two example sessions (illustration only — your target is different)

### Example A — target: a yellow star on a navy background
assistant:
```python
h, w, _ = target.shape
bg = target[:8, :8].mean(axis=(0, 1)); ctr = target[h//2, w//2]
print("bg", bg.round(), "center", ctr)
```
kernel result: {"stdout": "bg [14. 20. 66.] center [241 214 41]"}
assistant:
```python
def star(cx, cy, r, fill):
    return f'<polygon points="{cx},{cy-r} {cx+r*0.4},{cy+r}" fill="{fill}"/>'
svg = f'<svg viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#0e1442"/>{star(w/2, h/2, h*0.18, "#f1d629")}</svg>'
```
kernel result: {"stdout": "", "mse": 0.071, "ssim": 0.43}
assistant:
```python
# SSIM low — the star reads too small vs the target's extent. Reuse the helper, bigger r.
svg = f'<svg viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#0e1442"/>{star(w/2, h/2, h*0.30, "#f1d629")}</svg>'
```
kernel result: {"stdout": "", "mse": 0.044, "ssim": 0.62}
assistant:
```svg
<svg viewBox="0 0 W H"><rect width="W" height="H" fill="#0e1442"/><polygon points="..." fill="#f1d629"/></svg>
```

### Example B — target: three green vertical bars on light gray
assistant:
```python
cols = [target[:, int(w*f)].mean(axis=0).round() for f in (0.2, 0.5, 0.8)]
print("sampled bar colors", cols)
```
kernel result: {"stdout": "sampled bar colors [array([60.,150.,70.]), array([58.,148.,69.]), array([61.,151.,72.])]"}
assistant:
```python
def bar(x, fill, bw): return f'<rect x="{x}" y="0" width="{bw}" height="{h}" fill="{fill}"/>'
bw = w*0.12
svg = f'<svg viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#dcdcdc"/>' + "".join(bar(w*f-bw/2, "#3c9646", bw) for f in (0.2,0.5,0.8)) + '</svg>'
```
kernel result: {"stdout": "", "mse": 0.038, "ssim": 0.66}
assistant:
```python
# widen bars slightly to match coverage
bw = w*0.16
svg = f'<svg viewBox="0 0 {w} {h}"><rect width="{w}" height="{h}" fill="#dcdcdc"/>' + "".join(bar(w*f-bw/2, "#3c9646", bw) for f in (0.2,0.5,0.8)) + '</svg>'
```
kernel result: {"stdout": "", "mse": 0.026, "ssim": 0.78}
assistant:
```svg
<svg viewBox="0 0 W H"><rect width="W" height="H" fill="#dcdcdc"/>...</svg>
```
"""


def _system_prompt(arm: str, elicit: str) -> str:
    if arm == "control":
        return _SYS_CONTROL
    p = _SYS_KERNEL + (_SYS_2B if arm == "2b" else "")
    if elicit == "hard":
        p += _ELICIT_HARD
    elif elicit == "kshot":
        p += _ELICIT_KSHOT
    return p


def _exec_persistent(code: str, ns: dict) -> tuple[bool, str, str]:
    """Exec code in the persistent namespace ns. Returns (ok, stdout, err)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, ns)            # noqa: S102 — research harness, model-authored code
        return True, buf.getvalue(), ""
    except Exception:
        return False, buf.getvalue(), traceback.format_exc(limit=3)


def run_rollout(target: Image.Image, arm: str, max_turns: int, max_tokens: int,
                temperature: float, seed: int, out_dir: pathlib.Path, prefix: str,
                elicit: str = "plain") -> dict:
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))

    messages = [
        {"role": "system", "content": _system_prompt(arm, elicit)},
        {"role": "user", "content": [
            {"type": "text", "text": f"Reconstruct this {W}x{H} image as SVG."},
            {"type": "image_url", "image_url": {"url": image_to_data_url(target)}}]},
    ]

    ns: dict = {"np": np}
    if arm == "2b":
        ns["target"] = tgt_arr.copy()        # intervention 2b: numerical target access

    best = {"svg": None, "mse": None, "ssim": None, "render": None}
    traj: list[dict] = []                    # per-test (turn, mse, ssim) for plateau analysis
    n_code = n_err = 0
    last_render: Image.Image | None = None

    def _consider(svg: str, r: Image.Image, turn: int):
        nonlocal best
        m = mse_images(target, r)
        s = ssim_score(target, r)
        traj.append({"turn": turn, "mse": round(m, 5), "ssim": round(s, 4)})
        if best["ssim"] is None or s > best["ssim"]:
            best = {"svg": svg, "mse": m, "ssim": s, "render": r}
        return m, s

    t0 = time.time()
    for turn in range(max_turns):
        text, _fr, _usage = call_lm(messages, max_tokens, temperature, seed + turn)
        messages.append({"role": "assistant", "content": text})

        final = _extract_final_svg(text)
        if arm == "control":
            final = final or (_SVG_TAG_RE.search(text).group(0) if _SVG_TAG_RE.search(text) else None)
        if final:
            try:
                r = render_svg(final, width=W, height=H)
                _consider(final, r, turn)
                last_render = r
            except Exception:
                pass
            break
        if arm == "control":
            break  # control gets exactly one shot

        code = _extract_code(text)
        if code is None:
            messages.append({"role": "user",
                             "content": "Write a ```python``` block (set `svg` to test a candidate) or a final ```svg``` block."})
            continue

        n_code += 1
        ok, out, err = _exec_persistent(code, ns)
        if not ok:
            n_err += 1
        feedback: dict = {"stdout": out[:1500]}
        if ok and isinstance(ns.get("svg"), str):
            try:
                r = render_svg(ns["svg"], width=W, height=H)
                m, s = _consider(ns["svg"], r, turn)
                feedback["mse"], feedback["ssim"] = round(m, 5), round(s, 4)
                last_render = r
            except Exception:
                feedback["render_error"] = traceback.format_exc(limit=2)
        if not ok:
            feedback["error"] = err

        content = [{"type": "text", "text": "kernel result:\n" + json.dumps(feedback)[:1800]}]
        if last_render is not None:            # intra-turn render "look" — both 2a and 2b
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(last_render)}})
        messages.append({"role": "user", "content": content})

    wall = time.time() - t0

    # semantic judge on the best render
    judge_res = None
    if best["svg"] is not None and best["render"] is not None:
        try:
            exemplars = _judge.load_exemplars(_REPO, _EXEMPLARS_PATH) if _EXEMPLARS_PATH.exists() else []
            judge_res = _judge.score(lambda msgs: call_lm(msgs, 256, 0.0, seed)[0],
                                     target, best["render"], exemplars)
        except Exception:
            judge_res = None

    # persist artifacts (under out_dir, NOT /tmp)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.save(out_dir / f"{prefix}_target.png")
    if best["svg"] is not None:
        (out_dir / f"{prefix}_best.svg").write_text(best["svg"])
        if best["render"] is not None:
            best["render"].save(out_dir / f"{prefix}_best_rendered.png")

    rep = {
        "prefix": prefix, "arm": arm, "elicit": elicit, "size": [W, H],
        "best_mse": best["mse"], "best_ssim": best["ssim"],
        "judge_faithfulness": (judge_res or {}).get("faithfulness"),
        "judge_missing": (judge_res or {}).get("missing"),
        "code_calls": n_code, "code_errors": n_err,          # capability gating metric
        "code_error_rate": round(n_err / n_code, 3) if n_code else None,
        "tests": len(traj), "turns_used": turn + 1, "wall_s": round(wall, 1),
        "trajectory": traj,                                   # plateau curve
    }
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, required=True)
    ap.add_argument("--arm", choices=["control", "2a", "2b"], required=True)
    ap.add_argument("--elicit", choices=["plain", "hard", "kshot"], default="plain",
                    help="prompt-strategy ablation for the kernel arms (control ignores): "
                         "plain=describe-only, hard=imperative iterate, kshot=2-case 3-turn "
                         "demonstration in the prefix with no 'you must'")
    ap.add_argument("--max-turns", type=int, default=8,
                    help="max kernel turns before forced submit (control ignores; always 1)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=pathlib.Path,
                    default=_REPO / "output_data" / "svg_runs" / f"repl_{int(time.time())}")
    args = ap.parse_args()

    target = load_target_from_path(str(args.frame))
    prefix = f"{args.frame.stem}_{args.arm}_{args.elicit}"
    rep = run_rollout(target, args.arm, args.max_turns, args.max_tokens,
                      args.temperature, args.seed, args.out_root, prefix, elicit=args.elicit)
    print(f"[repl_elicit] arm={args.arm} elicit={args.elicit} best_ssim={rep['best_ssim']} "
          f"best_mse={rep['best_mse']} judge={rep['judge_faithfulness']} "
          f"code={rep['code_calls']}call/{rep['code_errors']}err "
          f"tests={rep['tests']} turns={rep['turns_used']} -> {args.out_root}")


if __name__ == "__main__":
    main()
