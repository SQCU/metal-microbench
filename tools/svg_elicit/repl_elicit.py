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

_PYREPL_RE = re.compile(r"<pyrepl>\s*(.*?)</pyrepl>", re.DOTALL | re.IGNORECASE)
_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)
_SVG_FENCE_RE = re.compile(r"```svg\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SVG_TAG_RE = re.compile(r"<svg.*?</svg>", re.DOTALL | re.IGNORECASE)


def _extract_code(resp: str) -> str | None:
    # Accept the <pyrepl>...</pyrepl> tag notation OR a ```python``` fence, so
    # prompt VOICE (which delimiter the prose instructs) is a pure prose
    # ablation and both register variants parse identically.
    m = _PYREPL_RE.search(resp) or _CODE_RE.search(resp)
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


# Grading disclosure — makes "best effort" literally true by stating the
# judge's AGENDA (not its exact prompt). Present in BOTH voices (content parity;
# only register differs) so the --voice axis is a clean prose ablation.
_JUDGE_TERSE = """

Grading: besides MSE/SSIM, a separate vision-language judge rates your best render for SEMANTIC faithfulness to the target — recognizable subject, pose, composition, colour story. So don't trade away recognizable content for a marginally lower MSE; best effort = close in the numbers AND still clearly the thing."""


def _forced_clause(n: int) -> str:
    return (f"\n\nThis is a multi-pass task. A \"pass\" = setting `svg` in a ```python``` block "
            f"and reading back its MSE/SSIM. You MUST complete at least {n} passes, and each pass "
            f"must be a GENUINE change aimed at lowering MSE / raising SSIM versus your best so "
            f"far — you are judged on best effort. Re-running an unchanged candidate is rejected "
            f"and does not count as a pass. A final ```svg``` block submitted before {n} passes is "
            f"bounced back to you. Only after at least {n} improving passes will your final be "
            f"accepted.")


# ===================== user-voice register (--voice user) =====================
# Authored to model how the harness designer writes: warm, lowercase-
# conversational, forthright; frames the harness as resources OFFERED to the
# model; <pyrepl> tag notation; names BOTH pixel error and semantics as
# objectives; discloses the judge's agenda. This whole register is itself an
# ablation dimension (does authorial voice change elicitation?).
_U_KERNEL = """right now we're working on image reconstruction. you'll be passed a reference image to look at through your vision encoding, and we'd like the clearest, most definite recovery of what you can see in it — written as SVG. you can build elements programmatically with a little scripting, 'hand-write' elements directly in svg, or mix both as suits you.

low pixel mean-squared-error is desirable; but with MSE held roughly equal, we'd much rather preserve the image's SEMANTICS — colour, composition, content, narrative significance, whatever you find yourself paying attention to — than shave off a little more error.

this is an exceptionally hard task, out past the frontier of what current ML training covers, so we've prepared a few resources to make it more tractable. the first is an interactive python repl — like a jupyter notebook, but a lot more convenient: <pyrepl>. anything you write between <pyrepl> and </pyrepl> runs in a scratch namespace that PERSISTS across your turns, so you can define a function once and just update its inputs on later turns instead of rewriting it every time. you'd be wise to lean on this. `np` (numpy) is already imported.

to try a candidate, set the string variable `svg` inside a <pyrepl> block; we'll render it and send back its MSE and SSIM against the target (lower MSE / higher SSIM = closer) plus the rendered image, so you can see the gap and close it. when you've got one you're happy with, send it as your final answer in a ```svg ...``` fenced block."""

_U_TARGET = """

a second resource: the reference image is sitting right in the repl as `target` — a numpy array, shape (H, W, 3), uint8, RGB. your vision encoding is a single lossy pass, so for anything fine-grained — exact colours, where an edge actually sits, how many of something there are — you'll do better measuring `target` directly with numpy than trusting your visual impression of it."""

_U_JUDGE = """

so it isn't a mystery how you're graded: besides the pixel scores, a separate vision-language judge looks at your best render beside the target and rates how faithfully it kept the SEMANTICS — is it recognizably the same subject, pose, composition, colour story, the stuff that makes the image what it is. it isn't pixel-peeping; it's asking 'did the meaning survive?'. so don't trade away recognizable content to chase a slightly lower MSE — best effort here is a reconstruction that's both close in the numbers and still clearly the thing."""


def _u_forced(n: int) -> str:
    return (f"\n\nthe repl is here to be used across turns, so we ask for at least {n} refinement passes "
            f"before you call it done (a pass = setting `svg` inside a <pyrepl> block and reading its "
            f"scores back). make each one a real attempt to move the numbers — we'll bounce an unchanged "
            f"candidate, or a final sent before {n} passes, right back to you. genuinely try to improve "
            f"{n}+ times, then submit your best.")


# user-voice kshot — same 2 fictional sessions, <pyrepl> notation, target
# measurement included only when arm == "2b" (so it never demonstrates a
# capability a 2a run withholds — fixes the kshot/2a NameError contradiction).
def _u_kshot(arm: str) -> str:
    measure_a = ("<pyrepl>\nh, w, _ = target.shape\nprint('bg', target[:8,:8].mean((0,1)).round(), "
                 "'center', target[h//2, w//2])\n</pyrepl>\nus: {\"stdout\": \"bg [14. 20. 66.] center "
                 "[241 214 41]\"}\n") if arm == "2b" else ""
    return f"""

## two example sessions (illustration only — your target will differ)

### A — a yellow star on a navy background
{measure_a}them:
<pyrepl>
def star(cx, cy, r, fill): return f'<polygon points="..." fill="{{fill}}"/>'
svg = f'<svg viewBox="0 0 {{w}} {{h}}"><rect width="{{w}}" height="{{h}}" fill="#0e1442"/>{{star(w/2,h/2,h*0.18,"#f1d629")}}</svg>'
</pyrepl>
us: {{"mse": 0.071, "ssim": 0.43}}
them:
<pyrepl>
# ssim's low — star reads too small. reuse the helper, bigger r.
svg = f'<svg viewBox="0 0 {{w}} {{h}}"><rect width="{{w}}" height="{{h}}" fill="#0e1442"/>{{star(w/2,h/2,h*0.30,"#f1d629")}}</svg>'
</pyrepl>
us: {{"mse": 0.044, "ssim": 0.62}}
them:
```svg
<svg viewBox="0 0 W H">...</svg>
```

### B — three green bars on light grey (note how they reuse one helper)
them:
<pyrepl>
def bar(x, fill, bw): return f'<rect x="{{x}}" y="0" width="{{bw}}" height="{{h}}" fill="{{fill}}"/>'
bw = w*0.12
svg = f'<svg viewBox="0 0 {{w}} {{h}}"><rect width="{{w}}" height="{{h}}" fill="#dcdcdc"/>' + "".join(bar(w*f-bw/2,"#3c9646",bw) for f in (0.2,0.5,0.8)) + '</svg>'
</pyrepl>
us: {{"mse": 0.038, "ssim": 0.66}}
them:
<pyrepl>
bw = w*0.16  # widen to match coverage
svg = f'<svg viewBox="0 0 {{w}} {{h}}"><rect width="{{w}}" height="{{h}}" fill="#dcdcdc"/>' + "".join(bar(w*f-bw/2,"#3c9646",bw) for f in (0.2,0.5,0.8)) + '</svg>'
</pyrepl>
us: {{"mse": 0.026, "ssim": 0.78}}
them:
```svg
<svg viewBox="0 0 W H">...</svg>
```
"""


def _system_prompt(arm: str, elicit: str, min_passes: int = 0, voice: str = "terse") -> str:
    if arm == "control":
        return _SYS_CONTROL
    if voice == "user":
        p = _U_KERNEL + (_U_TARGET if arm == "2b" else "") + _U_JUDGE
        if min_passes > 0:
            p += _u_forced(min_passes)
        if elicit == "kshot":
            p += _u_kshot(arm)
        elif elicit == "hard":
            p += ("\n\nbefore you submit, please try at least three candidates — measure, "
                  "adjust, re-test — and reuse a helper function rather than re-typing svg each time.")
        return p
    # terse register (default)
    p = _SYS_KERNEL + (_SYS_2B if arm == "2b" else "") + _JUDGE_TERSE
    if min_passes > 0:
        p += _forced_clause(min_passes)
    if elicit == "hard":
        p += _ELICIT_HARD
    elif elicit == "kshot" and arm == "2b":   # terse kshot uses target; 2b only
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
                elicit: str = "plain", min_passes: int = 0, voice: str = "terse") -> dict:
    W, H = target.size
    tgt_arr = np.asarray(target.convert("RGB"))

    messages = [
        {"role": "system", "content": _system_prompt(arm, elicit, min_passes, voice)},
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
    accepted_passes = 0                       # distinct scored candidates (forced-iter target)
    rejected_finals = 0                       # premature final submissions bounced back
    seen_sigs: set[str] = set()               # anti-laziness: unchanged resubmits don't count

    def _sig(svg: str) -> str:
        return " ".join(svg.split()).lower()  # whitespace-normalized candidate signature

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
            # Forced-iteration gate: refuse a final until min_passes distinct,
            # genuinely-different candidates have been scored. (control exempt.)
            if arm != "control" and accepted_passes < min_passes:
                rejected_finals += 1
                messages.append({"role": "user", "content":
                    f"Not accepted: you have completed {accepted_passes} of {min_passes} required "
                    f"refinement passes. You are judged on best effort. Do NOT submit a final "
                    f"```svg``` yet — set `svg` in a ```python``` block to test another candidate "
                    f"that genuinely tries to lower MSE / raise SSIM versus your best so far. "
                    f"Resubmitting an unchanged result will be rejected and will not count."})
                continue
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
            sig = _sig(ns["svg"])
            try:
                r = render_svg(ns["svg"], width=W, height=H)
                m, s = _consider(ns["svg"], r, turn)
                feedback["mse"], feedback["ssim"] = round(m, 5), round(s, 4)
                last_render = r
                if sig in seen_sigs:
                    feedback["rejected"] = ("unchanged from a previous candidate — this does NOT "
                                            "count as a refinement pass; make a substantive change "
                                            "aimed at improving the score")
                else:
                    seen_sigs.add(sig)
                    accepted_passes += 1
                    feedback["pass"] = (f"{accepted_passes}/{min_passes}" if min_passes
                                        else str(accepted_passes))
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
        "prefix": prefix, "arm": arm, "elicit": elicit, "voice": voice, "size": [W, H],
        "best_mse": best["mse"], "best_ssim": best["ssim"],
        "judge_faithfulness": (judge_res or {}).get("faithfulness"),
        "judge_missing": (judge_res or {}).get("missing"),
        "code_calls": n_code, "code_errors": n_err,          # capability gating metric
        "code_error_rate": round(n_err / n_code, 3) if n_code else None,
        "min_passes": min_passes, "accepted_passes": accepted_passes,
        "rejected_finals": rejected_finals,                   # forced-iteration enforcement
        "tests": len(traj), "turns_used": turn + 1, "wall_s": round(wall, 1),
        "trajectory": traj,                                   # plateau curve
    }
    (out_dir / f"{prefix}_report.json").write_text(json.dumps(rep, indent=2, default=str))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=pathlib.Path, required=True)
    ap.add_argument("--arm", choices=["control", "2a", "2b"], required=True)
    ap.add_argument("--voice", choices=["terse", "user"], default="terse",
                    help="prompt-authorship register ablation: terse=clinical default, "
                         "user=warm/forthright designer voice with <pyrepl> notation + judge "
                         "disclosure (same content, different register)")
    ap.add_argument("--elicit", choices=["plain", "hard", "kshot"], default="plain",
                    help="prompt-strategy ablation for the kernel arms (control ignores): "
                         "plain=describe-only, hard=imperative iterate, kshot=2-case 3-turn "
                         "demonstration in the prefix with no 'you must'")
    ap.add_argument("--max-turns", type=int, default=8,
                    help="max kernel turns before forced submit (control ignores; always 1)")
    ap.add_argument("--min-passes", type=int, default=0,
                    help="forced-iteration floor: refuse a final SVG until N distinct, "
                         "genuinely-changed candidates have been scored (0 = voluntary/off)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-root", type=pathlib.Path,
                    default=_REPO / "output_data" / "svg_runs" / f"repl_{int(time.time())}")
    args = ap.parse_args()

    target = load_target_from_path(str(args.frame))
    prefix = f"{args.frame.stem}_{args.voice}_{args.arm}_{args.elicit}_m{args.min_passes}"
    rep = run_rollout(target, args.arm, args.max_turns, args.max_tokens,
                      args.temperature, args.seed, args.out_root, prefix,
                      elicit=args.elicit, min_passes=args.min_passes, voice=args.voice)
    print(f"[repl_elicit] voice={args.voice} arm={args.arm} elicit={args.elicit} min_passes={args.min_passes} "
          f"best_ssim={rep['best_ssim']} best_mse={rep['best_mse']} judge={rep['judge_faithfulness']} "
          f"code={rep['code_calls']}call/{rep['code_errors']}err "
          f"passes={rep['accepted_passes']} rej_finals={rep['rejected_finals']} "
          f"tests={rep['tests']} turns={rep['turns_used']} -> {args.out_root}")


if __name__ == "__main__":
    main()
