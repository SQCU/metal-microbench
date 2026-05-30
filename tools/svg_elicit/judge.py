#!/usr/bin/env python3
"""Calibrated image VLM-judge: anchored rubric + on-policy k-shot exemplars.

The default judge in elicit.py floored to 1/5 (no scale to anchor to). The
judge_calibrate.py sweep showed: an anchored rubric removes floor-bias, and ~3
on-policy SVG-render exemplars that dissociate SSIM/MSE/semantic push rank-corr
vs ground truth to ~0.87. This module packages that calibrated judge so the
multi-turn elicitation loop can use it for BOTH in-loop feedback and best-of-N
selection. See docs/causal_ophthalmology_harness_design.md §8-9.
"""
from __future__ import annotations
import base64, io, json, pathlib, re
from PIL import Image

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

RUBRIC = ("Score how faithfully the CANDIDATE reproduces the TARGET's subject, "
          "pose, composition, color, and any text, on a 1-5 scale:\n"
          "  5 = near-identical; only minor differences\n"
          "  4 = clearly the same subject/scene; small style or color errors\n"
          "  3 = subject/pose recognizable but notable color/style/detail loss\n"
          "  2 = some shared elements but the subject is hard to recognize\n"
          "  1 = unrelated, blank, or unrecognizable\n"
          "Most reproductions fall in 2-4. Reserve 1 ONLY for unrelated/blank. "
          "Judge SUBJECT RECOGNIZABILITY, not pixel-exactness — a faithful "
          "drawing in the wrong color is still a 3-4, not a 1.")
SYS_RUBRIC = "You are a meticulous, calibrated visual critic. " + RUBRIC
ASK = ('Respond with ONLY JSON: {"score": <int 1-5>, '
       '"missing": ["<qualities in TARGET absent/wrong in CANDIDATE>"], '
       '"biggest_fix": "<the single most important change>"}.')


def _durl(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def load_exemplars(repo: pathlib.Path, json_path: pathlib.Path) -> list[dict]:
    """Load on-policy exemplars (target/candidate image paths + score + reason)
    into judge-ready data-urls. Returns [] if the path is missing."""
    if not json_path or not pathlib.Path(json_path).exists():
        return []
    out = []
    for e in json.loads(pathlib.Path(json_path).read_text()):
        tgt = Image.open(repo / e["target"]).convert("RGB")
        cand = Image.open(repo / e["candidate"]).convert("RGB")
        out.append({"target_url": _durl(tgt), "cand_url": _durl(cand),
                    "score": e["score"], "reason": e["reason"]})
    return out


def _exemplar_block(exemplars: list[dict]) -> list[dict]:
    content = []
    for i, ex in enumerate(exemplars):
        content += [
            {"type": "text", "text": f"--- Example {i+1} ---\nTARGET:"},
            {"type": "image_url", "image_url": {"url": ex["target_url"]}},
            {"type": "text", "text": "CANDIDATE:"},
            {"type": "image_url", "image_url": {"url": ex["cand_url"]}},
            {"type": "text", "text": "Correct answer: "
                + json.dumps({"score": ex["score"], "reason": ex["reason"]})},
        ]
    return content


def build_messages(exemplars: list[dict], target_img: Image.Image,
                   cand_img: Image.Image) -> list[dict]:
    uc = []
    if exemplars:
        uc.append({"type": "text", "text":
                   "Here are calibrated scoring examples. Study the scale, then "
                   "score the final pair the same way."})
        uc += _exemplar_block(exemplars)
        uc.append({"type": "text", "text": "--- Now score THIS pair ---"})
    uc += [
        {"type": "text", "text": "TARGET:"},
        {"type": "image_url", "image_url": {"url": _durl(target_img)}},
        {"type": "text", "text": "CANDIDATE:"},
        {"type": "image_url", "image_url": {"url": _durl(cand_img)}},
        {"type": "text", "text": ASK},
    ]
    return [{"role": "system", "content": SYS_RUBRIC}, {"role": "user", "content": uc}]


# ---------------------------------------------------------------------------
# Feature-Likert profile judge. A single 1-5 "faithfulness" scalar can't
# resolve semantic differences (a 0.5 mean delta is below its resolution and
# it conflates many axes). Instead: characterize the TARGET and the CANDIDATE
# *independently* on the same k monotonic 1-5 feature scales, then compare the
# two profiles. This (a) gives k× the resolution (mean of k deltas is near-
# continuous), (b) is less circular — the judge DESCRIBES each image rather
# than scoring success — and (c) decomposes "did the meaning survive" into
# interpretable per-feature gaps. Plus one comparative subject_match for raw
# recognizability. Designed for complex video frames, not the amongus canary.
FEATURES = {
    "detail":        "fine visual detail / texture density (1=flat or minimal, 5=intricate)",
    "color_variety": "spread of distinct hues present (1=near-monochrome, 5=many colours)",
    "saturation":    "colour vividness (1=muted/greyish, 5=vivid)",
    "brightness":    "overall luminance (1=dark, 5=bright)",
    "complexity":    "compositional busyness / number of distinct regions (1=simple, 5=busy)",
    "contrast":      "tonal contrast (1=flat, 5=high)",
    "depth":         "sense of foreground/background layering or 3-D structure (1=flat, 5=deep)",
}
FEATURE_SYS = ("You are a meticulous visual analyst. You will see a TARGET image and a CANDIDATE "
               "image. Characterize EACH image independently on a set of 1-5 feature scales — rate "
               "what each image ACTUALLY shows, do not reward or penalize the candidate for being a "
               "reconstruction. The two profiles will be compared feature-by-feature.")


def _feature_ask() -> str:
    lines = "\n".join(f"  {k}: {v}" for k, v in FEATURES.items())
    return ("Rate the TARGET and the CANDIDATE each, independently, on these 1-5 scales:\n"
            f"{lines}\n"
            "Then rate subject_match: does the CANDIDATE depict the same subject/scene/content as "
            "the TARGET? (1=unrelated, 3=partially recognizable, 5=clearly the same).\n"
            'Respond with ONLY JSON: {"target": {<feature>: <int 1-5>, ...}, '
            '"candidate": {<feature>: <int 1-5>, ...}, "subject_match": <int 1-5>}.')


def feature_score(chat, target_img: Image.Image, cand_img: Image.Image) -> dict | None:
    """`chat(messages) -> text`. Returns a feature-profile comparison:
    {target_profile, candidate_profile, feature_deltas, profile_mean_delta,
     subject_match, semantic_distance(0=identical..1=worst)} or None."""
    try:
        msgs = [{"role": "system", "content": FEATURE_SYS},
                {"role": "user", "content": [
                    {"type": "text", "text": "TARGET:"},
                    {"type": "image_url", "image_url": {"url": _durl(target_img)}},
                    {"type": "text", "text": "CANDIDATE:"},
                    {"type": "image_url", "image_url": {"url": _durl(cand_img)}},
                    {"type": "text", "text": _feature_ask()}]}]
        m = _JSON_RE.search(chat(msgs))
        if not m:
            return None
        obj = json.loads(m.group(0))
        tp, cp = obj.get("target", {}), obj.get("candidate", {})
        deltas = {f: abs(float(tp[f]) - float(cp[f]))
                  for f in FEATURES
                  if isinstance(tp.get(f), (int, float)) and isinstance(cp.get(f), (int, float))}
        prof = round(sum(deltas.values()) / len(deltas), 3) if deltas else None
        sm = obj.get("subject_match")
        sm = float(sm) if isinstance(sm, (int, float)) else None
        sem = (round(((prof / 4.0) + (1 - sm / 5.0)) / 2.0, 3)
               if (prof is not None and sm is not None) else None)
        return {"target_profile": tp, "candidate_profile": cp, "feature_deltas": deltas,
                "profile_mean_delta": prof, "subject_match": sm, "semantic_distance": sem}
    except Exception:
        return None


def score(chat, target_img: Image.Image, cand_img: Image.Image,
          exemplars: list[dict]) -> dict | None:
    """`chat(messages) -> text`. Returns {faithfulness:1-5, missing:[...],
    biggest_fix:str} or None on parse failure."""
    try:
        txt = chat(build_messages(exemplars, target_img, cand_img))
        m = _JSON_RE.search(txt)
        if not m:
            return None
        obj = json.loads(m.group(0))
        s = obj.get("score")
        return {"faithfulness": float(s) if isinstance(s, (int, float)) else None,
                "missing": obj.get("missing") or [],
                "biggest_fix": obj.get("biggest_fix")}
    except Exception:
        return None
