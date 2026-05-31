#!/usr/bin/env python3
"""Calibrate the image VLM-judge by sweeping prompt variants & k-shot count.

Mirrors tools/user-agent-harness/judge_prompt_ab.mjs (text-judge validation),
applied to the SVG-reconstruction image judge that floored to 1/5 in the first
elicit.py demo (see docs/causal_ophthalmology_harness_design.md §7).

The judge is calibrated against a CONTROLLED-DEGRADATION test set: each candidate
is a known perturbation of a reference frame, so its semantic-fidelity rank is
ground truth WITHOUT any human or external model in the loop. This lets us
measure, per judge-prompt variant:
  - floor_bias : fraction scored at the floor (1) when ground truth > 1
  - range_use  : distinct scores used / std of scores (degenerate judge ~ 0)
  - rank_corr  : Spearman(judge score, ground-truth rank)  [the headline metric]
  - discrim    : mean(score | gt>=4) - mean(score | gt<=2)

Variants:
  V0_holistic : the degenerate baseline prompt (no rubric, "be strict")
  V1_rubric   : anchored per-score rubric, 0-shot
  Vk_shot     : V1_rubric + k interleaved on-policy exemplars (the k-shot sweep)

k-shot exemplars default to controlled degradations of HELD-OUT frames (no leak;
known labels). Override with --exemplars-json to use on-policy SVG renders
(list of {target, candidate, score, reason}). Because exemplar images are FIXED
across every judge call, they hit the engine's per-image vision cache + the
content-hash prefix cache — so the long k-shot prefix amortizes to ~free; we
report the vision_cache_hits delta to show it.

Usage:
  GEMMA_BASE=http://127.0.0.1:8001 \\
    uv run --with numpy --with pillow --with scikit-image \\
      python tools/svg_elicit/judge_calibrate.py \\
      --frames-dir test_data/amongus_frames --max-shots 3
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, sys, time, urllib.request

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "archival"))
from svg_refinement_loop import mse_images, image_to_data_url  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

BASE = os.environ.get("GEMMA_BASE", "http://127.0.0.1:8001")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

import base64 as _b64  # noqa: E402
import recorder  # same-dir module  # noqa: E402
REC: "recorder.Recorder | None" = None   # set in main() once run_dir is known
URL2META: dict[str, dict] = {}


def img_url(img, **stash_kw) -> str:
    """Stash image (+ optional svg/code) via the recorder and return its data-url,
    registering url->meta so each judged call's images can be token-mapped."""
    url = "data:image/png;base64," + _b64.b64encode(recorder.png_bytes(img)).decode()
    if REC is not None and url not in URL2META:
        URL2META[url] = REC.stash_image(img, **stash_kw)
    return url


# ── Controlled degradations: candidate + ground-truth semantic rank (1-5) ──
def degradations(frame: Image.Image, wrong: Image.Image) -> list[tuple[str, Image.Image, int]]:
    """Each entry: (name, candidate_image, ground_truth_semantic_score). The
    score reflects how recognizable the ORIGINAL subject/pose is in the
    candidate — deliberately decoupled from pixel error where possible
    (grayscale/hue keep structure but wreck color = SSIM/semantic dissociation)."""
    w, h = frame.size
    g = frame.convert("RGB")
    def ds(factor):  # downsample-upsample (lose high-freq detail)
        return g.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR).resize((w, h), Image.NEAREST)
    gray = g.convert("L").convert("RGB")
    hue = Image.fromarray(np.roll(np.asarray(g), 1, axis=2))  # channel-rotate: pose intact, color wrong
    noise = np.clip(np.asarray(g).astype(np.float32) + np.random.RandomState(0).normal(0, 70, (h, w, 3)), 0, 255).astype("uint8")
    blank = Image.new("RGB", (w, h), (128, 128, 128))
    return [
        ("identity",   g,                                   5),
        ("blur_light", g.filter(ImageFilter.GaussianBlur(3)), 4),
        ("grayscale",  gray,                                 4),  # pose intact, color gone
        ("downsample", ds(10),                               3),
        ("blur_med",   g.filter(ImageFilter.GaussianBlur(9)), 3),
        ("hue_rotate", hue,                                  3),  # structure intact, color wrong
        ("blur_heavy", g.filter(ImageFilter.GaussianBlur(20)), 2),
        ("noise",      Image.fromarray(noise),               2),
        ("wrong_frame", wrong.convert("RGB").resize((w, h)), 1),
        ("blank",      blank,                                1),
    ]


def ssim_score(a: Image.Image, b: Image.Image) -> float:
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(np.asarray(a.convert("RGB")), np.asarray(b.convert("RGB")),
                      channel_axis=2, data_range=255))


# ── Judge prompt variants ────────────────────────────────────────────────
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

SYS_HOLISTIC = ("You are a meticulous visual critic. Score how faithfully the "
                "candidate reproduces the target. Be strict.")
SYS_RUBRIC = "You are a meticulous, calibrated visual critic. " + RUBRIC

ASK = ('Respond with ONLY JSON: {"score": <int 1-5>, "reason": "<one phrase>"}.')


def exemplar_block(exemplars: list[dict]) -> list[dict]:
    """Interleaved k-shot: for each, TARGET img, CANDIDATE img, the gold JSON."""
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


def build_messages(variant: str, exemplars: list[dict],
                   target_url: str, cand_url: str) -> list[dict]:
    system = SYS_HOLISTIC if variant == "V0_holistic" else SYS_RUBRIC
    user_content = []
    if exemplars:
        user_content += [{"type": "text", "text":
            "Here are calibrated scoring examples. Study the scale, then score "
            "the final pair the same way."}]
        user_content += exemplar_block(exemplars)
        user_content += [{"type": "text", "text": "--- Now score THIS pair ---"}]
    user_content += [
        {"type": "text", "text": "TARGET:"},
        {"type": "image_url", "image_url": {"url": target_url}},
        {"type": "text", "text": "CANDIDATE:"},
        {"type": "image_url", "image_url": {"url": cand_url}},
        {"type": "text", "text": ASK},
    ]
    return [{"role": "system", "content": system},
            {"role": "user", "content": user_content}]


def call_lm(messages, max_tokens=256, temperature=0.2, seed=0, timeout=900.0):
    """Non-streaming call with return_token_ids → (text, full_response_json).
    Non-stream so we receive usage.prompt_tokens plus the EXACT token_ids +
    prompt_token_layout the recorder logs (no re-tokenization, no log scraping)."""
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "stream": False, "seed": int(seed), "return_token_ids": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return text, resp


def parse_score(text: str) -> int | None:
    m = _JSON_RE.search(text)
    if m:
        try:
            v = json.loads(m.group(0)).get("score")
            if isinstance(v, (int, float)):
                return int(round(v))
        except Exception:
            pass
    m2 = re.search(r"score\D{0,4}([1-5])", text, re.I)
    return int(m2.group(1)) if m2 else None


def health_vision_hits() -> int:
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=4) as r:
            return json.load(r).get("vision_cache_hits", 0)
    except Exception:
        return 0


def spearman(pred: list, gt: list) -> float | None:
    pairs = [(p, g) for p, g in zip(pred, gt) if p is not None]
    if len(pairs) < 3:
        return None
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(xs):  # average ties
            j = i
            while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            avg = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    pr, gr = rank([p for p, _ in pairs]), rank([g for _, g in pairs])
    prm, grm = sum(pr) / len(pr), sum(gr) / len(gr)
    num = sum((a - prm) * (b - grm) for a, b in zip(pr, gr))
    den = (sum((a - prm) ** 2 for a in pr) * sum((b - grm) ** 2 for b in gr)) ** 0.5
    return num / den if den else None


def metrics(preds: list, gts: list, ssims: list) -> dict:
    valid = [(p, g) for p, g in zip(preds, gts) if p is not None]
    above = [(p, g) for p, g in valid if g > 1]
    floor = sum(1 for p, _ in above if p == 1) / len(above) if above else None
    got = [p for p, _ in valid]
    hi = [p for p, g in valid if g >= 4]
    lo = [p for p, g in valid if g <= 2]
    return {
        "n": len(valid), "parse_fail": preds.count(None),
        "floor_bias": floor,
        "distinct_scores": len(set(got)),
        "score_std": float(np.std(got)) if got else None,
        "rank_corr_vs_gt": spearman(preds, gts),
        "rank_corr_vs_ssim": spearman(preds, ssims),
        "discrim_hi_minus_lo": (sum(hi) / len(hi) - sum(lo) / len(lo)) if hi and lo else None,
    }


def make_exemplars(frames: list[pathlib.Path], picks: list[tuple[str, int, str]]) -> list[dict]:
    """Build k-shot exemplars from controlled degradations of HELD-OUT frames.
    picks = [(degradation_name, frame_idx, reason), ...] spanning the 1-5 range,
    including SSIM/semantic dissociation cases (grayscale/hue)."""
    out = []
    for name, fi, reason in picks:
        f = Image.open(frames[fi]).convert("RGB")
        wrong = Image.open(frames[(fi + 3) % len(frames)]).convert("RGB")
        cand = next(c for n, c, _ in degradations(f, wrong) if n == name)
        score = next(s for n, _, s in degradations(f, wrong) if n == name)
        out.append({"target_url": img_url(f), "cand_url": img_url(cand),
                    "score": score, "reason": reason, "name": name})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", type=pathlib.Path, default=REPO / "test_data" / "amongus_frames")
    ap.add_argument("--test-frames", type=int, default=3, help="how many frames form the test set")
    ap.add_argument("--max-shots", type=int, default=3)
    ap.add_argument("--exemplars-json", type=pathlib.Path, default=None,
                    help="override k-shot exemplars with on-policy renders "
                         "(list of {target, candidate, score, reason})")
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO / "output_data" / "svg_runs" / f"judge_calib_{int(time.time())}.json")
    args = ap.parse_args()

    global REC
    run_dir = args.out.parent / args.out.stem
    REC = recorder.Recorder(run_dir)
    print(f"[calib] recording transcripts + artifacts under {run_dir}/")

    frames = sorted(args.frames_dir.glob("frame_*.png"))
    assert len(frames) >= args.test_frames + 2, "need held-out frames for exemplars"
    # No leakage: any frame used as an exemplar TARGET is excluded from the test set.
    ex_target_names = set()
    if args.exemplars_json and args.exemplars_json.exists():
        for e in json.loads(args.exemplars_json.read_text()):
            ex_target_names.add(pathlib.Path(e["target"]).name)
    avail = [f for f in frames if f.name not in ex_target_names]
    test_frames = avail[:args.test_frames]
    pool = [f for f in frames if f not in test_frames]
    # Truly-unrelated source for the wrong_frame=1 anchor: a non-amongus frame
    # (a different video) so the floor anchor isn't "same character, other pose".
    _wrong_path = REPO / "test_data" / "frames_v2" / "KCrfDHS_YUw" / "frame_0000.png"
    wrong_src = (Image.open(_wrong_path).convert("RGB") if _wrong_path.exists()
                 else Image.open(pool[0]).convert("RGB"))

    # Build the controlled-degradation test items (known ground truth).
    test_items = []
    for tf in test_frames:
        f = Image.open(tf).convert("RGB")
        wrong = wrong_src
        for name, cand, gt in degradations(f, wrong):
            test_items.append({"frame": tf.name, "deg": name, "gt": gt,
                               "target_url": img_url(f, src_path=str(tf)),
                               "cand_url": img_url(cand),
                               "ssim": ssim_score(f, cand),
                               "mse": mse_images(f, cand)})
    print(f"[calib] {len(test_items)} test items from {len(test_frames)} frames; "
          f"exemplar pool {len(pool)} frames")

    # k-shot exemplars: span the range incl. dissociation cases. From pool frames.
    if args.exemplars_json and args.exemplars_json.exists():
        ex_all = json.loads(args.exemplars_json.read_text())
        exemplars = []
        for e in ex_all:
            tgt = Image.open(REPO / e["target"]).convert("RGB")
            cand_path = REPO / e["candidate"]
            cand = Image.open(cand_path).convert("RGB")
            # Link the on-policy render to its binary .svg source + parsed code
            # (elicit.py saves <prefix>_iter_NN.svg next to _rendered.png).
            svg_sib = pathlib.Path(str(cand_path).replace("_rendered.png", ".svg"))
            svg_src = svg_sib.read_text() if svg_sib.exists() else None
            exemplars.append({
                "target_url": img_url(tgt, src_path=e["target"]),
                "cand_url": img_url(cand, source_svg=svg_src, src_path=e["candidate"]),
                "score": e["score"], "reason": e["reason"], "name": e.get("name", "onpolicy")})
        print(f"[calib] using {len(exemplars)} on-policy exemplars from {args.exemplars_json.name}")
    else:
        exemplars = make_exemplars(pool, [
            ("identity",   0, "exact match — same crewmate, same twerk pose, same colors"),
            ("wrong_frame", 1, "a completely different image; the subject is absent"),
            ("grayscale",  2, "correct subject and pose but the yellow color is lost"),
            ("blur_med",   0, "the crewmate and pose are still recognizable through the blur"),
        ][:max(args.max_shots, 1) + 1])

    variants = ["V0_holistic", "V1_rubric"] + [f"V{k}_shot" for k in range(1, args.max_shots + 1)]
    results = {}
    for variant in variants:
        if variant == "V0_holistic" or variant == "V1_rubric":
            ex = []
        else:
            k = int(variant[1])
            ex = exemplars[:k]
        n_imgs = 2 + 2 * len(ex)
        hits0 = health_vision_hits()
        t0 = time.time()
        preds = []
        for i, it in enumerate(test_items):
            msgs = build_messages(variant if variant != "V0_holistic" else "V0_holistic",
                                  ex, it["target_url"], it["cand_url"])
            # Build with rubric for Vk_shot; V1_rubric is 0-shot rubric.
            if variant.startswith("V") and "shot" in variant:
                msgs = build_messages("V1_rubric", ex, it["target_url"], it["cand_url"])
            t_call = time.time()
            txt, resp = call_lm(msgs, seed=1000 + i)  # distinct seed/item → independent
            score = parse_score(txt)
            preds.append(score)
            # Record the full transcript: exact token in/out + artifacts.
            # Image order in the layout == message order, so map metas in order.
            metas = [URL2META.get(p["image_url"]["url"], {})
                     for m in msgs if isinstance(m["content"], list)
                     for p in m["content"] if p.get("type") == "image_url"]
            REC.record(call_id=f"{variant}_item{i:03d}", variant=variant, messages=msgs,
                       image_metas_ordered=metas, response=resp,
                       timing={"wall_s": round(time.time() - t_call, 2)},
                       extra={"test_item": {k: it[k] for k in ("frame", "deg", "gt", "ssim", "mse")},
                              "parsed_score": score, "n_shots": len(ex)})
        wall = time.time() - t0
        hits = health_vision_hits() - hits0
        m = metrics(preds, [it["gt"] for it in test_items], [it["ssim"] for it in test_items])
        m.update({"images_per_call": n_imgs, "wall_s": round(wall, 1),
                  "vision_cache_hits_delta": hits, "preds": preds})
        results[variant] = m
        print(f"  {variant:>11s}: floor={m['floor_bias']!r} rankρ={m['rank_corr_vs_gt']!r} "
              f"discrim={m['discrim_hi_minus_lo']!r} distinct={m['distinct_scores']} "
              f"imgs/call={n_imgs} cache_hits+={hits} wall={m['wall_s']}s")

    args.out.write_text(json.dumps({
        "test_items": [{k: v for k, v in it.items() if not k.endswith("_url")} for it in test_items],
        "results": results}, indent=2, default=str))
    print(f"\n[calib] → {args.out}")
    # Headline: shot-count → rank correlation (the help/hurt curve).
    print("shot-count → rank_corr_vs_gt (the help/hurt curve):")
    print(f"  0 (rubric): {results['V1_rubric']['rank_corr_vs_gt']!r}")
    for k in range(1, args.max_shots + 1):
        print(f"  {k}-shot   : {results[f'V{k}_shot']['rank_corr_vs_gt']!r}")


if __name__ == "__main__":
    main()
