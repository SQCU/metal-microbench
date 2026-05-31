#!/usr/bin/env python3
"""Direct ENGINE cross-stream image-isolation probe — no harness, no krepl import.

Reproduces the leak conditions that produced QTG content (麻雀) inside E4q output:
many CONCURRENT multi-turn multimodal streams, half carrying image A, half image B,
so their image soft tokens / KV pages are in flight together. Then it greps each
stream's output for the OTHER image's distinctive markers. A clean engine yields
zero cross-markers.

Run (engine up at GEMMA_BASE, default http://127.0.0.1:8001):
  ./server/.venv/bin/python tools/svg_elicit/leak_probe.py --seeds 6
"""
from __future__ import annotations
import argparse, base64, json, os, pathlib, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.environ.get("GEMMA_BASE", "http://127.0.0.1:8001")
REPO = pathlib.Path(__file__).resolve().parents[2]

IMAGES = {
    "QTG": {"path": REPO / "test_data/frames_v2/QTG2yY1znxM/frame_0010.png",
            "self": ["麻雀", "田村", "ゼミ", "mahjong", "南", "東", "北"],
            "other_key": "E4q"},
    "E4q": {"path": REPO / "test_data/frames_v2/E4qMxJJAszg/frame_0010.png",
            "self": ["crewmate", "among", "amongus", "sus", "visor", "astronaut",
                     "nebula", "twerk"],
            "other_key": "QTG"},
}
PROMPT = ("Reconstruct THIS image as an SVG. Include descriptive <!-- comments --> naming "
          "what you see, and render any visible text as <text> elements verbatim.")


def _data_url(p: pathlib.Path) -> str:
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()


def _chat(messages, seed, max_tokens=900):
    body = json.dumps({"model": "gemma", "max_tokens": max_tokens, "temperature": 0.7,
                       "seed": seed, "messages": messages}).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=300))
    return r["choices"][0]["message"]["content"]


def run_stream(key, seed, turns):
    img = IMAGES[key]
    durl = _data_url(img["path"])
    messages = [{"role": "user", "content": [
        {"type": "text", "text": PROMPT},
        {"type": "image_url", "image_url": {"url": durl}}]}]
    outputs = []
    for t in range(turns):
        out = _chat(messages, seed + t)
        outputs.append(out)
        messages.append({"role": "assistant", "content": out})
        messages.append({"role": "user", "content":
                         "Refine it: add any element you missed and fix wrong colours. Re-emit the SVG."})
    return key, seed, "\n".join(outputs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6, help="streams per image")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    jobs = [(k, 100 + i * 7, args.turns) for k in IMAGES for i in range(args.seeds)]
    print(f"[leak_probe] {len(jobs)} concurrent streams "
          f"({args.seeds}/image x {len(IMAGES)} images), {args.turns} turns each, base={BASE}",
          flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_stream, k, s, tn) for (k, s, tn) in jobs]
        for f in as_completed(futs):
            results.append(f.result())

    print("\n=== cross-contamination scan ===")
    leaks = []
    for key, seed, text in sorted(results):
        low = text.lower()
        other = IMAGES[IMAGES[key]["other_key"]]["self"]
        hits = [m for m in other if (m if any(ord(c) > 127 for c in m) else m.lower())
                in (text if any(ord(c) > 127 for c in m) else low)]
        tag = f"{key}_s{seed}"
        if hits:
            leaks.append((tag, hits))
            print(f"  [LEAK] {tag}: contains {IMAGES[key]['other_key']} markers {hits}")
        else:
            print(f"  ok     {tag}: clean")

    print(f"\n{'CLEAN — no cross-stream image leak' if not leaks else 'LEAK DETECTED'}: "
          f"{len(results) - len(leaks)}/{len(results)} streams clean")
    # persist outputs for forensic grep
    out = REPO / "output_data/svg_runs/leak_verify"
    out.mkdir(parents=True, exist_ok=True)
    for key, seed, text in results:
        (out / f"{key}_s{seed}.txt").write_text(text)
    print(f"-> outputs in {out}")
    return 1 if leaks else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
