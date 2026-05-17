#!/usr/bin/env python3
"""Concurrent vision-submit stress test.

Fires N distinct synthetic images (gaussian noise, per-seed unique bytes)
from K threads at once against /v1/media/extract. Verifies:

  - No deadlock / no corruption under parallel submission
  - All requests return n_tokens=280 and distinct cache_keys matching the
    SHA-256 of the input bytes
  - Soft-tokens are deterministic on re-submission (cache hit bit-exact)

Exit 0 on PASS, 1 on FAIL.
"""
from __future__ import annotations
import argparse, base64, concurrent.futures as cf, hashlib, io, json, sys
import time, urllib.request
import numpy as np
from PIL import Image

SERVER = "http://127.0.0.1:8000"


def make_image(seed: int, size: int = 64) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return buf.getvalue()


def extract(png: bytes) -> dict:
    req = urllib.request.Request(
        SERVER + "/v1/media/extract",
        data=json.dumps(
            {"image_url": "data:image/png;base64," + base64.b64encode(png).decode()}
        ).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return {
        "cache_key": r["cache_key"],
        "n_tokens": r["n_tokens"],
        "softs_sha": hashlib.sha256(base64.b64decode(r["softs_b64"])).hexdigest(),
        "elapsed_ms": r["elapsed_ms"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=12, help="distinct images")
    ap.add_argument("--K", type=int, default=8, help="concurrent threads")
    args = ap.parse_args()

    images = [make_image(1000 + i) for i in range(args.N)]
    cache_keys = [hashlib.sha256(p).hexdigest() for p in images]
    assert len(set(cache_keys)) == args.N, "images must be distinct"
    print(f"[stress] N={args.N} distinct images, K={args.K} concurrent threads")

    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.K) as exe:
        futs = [exe.submit(extract, p) for p in images]
        results = [f.result() for f in cf.as_completed(futs)]
    wall = time.time() - t0
    print(f"  wall = {wall:.2f}s  ({args.N / wall:.2f} img/s)")

    ok = True
    if {r["cache_key"] for r in results} != set(cache_keys):
        ok = False; print("  FAIL: cache_key set mismatch")
    for r in results:
        if r["n_tokens"] != 280:
            ok = False; print(f"  FAIL: n_tokens={r['n_tokens']}")
    el = [r["elapsed_ms"] for r in results]
    print(f"  per-req elapsed_ms: min={min(el)} max={max(el)} mean={sum(el)/len(el):.0f}")

    det = extract(images[0])
    orig_sha = next(r["softs_sha"] for r in results if r["cache_key"] == cache_keys[0])
    if det["softs_sha"] != orig_sha:
        ok = False; print("  FAIL: softs non-deterministic on re-submit")
    else:
        print(f"  OK: softs deterministic on cache hit (sha {det['softs_sha'][:16]}…)")

    print(f"\n{'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
