#!/usr/bin/env python3
"""Post-refactor vision-kernel parity validator.

Hits /v1/media/extract on the same two frames used to capture the
pre-refactor reference softs in /tmp/vision_refactor_refs/, compares
byte-for-byte and elementwise (fp32) against the reference.

Exits 0 on pass, nonzero on fail. No eyeballing — hard asserts only.
"""
from __future__ import annotations
import base64, hashlib, json, os, struct, sys, urllib.request

REFS_DIR = "/tmp/vision_refactor_refs"
SERVER = os.environ.get("SERVER", "http://127.0.0.1:8000")

# Post-refactor tolerance. Batched-B=1 *should* be bit-exact vs serial
# (same kernel dispatches, same buffer shapes). If it's not, these
# ceilings catch any meaningful drift. Tighten if the refactor turns
# out to preserve bit-equality.
MAX_ABS_TOL = 1e-6
MAX_REL_RMS = 1e-6


def fp32_from_bytes(raw: bytes) -> list[float]:
    assert len(raw) % 4 == 0, f"byte count {len(raw)} not aligned to fp32"
    n = len(raw) // 4
    return list(struct.unpack(f"{n}f", raw))


def compare(name: str, meta_path: str, ref_path: str) -> tuple[bool, str]:
    with open(meta_path) as f: meta = json.load(f)
    with open(ref_path, "rb") as f: ref = f.read()

    with open(meta["source_png"], "rb") as f: png = f.read()
    body = {"image_url": "data:image/png;base64," + base64.b64encode(png).decode()}
    req = urllib.request.Request(
        SERVER + "/v1/media/extract",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())

    got = base64.b64decode(r["softs_b64"])

    if r["n_tokens"] != meta["n_tokens"]:
        return False, f"n_tokens drift: {meta['n_tokens']} → {r['n_tokens']}"
    if r["is_fp32"] != meta["is_fp32"]:
        return False, f"is_fp32 drift: {meta['is_fp32']} → {r['is_fp32']}"
    if r["bytes"] != meta["bytes"]:
        return False, f"byte-count drift: {meta['bytes']} → {r['bytes']}"
    if r["cache_key"] != meta["cache_key"]:
        return False, (f"cache_key drift: {meta['cache_key']} → {r['cache_key']}"
                        " (image bytes changed?)")

    got_sha = hashlib.sha256(got).hexdigest()
    ref_sha = hashlib.sha256(ref).hexdigest()

    if got_sha == ref_sha:
        return True, f"BIT-EXACT match (sha256 {got_sha[:16]}…)"

    # Not bit-exact — check elementwise tolerance.
    g = fp32_from_bytes(got); rf = fp32_from_bytes(ref)
    if len(g) != len(rf):
        return False, f"length drift: {len(rf)} → {len(g)} floats"
    max_abs = 0.0; sum_sq_diff = 0.0; sum_sq_ref = 0.0
    for a, b in zip(g, rf):
        d = a - b
        max_abs = max(max_abs, abs(d))
        sum_sq_diff += d * d
        sum_sq_ref  += b * b
    rel_rms = (sum_sq_diff / max(sum_sq_ref, 1e-20)) ** 0.5
    if max_abs <= MAX_ABS_TOL and rel_rms <= MAX_REL_RMS:
        return True, (f"NOT bit-exact but within tol "
                       f"(max|Δ|={max_abs:.3e}, rel_rms={rel_rms:.3e})")
    return False, (f"DIVERGENCE: sha {ref_sha[:16]}→{got_sha[:16]}, "
                    f"max|Δ|={max_abs:.3e} (tol {MAX_ABS_TOL:.0e}), "
                    f"rel_rms={rel_rms:.3e} (tol {MAX_REL_RMS:.0e})")


def main():
    if not os.path.isdir(REFS_DIR):
        print(f"no references at {REFS_DIR}", file=sys.stderr); sys.exit(2)
    metas = sorted(p for p in os.listdir(REFS_DIR) if p.endswith(".meta.json"))
    if not metas:
        print(f"no .meta.json files in {REFS_DIR}", file=sys.stderr); sys.exit(2)

    all_ok = True
    for mfile in metas:
        name = mfile[: -len(".meta.json")]
        meta_path = os.path.join(REFS_DIR, mfile)
        ref_path = os.path.join(REFS_DIR, name)
        ok, msg = compare(name, meta_path, ref_path)
        marker = "PASS" if ok else "FAIL"
        print(f"[{marker}] {name}: {msg}")
        if not ok: all_ok = False

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
