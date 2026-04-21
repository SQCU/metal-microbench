"""Download Gemma-4 weights listed in server/config.toml into ./models/.

Reads the [fetch] section of config.toml, uses huggingface_hub to pull
each shard. Re-runs are idempotent — huggingface_hub skips files that
are already present and hash-match.

    uv run --with huggingface_hub python server/scripts/fetch-weights.py
    uv run --with huggingface_hub python server/scripts/fetch-weights.py --only gguf
    uv run --with huggingface_hub python server/scripts/fetch-weights.py --only safetensors

If the repo ids in config.toml aren't where you got your copies from,
override them there — this script doesn't bake in any hardcoded
locations beyond what the config says.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

try:
    import tomllib
except ImportError:
    import tomli as tomllib   # type: ignore

from huggingface_hub import hf_hub_download

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _resolve(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def fetch_gguf(cfg: dict) -> None:
    repo = cfg["gguf_repo"]
    filename = cfg["gguf_filename"]
    local_dir = _resolve(cfg["gguf_local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gguf] {repo}/{filename}  →  {local_dir}")
    path = hf_hub_download(repo_id=repo, filename=filename,
                            local_dir=str(local_dir))
    print(f"[gguf] ✓ {path}")


def fetch_safetensors(cfg: dict) -> None:
    repo = cfg["safetensors_repo"]
    files = cfg["safetensors_files"]
    local_dir = _resolve(cfg["safetensors_local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[safetensors] {repo} ({len(files)} files)  →  {local_dir}")
    for f in files:
        path = hf_hub_download(repo_id=repo, filename=f,
                                local_dir=str(local_dir))
        print(f"[safetensors] ✓ {pathlib.Path(path).name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "server" / "config.toml"),
                    help="path to config.toml")
    ap.add_argument("--only", choices=["gguf", "safetensors"],
                    help="fetch only one of the two components")
    args = ap.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        raise SystemExit(f"config not found: {cfg_path}")
    cfg = tomllib.loads(cfg_path.read_text())
    fetch = cfg.get("fetch", {})
    if not fetch:
        raise SystemExit("[fetch] section missing from config.toml")

    try:
        if args.only != "safetensors":
            fetch_gguf(fetch)
        if args.only != "gguf":
            fetch_safetensors(fetch)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        print("\nnotes:", file=sys.stderr)
        print("  · some Gemma repos are gated — run `huggingface-cli login` first", file=sys.stderr)
        print("  · config.toml [fetch] section lets you point at different repos", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
