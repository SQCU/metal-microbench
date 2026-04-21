"""Config-driven launcher for the gemma-metal bridge.

Reads server/config.toml, resolves model paths against the repo root,
lets env vars win, then runs uvicorn. Saves users from the 4-line
env-var incantation:

    ./server/serve.py                       # default config.toml
    ./server/serve.py --config myconfig.toml
    GEMMA_PORT=8080 ./server/serve.py       # env override
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

try:
    import tomllib   # stdlib in 3.11+
except ImportError:
    import tomli as tomllib   # type: ignore

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _resolve(path: str) -> str:
    """Absolute paths as-is; relative paths resolved against repo root."""
    if not path:
        return ""
    p = pathlib.Path(path)
    return str(p if p.is_absolute() else (REPO_ROOT / p).resolve())


def _load_config(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"config file not found: {path}\n"
            f"copy server/config.toml.example or pass --config <path>"
        )
    return tomllib.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "server" / "config.toml"),
                    help="path to config.toml (default: server/config.toml)")
    args = ap.parse_args()

    cfg = _load_config(pathlib.Path(args.config))
    model_cfg = cfg.get("model", {})
    server_cfg = cfg.get("server", {})

    # Resolve paths and export as env vars (bridge.py reads these).
    gguf = os.environ.get("GEMMA_GGUF") or _resolve(model_cfg.get("gguf_path", ""))
    st = os.environ.get("GEMMA_SAFETENSORS") or _resolve(model_cfg.get("safetensors_path", ""))
    display = os.environ.get("GEMMA_MODEL_NAME") or model_cfg.get("display_name", "gemma-metal")
    if not gguf or not pathlib.Path(gguf).exists():
        raise SystemExit(
            f"GGUF weights not found at {gguf!r}\n"
            f"  run `uv run server/scripts/fetch-weights.py` to download,\n"
            f"  or set gguf_path in config.toml / export GEMMA_GGUF=/path/to/.gguf"
        )
    os.environ["GEMMA_GGUF"] = gguf
    if st:
        if not pathlib.Path(st).exists():
            print(f"[warn] safetensors_path {st!r} doesn't exist — multimodal disabled", file=sys.stderr)
        else:
            os.environ["GEMMA_SAFETENSORS"] = st
    os.environ["GEMMA_MODEL_NAME"] = display

    host = os.environ.get("GEMMA_HOST") or server_cfg.get("host", "0.0.0.0")
    port = int(os.environ.get("GEMMA_PORT") or server_cfg.get("port", 8000))
    log_level = (os.environ.get("GEMMA_LOG_LEVEL") or server_cfg.get("log_level", "warning")).lower()

    # Run uvicorn. We chdir into the server dir first because bridge.py
    # resolves its static assets relative to __file__ and that's also
    # where uvicorn expects to find the module.
    os.chdir(REPO_ROOT / "server")
    import uvicorn
    print(f"[serve] gguf        = {gguf}")
    print(f"[serve] safetensors = {st or '(none — text-only)'}")
    print(f"[serve] listening   = http://{host}:{port}  ·  model name={display}")
    uvicorn.run("bridge:app", host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
