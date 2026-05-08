"""Single source of truth for bridge URL/host/port across all clients.

Reads `server/config.toml` (the same file `serve.py` reads to launch the
bridge) and exposes BRIDGE_URL / BRIDGE_HOST / BRIDGE_PORT plus URL
helpers. Env vars override (in priority order: BRIDGE_URL >
QUANT_BRIDGE_URL > GEMMA_BRIDGE > GEMMA_HOST/GEMMA_PORT > config file).

Why this exists
===============
Before this module landed, every client baked `http://127.0.0.1:8001`
into a constant. Adding a TLS proxy, switching ports for a parallel
study run, or pointing at a remote bridge meant patching ~15 files.
Worse, scripts had different defaults: the canonical config said
:8001, bridge.py's docstring still said :8000. This made the
"is the bridge online?" question genuinely confusing.

Now: change `port = 8001` in `server/config.toml` and every harness
follows. Or set BRIDGE_URL once in your shell and override the whole
thing for one run.

Use
===
    from bridge_config import (
        BRIDGE_URL, BRIDGE_HOST, BRIDGE_PORT,
        chat_completions_url, health_url, models_url,
    )

For tools/ packages outside server/, prepend server/ to sys.path or
import via the repo-root path:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[N] / "server"))
    from bridge_config import BRIDGE_URL
"""
from __future__ import annotations

import os
import pathlib
import urllib.parse

try:
    import tomllib   # stdlib in 3.11+
except ImportError:
    import tomli as tomllib   # type: ignore

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "server" / "config.toml"


def _load_server_section() -> dict:
    """Read [server] table from config.toml, or empty dict if missing."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return tomllib.loads(CONFIG_PATH.read_text()).get("server", {})
    except Exception:
        return {}


_cfg = _load_server_section()

# Env-var explicit URL overrides win over everything. Names match the
# legacy variables already scattered across the codebase so existing
# `BRIDGE_URL=... script.py` invocations keep working.
_explicit_url = (
    os.environ.get("BRIDGE_URL")
    or os.environ.get("QUANT_BRIDGE_URL")
    or os.environ.get("GEMMA_BRIDGE")
)

if _explicit_url:
    parsed = urllib.parse.urlparse(_explicit_url.rstrip("/"))
    BRIDGE_HOST = parsed.hostname or "127.0.0.1"
    BRIDGE_PORT = parsed.port or 8001
    BRIDGE_SCHEME = parsed.scheme or "http"
    BRIDGE_URL = f"{BRIDGE_SCHEME}://{BRIDGE_HOST}:{BRIDGE_PORT}"
else:
    # Resolve from config + env. Server-side bind host (0.0.0.0) means
    # "listen on all interfaces" but client-side we need a routable
    # address — collapse 0.0.0.0 to 127.0.0.1 for client URL building.
    raw_host = (
        os.environ.get("GEMMA_HOST")
        or _cfg.get("host", "127.0.0.1")
    )
    BRIDGE_HOST = "127.0.0.1" if raw_host == "0.0.0.0" else raw_host
    BRIDGE_PORT = int(os.environ.get("GEMMA_PORT") or _cfg.get("port", 8001))
    BRIDGE_SCHEME = "http"
    BRIDGE_URL = f"{BRIDGE_SCHEME}://{BRIDGE_HOST}:{BRIDGE_PORT}"


def chat_completions_url() -> str:
    return f"{BRIDGE_URL}/v1/chat/completions"


def health_url() -> str:
    return f"{BRIDGE_URL}/health"


def models_url() -> str:
    return f"{BRIDGE_URL}/v1/models"


def tokenize_url() -> str:
    return f"{BRIDGE_URL}/v1/tokenize"


# Display function for diagnostic logging — every harness should print
# this once at startup so a misconfigured port shows up immediately.
def describe() -> str:
    src = "BRIDGE_URL/QUANT_BRIDGE_URL/GEMMA_BRIDGE env" if _explicit_url \
          else f"config.toml [server] (or GEMMA_HOST/GEMMA_PORT env)"
    return f"bridge → {BRIDGE_URL}  (source: {src})"


if __name__ == "__main__":
    # Smoke: print resolved URL when run directly.
    print(describe())
    print(f"  host = {BRIDGE_HOST}")
    print(f"  port = {BRIDGE_PORT}")
    print(f"  health = {health_url()}")
    print(f"  models = {models_url()}")
    print(f"  chat   = {chat_completions_url()}")
