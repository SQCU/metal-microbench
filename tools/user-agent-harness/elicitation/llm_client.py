"""ST-owned LLM client for Python user-personas harnesses.

Python elicitation scripts are children of the user-personas plugin. They
should not know whether the active model provider is the local bridge,
OpenRouter, or any other OpenAI-compatible backend. The plugin owns that
routing through SillyTavern's configured provider profile.
"""

import json
import os
import urllib.request


def _base_url() -> str:
    plugin = os.environ.get("PLUGIN_URL")
    if plugin:
        return plugin.rstrip("/")
    st = os.environ.get("USER_PERSONAS_ST_URL") or os.environ.get("ST_URL")
    if not st:
        raise RuntimeError("missing PLUGIN_URL, USER_PERSONAS_ST_URL, or ST_URL")
    return st.rstrip("/") + "/api/plugins/user-personas"


def _st_url() -> str:
    st = os.environ.get("USER_PERSONAS_ST_URL") or os.environ.get("ST_URL")
    if not st:
        raise RuntimeError("missing USER_PERSONAS_ST_URL or ST_URL")
    return st.rstrip("/")


def plugin_get(path: str, timeout=30) -> dict:
    req = urllib.request.Request(_base_url() + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def st_character_card(avatar: str, timeout=30) -> dict:
    avatar_url = avatar if avatar.endswith(".png") else f"{avatar}.png"
    req = urllib.request.Request(
        _st_url() + "/api/characters/get",
        data=json.dumps({"avatar_url": avatar_url}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("data") or data


def llm_call(messages, max_tokens=None, seed=None, timeout=180) -> str:
    body = {"messages": messages}
    if max_tokens is not None and max_tokens > 0:
        body["max_tokens"] = max_tokens
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        _base_url() + "/llm-call",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    text = data.get("text")
    if not isinstance(text, str):
        raise RuntimeError(f"/llm-call returned no text: {data!r}")
    return text


def judge_metadata() -> tuple[str, str]:
    """Best-effort diagnostic metadata; never part of generation routing."""
    diagnostics = os.environ.get("USER_PERSONAS_BRIDGE_DIAGNOSTICS_URL")
    if diagnostics:
        try:
            req = urllib.request.Request(diagnostics.rstrip("/") + "/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                health = json.loads(resp.read())
            return health.get("model", "unknown"), health.get("gguf", "unknown")
        except Exception:
            pass
    return os.environ.get("USER_PERSONAS_MODEL", "unknown"), "unknown"
