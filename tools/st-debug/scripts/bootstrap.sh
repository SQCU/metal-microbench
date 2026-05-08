#!/usr/bin/env bash
# Seed tools/st-debug/_data/ as a clean ST dataRoot pointing at our bridge.
#
# Steps:
#   1. ensure _data/ exists (or recreate if --fresh)
#   2. spin ST briefly to populate defaults (sysprompts, etc.)
#   3. patch _data/default-user/settings.json: chat_completion_source=custom,
#      custom_url=http://127.0.0.1:8001
#   4. drop a minimal default character so the chat UI is usable on first load
#   5. leave ST stopped — caller invokes scripts/run.sh
#
# Idempotent: safe to run multiple times. Skips re-seeding if _data/ already
# has a settings.json.

set -euo pipefail

# Where we live.
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
ST_SRC="${ST_SRC:-/Users/mdot/sillytavern-fork}"
ST_PORT="${ST_PORT:-8002}"
BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8001}"

# --fresh wipes _data/ entirely.
if [[ "${1:-}" == "--fresh" ]]; then
    echo "[bootstrap] --fresh: wiping $DATA_ROOT"
    rm -rf "$DATA_ROOT"
fi

mkdir -p "$DATA_ROOT"

# If settings.json already exists and we're not --fresh, re-patch but skip
# the seed step (faster reruns).
if [[ ! -f "$DATA_ROOT/default-user/settings.json" ]]; then
    echo "[bootstrap] seeding defaults via ST first-launch (this takes ~5s)..."
    cd "$ST_SRC"
    node server.js --dataRoot "$DATA_ROOT" --port "$ST_PORT" > "$DATA_ROOT/_first_launch.log" 2>&1 &
    SEED_PID=$!
    # Wait for settings.json to appear (signal that defaults are written).
    for i in $(seq 1 30); do
        if [[ -f "$DATA_ROOT/default-user/settings.json" ]]; then
            echo "[bootstrap]   defaults seeded after ${i}s"
            break
        fi
        sleep 1
    done
    kill "$SEED_PID" 2>/dev/null || true
    wait "$SEED_PID" 2>/dev/null || true
    cd "$HERE"
fi

# Patch settings.json: route the chat-completions traffic to our bridge.
# These are the exact field names SillyTavern reads (see public/scripts/openai.js).
SETTINGS="$DATA_ROOT/default-user/settings.json"
if [[ ! -f "$SETTINGS" ]]; then
    echo "[bootstrap] FAIL: $SETTINGS still doesn't exist after seed"
    exit 1
fi

echo "[bootstrap] patching $SETTINGS to talk to bridge at $BRIDGE_URL"
python3 <<PY
import json, sys
p = "$SETTINGS"
with open(p) as f:
    s = json.load(f)
s["main_api"] = "openai"
# Suppress the firstRun "Welcome to SillyTavern!" popup that blocks
# all UI interaction on a fresh dataRoot. We're a non-interactive
# instance; the welcome panel is dead weight.
s["firstRun"] = False
# OpenAI / chat-completion section
s.setdefault("openai_setting_names", [])
oai = {
    "chat_completion_source": "custom",
    "custom_url": "$BRIDGE_URL",
    "custom_model": "gemma-4-a4b",
    "openai_model": "gemma-4-a4b",
    "temperature_openai": 1.0,
    "openai_max_tokens": 2048,
    "stream_openai": True,
    "function_calling": True,
}
# Merge into existing oai_settings if present, else create.
existing = s.get("oai_settings", {})
existing.update(oai)
s["oai_settings"] = existing
# Also surface at top-level for any code path that reads there.
for k, v in oai.items():
    s[k] = v
# username left as default ("User") — chat UI uses it for the user-side
# avatar; doesn't affect bridge traffic.
with open(p, "w") as f:
    json.dump(s, f, indent=2)
print(f"  oai_settings.chat_completion_source = {existing['chat_completion_source']}")
print(f"  oai_settings.custom_url             = {existing['custom_url']}")
print(f"  oai_settings.function_calling       = {existing['function_calling']}")
print(f"  firstRun                            = False (welcome popup suppressed)")
PY

# Symlink toolcards from main ST install. Plugin reads them from
# $DATA_ROOT/toolcards/{cards,installed}/. Keeping them as symlinks
# means the debug instance picks up changes the user makes to their
# main install's cards (e.g., editing a service.py to reproduce a
# bug). If isolation matters in a future scenario, switch to `cp -r`.
TOOLCARDS_SRC="$ST_SRC/data/toolcards"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
mkdir -p "$TOOLCARDS_DST"
if [[ -d "$TOOLCARDS_SRC/cards" ]]; then
    rm -rf "$TOOLCARDS_DST/cards"
    ln -sf "$TOOLCARDS_SRC/cards" "$TOOLCARDS_DST/cards"
    echo "[bootstrap] toolcards/cards → $TOOLCARDS_SRC/cards (symlinked)"
fi
if [[ -d "$TOOLCARDS_SRC/installed" ]]; then
    rm -rf "$TOOLCARDS_DST/installed"
    ln -sf "$TOOLCARDS_SRC/installed" "$TOOLCARDS_DST/installed"
    echo "[bootstrap] toolcards/installed → $TOOLCARDS_SRC/installed (symlinked)"
fi
if [[ -d "$TOOLCARDS_SRC/sources" ]]; then
    # Sources directory is referenced by some toolcards too.
    rm -rf "$TOOLCARDS_DST/sources"
    ln -sf "$TOOLCARDS_SRC/sources" "$TOOLCARDS_DST/sources"
fi

# Drop a minimal default character so the chat UI is usable.
CHARS_DIR="$DATA_ROOT/default-user/characters"
mkdir -p "$CHARS_DIR"
cat > "$CHARS_DIR/Debug.json" <<'JSON'
{
    "name": "Debug",
    "description": "A minimal character for autonomous testing. Responds plainly.",
    "first_mes": "ready.",
    "personality": "concise, factual",
    "scenario": "automated integration test",
    "mes_example": "",
    "creator_notes": "Generated by tools/st-debug/scripts/bootstrap.sh — do not edit by hand.",
    "talkativeness": "0.5",
    "tags": ["debug", "autotest"],
    "spec": "chara_card_v3",
    "spec_version": "3.0",
    "data": {
        "name": "Debug",
        "description": "A minimal character for autonomous testing. Responds plainly.",
        "first_mes": "ready.",
        "personality": "concise, factual",
        "scenario": "automated integration test",
        "mes_example": ""
    }
}
JSON

echo "[bootstrap] done. data root: $DATA_ROOT"
echo "[bootstrap] next: ./scripts/run.sh  →  http://127.0.0.1:$ST_PORT"
