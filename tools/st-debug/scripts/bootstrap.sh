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
# See run.sh / CLAUDE.md: st-debug owns its own clone of sillytavern-fork
# under tools/st-debug/sillytavern-fork/. NEVER point this at the root
# /Users/mdot/sillytavern-fork — that breaks instance isolation.
ST_SRC="${ST_SRC:-$HERE/sillytavern-fork}"
ST_PORT="${ST_PORT:-8002}"
BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8001}"

if [[ ! -d "$ST_SRC" ]]; then
    echo "[bootstrap] FAIL: ST source clone not found at $ST_SRC"
    echo "[bootstrap] Run: git clone /Users/mdot/sillytavern-fork \"$ST_SRC\""
    echo "[bootstrap]      cd \"$ST_SRC\" && npm install"
    echo "[bootstrap] See $HERE/CLAUDE.md for full setup."
    exit 1
fi

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
    SERVER_PORT="$ST_PORT" \
    USER_PERSONAS_ST_URL="http://127.0.0.1:$ST_PORT" \
    ST_URL="http://127.0.0.1:$ST_PORT" \
    USER_PERSONAS_BRIDGE_URL="$BRIDGE_URL" \
    BRIDGE_URL="$BRIDGE_URL" \
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
    # gemma-4-a4b supports 128k. The default 4095 is too small once
    # 11 toolcards worth of schemas + persona description are stuffed
    # into the prompt — ST raises "Mandatory prompts exceed the
    # context size" and silently truncates the user message out of
    # the request, which manifests as the model replying "no previous
    # content provided" instead of engaging with the prompt.
    "openai_max_context": 32768,
    "max_context_unlocked": True,
    "stream_openai": True,
    "function_calling": True,
    # Reasoning channel: ST sends reasoning_effort to the bridge,
    # bridge enables the model's thinking-channel and emits
    # delta.reasoning_content chunks back. ST renders the trace in
    # a collapsible "thoughts" section on the assistant turn. With
    # "auto", ST passes the param through unchanged; with high/
    # medium/low the user can dial it. Default to "auto" so the
    # operator decides per-conversation; the bootstrap-provided
    # default just makes sure show_thoughts is enabled so the UI
    # actually renders whatever comes back.
    "show_thoughts": True,
    "reasoning_effort": "auto",
}
# Merge into existing oai_settings if present, else create.
existing = s.get("oai_settings", {})
existing.update(oai)
s["oai_settings"] = existing
# Also surface at top-level for any code path that reads there.
for k, v in oai.items():
    s[k] = v
# username / persona: the scringlo scrambler persona should be active
# so the welcome chat's user turns render with the right avatar.
s["username"] = "scringlo scrambler"
s["user_avatar"] = "1779035204660-scringloscrambler.png"
# active_character: dicemother is the default character for st-debug.
# The welcome chat lives under chats/dicemother/ so this must match.
s["active_character"] = "dicemother.png"
# auto_load_chat: ST will restore the last-active character + its most
# recent chat on startup. Without this, the operator sees a blank
# character-selection screen and the suggester has no chat to score
# against — a P-NO-EMPTY-FIRST-PAINT violation.
pu = s.setdefault("power_user", {})
pu["auto_load_chat"] = True
with open(p, "w") as f:
    json.dump(s, f, indent=2)
print(f"  oai_settings.chat_completion_source = {existing['chat_completion_source']}")
print(f"  oai_settings.custom_url             = {existing['custom_url']}")
print(f"  oai_settings.function_calling       = {existing['function_calling']}")
print(f"  firstRun                            = False (welcome popup suppressed)")
print(f"  active_character                    = dicemother.png")
print(f"  power_user.auto_load_chat           = True (P-NO-EMPTY-FIRST-PAINT)")
PY

# UX-T4 P-NO-EMPTY-FIRST-PAINT: seed the welcome chat so a fresh install
# opens ST with an active chat already loaded.  The content-manager does
# not handle "chat" type seeds (no case in getTargetByType), so we copy
# manually here.  The file lives in default/content/chats/dicemother/
# (alongside the existing seed_accusation / seed_cistern / seed_geas files).
# Timestamp in the filename is 2026-05-24T10:00:00 — after all existing
# dicemother session timestamps — so it sorts as the most recent chat
# and ST loads it first on startup when active_character = dicemother.png.
#
# Idempotent: we skip the copy if the file already exists in _data/.
WELCOME_SRC="$ST_SRC/default/content/chats/dicemother/welcome_suggester_demo.jsonl"
WELCOME_DEST_DIR="$DATA_ROOT/default-user/chats/dicemother"
WELCOME_DEST="$WELCOME_DEST_DIR/dicemother - 2026-05-24@10h00m00s000ms.jsonl"
if [[ -f "$WELCOME_SRC" ]]; then
    mkdir -p "$WELCOME_DEST_DIR"
    if [[ ! -f "$WELCOME_DEST" ]]; then
        cp "$WELCOME_SRC" "$WELCOME_DEST"
        echo "[bootstrap] welcome chat seeded: $WELCOME_DEST"
    else
        echo "[bootstrap] welcome chat already present (idempotent skip)"
    fi
else
    echo "[bootstrap] WARN: welcome chat source missing at $WELCOME_SRC"
    echo "[bootstrap]       sync: cd $ST_SRC && git pull"
fi

# Example personas use the upstream SillyTavern content-manager seed:
# they live at $ST_SRC/default/content/<persona>.png and are listed in
# $ST_SRC/default/content/index.json with type="character".
# content-manager.js copies them into <DATA_ROOT>/default-user/characters/
# during the brief ST first-launch above (line ~35).
#
# Toolcards live INSIDE the plugin directory at
# $ST_SRC/plugins/toolcards/cards/<id>.toolcard.json (a regular tracked
# location, no seed indirection). The plugin reads them on every boot
# directly — no copy step needed. Updating a manifest IS the publish.
if [[ ! -d "$ST_SRC/plugins/toolcards/cards" ]]; then
    echo "[bootstrap] WARN: $ST_SRC/plugins/toolcards/cards/ missing — toolcards plugin will boot with 0 cards."
    echo "[bootstrap]       sync: cd $ST_SRC && git pull  (or rsync from /Users/mdot/sillytavern-fork/plugins/toolcards/)"
fi
if [[ ! -f "$ST_SRC/default/content/scringlo_scrambler.png" ]] || \
   [[ ! -f "$ST_SRC/default/content/dicemother.png" ]]; then
    echo "[bootstrap] WARN: example personas missing from $ST_SRC/default/content/."
    echo "[bootstrap]       sync: cd $ST_SRC && git pull"
fi

echo "[bootstrap] done. data root: $DATA_ROOT"
echo "[bootstrap] next: ./scripts/run.sh  →  http://127.0.0.1:$ST_PORT"
