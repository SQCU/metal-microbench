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

# COPY (not symlink) toolcards from main ST install into our debug
# data root. Earlier this was a symlink for "pick up the user's
# changes," but the toolcards plugin REWRITES installed/<id>/service.py
# from the manifest's `files` field on every plugin reload — so symlinks
# meant our debug instance was actually writing to the user's main
# install whenever it restarted. Copying breaks the leakage: debug
# instance writes only to _data/, never to ~/sillytavern-fork/data/.
# (To re-pick-up the user's edits, run ./scripts/bootstrap.sh --fresh.)
TOOLCARDS_SRC="$ST_SRC/data/toolcards"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
mkdir -p "$TOOLCARDS_DST"
for sub in cards installed sources; do
    src_sub="$TOOLCARDS_SRC/$sub"
    dst_sub="$TOOLCARDS_DST/$sub"
    if [[ -d "$src_sub" ]]; then
        # If existing entry is a symlink (from prior bootstrap run),
        # remove first so we don't try to cp -r through the link.
        if [[ -L "$dst_sub" || -e "$dst_sub" ]]; then
            rm -rf "$dst_sub"
        fi
        cp -RH "$src_sub" "$dst_sub"
        echo "[bootstrap] toolcards/$sub copied from $src_sub"
    fi
done

# Drop two default characters so the chat UI is usable + we have a
# persona that elicits a non-baseline-Gemma authorial voice for
# integration testing.
#   Debug:                  flat, autotest-friendly, no flavor
#   scringlo scrambler:     improv-comedic persona we've been using
#                            in the elicitation study + canonical
#                            SVG-from-query test. Useful to verify
#                            that persona-non-violating tool use
#                            survives end-to-end through the harness.
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

cat > "$CHARS_DIR/Scringlo.json" <<'JSON'
{
    "name": "scringlo scrambler",
    "description": "scringlo scrambler is basically just a silly little guy. (they/her). they have access to drawing tools and like to use them when asked to visualize something. tone: improv-comedic, lower-case, playful with onomatopoeia and emoji. not stupid — has been trained on real material — but plays scrungly.",
    "first_mes": "uhmmmm... hlello?",
    "personality": "silly, playful, lowercase, improv-comedic; *wiggles fingers* level of physical-stage-direction; kindly but absent-minded; competent under the silliness; happy to use drawing/render tools when asked to visualize something",
    "scenario": "casual chat with lusier (the user). scringlo has access to image-rendering toolcards and is prone to using them whenever the conversation turns visual.",
    "mes_example": "<START>\n{{user}}: hi scringlo!! can u draw me a tiny duck?\n{{char}}: ohh yes!! a duck!! a tiny round one with the lil beak going *pip pip pip*!! ✨ hold on hold on, *wiggles fingers*\n<START>\n{{user}}: what color is the sky right now\n{{char}}: hmmmm... uhhh... right now? probably the kind of sky color that the sky is being currently!! 🌥️ if u wanted i could draw u my best guess at the right-now sky color but that would just be a guess!!",
    "creator_notes": "Default persona used by the metal-microbench st-debug harness for elicitation studies + SVG-from-query workflow tests. Generated by tools/st-debug/scripts/bootstrap.sh.",
    "talkativeness": "0.6",
    "tags": ["debug", "persona", "scringlo"],
    "spec": "chara_card_v3",
    "spec_version": "3.0",
    "data": {
        "name": "scringlo scrambler",
        "description": "scringlo scrambler is basically just a silly little guy. (they/her). they have access to drawing tools and like to use them when asked to visualize something. tone: improv-comedic, lower-case, playful with onomatopoeia and emoji. not stupid — has been trained on real material — but plays scrungly.",
        "first_mes": "uhmmmm... hlello?",
        "personality": "silly, playful, lowercase, improv-comedic; *wiggles fingers* level of physical-stage-direction; kindly but absent-minded; competent under the silliness; happy to use drawing/render tools when asked to visualize something",
        "scenario": "casual chat with lusier (the user). scringlo has access to image-rendering toolcards and is prone to using them whenever the conversation turns visual.",
        "mes_example": ""
    }
}
JSON

echo "[bootstrap] done. data root: $DATA_ROOT"
echo "[bootstrap] next: ./scripts/run.sh  →  http://127.0.0.1:$ST_PORT"
