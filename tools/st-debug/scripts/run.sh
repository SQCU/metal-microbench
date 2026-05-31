#!/usr/bin/env bash
# Launch the dev/debug ST instance against our bridge.
# Foreground by default; pass --bg to run in background and tail the log.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
# ST source — st-debug owns its own clone under tools/st-debug/sillytavern-fork/.
# This used to point at /Users/mdot/sillytavern-fork (root). That sharing
# was the source of the instance-isolation bug (2026-05-19): the plugin
# couldn't tell which ST instance loaded it from filesystem geometry
# (both instances had the same PLUGIN_DIR), so write-throughs landed in
# root's data dir regardless of which instance was running.
#
# Sync workflow: edit at root → `git commit` → `cd <clone> && git pull`
# → restart this instance. The clone's `origin` points at the local root
# checkout, so pull doesn't require network.
ST_SRC="${ST_SRC:-$HERE/sillytavern-fork}"
ST_PORT="${ST_PORT:-8002}"
BRIDGE_URL="${BRIDGE_URL:-http://127.0.0.1:8001}"
BG="${1:-}"

if [[ ! -d "$ST_SRC" ]]; then
    echo "[run] FAIL: ST source clone not found at $ST_SRC"
    echo "[run] Bootstrap with: git clone /Users/mdot/sillytavern-fork \"$ST_SRC\""
    echo "[run] See $HERE/CLAUDE.md for the full sync workflow."
    exit 1
fi

if [[ ! -f "$DATA_ROOT/default-user/settings.json" ]]; then
    echo "[run] no _data/ — run ./scripts/bootstrap.sh first"
    exit 1
fi

# Kill orphan toolcard service.py processes from previous interrupted
# test runs. Without this, the toolcards plugin's per-card service queue
# (plugins/toolcards/index.mjs:338) parks new invocations behind the
# orphan's still-active session — observable as multi-minute hangs in
# test 05 / test 06 with no error events. Found via static analysis of
# the plugin's startSession flow, 2026-05-08.
pkill -f 'uv run --no-project --with.*python service.py' 2>/dev/null || true
pkill -f 'uv run --with .*python service.py' 2>/dev/null || true
sleep 0.5

# Sweep orphan Playwright / MCP browser trees from previous sessions.
# These accumulate when Claude Code terminals exit without reaping the
# @playwright/mcp server tree (audited 2026-05-21: 14 orphans across 8
# days). Without this sweep, every restart adds another 6-process
# chromium tree + leftover MCP servers. The cleanup script is
# conservative — it leaves anything < 5 minutes old alone, so a
# currently-active MCP session isn't disturbed.
"$HERE/scripts/cleanup_playwright.sh" --apply 2>&1 | sed 's/^/[run] /' || true

# Sanity: is the bridge reachable? (warn-only; ST will start regardless)
if ! curl -sS --max-time 2 "$BRIDGE_URL/health" > /dev/null 2>&1; then
    echo "[run] WARN: bridge at $BRIDGE_URL unreachable — chat will 500"
fi

cd "$ST_SRC"
LOG="$DATA_ROOT/_run.log"
export ST_PORT
export SERVER_PORT="$ST_PORT"
export USER_PERSONAS_ST_URL="http://127.0.0.1:$ST_PORT"
export ST_URL="$USER_PERSONAS_ST_URL"
export USER_PERSONAS_BRIDGE_URL="$BRIDGE_URL"
export BRIDGE_URL
echo "[run] starting ST on port $ST_PORT (dataRoot=$DATA_ROOT)"
echo "[run]   UI:   http://127.0.0.1:$ST_PORT"
echo "[run]   bridge: $BRIDGE_URL"
echo "[run]   log:  $LOG"

# --disableCsrf is REQUIRED for autonomous (curl/Playwright/MCP) clients
# of ST's /api/... endpoints. Without it, every POST to a backend
# endpoint requires the CSRF token that the homepage puts in a cookie,
# which means you'd need a browser session before any HTTP call. This
# instance is dev/debug-only, bound to listen=false (loopback), so the
# threat model that CSRF defends against doesn't apply.
ST_FLAGS=( --dataRoot "$DATA_ROOT" --port "$ST_PORT"
           --browserLaunchEnabled false --listen false --disableCsrf )

if [[ "$BG" == "--bg" ]]; then
    nohup node server.js "${ST_FLAGS[@]}" > "$LOG" 2>&1 &
    PID=$!
    echo "[run]   pid:  $PID"
    echo "$PID" > "$DATA_ROOT/_run.pid"
    # Wait briefly for readiness
    for i in $(seq 1 20); do
        if curl -sS --max-time 1 "http://127.0.0.1:$ST_PORT/" > /dev/null 2>&1; then
            echo "[run]   ready after ${i}s (csrf disabled)"
            exit 0
        fi
        sleep 1
    done
    echo "[run]   WARN: didn't reach ready in 20s; check $LOG"
    exit 1
else
    exec node server.js "${ST_FLAGS[@]}"
fi
