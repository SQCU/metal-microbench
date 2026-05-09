#!/usr/bin/env bash
# Launch the dev/debug ST instance against our bridge.
# Foreground by default; pass --bg to run in background and tail the log.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
ST_SRC="${ST_SRC:-/Users/mdot/sillytavern-fork}"
ST_PORT="${ST_PORT:-8002}"
BG="${1:-}"

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

# Sanity: is the bridge reachable? (warn-only; ST will start regardless)
if ! curl -sS --max-time 2 http://127.0.0.1:8001/health > /dev/null 2>&1; then
    echo "[run] WARN: bridge at http://127.0.0.1:8001 unreachable — chat will 500"
fi

cd "$ST_SRC"
LOG="$DATA_ROOT/_run.log"
echo "[run] starting ST on port $ST_PORT (dataRoot=$DATA_ROOT)"
echo "[run]   UI:   http://127.0.0.1:$ST_PORT"
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
