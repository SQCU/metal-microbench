#!/usr/bin/env bash
# Wipe state. Two levels:
#   ./reset.sh          → just wipes chats/ + thumbnails/ (preserves settings)
#   ./reset.sh --fresh  → re-runs bootstrap from scratch (incl. ST seed)

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"

# Always: kill any background ST instance from this workspace.
if [[ -f "$DATA_ROOT/_run.pid" ]]; then
    PID=$(cat "$DATA_ROOT/_run.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[reset] killing ST pid=$PID"
        kill "$PID" 2>/dev/null || true
        sleep 1
    fi
    rm -f "$DATA_ROOT/_run.pid"
fi

if [[ "${1:-}" == "--fresh" ]]; then
    echo "[reset] --fresh: full wipe + re-bootstrap"
    exec "$HERE/scripts/bootstrap.sh" --fresh
fi

echo "[reset] light wipe: chats/ thumbnails/ backups/ extensions/ logs"
rm -rf "$DATA_ROOT/default-user/chats/"*
rm -rf "$DATA_ROOT/default-user/thumbnails/"*
rm -rf "$DATA_ROOT/default-user/backups/"*
rm -rf "$DATA_ROOT/default-user/group chats/"*
rm -f "$DATA_ROOT/_run.log" "$DATA_ROOT/_first_launch.log"
echo "[reset] done. settings.json + characters preserved."
