#!/usr/bin/env bash
# Real cleanup for orphaned Playwright / MCP browser processes.
#
# What this kills (and what it explicitly does NOT kill):
#
# 1. MCP-spawned chromium + helpers. The @playwright/mcp server (which
#    Claude Code uses for browser_navigate / browser_evaluate / etc.)
#    spawns a Chrome+helpers tree with --user-data-dir under
#    ~/Library/Caches/ms-playwright/mcp-chrome-* . When a Claude Code
#    session ends, the MCP node process may exit cleanly OR it may
#    leave the chromium tree behind. Lingering across days is the
#    failure mode we keep observing (audited 2026-05-21: 6 MCP server
#    pairs from days-old sessions, plus chromium trees with etime > 24h).
#
# 2. Stale @playwright/mcp node processes themselves whose parent TTY
#    no longer has an active shell (the Claude Code session that
#    spawned them exited but the npm/node didn't get reaped).
#
# 3. Stray playwright-test chromium-headless-shell processes whose
#    --user-data-dir matches /var/folders/.../playwright_chromiumdev_profile-*
#    AND whose etime > MAX_TEST_BROWSER_AGE_MIN (default 60m). Healthy
#    test runs always tear these down at end-of-test; anything older
#    than an hour is residue from a killed/crashed test run.
#
# What it does NOT kill:
#
# - Chromium processes whose etime is under MIN_AGE_MIN (default 5m) —
#   they're likely actively in use by a running session.
# - The user's own Google Chrome browser (no --user-data-dir matching
#   playwright/MCP patterns).
# - The st-debug ST instance, the bridge, or any other non-Playwright
#   process.
#
# Usage:
#
#   ./scripts/cleanup_playwright.sh             # dry run, report only
#   ./scripts/cleanup_playwright.sh --apply     # actually kill
#   ./scripts/cleanup_playwright.sh --apply -v  # verbose
#
# Environment overrides:
#
#   MIN_AGE_MIN=N             # don't touch procs younger than N minutes (default: 5)
#   MAX_TEST_BROWSER_AGE_MIN  # test-spawned chromium older than this is residue (default: 60)

set -euo pipefail

DRY_RUN=true
VERBOSE=false
MIN_AGE_MIN="${MIN_AGE_MIN:-5}"
MAX_TEST_BROWSER_AGE_MIN="${MAX_TEST_BROWSER_AGE_MIN:-60}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --apply) DRY_RUN=false ;;
        -v|--verbose) VERBOSE=true ;;
        -h|--help)
            sed -n '2,/^set/p' "$0" | sed -n '/^#/p' | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "[cleanup_playwright] unknown arg: $1" >&2; exit 2 ;;
    esac
    shift
done

log()  { echo "[cleanup_playwright] $*" >&2; }
vlog() { $VERBOSE && echo "[cleanup_playwright] $*" >&2 || true; }

# Convert macOS etime (DD-HH:MM:SS, HH:MM:SS, MM:SS) to seconds.
# Each component is forced to base-10 (10#…) so leading zeros like "09"
# don't get misparsed as octal by bash arithmetic.
etime_to_sec() {
    local s="$1"
    local days=0 hours=0 mins=0 secs=0
    if [[ "$s" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        days=${BASH_REMATCH[1]}; hours=${BASH_REMATCH[2]}; mins=${BASH_REMATCH[3]}; secs=${BASH_REMATCH[4]}
    elif [[ "$s" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
        hours=${BASH_REMATCH[1]}; mins=${BASH_REMATCH[2]}; secs=${BASH_REMATCH[3]}
    elif [[ "$s" =~ ^([0-9]+):([0-9]+)$ ]]; then
        mins=${BASH_REMATCH[1]}; secs=${BASH_REMATCH[2]}
    else
        echo "0"; return
    fi
    echo $(( 10#$days*86400 + 10#$hours*3600 + 10#$mins*60 + 10#$secs ))
}

MIN_AGE_SEC=$(( MIN_AGE_MIN * 60 ))
MAX_TEST_BROWSER_AGE_SEC=$(( MAX_TEST_BROWSER_AGE_MIN * 60 ))

# Pattern collection. We grep against `ps -e -o pid,etime,command`.
# Each predicate function takes (pid, etime_sec, command) and emits
# the pid if it should be killed, along with a reason tag.
collect_targets() {
    local TARGETS=()

    # Read all processes once.
    # macOS ps output: "  PID    ELAPSED COMMAND..."
    while IFS= read -r line; do
        # Trim leading whitespace, split off pid + etime + rest
        local trimmed pid etime rest
        trimmed=$(echo "$line" | sed -E 's/^[[:space:]]+//')
        pid=$(echo "$trimmed" | awk '{print $1}')
        etime=$(echo "$trimmed" | awk '{print $2}')
        rest=$(echo "$trimmed" | cut -d' ' -f3-)
        [[ "$pid" == "PID" ]] && continue   # header
        [[ -z "$pid" || -z "$etime" ]] && continue
        local age_sec
        age_sec=$(etime_to_sec "$etime")

        # Skip very-recently-started processes (probably current session).
        if (( age_sec < MIN_AGE_SEC )); then
            continue
        fi

        # Predicate 1: MCP-spawned chromium tree.
        # Matches user-data-dir under ~/Library/Caches/ms-playwright/mcp-chrome-*
        if [[ "$rest" == *"--user-data-dir=$HOME/Library/Caches/ms-playwright/mcp-chrome-"* ]]; then
            TARGETS+=("$pid|$age_sec|mcp-chromium|$rest")
            continue
        fi

        # Predicate 2: stale @playwright/mcp node process or its npm wrapper.
        # These are long-running by design — the current Claude Code
        # session's own MCP server can survive hours/days of idle quiet
        # between browser tool calls. So the ONLY kill criterion is
        # "parent is dead or reparented to launchd (pid 1)." Age alone
        # is not enough — a multi-day live session is legitimate.
        # Active sessions can additionally register their PID in
        # /tmp/active_mcp_pids (newline-separated) for unconditional
        # protection. Bumped age gate from 1h → 4h to avoid touching
        # very-recently-started MCPs even if reparenting somehow
        # happened immediately (2026-05-22 fix).
        if [[ "$rest" == *"playwright-mcp"* ]] || [[ "$rest" == *"@playwright/mcp@latest"* ]]; then
            # Active-PID exclusion list — if the user / a session-start
            # hook writes the current MCP PID here, we never touch it.
            if [[ -f /tmp/active_mcp_pids ]] && grep -q "^${pid}$" /tmp/active_mcp_pids 2>/dev/null; then
                continue
            fi
            if (( age_sec > 14400 )); then
                # Check parent process. The ONLY orphan signal we trust:
                # ppid is gone, or ppid is launchd (pid 1). A live parent
                # — even after many hours — means the session that spawned
                # this MCP is still around.
                local ppid
                ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ' || echo "")
                if [[ -z "$ppid" || "$ppid" == "1" ]] || ! ps -p "$ppid" > /dev/null 2>&1; then
                    TARGETS+=("$pid|$age_sec|mcp-server-orphan|$rest")
                    continue
                fi
                # Parent alive — DO NOT kill, regardless of age.
                # Earlier versions of this script had a "12h+ stale with
                # parent alive" fallback that misfired on long-running
                # sessions. Removed 2026-05-22 after it reaped this
                # session's own MCP. If you want to kill long-lived MCPs
                # whose parent is alive, do it manually with `kill <pid>`.
            fi
            continue
        fi

        # Predicate 3: orphan playwright-test chromium-headless-shell.
        # These should always be reaped by playwright at end-of-test.
        # If we see one that's old, the test died without cleanup.
        if [[ "$rest" == *"chromium_headless_shell"* ]] || [[ "$rest" == *"chrome-headless-shell"* ]]; then
            if [[ "$rest" == *"playwright_chromiumdev_profile-"* ]] || [[ "$rest" == *"--user-data-dir=/var/folders/"* ]]; then
                if (( age_sec > MAX_TEST_BROWSER_AGE_SEC )); then
                    TARGETS+=("$pid|$age_sec|orphan-test-browser|$rest")
                    continue
                fi
            fi
        fi
    done < <(ps -e -o pid,etime,command 2>/dev/null)

    # printf '%s\n' on an empty array still emits a single empty line on
    # bash 3.2 (macOS default). Guarding the printf avoids the phantom
    # "1 target process(es)" with empty fields the post-cleanup dry-run
    # reported on 2026-05-21.
    if (( ${#TARGETS[@]} > 0 )); then
        printf '%s\n' "${TARGETS[@]}"
    fi
}

# Build target list. The `| grep -v '^$'` is belt-and-suspenders against
# any other path that might emit a blank line — the empty-line guard in
# collect_targets is the load-bearing fix.
mapfile -t TARGETS < <(collect_targets | grep -v '^$' || true)

if (( ${#TARGETS[@]} == 0 )); then
    log "no orphan playwright / MCP processes found (MIN_AGE_MIN=$MIN_AGE_MIN, MAX_TEST_BROWSER_AGE_MIN=$MAX_TEST_BROWSER_AGE_MIN)"
    exit 0
fi

# Report.
if $DRY_RUN; then
    log "found ${#TARGETS[@]} target process(es) [DRY RUN — re-run with --apply to kill]"
else
    log "found ${#TARGETS[@]} target process(es)"
fi
for entry in "${TARGETS[@]}"; do
    IFS='|' read -r tpid tage_sec ttag tcmd <<< "$entry"
    tage_min=$(( tage_sec / 60 ))
    # Truncate command for display (printf %.100s, no echo + cut to avoid
    # subshell parsing quirks under set -u).
    printf '  pid=%s age=%sm tag=%s cmd=%.100s…\n' "$tpid" "$tage_min" "$ttag" "$tcmd"
done

$DRY_RUN && exit 0

# Apply. SIGTERM first, then SIGKILL after grace period.
log "sending SIGTERM..."
PIDS=()
for entry in "${TARGETS[@]}"; do
    IFS='|' read -r pid _ _ _ <<< "$entry"
    PIDS+=("$pid")
    kill -TERM "$pid" 2>/dev/null || true
done

sleep 2

# Check survivors + escalate.
SURVIVORS=()
for pid in "${PIDS[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
        SURVIVORS+=("$pid")
    fi
done

if (( ${#SURVIVORS[@]} > 0 )); then
    log "escalating to SIGKILL for ${#SURVIVORS[@]} survivor(s): ${SURVIVORS[*]}"
    for pid in "${SURVIVORS[@]}"; do
        kill -KILL "$pid" 2>/dev/null || true
    done
    sleep 1
fi

# Final verification.
FINAL_SURVIVORS=()
for pid in "${PIDS[@]}"; do
    if ps -p "$pid" > /dev/null 2>&1; then
        FINAL_SURVIVORS+=("$pid")
    fi
done

if (( ${#FINAL_SURVIVORS[@]} > 0 )); then
    log "WARNING: ${#FINAL_SURVIVORS[@]} process(es) survived SIGKILL: ${FINAL_SURVIVORS[*]}"
    log "  (likely zombie state or kernel hold; manual inspection needed)"
    exit 1
fi

log "all ${#TARGETS[@]} target process(es) terminated cleanly"

# Optional: clean up the ms-playwright cache profile dirs left behind
# by MCP-spawned chromium. These can grow to 100s of MB per profile.
if [[ -d "$HOME/Library/Caches/ms-playwright" ]]; then
    # Only delete mcp-chrome-* profile dirs older than 1 day. Don't touch
    # the playwright browser binaries (chromium-* dirs without "mcp" prefix).
    STALE_PROFILES=$(find "$HOME/Library/Caches/ms-playwright" -maxdepth 1 -type d -name "mcp-chrome-*" -mtime +1 2>/dev/null || true)
    if [[ -n "$STALE_PROFILES" ]]; then
        local_n=$(echo "$STALE_PROFILES" | wc -l | tr -d ' ')
        log "removing $local_n stale mcp-chrome-* profile dir(s) (older than 1 day)"
        echo "$STALE_PROFILES" | xargs rm -rf
    fi
fi

exit 0
