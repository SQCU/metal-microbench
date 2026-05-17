#!/usr/bin/env bash
# yeet_to_runpod.sh — provision a 7×RTX5090 (or other Blackwell) workstation
# on RunPod, build the inference worker image, ship the GGUFs, start one
# llama-server per GPU, and print the URLs the search driver should target.
#
# Two modes:
#   1. local-only mode (this script's first ~80% of work): just builds the
#      container image and runs N workers on a local box that already has
#      GPUs. Useful for testing on an in-house workstation before going
#      to the cloud.
#
#   2. RunPod mode: provision a pod via runpod CLI, scp the GGUFs in,
#      docker-compose up the workers, return the public URLs.
#
# Prerequisites for RunPod mode:
#   - `runpodctl` installed (https://docs.runpod.io/cli/install-runpodctl)
#   - RunPod API key in $RUNPOD_API_KEY
#   - GGUFs already materialized locally via 03_materialize_grid.sh
#
# Usage:
#   tools/quant_search/cloud/yeet_to_runpod.sh local 4 /path/to/ggufs
#     → 4 workers on local GPUs, ports 8080..8083
#
#   tools/quant_search/cloud/yeet_to_runpod.sh runpod 7 \
#       /Users/mdot/models/gemma-4-a4b-quant-search rtx-5090
#     → 7×RTX5090 spot instance, GGUFs uploaded, workers started
#
# Output: writes a workers.json file with [{gpu, gguf, url}, ...].
#
# TODO(cloud-redesign): the cloud-distribution path needs redesign.
# It was meaningful only with the deleted search.py driver, which had
# multi-bridge fan-out and consumed workers.json via --workers. The
# current canonical orchestrator is tools/quant_search/scripts/08_long_run.py,
# which restarts a SINGLE local bridge sequentially per config and has
# no notion of a remote worker fleet. Either:
#   (a) extend 08_long_run.py to accept --workers and fan configs out
#       across the fleet, or
#   (b) refactor this script to materialize per-worker GGUF subsets and
#       launch one 08_long_run.py per remote pod against its local bridge.
# Until that lands, the workers.json this script writes is unconsumed —
# the script is preserved as the only artifact of the cloud-distribution
# plan, but the invocation hint at the bottom is commented out.

set -euo pipefail

MODE="${1:-local}"
N_WORKERS="${2:-4}"
GGUF_DIR="${3:-/Users/mdot/models/gemma-4-a4b-quant-search}"
GPU_TYPE="${4:-rtx-5090}"      # only used in runpod mode
IMAGE_TAG="metal-microbench/quant-worker:latest"
WORKERS_JSON="${WORKERS_JSON:-tools/quant_search/cloud/workers.json}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DOCKERFILE="$REPO_ROOT/tools/quant_search/cloud/Dockerfile"

# Find available GGUFs and pick N of them — one per worker.
mapfile -t GGUFS < <(ls "$GGUF_DIR"/*.gguf 2>/dev/null | head -n "$N_WORKERS")
if [[ ${#GGUFS[@]} -lt "$N_WORKERS" ]]; then
    echo "[yeet] FAIL — only ${#GGUFS[@]} GGUFs in $GGUF_DIR, need $N_WORKERS"
    echo "[yeet]   run 03_materialize_grid.sh to produce the search grid"
    exit 1
fi
echo "[yeet] GGUFs to deploy: ${#GGUFS[@]}"
for g in "${GGUFS[@]}"; do echo "  - $g ($(du -h "$g" | awk '{print $1}'))"; done

# ── Function definitions (bash needs these BEFORE the dispatch case) ──

deploy_local() {
    echo
    echo "[yeet] deploying ${#GGUFS[@]} workers locally..."
    # Tear down any previous workers from a prior run.
    for i in $(seq 0 $((${#GGUFS[@]} - 1))); do
        docker rm -f "quant-worker-$i" 2>/dev/null || true
    done

    # Spin up new ones.
    URLS=()
    for i in "${!GGUFS[@]}"; do
        GGUF="${GGUFS[$i]}"
        PORT=$((8080 + i))
        echo "[yeet]   gpu=$i port=$PORT gguf=$(basename "$GGUF")"
        docker run -d \
            --gpus "\"device=$i\"" \
            -p "$PORT:8080" \
            -v "$GGUF_DIR:/models:ro" \
            -e "GGUF=/models/$(basename "$GGUF")" \
            --name "quant-worker-$i" \
            "$IMAGE_TAG"
        URLS+=("http://127.0.0.1:$PORT")
    done

    # Wait for all workers to come up.
    echo "[yeet] waiting for workers to be ready (timeout 5min)..."
    for i in "${!URLS[@]}"; do
        URL="${URLS[$i]}"
        until curl -sf "$URL/health" > /dev/null 2>&1; do
            sleep 5
            if ! docker ps -q --filter "name=quant-worker-$i" | grep -q .; then
                echo "[yeet]   ! quant-worker-$i died, see: docker logs quant-worker-$i"
                exit 1
            fi
        done
        echo "[yeet]   ✓ $URL ready"
    done

    write_workers_json "${GGUFS[@]}" "${URLS[@]}"
}

# ── Deploy on RunPod ───────────────────────────────────────────────────

deploy_runpod() {
    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        echo "[yeet] FAIL — set RUNPOD_API_KEY"; exit 1
    fi
    if ! command -v runpodctl &>/dev/null; then
        echo "[yeet] FAIL — runpodctl not in PATH"; exit 1
    fi

    echo
    echo "[yeet] launching RunPod $GPU_TYPE pod with $N_WORKERS GPUs..."
    # NOTE: runpodctl pod create flags are evolving; this is a sketch.
    # Adjust to current runpodctl syntax. The intent: spot instance,
    # docker-runtime, sufficient disk for the GGUFs (10-50GB each).
    POD_ID=$(runpodctl pod create \
        --gpuType "$GPU_TYPE" --gpuCount "$N_WORKERS" \
        --imageName "$IMAGE_TAG" \
        --containerDiskInGb 200 \
        --bidPerGpu 0.50 \
        --secureCloud false \
        --tag "metal-microbench-quant-search" \
        --output json | jq -r '.id')
    echo "[yeet]   POD_ID=$POD_ID"

    # Wait for pod to be running.
    until [[ "$(runpodctl pod get "$POD_ID" --output json | jq -r '.status')" == "RUNNING" ]]; do
        sleep 10
        echo "[yeet]   waiting for pod..."
    done

    # rsync GGUFs.
    POD_HOST=$(runpodctl pod get "$POD_ID" --output json | jq -r '.publicHost')
    POD_PORT=$(runpodctl pod get "$POD_ID" --output json | jq -r '.publicPort')
    echo "[yeet]   rsync'ing GGUFs to $POD_HOST..."
    rsync -avz -e "ssh -p $POD_PORT" "$GGUF_DIR/" "root@$POD_HOST:/workspace/ggufs/"

    # Start workers via SSH.
    echo "[yeet]   starting $N_WORKERS workers..."
    URLS=()
    for i in "${!GGUFS[@]}"; do
        GGUF="${GGUFS[$i]}"
        PORT=$((8080 + i))
        ssh -p "$POD_PORT" "root@$POD_HOST" \
            "docker run -d --gpus '\"device=$i\"' \
                -p $PORT:8080 \
                -v /workspace/ggufs:/models:ro \
                -e GGUF=/models/$(basename "$GGUF") \
                --name quant-worker-$i \
                $IMAGE_TAG"
        URLS+=("http://$POD_HOST:$PORT")
    done

    # Wait for them.
    for URL in "${URLS[@]}"; do
        until curl -sf "$URL/health" > /dev/null 2>&1; do sleep 5; done
        echo "[yeet]   ✓ $URL ready"
    done

    write_workers_json "${GGUFS[@]}" "${URLS[@]}"
    echo "[yeet]   pod_id=$POD_ID  (terminate via: runpodctl pod stop $POD_ID)"
}

# ── Write workers.json ─────────────────────────────────────────────────

write_workers_json() {
    local n=$(( $# / 2 ))
    local ggufs=("${@:1:$n}")
    local urls=("${@:$((n+1))}")

    {
        echo "["
        for i in $(seq 0 $((n - 1))); do
            comma=$([ "$i" -lt $((n-1)) ] && echo "," || echo "")
            cat <<EOF
  {"gpu": $i, "gguf": "${ggufs[$i]}", "url": "${urls[$i]}"}$comma
EOF
        done
        echo "]"
    } > "$WORKERS_JSON"
    echo
    echo "[yeet] OK — workers.json written to $WORKERS_JSON"
    echo
    cat "$WORKERS_JSON"
    echo
    # TODO(cloud-redesign): no current orchestrator consumes workers.json.
    # See top-of-file note. Re-enable once 08_long_run.py (or a successor)
    # gains a --workers flag.
    # echo "Run search against this fleet:"
    # echo "  python3 tools/quant_search/search.py --workers $WORKERS_JSON ..."
}

# ── Build + dispatch ───────────────────────────────────────────────────

echo
echo "[yeet] building $IMAGE_TAG ..."
docker build -t "$IMAGE_TAG" -f "$DOCKERFILE" "$REPO_ROOT"
echo "[yeet] image built"

case "$MODE" in
    local)  deploy_local ;;
    runpod) deploy_runpod ;;
    *) echo "unknown mode: $MODE (expected: local | runpod)"; exit 1 ;;
esac
