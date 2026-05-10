#!/usr/bin/env bash
# Generate the reference image set for SVG MSE scoring.
#
# Architecture:
#   - The bridge serves /v1/chat/completions on $QUANT_BRIDGE_URL (8001).
#   - The toolcards runner (scripts/04_toolcards_runner.mjs) imports the
#     SillyTavern plugin and serves toolcard APIs on $TOOLCARDS_URL (8002).
#   - We POST start_invoke and poll /sessions for the result. No CLI shim.
#
# Idempotent: skips prompts whose ref PNG already exists.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/Users/mdot/metal-microbench}"
BRIDGE_URL="${QUANT_BRIDGE_URL:-http://127.0.0.1:8001}"
TOOLCARDS_URL="${TOOLCARDS_URL:-http://127.0.0.1:8002}"
REFS_DIR="${REFS_DIR:-$REPO_ROOT/test_data/svg_quant_refs}"

declare -a PROMPTS=(
    "a smiley face"
    "a red circle on white background"
    "three concentric squares"
    "a simple house with a triangular roof"
    "a yellow sun with rays"
)

mkdir -p "$REFS_DIR"

# Sanity-check both services.
if ! curl -sf "$BRIDGE_URL/health" > /dev/null; then
    echo "[05] FAIL — bridge at $BRIDGE_URL not responding"
    echo "[05]   start it via: server/.venv/bin/python server/serve.py"
    exit 1
fi
if ! curl -sf "$TOOLCARDS_URL/health" > /dev/null; then
    echo "[05] FAIL — toolcards runner at $TOOLCARDS_URL not responding"
    echo "[05]   start it via: node tools/quant_search/scripts/04_toolcards_runner.mjs"
    exit 1
fi
MODEL=$(curl -s "$BRIDGE_URL/health" | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])")
echo "[05] bridge live: model=$MODEL"
echo "[05] toolcards runner live"
echo "[05] generating refs to $REFS_DIR using $MODEL as ground truth"

# A small Python helper does the start_invoke + poll-for-result dance
# inline. Could be its own .py but it's simple enough to embed.
ref_one() {
    local prompt="$1"
    local out_png="$2"
    local out_json="$3"
    python3 - "$prompt" "$out_png" "$out_json" "$BRIDGE_URL" "$TOOLCARDS_URL" <<'PYEOF'
import base64, json, sys, time, urllib.request, uuid
prompt, out_png, out_json, bridge_url, toolcards_url = sys.argv[1:6]

chat_id = f"quant_search_ref_{uuid.uuid4().hex[:12]}"
body = {
    "args": {"query": prompt, "max_iters": 3, "width": 512, "height": 512},
    "profile": {
        "chat_completion_source": "custom",
        "custom_url": bridge_url,
        "openai_model": "gemma-4-a4b",
        "temperature": 0.7,
    },
    "chat_id": chat_id,
}
req = urllib.request.Request(
    f"{toolcards_url}/api/plugins/toolcards/start_invoke/query-to-svg/generate",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=10) as r:
    sid = json.loads(r.read())["session_id"]
print(f"  session={sid}", flush=True)

deadline = time.time() + 600
while time.time() < deadline:
    poll_req = urllib.request.Request(
        f"{toolcards_url}/api/plugins/toolcards/sessions?chat_id={chat_id}",
    )
    with urllib.request.urlopen(poll_req, timeout=10) as r:
        resp = json.loads(r.read())
    for rec in resp.get("results", []):
        if rec.get("session_id") == sid:
            if not rec.get("ok"):
                print(f"  ! tool failed: {rec.get('error')}", flush=True)
                sys.exit(1)
            result = rec["result"]
            png_url = result.get("rendered_png_url")
            if not png_url:
                for part in result.get("embed", []):
                    if part.get("type") == "image_url":
                        png_url = part.get("image_url", {}).get("url")
                        break
            if not png_url or not png_url.startswith("data:image/png;base64,"):
                print(f"  ! no png url in result: {list(result.keys())}", flush=True)
                sys.exit(2)
            png_bytes = base64.b64decode(png_url.split(",", 1)[1])
            with open(out_png, "wb") as f: f.write(png_bytes)
            with open(out_json, "w") as f: json.dump(result, f, indent=2)
            print(f"  ✓ saved {out_png} ({len(png_bytes)} bytes)", flush=True)
            sys.exit(0)
    time.sleep(3)

print(f"  ! timed out waiting for {sid}", flush=True)
sys.exit(3)
PYEOF
}

for PROMPT in "${PROMPTS[@]}"; do
    SLUG=$(echo "$PROMPT" | tr '[:upper:] ' '[:lower:]_' | head -c 40)
    REF_PNG="$REFS_DIR/${SLUG}.png"
    REF_JSON="$REFS_DIR/${SLUG}.json"
    if [[ -f "$REF_PNG" ]]; then
        echo "[05]   ✓ $PROMPT (cached)"
        continue
    fi
    echo "[05]   → generating ref for: $PROMPT"
    ref_one "$PROMPT" "$REF_PNG" "$REF_JSON"
done

echo "[05] OK — ${#PROMPTS[@]} reference images at $REFS_DIR"
ls -la "$REFS_DIR"
