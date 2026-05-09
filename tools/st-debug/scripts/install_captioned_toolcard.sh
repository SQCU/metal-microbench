#!/usr/bin/env bash
# Install a CAPTIONED variant of query-to-svg into the debug data root.
#
# Same multi-iter SVG refinement loop, but fires a tiny "describe what
# just happened" llm_call after each iter and emits the result as a
# progress event with a `caption: ` prefix. Replaces dead air during
# the refinement loop with a continuously-updating user-facing
# affordance.
#
# Idempotent. Lives under tools/st-debug/_data/toolcards/, NEVER touches
# the user's main install at ~/sillytavern-fork/data/toolcards/.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
SRC_INSTALLED="/Users/mdot/sillytavern-fork/data/toolcards/installed/query-to-svg"
SRC_CARD="/Users/mdot/sillytavern-fork/data/toolcards/cards/query-to-svg.toolcard.json"

DST_INSTALLED="$DATA_ROOT/toolcards/installed/query-to-svg-captioned"
DST_CARD="$DATA_ROOT/toolcards/cards/query-to-svg-captioned.toolcard.json"

# Export so the python subprocesses (heredocs below) can read them.
export SRC_CARD SRC_INSTALLED DST_CARD DST_INSTALLED

# The cards/ and installed/ symlinks the bootstrap created point at
# the user's main install. We need WRITE access for our own card
# without poisoning that. So: replace the cards/ symlink with a
# directory that contains existing cards (as symlinks) PLUS our new
# captioned card (as a real file). Same for installed/.
TOOLCARDS_DST="$DATA_ROOT/toolcards"

# Step 1: turn cards/ into a real dir, with symlinks to original cards.
if [[ -L "$TOOLCARDS_DST/cards" ]]; then
    SRC_CARDS_DIR="$(readlink "$TOOLCARDS_DST/cards")"
    rm "$TOOLCARDS_DST/cards"
    mkdir -p "$TOOLCARDS_DST/cards"
    for f in "$SRC_CARDS_DIR"/*.toolcard.json; do
        [[ -f "$f" ]] || continue
        ln -sf "$f" "$TOOLCARDS_DST/cards/$(basename "$f")"
    done
    echo "[install-captioned] cards/ → real dir + symlinks to $SRC_CARDS_DIR"
fi
if [[ -L "$TOOLCARDS_DST/installed" ]]; then
    SRC_INSTALLED_DIR="$(readlink "$TOOLCARDS_DST/installed")"
    rm "$TOOLCARDS_DST/installed"
    mkdir -p "$TOOLCARDS_DST/installed"
    for d in "$SRC_INSTALLED_DIR"/*/; do
        [[ -d "$d" ]] || continue
        ln -sf "$d" "$TOOLCARDS_DST/installed/$(basename "$d")"
    done
    echo "[install-captioned] installed/ → real dir + symlinks to $SRC_INSTALLED_DIR"
fi

# Step 2: write the captioned card manifest, with the PATCHED
# service.py embedded in the manifest's `files.service.py` field.
#
# Critical: the toolcards plugin treats the manifest's `files`
# dictionary as the source of truth — on startup (and when serving
# /list) it WRITES `files.service.py` to installed/<id>/service.py,
# overwriting any local edits. So we must patch the source BEFORE
# embedding it in the manifest, and we should NOT separately write
# installed/<id>/service.py (the plugin owns that path).

python3 <<'PY'
import json
import os
import pathlib

src_card = os.environ["SRC_CARD"]
src_installed = os.environ["SRC_INSTALLED"]
dst_card = os.environ["DST_CARD"]

# Read the original manifest.
with open(src_card) as f:
    card = json.load(f)

# Read the original service.py.
service_src = pathlib.Path(src_installed, "service.py").read_text()

# ── Patch 1: insert the caption helper block AFTER the existing
# llm_call() function definition. Anchor on `def extract_svg` (the
# function right after llm_call).
CAPTION_BLOCK = r'''

# ── Spur-subagent caption helper (added by install_captioned_toolcard.sh) ─
#
# After each iter we ask a separate small llm_call to summarize the
# iter's latest SVG output in one sentence, and emit it as a progress
# event so the FE surfaces it in the placeholder bubble. The caption
# is decoded at low max_tokens so the descendant cost is small.
CAPTION_SYSTEM_PROMPT = (
    "You are a caption-writer for a live activity feed. The user shows "
    "you the most recent text output from a different agent that is "
    "doing some longer-running work, and you write ONE short sentence "
    "(under 20 words) describing what that agent is currently doing — "
    "present-progressive, informal but precise. No preamble, no "
    "quoting, no commentary about your own role.\n"
    "\n"
    "Example input:\n"
    "    <svg width=\"200\" height=\"200\" xmlns=\"http://www.w3.org/2000/svg\">\n"
    "      <defs>\n"
    "        <radialGradient id=\"sun\" cx=\"50%\" cy=\"40%\" r=\"40%\">\n"
    "Example caption: drawing a radial-gradient sun in the upper-left.\n"
    "\n"
    "Example input:\n"
    "    iter 2/3: rendering 502-char SVG to PNG via playwright\n"
    "Example caption: on iteration 2, rendering the SVG to a PNG preview."
)


def emit_caption(activity_text: str) -> None:
    """Fire a tiny llm_call to summarize, emit result as progress."""
    try:
        snip = activity_text[-600:] if len(activity_text) > 600 else activity_text
        resp = llm_call([
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": snip},
        ], max_tokens=32)
        text = (resp.get("text") or "").strip()
        if text:
            progress(f"caption: {text}")
    except Exception as e:
        progress(f"caption-subagent error: {type(e).__name__}: {e}")

'''

anchor = "def extract_svg"
idx = service_src.find(anchor)
if idx < 0:
    raise SystemExit("could not find anchor 'def extract_svg' in service.py")
service_src = service_src[:idx] + CAPTION_BLOCK + "\n" + service_src[idx:]

# ── Patch 2: hook the caption call after each iter's "done in X.Xs"
# progress event. We anchor on the progress() call that includes
# 'iter {i+1}: done in' and insert an emit_caption(svg) right after.
hook_marker = 'progress(f"iter {i+1}: done in'
hook_idx = service_src.find(hook_marker)
if hook_idx < 0:
    raise SystemExit("could not find hook 'iter ... done in' in service.py")
line_end = service_src.find("\n", hook_idx)
# Match the indent of the hook line.
ind_start = service_src.rfind("\n", 0, hook_idx) + 1
indent = service_src[ind_start:hook_idx]

INSERT = "\n" + indent + "emit_caption(svg)"
service_src = service_src[:line_end] + INSERT + service_src[line_end:]

# Sanity: make sure the patched source is valid Python.
import ast
try:
    ast.parse(service_src)
except SyntaxError as e:
    raise SystemExit(f"patched service.py has syntax error: {e}")

# ── Update the manifest with the patched source embedded.
card["id"] = "query-to-svg-captioned"
card["display_name"] = card.get("display_name", "Query → SVG") + " (captioned)"
card["description"] = (
    "Same iterative SVG refinement as query-to-svg, but emits live "
    "captions of each iter's progress via a tiny spur-subagent that "
    "summarizes the agent's recent activity in one sentence per "
    "iteration. Useful for long workflows where the user would "
    "otherwise see dead air for tens of seconds."
)
card["files"] = {"service.py": service_src}

with open(dst_card, "w") as f:
    json.dump(card, f, indent=2)
print(f"  wrote {dst_card}")
print(f"    embedded patched service.py: {len(service_src)} chars")
print(f"    contains CAPTION_SYSTEM_PROMPT: {'CAPTION_SYSTEM_PROMPT' in service_src}")
print(f"    contains emit_caption hook: {'emit_caption(svg)' in service_src}")
PY

# Step 3: ALSO write the patched service.py to installed/ for the
# plugin's first-load path. The plugin writes from the manifest on
# install, but if the directory exists already (from prior install)
# it may not re-emit. Belt and suspenders.
mkdir -p "$DST_INSTALLED"
python3 <<'PY' > /dev/null
import json, os, pathlib
dst_card = os.environ["DST_CARD"]
dst_installed = os.environ["DST_INSTALLED"]
card = json.loads(pathlib.Path(dst_card).read_text())
service_src = card["files"]["service.py"]
pathlib.Path(dst_installed, "service.py").write_text(service_src)
PY

echo "[install-captioned] done."
echo "  Card: $DST_CARD"
echo "  Service: $DST_INSTALLED/service.py (mirrored from manifest's files.service.py)"
echo "  Restart ST and the plugin will discover query-to-svg-captioned."
