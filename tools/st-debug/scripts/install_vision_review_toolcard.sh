#!/usr/bin/env bash
# Install vision-review: model-on-model review of recorded
# playwright transcripts (or any video file). Closes the visual-
# oversight gap flagged in test 17's README.
#
# Topology: spawn-only (no parent context — visual review is a
# focused technical task; the persona's voice would be noise).
#   pre: ffmpeg-extract N frames from the input video at uniform
#        intervals; encode each as data: PNG URL
#   spawn S_i (parallel via parallel_llm_call): one per frame.
#       prefix = "You are a visual transcript reviewer. Describe
#                what is visible in the given frame, focusing on
#                shapes, colors, and text. Be concise."
#       input = the frame's image_url
#       output = ~50-150 tokens of description
#       (siblings share the prefix → KV cache hits)
#   spawn S_synth (after collecting per-frame obs):
#       prefix = same shared one
#       input = a structured list of (frame_n, description) +
#               the user's claim under review
#       output = pass/fail + 1-2 sentence reasoning
#
# Each frame description emits a `summary_progress` line so the
# parent's trace surfaces "what was seen at each timestamp".
# Final synthesis's pass/fail emits the closing summary_progress.
#
# This card is the vision-validator stage missing from
# render-visual: it can be invoked downstream of any video-
# producing tool to score whether the recording shows the
# claimed feature.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HERE/_data}"
CARDS_DIR="$DATA_ROOT/toolcards/cards"
INSTALLED_DIR="$DATA_ROOT/toolcards/installed/vision-review"

mkdir -p "$CARDS_DIR" "$INSTALLED_DIR"

SERVICE_PY="$INSTALLED_DIR/service.py"

cat > "$SERVICE_PY" << 'PY'
"""vision-review: per-frame multimodal review of a video file.

Spawn-only card. Extracts N frames from a video, fires N parallel
vision llm_calls (one per frame), then a final synthesis call to
score whether the user's claim about the video holds.

Inputs:
  video_path: filesystem path to a .webm/.mp4
  claim: natural-language claim under review (e.g. "the recording
         shows a Lissajous curve being drawn")
  num_frames: optional, default 5

Output:
  {
    "pass": bool,
    "reasoning": "...",
    "frames": [{"frame": i, "description": "...", "tMs": ms}, ...],
  }

Per-frame `summary_progress` events emit during execution; the
final pass/fail also emits a `summary_progress` with the verdict.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any


REVIEWER_PREFIX = (
    "You are a visual transcript reviewer. You receive a single "
    "frame from a recording and describe what is visible in 1-2 "
    "concise sentences. Focus on: SHAPES (geometric, curved, "
    "tessellated, scattered), COLORS, and any TEXT visible. If a "
    "user interface is visible, name what part (chat bubble, "
    "menu, modal, image-embed, etc.). Do not speculate beyond "
    "what is in the frame."
)

VERDICT_PREFIX = (
    "You are a verdict synthesizer. You receive a list of frame "
    "descriptions from a recording AND a user's claim about that "
    "recording. Decide whether the claim is supported by the "
    "evidence. Output EXACTLY this format:\n"
    "VERDICT: PASS|FAIL\n"
    "REASONING: <1-2 sentences>"
)

VERDICT_RE = re.compile(r"VERDICT:\s*(PASS|FAIL)", re.IGNORECASE)
REASON_RE = re.compile(r"REASONING:\s*(.+?)(?:\n\n|$)", re.IGNORECASE | re.DOTALL)


_NEXT_CALL_ID = 0


def emit(event: dict[str, Any]) -> None:
    print(json.dumps(event), flush=True)


def progress(text: str) -> None:
    emit({"type": "progress", "text": text})


def summary_progress(scope: str, summary: str,
                     compressed_lines: int | None = None) -> None:
    event: dict[str, Any] = {
        "type": "summary_progress",
        "scope": scope,
        "summary": summary.strip()[:300],
    }
    if compressed_lines is not None:
        event["compressed_lines"] = compressed_lines
    emit(event)


def next_call_id() -> int:
    global _NEXT_CALL_ID
    _NEXT_CALL_ID += 1
    return _NEXT_CALL_ID


def parse_stdin_json() -> dict[str, Any]:
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("stdin closed during llm_call")
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except Exception:
            print(f"[vision-review] non-JSON stdin: {line[:120]!r}",
                  file=sys.stderr)


def parallel_llm_call(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not calls:
        return []
    cids: list[int] = []
    starts: dict[int, float] = {}
    pending: set[int] = set()
    responses: dict[int, dict[str, Any]] = {}

    for call in calls:
        cid = next_call_id()
        cids.append(cid)
        pending.add(cid)
        starts[cid] = time.time()
        emit({"type": "llm_call", "id": cid,
              "messages": call["messages"],
              "max_tokens": call.get("max_tokens", 256)})

    while pending:
        msg = parse_stdin_json()
        if msg.get("type") != "llm_response":
            continue
        cid = msg.get("id")
        if cid not in pending:
            continue
        pending.remove(cid)
        if msg.get("ok"):
            responses[cid] = {
                "text": str(msg.get("data", "") or ""),
                "elapsed_s": time.time() - starts[cid],
            }
        else:
            raise RuntimeError(f"llm_call {cid} failed: {msg.get('error', 'unknown')}")

    return [responses[cid] for cid in cids]


def llm_call(messages: list[dict[str, Any]],
             max_tokens: int = 256) -> dict[str, Any]:
    return parallel_llm_call([{"messages": messages, "max_tokens": max_tokens}])[0]


def extract_frames(video_path: str, num_frames: int,
                   tmp_dir: str) -> list[str]:
    """Use ffmpeg to extract num_frames evenly-spaced frames from the
    video. Returns a list of absolute paths to the PNG files.

    We compute a fps rate that produces ~num_frames over the video's
    duration. ffmpeg's `-vf fps=N/duration` doesn't work directly
    so we probe duration first.
    """
    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=15,
    )
    if probe.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {probe.stderr}")
    duration_s = float(probe.stdout.strip())
    if duration_s <= 0.5:
        raise RuntimeError(f"video too short to review: {duration_s:.2f}s")

    # Ask ffmpeg for one frame every (duration/num_frames) seconds
    interval_s = max(0.5, duration_s / max(num_frames, 1))
    fps = 1.0 / interval_s
    out_pattern = os.path.join(tmp_dir, "frame_%03d.png")
    cmd = ["ffmpeg", "-loglevel", "error", "-y",
           "-i", video_path,
           "-vf", f"fps={fps}",
           "-frames:v", str(num_frames),
           out_pattern]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr}")
    frames = sorted(
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.startswith("frame_") and f.endswith(".png")
    )
    return frames


def encode_png_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def parse_verdict(text: str) -> tuple[bool, str]:
    verdict_match = VERDICT_RE.search(text)
    pass_ = bool(verdict_match) and verdict_match.group(1).upper() == "PASS"
    reason_match = REASON_RE.search(text)
    reason = (reason_match.group(1).strip() if reason_match
              else text.strip()[:300])
    return pass_, reason


def handle(args: dict[str, Any], _caller_messages: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")
    video_path = args.get("video_path")
    claim = args.get("claim")
    num_frames = int(args.get("num_frames", 5))
    if not isinstance(video_path, str) or not video_path:
        raise ValueError("video_path is required")
    if not isinstance(claim, str) or not claim.strip():
        raise ValueError("claim is required")
    claim = claim.strip()
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"video not found: {video_path}")

    progress(f"extracting {num_frames} frames from video")
    with tempfile.TemporaryDirectory() as tmp_dir:
        frames = extract_frames(video_path, num_frames, tmp_dir)
        progress(f"got {len(frames)} frame(s); firing vision review per frame")

        # Spawn vision llm_calls SEQUENTIALLY. We initially tried
        # parallel via parallel_llm_call to exploit the shared
        # REVIEWER_PREFIX, but parallel multimodal calls reproducibly
        # produced `<unused6226>` placeholder-token outputs at the
        # bridge level (see /tmp/bridge.log post 2026-05-10 probe);
        # single-frame multimodal works correctly. Empirical workaround
        # until the bridge's parallel-vision path is debugged: run
        # PARALLEL. Each call still benefits from the system-prompt
        # prefix cache hit (REVIEWER_PREFIX is identical across calls);
        # only the per-frame image bytes change.
        frame_results: list[dict[str, Any]] = []
        for i, path in enumerate(frames):
            data_url = encode_png_data_url(path)
            resp = llm_call(
                [
                    {"role": "system", "content": REVIEWER_PREFIX},
                    {"role": "user", "content": [
                        {"type": "text",
                         "text": "What is visible in this frame?"},
                        {"type": "image_url",
                         "image_url": {"url": data_url}},
                    ]},
                ],
                max_tokens=256,
            )
            description = resp["text"].strip().split("\n")[0][:300]
            frame_results.append({
                "frame": i,
                "description": resp["text"].strip(),
                "elapsed_s": resp["elapsed_s"],
            })
            summary_progress(f"frame:{i}", description)

        progress("synthesizing verdict from per-frame observations")
        # Build the verdict-input message
        evidence = "\n".join(
            f"frame {fr['frame']}: {fr['description'][:300]}"
            for fr in frame_results
        )
        verdict_resp = llm_call(
            [
                {"role": "system", "content": VERDICT_PREFIX},
                {"role": "user", "content": (
                    f"CLAIM: {claim}\n\nEVIDENCE (per-frame observations):\n"
                    f"{evidence}\n\n"
                    f"Decide whether the claim is supported."
                )},
            ],
            max_tokens=256,
        )
        verdict_text = verdict_resp["text"]
        pass_, reasoning = parse_verdict(verdict_text)

        verdict_label = "PASS" if pass_ else "FAIL"
        summary_progress(
            "verdict",
            f"{verdict_label}: {reasoning[:200]}",
        )

        return {
            "pass": pass_,
            "verdict_label": verdict_label,
            "reasoning": reasoning,
            "claim": claim,
            "video_path": video_path,
            "num_frames": len(frame_results),
            "frames": frame_results,
            "summary": "\n".join([
                f"[claim]   {claim}",
                f"[verdict] {verdict_label}",
                f"[reason]  {reasoning[:200]}",
                f"[frames]  reviewed {len(frame_results)} frames",
            ]),
        }


def emit_result(rid: Any, ok: bool, result: Any = None,
                 error: str | None = None) -> None:
    msg: dict[str, Any] = {"type": "result", "id": rid, "ok": ok}
    if ok:
        msg["result"] = result
    else:
        msg["error"] = str(error or "unknown")
    emit(msg)


def main() -> None:
    emit({"type": "ready"})
    while True:
        try:
            msg = parse_stdin_json()
        except EOFError:
            return
        if msg.get("type") != "invoke":
            continue
        rid = msg.get("id")
        try:
            result = handle(msg.get("args") or {},
                            msg.get("caller_messages"))
            emit_result(rid, True, result)
        except Exception as e:
            emit_result(rid, False, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
PY

# Embed service.py into the manifest's files field.
SERVICE_CONTENT=$(python3 -c "import json,sys; print(json.dumps(open('$SERVICE_PY').read()))")

cat > "$CARDS_DIR/vision-review.toolcard.json" << JSON
{
  "card_format_version": "1",
  "id": "vision-review",
  "version": "0.1.0",
  "display_name": "Vision Review",
  "description": "Inspect a recorded video file frame-by-frame and decide whether a user-supplied claim about its visual content holds. Spawns parallel multimodal LLM calls (one per frame) sharing a 'visual transcript reviewer' prefix; rolls up to a final pass/fail verdict via a synthesis call. Use to score model-produced visual outputs against expected behavior — closes the visual-oversight gap when validator-level invariants don't catch shape correctness. See docs/scalable_oversight_recursive_decomposition.md.",
  "tools": [
    {
      "name": "review",
      "display_name": "Review video against a claim",
      "description": "Use when there is a recorded video file (e.g. a playwright run's video.webm) and a claim about what should be visible in it. The card extracts N frames, runs a multimodal review per frame, and reports whether the claim is supported. Example: review(video_path='/path/to/run.webm', claim='this recording shows a Lissajous curve being drawn').",
      "parameters": {
        "type": "object",
        "properties": {
          "video_path": {
            "type": "string",
            "description": "Absolute filesystem path to the video file (.webm/.mp4)."
          },
          "claim": {
            "type": "string",
            "description": "Natural-language claim about what the video should show."
          },
          "num_frames": {
            "type": "integer",
            "description": "Frames to extract evenly across the video. Default 5.",
            "default": 5
          }
        },
        "required": ["video_path", "claim"]
      },
      "async": false
    }
  ],
  "runtime": {
    "kind": "python",
    "deps": [],
    "entrypoint": "service.py",
    "idle_timeout_s": 300
  },
  "files": {
    "service.py": $SERVICE_CONTENT
  }
}
JSON

# Mirror service.py from manifest's files field
python3 -c "
import json, pathlib
manifest = json.load(open('$CARDS_DIR/vision-review.toolcard.json'))
content = manifest['files']['service.py']
target = pathlib.Path('$SERVICE_PY')
target.write_text(content)
print(f'  embedded service.py: {len(content)} chars')
"

echo "[install-vision-review] done."
echo "  Card: $CARDS_DIR/vision-review.toolcard.json"
echo "  Service: $SERVICE_PY"
