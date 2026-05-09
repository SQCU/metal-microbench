#!/usr/bin/env bash
# Install the random-choice toolcard into the debug data root.
#
# Pure programmatic RNG-backed sampling. No LLM calls, no persistent state.
# Idempotent. Lives under tools/st-debug/_data/toolcards/, NEVER touches
# the user's main install at ~/sillytavern-fork/data/toolcards/.

set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="$HERE/_data"
TOOLCARDS_DST="$DATA_ROOT/toolcards"
DST_INSTALLED="$TOOLCARDS_DST/installed/random-choice"
DST_CARD="$TOOLCARDS_DST/cards/random-choice.toolcard.json"

export DST_CARD DST_INSTALLED

# The cards/ and installed/ symlinks the bootstrap created point at the
# user's main install. Replace each symlink with a real directory that
# keeps the original entries as symlinks, then add our debug-only card as
# real files under _data/toolcards/.
mkdir -p "$TOOLCARDS_DST"
if [[ -L "$TOOLCARDS_DST/cards" ]]; then
    SRC_CARDS_DIR="$(readlink "$TOOLCARDS_DST/cards")"
    rm "$TOOLCARDS_DST/cards"
    mkdir -p "$TOOLCARDS_DST/cards"
    for f in "$SRC_CARDS_DIR"/*.toolcard.json; do
        [[ -f "$f" ]] || continue
        ln -sf "$f" "$TOOLCARDS_DST/cards/$(basename "$f")"
    done
    echo "[install-random-choice] cards/ -> real dir + symlinks to $SRC_CARDS_DIR"
fi
if [[ -L "$TOOLCARDS_DST/installed" ]]; then
    SRC_INSTALLED_DIR="$(readlink "$TOOLCARDS_DST/installed")"
    rm "$TOOLCARDS_DST/installed"
    mkdir -p "$TOOLCARDS_DST/installed"
    for d in "$SRC_INSTALLED_DIR"/*/; do
        [[ -d "$d" ]] || continue
        ln -sf "$d" "$TOOLCARDS_DST/installed/$(basename "$d")"
    done
    echo "[install-random-choice] installed/ -> real dir + symlinks to $SRC_INSTALLED_DIR"
fi

mkdir -p "$(dirname "$DST_CARD")" "$DST_INSTALLED"

python3 <<'PY'
import json
import os
import pathlib

dst_card = pathlib.Path(os.environ["DST_CARD"])
dst_installed = pathlib.Path(os.environ["DST_INSTALLED"])

service_src = '''"""random-choice tool service.

Uniformly samples items from a caller-provided list using Python's
SystemRandom. This card is intentionally non-LLM: the service receives
only the invocation args, performs local validation, rolls the RNG, and
returns the selected items.
"""
from __future__ import annotations

import json
import random
import sys
from typing import Any


RNG = random.SystemRandom()


def _validate_items(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("items must be a non-empty list of strings")
    if not value:
        raise ValueError("items must be a non-empty list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError("items must contain only strings")
    return value


def _validate_n(value: Any) -> int:
    if value is None:
        return 1
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("n must be an integer greater than 0")
    if value <= 0:
        raise ValueError("n must be greater than 0")
    return value


def _validate_with_replacement(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("with_replacement must be a boolean")
    return value


def handle(args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("args must be an object")

    items = _validate_items(args.get("items"))
    n = _validate_n(args.get("n", 1))
    with_replacement = _validate_with_replacement(
        args.get("with_replacement", False)
    )

    if not with_replacement and n > len(items):
        raise ValueError(
            "n must be less than or equal to len(items) when "
            "with_replacement is false"
        )

    if with_replacement:
        sampled = [RNG.choice(items) for _ in range(n)]
    else:
        sampled = RNG.sample(items, n)

    summary = "Selected " + str(n) + " item(s): " + ", ".join(sampled)
    return {
        "summary": summary,
        "items": sampled,
        "embed": [{"type": "text", "text": summary}],
        "n": n,
        "with_replacement": with_replacement,
    }


def emit_result(rid: Any, ok: bool, result: Any = None, error: str | None = None) -> None:
    msg = {"type": "result", "id": rid, "ok": ok}
    if ok:
        msg["result"] = result
    else:
        msg["error"] = error or "unknown error"
    print(json.dumps(msg), flush=True)


def main() -> None:
    print(json.dumps({"type": "ready"}), flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception as exc:
            emit_result(0, False, error=f"bad request json: {exc}")
            continue
        if req.get("type") != "invoke":
            continue
        rid = req.get("id", 0)
        try:
            result = handle(req.get("args") or {})
            emit_result(rid, True, result=result)
        except ValueError as exc:
            emit_result(rid, False, error=str(exc))
        except Exception as exc:
            emit_result(rid, False, error=f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
'''

card = {
    "card_format_version": "1",
    "id": "random-choice",
    "display_name": "Random Choice",
    "description": (
        "Uniformly samples N items from a list. "
        "When to call: a user asks for a random pick (a tarot card, "
        "a dice roll, a name from a list, a song from a playlist), or "
        "anywhere that asking the model to choose one would skew the "
        "distribution. "
        "How to call: pass items as a list of strings. Example call: "
        "items=['cinnamon roll', 'crepe', 'belgian waffle', 'oatmeal'], n=2."
    ),
    "version": "0.1.0",
    "author": "fork",
    "runtime": {
        "kind": "python",
        "deps": [],
        "entrypoint": "service.py",
        "idle_timeout_s": 60,
    },
    "tools": [
        {
            "name": "uniform",
            "display_name": "Uniform random sample",
            "description": (
                "Uniformly samples N items from a list. "
                "When to call: a user asks for a random pick (a tarot card, "
                "a dice roll, a name from a list, a song from a playlist), "
                "or anywhere that asking the model to choose one would skew "
                "the distribution. "
                "How to call: pass items as a list of strings. Example call: "
                "items=['cinnamon roll', 'crepe', 'belgian waffle', "
                "'oatmeal'], n=2."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Candidate strings to sample from.",
                    },
                    "n": {
                        "type": "integer",
                        "default": 1,
                        "description": "Number of items to sample.",
                    },
                    "with_replacement": {
                        "type": "boolean",
                        "default": False,
                        "description": "Allow the same item to be selected more than once.",
                    },
                },
                "required": ["items"],
            },
        }
    ],
    "files": {"service.py": service_src},
}

dst_card.write_text(json.dumps(card, indent=2) + "\n")
dst_installed.mkdir(parents=True, exist_ok=True)
(dst_installed / "service.py").write_text(service_src)

print(f"  wrote {dst_card}")
print(f"  wrote {dst_installed / 'service.py'}")
print(f"  embedded service.py: {len(service_src)} chars")
PY

echo "[install-random-choice] done."
echo "  Card: $DST_CARD"
echo "  Service: $DST_INSTALLED/service.py (mirrored from manifest's files.service.py)"
