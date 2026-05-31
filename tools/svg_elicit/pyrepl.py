#!/usr/bin/env python3
"""Shared, clean core for SVG-reconstruction harnesses: a tag-delimited PERSISTENT
Python REPL. One protocol, imported by every harness — so we stop reinventing (and
re-breaking) extraction/edit/REPL logic per harness.

Protocol: the model emits Python inside ONE <pyrepl> ... </pyrepl> block per turn.
The cell is exec'd into a namespace that PERSISTS across turns (Jupyter-style):
variables set last turn are still defined, so a turn is a small delta — no
re-emission, no text-edit grammar, no string-literal-boundary hazard. The finished
image must be left in the string variable `svg`.

Transactional: a cell that raises, or that fails to leave a non-empty string in
`svg`, is ROLLED BACK — the namespace is restored to its pre-cell snapshot and the
error is returned for feedback. REPL state is never corrupted by a bad cell.

Why a single <pyrepl> tag and not a tolerant multi-fallback parser: extraction must
be UNAMBIGUOUS. Trailing turn markers (<turn|>), prose, and markdown fences fall
OUTSIDE the captured group, so there is nothing to "tolerantly" guess.
"""
from __future__ import annotations
import copy
import re

__all__ = ["extract_pyrepl", "PyRepl", "PYREPL_TAG_HELP"]

_PYREPL_RE = re.compile(r"<pyrepl>(.*?)</pyrepl>", re.DOTALL | re.IGNORECASE)
_OPEN = "<pyrepl>"

PYREPL_TAG_HELP = (
    "Emit your Python inside ONE  <pyrepl> ... </pyrepl>  block. The code runs in a "
    "PERSISTENT namespace that carries over between turns — variables you defined a "
    "previous turn are still in scope, so each turn is a small delta, not a rewrite. "
    "Your code must leave the finished picture in the string variable `svg`. `np` and "
    "the reference array `target` (H,W,3 uint8 RGB) are pre-defined."
)


def extract_pyrepl(text: str) -> str | None:
    """Code inside the LAST <pyrepl>...</pyrepl> block (case-insensitive), or None.

    One clean rule, no fenced/heuristic fallbacks. Tolerates exactly one degenerate
    case — a single UNCLOSED final tag from a truncated generation — by reading to
    end-of-text; anything else without the tag returns None (caller re-asks)."""
    if not text:
        return None
    blocks = _PYREPL_RE.findall(text)
    if blocks:
        return blocks[-1].strip("\n")
    i = text.lower().rfind(_OPEN)
    if i >= 0:                       # unclosed final tag (truncation) -> take to EOF
        return text[i + len(_OPEN):].strip("\n")
    return None


class PyRepl:
    """A persistent Python namespace mutated across turns by <pyrepl> cells."""

    def __init__(self, seed_vars: dict | None = None, out_var: str = "svg"):
        self.out_var = out_var
        self.ns: dict = {"__name__": "__pyrepl__"}
        if seed_vars:
            self.ns.update(seed_vars)
        self.cells: list[str] = []        # accepted cells, in order (the history)
        self.last_svg: str | None = None

    def _snapshot(self) -> dict:
        snap = {}
        for k, v in self.ns.items():
            try:
                snap[k] = copy.deepcopy(v)
            except Exception:
                snap[k] = v               # modules/functions/unpicklables: keep ref
        return snap

    def _restore(self, snap: dict) -> None:
        self.ns.clear()
        self.ns.update(snap)

    def run_cell(self, code: str) -> dict:
        """Exec `code` into the persistent namespace, transactionally.

        Returns {ok, error, svg, rolled_back}. On any failure the namespace is
        restored to its pre-cell state and `svg` is the last good output (or None)."""
        if not code or not code.strip():
            return {"ok": False, "error": "empty <pyrepl> cell",
                    "svg": self.last_svg, "rolled_back": False}
        snap = self._snapshot()
        try:
            exec(compile(code, "<pyrepl>", "exec"), self.ns)
        except Exception as e:
            self._restore(snap)
            return {"ok": False, "error": f"{type(e).__name__}: {e}",
                    "svg": self.last_svg, "rolled_back": True}
        svg = self.ns.get(self.out_var)
        if not isinstance(svg, str) or not svg.strip():
            self._restore(snap)
            return {"ok": False,
                    "error": (f"cell ran but `{self.out_var}` is not a non-empty "
                              f"string (got {type(svg).__name__})"),
                    "svg": self.last_svg, "rolled_back": True}
        self.cells.append(code)
        self.last_svg = svg
        return {"ok": True, "error": None, "svg": svg, "rolled_back": False}

    @property
    def canonical_py(self) -> str:
        """The accepted cells concatenated. Re-running this in a fresh namespace with
        the same seeds reproduces the current `svg` — the dump/debug artifact and the
        'canonical python for this turn'."""
        return "\n\n# ---- turn boundary ----\n".join(self.cells)


# --------------------------------------------------------------------------- #
# SERVERLESS self-test. No LM, no render, no network — pure namespace assertions.
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    import numpy as np

    checks: list[bool] = []

    def check(name: str, cond: bool, detail: str = ""):
        checks.append(bool(cond))
        print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))

    # ---- extraction: one clean rule, robust to prose/fences/turn-markers ----
    check("extract: simple block",
          extract_pyrepl("blah\n<pyrepl>\nsvg='a'\n</pyrepl>\nthanks") == "svg='a'")
    check("extract: LAST of multiple blocks wins",
          extract_pyrepl("<pyrepl>\nsvg='1'\n</pyrepl> then <pyrepl>\nsvg='2'\n</pyrepl>") == "svg='2'")
    check("extract: trailing <turn|> is OUTSIDE the capture (no stripping needed)",
          extract_pyrepl("<pyrepl>\nsvg='x'\n</pyrepl><turn|>") == "svg='x'")
    check("extract: case-insensitive tags",
          extract_pyrepl("<PyRepl>\nsvg='y'\n</PYREPL>") == "svg='y'")
    check("extract: unclosed final tag (truncation) reads to EOF",
          extract_pyrepl("ok <pyrepl>\nsvg='trunc'") == "svg='trunc'")
    check("extract: no tag -> None", extract_pyrepl("just prose, no code") is None)
    check("extract: markdown fence INSIDE tag is preserved verbatim (caller's problem, not ours)",
          extract_pyrepl("<pyrepl>\n```python\nsvg='z'\n```\n</pyrepl>") == "```python\nsvg='z'\n```")

    # ---- persistent namespace + transactional rollback ----
    target = np.zeros((4, 4, 3), dtype=np.uint8)
    repl = PyRepl(seed_vars={"np": np, "target": target})

    r0 = repl.run_cell("parts = []\n"
                       "parts.append('<rect width=\"10\" height=\"10\" fill=\"#222\"/>')\n"
                       "svg = '<svg>' + ''.join(parts) + '</svg>'")
    check("cell0 ok, svg set", r0["ok"] and r0["svg"].startswith("<svg>") and "fill=\"#222\"" in r0["svg"])
    svg0 = r0["svg"]

    # incremental delta: mutate persisted `parts` (state carried over)
    r1 = repl.run_cell("parts[0] = '<rect width=\"10\" height=\"10\" fill=\"#3A7D35\"/>'\n"
                       "parts.append('<circle cx=\"5\" cy=\"5\" r=\"3\"/>')\n"
                       "svg = '<svg>' + ''.join(parts) + '</svg>'")
    check("cell1 ok, namespace PERSISTED (parts mutated across turns)",
          r1["ok"] and "fill=\"#3A7D35\"" in r1["svg"] and "<circle" in r1["svg"] and r1["svg"] != svg0)
    svg1 = r1["svg"]

    # breaking cell (raises) -> rollback; namespace + last_svg intact, not recorded
    r2 = repl.run_cell("parts.append('SHOULD_BE_ROLLED_BACK')\n"
                       "svg = parts[999]   # IndexError")
    check("cell2 raises -> rolled_back, last svg preserved",
          (not r2["ok"]) and r2["rolled_back"] and r2["svg"] == svg1 and "IndexError" in r2["error"])
    check("cell2 rollback UNDID the in-place mutation (parts restored)",
          repl.run_cell("svg = '<svg>'+''.join(parts)+'</svg>'")["svg"] == svg1)
    # (the line above is itself a cell; it recomputes svg from the restored parts ==> == svg1)

    # cell that runs but leaves svg empty -> rollback
    r4 = repl.run_cell("svg = ''")
    check("cell leaving empty svg -> rolled_back (no-empty-output rule)",
          (not r4["ok"]) and r4["rolled_back"] and r4["svg"] == svg1)

    # cell that runs but never assigns a string svg -> rollback
    r5 = repl.run_cell("svg = 123")
    check("cell leaving non-string svg -> rolled_back",
          (not r5["ok"]) and r5["rolled_back"])

    # history holds only ACCEPTED cells; canonical_py re-runs to the same svg
    accepted = len(repl.cells)
    fresh = PyRepl(seed_vars={"np": np, "target": target})
    fr = fresh.run_cell(repl.canonical_py)
    check("canonical_py re-runs in a fresh namespace to the SAME svg",
          fr["ok"] and fr["svg"] == repl.last_svg, f"accepted_cells={accepted}")

    ok = all(checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}: {sum(checks)}/{len(checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
