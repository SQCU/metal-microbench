#!/usr/bin/env python3
"""DEPRECATED: superseded by pyrepl.py (extraction+REPL) and pyrepl_refine.py
(the incremental harness). Kept for reference; do not build new harnesses on this.

The REAL core of an incremental SVG-editing REPL — the edit-suite a prior
harness only NAMED ("edit-in-REPL") but never built (it forced the model to
re-emit the ENTIRE program every turn, see krepl.py / edit_elicit.py 'rewrite').

Here the artifact PERSISTS and is mutated by atomic, tolerantly-parsed EDIT
blocks. This module is PURE TEXT: no LM, no render, no HTTP, no API backend —
fully serverless — so the apply semantics can be unit-tested deterministically
and reused by any caller (the elicitation harness wires call_lm + render around
`SvgRepl`).

EDIT-SUITE GRAMMAR. The model emits one or more edit blocks; prose/fences around
them are ignored. Op names are case-insensitive and the marker rune may be any of
### / *** / === :

    ### EDIT <OP> ###
    --- FIND ---
    <one or more exact lines from the CURRENT program>   (omit FIND for APPEND)
    --- REPLACE ---
    <replacement lines>                                  (omit REPLACE for DELETE)
    ### END ###

Ops (FIND must be UNIQUE in the running program where required):
  REPLACE      : FIND exact substring -> substitute REPLACE (REPLACE may be empty).
  INSERT_AFTER : FIND anchor -> insert REPLACE lines immediately AFTER it.
  DELETE       : FIND exact substring -> remove it.
  APPEND       : no FIND -> insert REPLACE immediately before the LAST line
                 beginning with "svg " (the assignment); else at EOF.

Apply semantics: ops are applied IN ORDER to the running program text. An op
whose FIND is missing, matches MORE THAN ONCE (ambiguous), or is empty where
required is NOT applied; instead an error
{index, op, reason in {not_found, ambiguous, empty}, snippet} is appended to the
errors list. Ops are NEVER silently dropped.
"""
from __future__ import annotations

import re

__all__ = ["parse_edits", "apply_edits", "SvgRepl"]

# Ops the grammar exposes. Normalised to UPPER for case-insensitive matching.
_OPS = {"REPLACE", "INSERT_AFTER", "DELETE", "APPEND"}
# Common spelling drift we accept and canonicalise (still tolerant, not lax).
_OP_ALIASES = {
    "INSERTAFTER": "INSERT_AFTER",
    "INSERT-AFTER": "INSERT_AFTER",
    "INSERT": "INSERT_AFTER",
    "ADD": "APPEND",
}

# A marker rune is a run (>=3) of one of # * = . The header carries the op name,
# the END line closes the block, and the two section markers name FIND / REPLACE.
# Everything is whitespace- and case-tolerant; surrounding prose is ignored
# because we scan for these lines rather than requiring the whole message match.
_RUNE = r"[#*=]{3,}"
_HEADER_RE = re.compile(rf"^\s*{_RUNE}\s*EDIT\s+([A-Za-z_][A-Za-z_\-]*)\s*{_RUNE}\s*$", re.I)
_END_RE = re.compile(rf"^\s*{_RUNE}\s*END\s*{_RUNE}\s*$", re.I)
# Section dividers: "--- FIND ---" / "--- REPLACE ---". Dashes optional, the
# word is what matters; accept the marker rune too in case the model reuses it.
_SECTION_RE = re.compile(rf"^\s*(?:-{{2,}}|{_RUNE})?\s*(FIND|REPLACE)\s*(?:-{{2,}}|{_RUNE})?\s*$", re.I)
# A line beginning with `svg ` (the assignment) — used by APPEND.
_SVG_ASSIGN_LINE_RE = re.compile(r"^\s*svg\b")


def _canon_op(raw: str) -> str | None:
    key = raw.strip().upper().replace(" ", "_")
    if key in _OPS:
        return key
    return _OP_ALIASES.get(key)


def _strip_one_trailing_newline(s: str) -> str:
    # The block body is captured between marker lines; a single trailing newline
    # from the section join is not part of the content.
    return s[:-1] if s.endswith("\n") else s


def parse_edits(text: str):
    """Tolerantly parse EDIT blocks out of `text`, ignoring surrounding prose and
    optional triple-backtick fences. Returns a list of dicts
    {op, find, replace} in document order. `find`/`replace` are strings (possibly
    empty); for APPEND `find` is "" (unused); for DELETE `replace` is "".

    Robustness: a malformed/incomplete block (header with no END before the next
    header or EOF) is skipped rather than raising — the apply stage reports
    substantive errors; the parse stage just extracts what's well-formed.
    """
    if text is None:
        return []
    # Drop triple-backtick fence lines wholesale — they may wrap the whole reply
    # or individual blocks; either way the fence line itself is never content.
    raw_lines = text.split("\n")
    lines = [ln for ln in raw_lines if not ln.lstrip().startswith("```")]

    edits: list[dict] = []
    i, n = 0, len(lines)
    while i < n:
        mh = _HEADER_RE.match(lines[i])
        if not mh:
            i += 1
            continue
        op = _canon_op(mh.group(1))
        i += 1
        if op is None:
            # Unknown op name: still consume to END (if any) so its body can't be
            # mis-read as a new block, but emit nothing.
            while i < n and not _END_RE.match(lines[i]) and not _HEADER_RE.match(lines[i]):
                i += 1
            if i < n and _END_RE.match(lines[i]):
                i += 1
            continue

        find_lines: list[str] | None = None
        repl_lines: list[str] | None = None
        cur: list[str] | None = None        # which section we're accumulating into
        closed = False
        while i < n:
            ln = lines[i]
            if _END_RE.match(ln):
                i += 1
                closed = True
                break
            if _HEADER_RE.match(ln):
                # Next block began without an END — stop here, leave i on the
                # header so the outer loop re-reads it.
                break
            ms = _SECTION_RE.match(ln)
            if ms:
                if ms.group(1).upper() == "FIND":
                    find_lines = find_lines or []
                    cur = find_lines
                else:
                    repl_lines = repl_lines or []
                    cur = repl_lines
                i += 1
                continue
            if cur is not None:
                cur.append(ln)
            # lines before any section marker (stray prose inside a block) are ignored
            i += 1

        if not closed and find_lines is None and repl_lines is None:
            # Header with no usable body and no END: malformed, drop it.
            continue

        find = "\n".join(find_lines) if find_lines is not None else ""
        replace = "\n".join(repl_lines) if repl_lines is not None else ""
        edits.append({"op": op, "find": find, "replace": replace})
    return edits


def _count_occurrences(haystack: str, needle: str) -> int:
    if not needle:
        return 0
    count = start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            return count
        count += 1
        start = idx + 1        # overlapping count -> 2+ still reads as ambiguous
    return count


def _append_index(program: str) -> int:
    """Character offset at which APPEND inserts: at the start of the LAST line
    beginning with `svg ` (the assignment); else end-of-file."""
    last = -1
    pos = 0
    for ln in program.split("\n"):
        if _SVG_ASSIGN_LINE_RE.match(ln):
            last = pos
        pos += len(ln) + 1     # +1 for the '\n' that split removed
    return last if last >= 0 else len(program)


def apply_edits(program_text: str, ops):
    """Apply `ops` (as produced by parse_edits) IN ORDER to program_text.

    Returns {new_text, applied, errors}:
      applied : count of ops that mutated the program.
      errors  : list of {index, op, reason, snippet} for ops NOT applied.
                reason in {not_found, ambiguous, empty}.
    An op is never silently dropped: it either mutates the text or yields an error.
    """
    text = program_text if program_text is not None else ""
    applied = 0
    errors: list[dict] = []

    for index, ed in enumerate(ops):
        op = ed.get("op")
        find = ed.get("find", "") or ""
        replace = ed.get("replace", "") or ""

        if op == "APPEND":
            ins = replace
            if ins == "":
                errors.append({"index": index, "op": op, "reason": "empty",
                               "snippet": ""})
                continue
            at = _append_index(text)
            block = ins if ins.endswith("\n") else ins + "\n"
            text = text[:at] + block + text[at:]
            applied += 1
            continue

        # All remaining ops require a FIND.
        if find == "":
            errors.append({"index": index, "op": op, "reason": "empty",
                           "snippet": ""})
            continue
        occ = _count_occurrences(text, find)
        if occ == 0:
            errors.append({"index": index, "op": op, "reason": "not_found",
                           "snippet": find[:80]})
            continue
        if occ > 1:
            errors.append({"index": index, "op": op, "reason": "ambiguous",
                           "snippet": find[:80]})
            continue

        idx = text.find(find)
        if op == "REPLACE":
            text = text[:idx] + replace + text[idx + len(find):]
            applied += 1
        elif op == "DELETE":
            text = text[:idx] + text[idx + len(find):]
            applied += 1
        elif op == "INSERT_AFTER":
            end = idx + len(find)
            ins = replace
            # Insert on the line(s) immediately AFTER the anchor: if the anchor is
            # not already at a line boundary, start a new line first; otherwise we
            # land right after the anchor's trailing newline.
            if end < len(text) and text[end] == "\n":
                # anchor ends exactly at a newline -> insert after that newline
                prefix = text[:end + 1]
                suffix = text[end + 1:]
                block = ins if ins.endswith("\n") else ins + "\n"
                text = prefix + block + suffix
            else:
                # anchor ends mid-line (e.g. a substring) -> break the line
                block = "\n" + ins
                text = text[:end] + block + text[end:]
            applied += 1
        else:
            # parse_edits only yields known ops; defensive.
            errors.append({"index": index, "op": op, "reason": "not_found",
                           "snippet": find[:80]})

    return {"new_text": text, "applied": applied, "errors": errors}


class SvgRepl:
    """Holds the current program text and mutates it via edit-suite blocks.

    The artifact PERSISTS across .apply() calls — the whole point of an
    edit-in-REPL (vs re-emitting the full program each turn). Each .apply returns
    the same {new_text, applied, errors} shape; .program is the current text.
    """

    def __init__(self, program_text: str = ""):
        self._text = program_text if program_text is not None else ""

    @property
    def program(self) -> str:
        return self._text

    def apply(self, edits_text: str):
        ops = parse_edits(edits_text)
        res = apply_edits(self._text, ops)
        self._text = res["new_text"]
        return res


# --------------------------------------------------------------------------- #
# SERVERLESS self-test. No LM, no render, no network — pure text assertions.   #
# --------------------------------------------------------------------------- #
def _selftest() -> int:
    start_program = (
        "import math\n"
        "W, H = 320, 180\n"
        "parts = []\n"
        "parts.append('<rect x=\"0\" y=\"0\" width=\"320\" height=\"180\" fill=\"#222\"/>')\n"
        "parts.append('<circle cx=\"160\" cy=\"90\" r=\"40\" fill=\"#e44\"/>')\n"
        "svg = '<svg xmlns=\"http://www.w3.org/2000/svg\">' + ''.join(parts) + '</svg>'"
    )

    # The model's reply: prose + fences around four edit blocks, mixed marker
    # runes and case, exercising tolerant parsing. Block 4's FIND does not exist.
    edits_text = '''Sure — here are my edits to push the residual down.

```
### EDIT replace ###
--- FIND ---
parts.append('<circle cx="160" cy="90" r="40" fill="#e44"/>')
--- REPLACE ---
parts.append('<circle cx="160" cy="90" r="55" fill="#e44"/>')
### END ###
```

Now add an outline ring just after the background:

*** EDIT INSERT_AFTER ***
--- find ---
parts.append('<rect x="0" y="0" width="320" height="180" fill="#222"/>')
--- replace ---
parts.append('<circle cx="160" cy="90" r="70" fill="none" stroke="#fff"/>')
*** END ***

And a caption, appended before the svg assignment:

=== EDIT append ===
--- REPLACE ---
parts.append('<text x="10" y="170" fill="#fff">hi</text>')
=== END ===

Finally tweak a colour that isn't actually there (should error, not crash):

### EDIT REPLACE ###
--- FIND ---
parts.append('<ellipse cx="1" cy="2" rx="3" ry="4" fill="#0f0"/>')
--- REPLACE ---
parts.append('<ellipse cx="1" cy="2" rx="3" ry="4" fill="#00f"/>')
### END ###
'''

    repl = SvgRepl(start_program)
    res = repl.apply(edits_text)

    expected = (
        "import math\n"
        "W, H = 320, 180\n"
        "parts = []\n"
        "parts.append('<rect x=\"0\" y=\"0\" width=\"320\" height=\"180\" fill=\"#222\"/>')\n"
        "parts.append('<circle cx=\"160\" cy=\"90\" r=\"70\" fill=\"none\" stroke=\"#fff\"/>')\n"
        "parts.append('<circle cx=\"160\" cy=\"90\" r=\"55\" fill=\"#e44\"/>')\n"
        "parts.append('<text x=\"10\" y=\"170\" fill=\"#fff\">hi</text>')\n"
        "svg = '<svg xmlns=\"http://www.w3.org/2000/svg\">' + ''.join(parts) + '</svg>'"
    )

    checks = []

    def check(name: str, cond: bool, detail: str = ""):
        checks.append(cond)
        status = "PASS" if cond else "FAIL"
        line = f"[{status}] {name}"
        if detail and not cond:
            line += f"  -- {detail}"
        print(line)

    # parse tolerance: exactly 4 blocks recovered from prose+fences+mixed runes
    ops = parse_edits(edits_text)
    check("parse: 4 edit blocks recovered from prose/fences/mixed-runes",
          len(ops) == 4, f"got {len(ops)}: {[o['op'] for o in ops]}")
    check("parse: ops canonicalised to REPLACE/INSERT_AFTER/APPEND/REPLACE",
          [o["op"] for o in ops] == ["REPLACE", "INSERT_AFTER", "APPEND", "REPLACE"],
          str([o["op"] for o in ops]))

    # apply semantics
    check("apply: exactly 3 ops applied", res["applied"] == 3,
          f"applied={res['applied']}")
    check("apply: exactly 1 error", len(res["errors"]) == 1,
          f"errors={res['errors']}")
    check("apply: the single error is reason=not_found",
          len(res["errors"]) == 1 and res["errors"][0]["reason"] == "not_found",
          str(res["errors"]))
    check("apply: error carries {index, op, reason, snippet}",
          len(res["errors"]) == 1
          and set(res["errors"][0]) == {"index", "op", "reason", "snippet"}
          and res["errors"][0]["index"] == 3
          and res["errors"][0]["op"] == "REPLACE",
          str(res["errors"]))

    # the load-bearing assertion: exact resulting program text
    if res["new_text"] != expected:
        print("---- EXPECTED ----")
        print(expected)
        print("---- ACTUAL ----")
        print(res["new_text"])
    check("apply: new_text equals the explicit expected program",
          res["new_text"] == expected)

    # persistence: the SvgRepl now holds the mutated text
    check("repl: .program persists the mutated text",
          repl.program == expected)

    # APPEND landed BEFORE the `svg = ...` assignment (not at EOF)
    txt = repl.program
    check("append: text() element inserted before the svg assignment line",
          txt.index("<text") < txt.index("svg = '<svg"))

    # INSERT_AFTER placed the ring immediately after the background rect, and
    # BEFORE the (now resized) red circle.
    check("insert_after: ring placed right after the background rect",
          txt.index('fill="none" stroke="#fff"') > txt.index('fill="#222"')
          and txt.index('fill="none" stroke="#fff"') < txt.index('r="55"'))

    # ---- extra apply-semantics coverage (still serverless) ----
    # ambiguous FIND (appears twice) -> reason=ambiguous, op not applied
    amb_prog = "a = 1\nx = 0\nx = 0\nb = 2"
    amb = apply_edits(amb_prog, [{"op": "REPLACE", "find": "x = 0", "replace": "x = 9"}])
    check("ambiguous: duplicate FIND -> reason=ambiguous, 0 applied",
          amb["applied"] == 0 and len(amb["errors"]) == 1
          and amb["errors"][0]["reason"] == "ambiguous"
          and amb["new_text"] == amb_prog,
          str(amb))

    # empty REPLACE value is allowed (deletes the found text via REPLACE)
    emp = apply_edits("keep\nDROPME\nkeep2",
                      [{"op": "REPLACE", "find": "DROPME\n", "replace": ""}])
    check("replace-empty: REPLACE with empty value removes the found text",
          emp["applied"] == 1 and emp["new_text"] == "keep\nkeep2"
          and emp["errors"] == [],
          str(emp))

    # DELETE removes the found substring
    dele = apply_edits("alpha\nbeta\ngamma",
                       [{"op": "DELETE", "find": "beta\n", "replace": ""}])
    check("delete: removes the found substring",
          dele["applied"] == 1 and dele["new_text"] == "alpha\ngamma",
          str(dele))

    # APPEND with EOF fallback (no svg-assignment line present)
    app_eof = apply_edits("a = 1\nb = 2",
                          [{"op": "APPEND", "find": "", "replace": "c = 3"}])
    check("append-eof: no svg assignment -> appended at EOF",
          app_eof["applied"] == 1 and app_eof["new_text"] == "a = 1\nb = 2c = 3\n",
          str(app_eof))

    # empty-where-required: REPLACE with no FIND -> reason=empty
    req = apply_edits("z = 0", [{"op": "REPLACE", "find": "", "replace": "z = 1"}])
    check("empty-required: REPLACE with no FIND -> reason=empty, 0 applied",
          req["applied"] == 0 and len(req["errors"]) == 1
          and req["errors"][0]["reason"] == "empty",
          str(req))

    ok = all(checks)
    print(f"\n{'ALL PASS' if ok else 'SOME FAILED'}: {sum(checks)}/{len(checks)} checks")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
