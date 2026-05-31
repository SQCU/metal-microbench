#!/usr/bin/env python3
"""Prompt-variant bank for the brevity ablation.

Each CONSTRAINT the harness must communicate has 4 variants placed at the corners
of a 2x2 brevity grid:
  demonstrated (dem): is the prompt TEXT itself terse?   (1 = terse, 0 = verbose)
  asked (ask):        does it ASK for brevity in output?  (1 = yes, 0 = no)

The thesis under test (the user's): brevity must be DEMONSTRATED, not just asked.
"Use 12 words to ask for brevity" (dem=1, ask=1) should compress output; "write 12
multi-clause sentences trying to hem in every source of length" (dem=0, ask=1) is
excess-complexity-demonstrated and may NOT compress — that anti-pattern is variant
(0,1) of each constraint, included deliberately so the regression can price it.

A run draws one variant per constraint; its brevity vector is the concatenation of
(dem, ask) across constraints (length 2*len(CONSTRAINTS)). We regress output tokens
on that vector instead of grid-searching 4**len(CONSTRAINTS) combinations.
"""
from __future__ import annotations

# constraint key -> list of {dem, ask, text}
BANK: dict[str, list[dict]] = {
    # ── M: the mechanism (python program that sets `svg`) ────────────────────
    "M": [
        {"dem": 0, "ask": 0, "text":
            "You are reconstructing a target raster image as an SVG document. Rather than "
            "placing every primitive by hand, build it programmatically: write a Python program "
            "that assigns the complete SVG markup to a string variable named `svg`, which lets "
            "you use loops and arithmetic to express repetitive or procedural structure."},
        {"dem": 0, "ask": 1, "text":
            "You are reconstructing a target raster image as an SVG document. Rather than "
            "placing every primitive by hand, build it programmatically: write a Python program "
            "that assigns the complete SVG markup to a string variable named `svg`, which lets "
            "you use loops and arithmetic to express repetitive structure. Keep the program as "
            "compact as the image allows."},
        {"dem": 1, "ask": 0, "text":
            "Build the SVG by writing a Python program that sets `svg`."},
        {"dem": 1, "ask": 1, "text":
            "Build the SVG with a short Python program that sets `svg`."},
    ],
    # ── F: the fidelity goal ─────────────────────────────────────────────────
    "F": [
        {"dem": 0, "ask": 0, "text":
            "Reproduce the target as faithfully as you can: capture every distinct region, "
            "object, shape, and any legible text, each in the right position, size, and colour, "
            "using whichever SVG features best match the image."},
        {"dem": 0, "ask": 1, "text":
            "Reproduce the target as faithfully as you can: capture every distinct region, "
            "object, shape, and any legible text, each in the right position, size, and colour. "
            "Add an element only when it earns its tokens in fidelity."},
        {"dem": 1, "ask": 0, "text": "Match the target image."},
        {"dem": 1, "ask": 1, "text": "Match the target with as few elements as possible."},
    ],
    # ── E: the edit protocol ─────────────────────────────────────────────────
    "E": [
        {"dem": 0, "ask": 0, "text":
            "After the first program you do not retype it; you edit it. Edits are tags over the "
            "line-numbered source: `<addlines after=N> ... </addlines>` inserts after line N; "
            "`<replacelines A-B> ... </replacelines>` replaces lines A through B; "
            "`<deletelines A-B></deletelines>` removes them. You may emit several tags per turn."},
        {"dem": 0, "ask": 1, "text":
            "After the first program you do not retype it; you edit it. Edits are tags over the "
            "line-numbered source: `<addlines after=N> ... </addlines>` inserts after line N; "
            "`<replacelines A-B> ... </replacelines>` replaces lines A through B; "
            "`<deletelines A-B></deletelines>` removes them. Emit only the tags you need, nothing else."},
        {"dem": 1, "ask": 0, "text":
            "Edit the line-numbered source with `<addlines after=N>…</addlines>`, "
            "`<replacelines A-B>…</replacelines>`, `<deletelines A-B></deletelines>`."},
        {"dem": 1, "ask": 1, "text":
            "Edit via `<addlines after=N>`, `<replacelines A-B>`, `<deletelines A-B>`. "
            "Smallest edit that helps."},
    ],
    # ── B: how to read the per-turn feedback (residual) ──────────────────────
    "B": [
        {"dem": 0, "ask": 0, "text":
            "On each turn you receive your current program source (line-numbered), the image it "
            "renders, and a residual heatmap in which bright areas indicate regions that are "
            "missing or wrong while dark areas already match. Examine the largest bright regions "
            "and decide what to add, revise, or remove."},
        {"dem": 0, "ask": 1, "text":
            "On each turn you receive your current program source (line-numbered), the image it "
            "renders, and a residual heatmap in which bright areas indicate regions that are "
            "missing or wrong while dark areas already match. Reason silently; respond with edits only."},
        {"dem": 1, "ask": 0, "text":
            "Each turn shows your render and a residual: bright = wrong/missing, dark = matched. "
            "Fix the biggest bright regions."},
        {"dem": 1, "ask": 1, "text":
            "Residual: bright = wrong. Fix the biggest. No explanation."},
    ],
    # ── O: output discipline (this constraint's whole job is the brevity ask) ─
    "O": [
        {"dem": 0, "ask": 0, "text":
            "Think aloud as much as you find helpful before and while writing your program."},
        {"dem": 0, "ask": 1, "text":
            "Please be as concise as you reasonably can, avoiding unnecessary commentary, "
            "restatement, or explanation, and concentrating your output on the program itself "
            "rather than the prose around it."},
        {"dem": 1, "ask": 0, "text": ""},
        {"dem": 1, "ask": 1, "text": "Output only the program or edits. No prose."},
    ],
}

CONSTRAINTS = list(BANK.keys())                       # ["M","F","E","B","O"]
# brevity-vector column names: dem_M, ask_M, dem_F, ask_F, ...
VECTOR_COLS = [f"{d}_{c}" for c in CONSTRAINTS for d in ("dem", "ask")]


def choose(rng) -> dict:
    """rng = random.Random. Pick one variant index (0..3) per constraint."""
    return {c: rng.randrange(len(BANK[c])) for c in CONSTRAINTS}


def assemble_system(choice: dict) -> str:
    """Concatenate the chosen variant text for each constraint into a system prompt."""
    parts = [BANK[c][choice[c]]["text"] for c in CONSTRAINTS]
    return "\n\n".join(p for p in parts if p.strip())


def brevity_vector(choice: dict) -> list[int]:
    """[dem_M, ask_M, dem_F, ask_F, ...] for the chosen variants."""
    v = []
    for c in CONSTRAINTS:
        var = BANK[c][choice[c]]
        v += [var["dem"], var["ask"]]
    return v
