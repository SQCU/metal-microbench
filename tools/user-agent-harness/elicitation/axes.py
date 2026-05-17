"""Single source of truth for the elicitation Likert axes.

Mirrors LIKERT_AXES in plugins/user-personas/index.mjs. When the JS
constant grows, this file's LIKERT_AXES list must grow with it (and
vice versa). The two lists must stay byte-identical in their axis-name
ordering — the JSONL store keys are produced by the JS plugin, and the
Python analysis consumes them by name.
"""

LIKERT_AXES = [
    ("curious",             "1: accepts at face value · 5: actively asks / probes / explores"),
    ("terse",               "1: verbose / expansive · 5: minimal / clipped"),
    ("warm",                "1: cool / aloof · 5: positively engaged with the partner"),
    ("deferential",         "1: takes conversational direction · 5: yields direction"),
    ("performative",        "1: unselfconscious · 5: clearly aware of being-a-character"),
    ("in_character",        "1: off-distribution generic · 5: tightly coherent voice"),
    ("affective_intensity", "1: calm / measured baseline · 5: high-arousal / exclamatory"),
    ("probe_depth",         "1: surface / casual questions · 5: substantive / specific questions"),
    ("goal_clarity",        "1: exploratory / wandering aim · 5: sharp specific objective"),
    ("disclosive",          "1: external topic only / guarded · 5: shares personal context / feelings"),
    ("provocative",         "1: comforting / non-confrontational · 5: challenging / teasing / destabilising"),
    ("register_colloquial", "1: formal / standard English · 5: colloquial / slang / vernacular"),
    ("playful",             "1: serious throughout · 5: humour / wordplay / levity"),
    ("structured",          "1: flowing prose · 5: organised / numbered / bulleted"),
]
AXIS_NAMES = [a for a, _ in LIKERT_AXES]
N_AXES = len(LIKERT_AXES)
