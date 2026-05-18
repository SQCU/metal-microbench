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
    ("trope_density",       "1: novel / specific / texture-this-writer-couldn't-have-anticipated · 5: fully tropey / anonymous-genre-material / no individuating texture"),
    # ── "Plays nicely with others" triplet (added 2026-05-17) ────────────
    # DESCRIPTIVE, NOT NORMATIVE. We usually want most agents median or
    # above on these — but some agents are intentionally floor-low and
    # that's a feature (and often funny). The Rock as a USER-agent in
    # a multi-user chat would score floor on all three by design and
    # that radical-bio-adherence IS the comedic point. Score what's
    # there; don't push toward 5.
    ("ludic_engagement",    "1: bio-anchored / self-presentation regardless of scene cues (the rock-as-user-agent would be hilariously low) · 5: scene-coupled / actively engages with environmental cues, objects, partner's frame"),
    ("user_multipolarity",  "1: dyadic-only / addresses one partner (often the assistant) regardless of who else is in the room · 5: multi-pole / treats each participant as a separately-addressable counterparty with own stakes"),
    ("other_awareness",     "1: porous / mirrors-or-becomes the partner's frame, bio, or motivation (a feature for some designs, a bug for others) · 5: anchored / stays in own bio + own motivation while still acknowledging the partner's frame"),
]
AXIS_NAMES = [a for a, _ in LIKERT_AXES]
N_AXES = len(LIKERT_AXES)
