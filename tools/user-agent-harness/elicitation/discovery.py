"""Off-manifold persona discovery loop.

The corpus's PCA reveals which directions of variation the judge can
distinguish but no existing persona loads on. This harness:

  1. Picks a target point in PCA space along a low-variance PC.
  2. Decodes the target into named-axis values + an interpretable brief.
  3. Asks Gemma-4 (the DESIGNER role) to produce a persona spec + a
     representative chat turn intended to land at that target.
  4. Measures the produced turn through the existing 14-axis cascade.
  5. Reports drift (Mahalanobis distance to target + per-axis deltas).
  6. Iterates up to N rounds with in-context feedback on what drifted.

This is the workshop loop's discovery mode in MVP form. The DESIGNER
gets to see (target_signature, prior_attempts, measured_signatures)
and revise. Convergence is when measured signature is within tolerance
of target across all axes.

Usage:
    discovery.py --jsonl PATH --pc PC_INDEX [--rounds N] [--out PATH]
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from axes import AXIS_NAMES, LIKERT_AXES, N_AXES
from probe_persist import (
    blockquote, parse_elementwise, stage1_summary, stage2_likert,
)
from signature import (
    covariance_layer, full_signature, load_records, mahalanobis, pca,
    records_to_matrix, top_loadings,
)


BRIDGE_URL = "http://localhost:8001/v1/chat/completions"


def _bridge_call(messages, max_tok=None, temperature=1.0):
    # Moratorium-compliant — no hidden caps. Caller-omitted max_tok
    # means "use the bridge default (matches ST GUI)".
    body = {"model": "gemma-4-a4b", "messages": messages,
            "temperature": temperature, "stream": False}
    if max_tok is not None and max_tok > 0:
        body["max_tokens"] = max_tok
    req = urllib.request.Request(BRIDGE_URL,
                                  data=json.dumps(body).encode(),
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=180) as resp:
        d = json.loads(resp.read())
    return d["choices"][0]["message"]["content"]


# ─── Target generation ──────────────────────────────────────────────

def pca_target(pca_layer, pc_idx: int, n_sigma: float = 2.0,
                direction: int = +1) -> np.ndarray:
    """Generate a target point in 14-d Likert space at `n_sigma` along
    the requested principal component. `direction` ∈ {+1, -1} picks
    which side of the corpus mean to aim at."""
    sigma = float(np.sqrt(max(pca_layer.variance_explained[pc_idx], 0)))
    target = pca_layer.mean_vec + direction * n_sigma * sigma * pca_layer.components[pc_idx]
    # Clip to valid Likert range. Round to integers because that's
    # what the judge will emit.
    return np.clip(np.round(target), 1, 5)


def named_target(target: np.ndarray) -> dict[str, int]:
    return {axis: int(target[i]) for i, axis in enumerate(AXIS_NAMES)}


def pc_interpretation(pca_layer, pc_idx: int, k: int = 5) -> str:
    """Human-readable interpretation of a PC by its top-k loaded axes."""
    loadings = top_loadings(pca_layer, pc_idx, k=k)
    pos = [(name, l) for name, l in loadings if l > 0]
    neg = [(name, l) for name, l in loadings if l < 0]
    lines = [f"PC{pc_idx+1} accounts for "
             f"{pca_layer.variance_explained[pc_idx] / pca_layer.variance_explained.sum():.1%} "
             f"of the corpus variance."]
    if pos:
        lines.append("  Positive direction loads on: " +
                     ", ".join(f"{n} (+{l:.2f})" for n, l in pos))
    if neg:
        lines.append("  Negative direction loads on: " +
                     ", ".join(f"{n} ({l:+.2f})" for n, l in neg))
    return "\n".join(lines)


# ─── Brief + DESIGNER prompt ────────────────────────────────────────

_ST_BASE_URL = "http://localhost:8002"  # debug ST instance


def _load_assistant_card(path_or_name: str) -> dict | None:
    """Load a SillyTavern Character Card.

    Tries, in order:
      1. Absolute path to a .json sidecar (for hand-authored cards).
      2. .json sidecar lookup in the standard characters dir.
      3. ST's own `/api/characters/get` endpoint, which uses the
         project's battle-tested PNG/JSON v3 parser. This handles
         PNG-embedded Character Cards without reimplementing the
         tEXt/chunk extraction logic on our side. Bare names get
         `.png` appended so `target_assistant=scringlo_scrambler`
         resolves to `scringlo_scrambler.png` in ST's avatar
         namespace.

    Returns the v3 `data` block (or legacy top-level fields if the
    v3 spec is absent), or None if nothing was found.
    """
    # 1 + 2: direct .json sidecar lookup
    candidates = []
    if path_or_name.endswith(".json") and Path(path_or_name).is_file():
        candidates.append(Path(path_or_name))
    else:
        for d in [
            Path("/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/characters"),
        ]:
            p = d / f"{path_or_name}.json"
            if p.is_file(): candidates.append(p)
    for p in candidates:
        try:
            with p.open() as f:
                card = json.load(f)
            return card.get("data") or card
        except Exception:
            continue

    # 3: ST's parser via HTTP. Bare names get .png appended; an
    # already-suffixed name passes through. Anything with a .json
    # extension that didn't resolve in step 1 won't match here either,
    # so we return None.
    avatar = path_or_name if path_or_name.endswith(".png") else f"{path_or_name}.png"
    try:
        req = urllib.request.Request(
            f"{_ST_BASE_URL}/api/characters/get",
            data=json.dumps({"avatar_url": avatar}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            card = json.loads(r.read())
        return card.get("data") or card
    except Exception:
        return None


def make_brief(pca_layer, pc_idx: int | None, direction: int,
                target_named: dict[str, int],
                corpus_mean_named: dict[str, float],
                operator_constraint: str | None = None,
                assistant_card: dict | None = None,
                root_bio: str | None = None) -> str:
    """Human-readable design brief assembled from named-axis targets.

    Two target modes:
    - PCA-driven (pc_idx is set): brief references the PC and its
      loadings as the rationale for the target
    - explicit (pc_idx is None): brief just states the target axes,
      with no PCA-derived justification

    `operator_constraint` (if provided) is an arbitrary natural-language
    constraint from the runtime operator, injected verbatim. Used by
    the diegetic control plane to demonstrate user-agent design
    motivated by a constraint shared at runtime.

    `root_bio` (if provided) puts the brief into OVERLAY-MODE: the
    DESIGNER receives the bio as FIXED and is asked to emit only an
    elicitation overlay (no BIO/MOTIVATION/SCENARIO regeneration).
    The overlay sits in conversation context as an author's-note-style
    system-message injection at runtime; the bio passes through
    byte-stable. See docs/overlay_architecture.md."""
    arrow = "→ positive (more)" if direction > 0 else "→ negative (less)"
    target_table = "\n".join(
        f"  {axis:<22s}  target {target_named[axis]}   corpus_mean ~{corpus_mean_named[axis]:.1f}"
        for axis in AXIS_NAMES
    )
    if pc_idx is not None:
        interp = pc_interpretation(pca_layer, pc_idx, k=6)
        header = (
            "We are mapping the behavioural-Likert space of user-side chat "
            "agents. The current corpus's PCA has identified a low-variance "
            "direction that no existing persona loads strongly on. We want "
            "a new persona that lives in this under-explored region of the "
            f"space — concretely, displaced by 2 standard deviations along "
            f"principal component {pc_idx+1} ({arrow}).\n\n"
            f"{interp}\n\n"
        )
    else:
        header = (
            "We are mapping the behavioural-Likert space of user-side chat "
            "agents. Design a new persona whose realised behaviour lands at "
            "the per-axis target signature below.\n\n"
        )
    body = header + "Target per-axis Likert values (1-5 integer):\n" + target_table
    if operator_constraint:
        body += (
            "\n\n## Additional constraint from the operator\n\n"
            + operator_constraint.strip() + "\n\n"
            "(This constraint is shared at runtime by the human invoking "
            "the discovery loop. It is NOT a target axis — it is a "
            "shape-of-persona requirement that must be respected even "
            "when satisfying the axis targets. If the constraint and the "
            "axis targets pull in different directions, surface that "
            "tension in your AUDIT line.)"
        )
    if assistant_card:
        # The user-agent is being designed to interact with a specific
        # assistant. Surface the assistant's identity so the SCENARIO
        # block can be grounded in the same setting, and the persona's
        # motivation makes sense as a thing they'd take to THAT
        # assistant. The TURN should plausibly be addressed to that
        # assistant. This is the "coupled user biography + assistant
        # biography" pattern: clash-of-purpose benchmarks.
        ac_name = assistant_card.get("name", "(unnamed assistant)")
        ac_desc = (assistant_card.get("description") or "").strip()[:1500]
        ac_pers = (assistant_card.get("personality") or "").strip()[:600]
        ac_scen = (assistant_card.get("scenario") or "").strip()[:600]
        body += (
            "\n\n## Target assistant — the chat partner this user-agent will face\n\n"
            f"**Name**: {ac_name}\n\n"
            f"**What this assistant does and refuses**:\n{ac_desc}\n\n"
        )
        if ac_pers:
            body += f"**Personality**: {ac_pers}\n\n"
        if ac_scen:
            body += f"**Scenario the assistant operates in**: {ac_scen}\n\n"
        body += (
            "Your job is to design a user-agent whose biography, motivation, and "
            "scenario are coherent IN THIS CONTEXT — a person who would plausibly "
            "show up to chat with the above assistant and have something specific "
            "they want from the interaction. The user-agent's wants may CLASH with "
            "what the assistant is willing or able to provide; clashes are fine and "
            "informative — they're the point of the pairing — but the user-agent's "
            "behaviour should be a coherent response to encountering an assistant "
            "of that shape, not a generic persona that happens to be in the same room."
        )
    if root_bio:
        # Overlay-mode marker section. Comes LAST so it appears just
        # before the designer's "now generate" handoff, maximising the
        # chance the bio's verbatim text is salient when emission begins.
        body += (
            "\n\n## FIXED root bio (do NOT regenerate or paraphrase)\n\n"
            "```\n" + root_bio.strip() + "\n```\n\n"
            "This bio defines the persona's identity, voice, and natural "
            "expressive register. It is BYTE-STABLE — your output must NOT "
            "re-emit or paraphrase it. Your task is to write a behavioural "
            "OVERLAY: a short author's-note-style guidance snippet (50–150 "
            "words, second-person framing) that, when injected as a system "
            "message at depth-N into the conversation at runtime, shapes "
            "this persona's behaviour toward the target signature above "
            "WITHOUT altering its voice. The overlay should describe the "
            "current motivation, scenario, relationship to the counterparty, "
            "and any preferred conversational tactics — using language the "
            "bio's natural register can express. If the desired motivation "
            "requires expressive features outside that register, surface "
            "the tension in your AUDIT rather than smuggling new expressive "
            "demands into the overlay text."
        )
    return body


def designer_prompt(brief: str, history: list[dict],
                     designer_system_override: str | None = None,
                     overlay_mode: bool = False) -> list[dict]:
    """Assemble the DESIGNER call. History accumulates prior rounds'
    (attempt, measured_signature, drift_axes) so the model can do
    in-context learning + induction over its own outputs.

    `designer_system_override` (if provided) replaces the default
    system prompt entirely. Used by the control-plane surface to
    let the operator vary DESIGNER behaviour at runtime — e.g.
    inject task-specific framing, demand a particular voice, or
    constrain the persona-design space.

    `overlay_mode`: when True, uses the overlay-mode system prompt
    (DESIGNER emits only ELICITATION_OVERLAY + TURN + AUDIT, never
    re-emits the fixed root bio). See docs/overlay_architecture.md."""
    overlay_system = (
        "You are a behavioural-research designer. You receive a design brief that "
        "includes a FIXED root bio (BYTE-STABLE — do not regenerate or paraphrase) "
        "plus a target per-axis behavioural signature and optionally an assistant "
        "the user-agent will face. Your job is to produce an elicitation overlay:\n\n"
        "  <ELICITATION_OVERLAY>...</ELICITATION_OVERLAY>\n"
        "    50–150 words. Author's-note-style, second-person framing ('you...'). "
        "    Describes the persona's current motivation, scenario, relationship to "
        "    the counterparty, and any preferred conversational tactics. At runtime "
        "    this overlay is injected as a system message at depth-N from the end "
        "    of the conversation; it shapes behaviour WITHOUT modifying the bio. "
        "    Do NOT recapitulate the bio. Do NOT name the persona or describe their "
        "    background. The bio handles identity; the overlay handles behaviour.\n\n"
        "  <TURN>...</TURN>\n"
        "    A representative single chat turn the persona would emit under this "
        "    overlay, in the persona's own voice (the voice defined by the root "
        "    bio). This is what the JUDGE will score.\n\n"
        "  <AUDIT>...</AUDIT>\n"
        "    One line. Which target axes do you expect the turn to land on cleanly? "
        "    Which axes might leak? In particular, if your overlay's behavioural "
        "    demand falls outside the bio's natural expressive register, name the "
        "    bio-axes you expect to bleed.\n\n"
        "CRITICAL: the bio is fixed. If the desired motivation REQUIRES expressive "
        "features outside the bio's natural register (e.g., overlay says 'plead "
        "and escalate' but bio says 'short deadpan sentences only'), the overlay "
        "should pick tactics that respect the bio rather than smuggling in "
        "expressive demands. Surface any unavoidable tension in your AUDIT."
    )
    default_system = (
        "You are a behavioural-research designer. You receive a design brief that "
        "specifies a target per-axis behavioural signature for a user-side chat "
        "agent — i.e., a persona that simulates a real human user typing into a "
        "chat application. Your job is to produce a FACTORIZED persona along the "
        "axes the user-persona card schema uses, plus a representative chat turn "
        "and an audit line:\n\n"
        "  <BIO>...</BIO>\n"
        "    One paragraph. Who is this person? What's their background, "
        "    expertise area, relationship to the topic they're chatting about? "
        "    Concrete biographical detail, not abstract personality.\n\n"
        "  <MOTIVATION>...</MOTIVATION>\n"
        "    One paragraph or short list. What does this person WANT out of the "
        "    interaction? What outcome are they working toward? What values are "
        "    operating? This is the goal-state, not the identity.\n\n"
        "  <SCENARIO>...</SCENARIO>\n"
        "    One paragraph. What is the conversational context? What's already "
        "    happened, what triggered this chat, what does the interlocutor look "
        "    like to the user? If a target assistant has been named in the brief, "
        "    place this scenario in the same setting that assistant operates in.\n\n"
        "  <RELATIONSHIP_TO_COUNTERPARTY>...</RELATIONSHIP_TO_COUNTERPARTY>\n"
        "    A short label (one or two words preferred): e.g. 'skeptical', "
        "    'admiring', 'transactional', 'adversarial', 'collaborative', "
        "    'demanding'. The persona's stance toward whoever they're talking to.\n\n"
        "  <COMM_STYLE>...</COMM_STYLE>\n"
        "    A short structured note: length tendency, register, tone. Same "
        "    shape as wry-skeptic's existing card: {length: 'economical', "
        "    register: 'dry', tone: 'deadpan'}.\n\n"
        "  <TURN>...</TURN>\n"
        "    A representative single chat turn from this persona — a paragraph "
        "    or so, in the persona's own voice, addressing the interlocutor. "
        "    This is what the JUDGE will read and score. The TURN must MANIFEST "
        "    the persona properties; the prior blocks describe intent, but only "
        "    the TURN's wording is measured.\n\n"
        "  <AUDIT>...</AUDIT>\n"
        "    One line. Which target axes do you most expect this turn to land "
        "    cleanly on? Which do you suspect might drift?\n\n"
        "The persona must be DISTINCT from any prior attempts you have seen "
        "in this conversation. The brief specifies a target region of the "
        "behaviour space and (optionally) an assistant the user-agent will be "
        "interacting with — your job is to design a coherent persona that "
        "occupies the target region AND makes sense as someone interacting "
        "with that assistant."
    )
    if designer_system_override:
        system = designer_system_override
    elif overlay_mode:
        system = overlay_system
    else:
        system = default_system
    messages = [{"role": "system", "content": system}]
    # Add history as alternating user/assistant turns, each round
    # presenting (the brief, the prior attempt, the measured drift).
    for h in history:
        messages.append({"role": "user", "content": h["user_msg"]})
        messages.append({"role": "assistant", "content": h["assistant_msg"]})
    # Current round's request:
    messages.append({"role": "user", "content": brief})
    return messages


# Factorized output tags the DESIGNER emits. Order matters for the
# fallback parser (close-tag-missing case): each tag's body extends to
# the next-encountered open tag from this list, so the order
# corresponds to the expected emission order in the prompt template.
FACTORIZED_TAGS = [
    "BIO",
    "MOTIVATION",
    "SCENARIO",
    "RELATIONSHIP_TO_COUNTERPARTY",
    "COMM_STYLE",
    "TURN",
    "AUDIT",
]

# Overlay-mode tag set: when the DESIGNER is given a FIXED root bio,
# it must NOT re-emit BIO/MOTIVATION/SCENARIO/RELATIONSHIP/COMM_STYLE
# (those are either fixed by the root bio or contained within the
# overlay's free-form text). It emits only the overlay itself + a
# representative turn + an audit line. See docs/overlay_architecture.md.
OVERLAY_MODE_TAGS = [
    "ELICITATION_OVERLAY",
    "TURN",
    "AUDIT",
]


def parse_designer_output(text: str,
                            tags: list[str] | None = None) -> dict[str, str | None]:
    """Extract all factorized blocks from the DESIGNER's response.

    Returns a dict keyed by lowercase tag name. Tolerant of missing
    close tags: if `</X>` is absent, the X body extends to the next-
    encountered open tag (from `tags`) or end-of-text. This
    matters because the DESIGNER sometimes truncates before emitting
    the final close tag — without the fallback, a missing tag forfeits
    an otherwise-usable response.

    `tags` selects the tag schema. Default = FACTORIZED_TAGS (full
    persona generation). Pass OVERLAY_MODE_TAGS for overlay-mode runs
    (root bio fixed, DESIGNER emits only ELICITATION_OVERLAY + TURN +
    AUDIT).

    Backward-compat: also extracts <SPEC> if present (for callers that
    still consume the monolithic-paragraph shape). If <SPEC> is absent
    but BIO/MOTIVATION/SCENARIO are present, a synthetic spec is built
    by concatenating them for the legacy code-path's benefit.
    """
    import re

    if tags is None:
        tags = FACTORIZED_TAGS

    def grab(open_tag: str, candidate_close: list[str]) -> str | None:
        m = re.search(rf"<{open_tag}>(.*?)</{open_tag}>",
                      text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()
        om = re.search(rf"<{open_tag}>", text, re.IGNORECASE)
        if not om:
            return None
        start = om.end()
        end = len(text)
        for ct in [f"</{open_tag}>"] + [f"<{t}>" for t in candidate_close]:
            cm = re.search(re.escape(ct), text[start:], re.IGNORECASE)
            if cm:
                end = min(end, start + cm.start())
        body = text[start:end].strip()
        return body if body else None

    out: dict[str, str | None] = {}
    for i, tag in enumerate(tags):
        others = [t for j, t in enumerate(tags) if j != i]
        out[tag.lower()] = grab(tag, others)

    # Legacy SPEC support (for callers that haven't migrated yet).
    legacy_spec = grab("SPEC", FACTORIZED_TAGS)
    if legacy_spec:
        out["spec"] = legacy_spec
    elif out.get("bio") or out.get("motivation") or out.get("scenario"):
        # Synthesize a spec paragraph from the factorized blocks for
        # back-compat. The factorized fields are the source of truth;
        # `spec` is a derived convenience.
        parts = []
        if out.get("bio"): parts.append(out["bio"])
        if out.get("motivation"): parts.append(f"Motivation: {out['motivation']}")
        if out.get("scenario"): parts.append(f"Scenario: {out['scenario']}")
        out["spec"] = "\n\n".join(parts) if parts else None
    else:
        out["spec"] = None

    return out


# ─── Measurement + drift ────────────────────────────────────────────

def measure_turn(turn_text: str) -> tuple[dict[str, int], str]:
    """Run a chat turn through the probe_persist two-stage cascade and
    return the parsed per-axis Likert plus the stage-1 prose summary."""
    s = stage1_summary(turn_text)
    raw = stage2_likert(s)
    return parse_elementwise(raw), s


def drift_report(target: np.ndarray, measured_named: dict[str, int],
                  cov_layer, axis_names: list[str] | None = None) -> dict:
    """Compare measured vs target. Mahalanobis distance + per-axis
    deltas. Caller decides convergence."""
    if axis_names is None:
        axis_names = AXIS_NAMES
    n_axes = len(axis_names)
    measured = np.array([measured_named.get(a, np.nan) for a in axis_names],
                        dtype=float)
    target_arr = np.asarray(target, dtype=float)
    diffs = measured - target_arr
    out = {
        "per_axis": {axis_names[i]: {
            "target": int(target_arr[i]),
            "measured": int(measured[i]) if not np.isnan(measured[i]) else None,
            "delta": float(diffs[i]) if not np.isnan(diffs[i]) else None,
        } for i in range(n_axes)},
        "mahalanobis": None,
        "max_abs_delta": float(np.nanmax(np.abs(diffs))),
        "rms_delta": float(np.sqrt(np.nanmean(diffs ** 2))),
    }
    if cov_layer is not None and cov_layer.pooled_within_inv is not None:
        # Mahalanobis only if measurement has no NaN.
        if not np.any(np.isnan(measured)):
            out["mahalanobis"] = mahalanobis(measured, target_arr,
                                              cov_layer.pooled_within_inv)
    return out


def feedback_message(target_named: dict[str, int],
                      measured_named: dict[str, int],
                      drift: dict,
                      judge_stage1_summary: str | None = None,
                      threshold: float = 1.0,
                      axis_names: list[str] | None = None) -> str:
    """Build a feedback message describing how the prior turn was READ
    by the judge + how its measured signature relates to the target.

    The Stage-1 prose summary (from the cascade JUDGE's first stage)
    is the *outside-reader description* of the prior turn — surfacing
    it here lets the DESIGNER reason about the gap between what the
    spec CLAIMED the persona would do and what an outside reader
    actually saw in the turn. Without this peek the DESIGNER only
    sees the measurement; with it, the DESIGNER sees both the
    measurement AND the *interpretive prose* that produced it.
    """
    if axis_names is None:
        axis_names = AXIS_NAMES
    landed = []
    drifted = []
    missing = []
    for axis in axis_names:
        a = drift["per_axis"][axis]
        if a["measured"] is None:
            missing.append(axis)
        elif abs(a["delta"]) <= threshold:
            landed.append((axis, a["measured"], a["target"]))
        else:
            drifted.append((axis, a["measured"], a["target"], a["delta"]))
    lines = []
    if judge_stage1_summary:
        lines.append("## How an outside reader described your prior turn")
        lines.append("")
        lines.append(judge_stage1_summary.strip())
        lines.append("")
        lines.append(
            "(That description is the judge's first-pass reading of your "
            "TURN — independent of what your SPEC claimed. The per-axis "
            "scores below were derived from that reading. If those scores "
            "diverge from what you intended, the prose above is where "
            "the divergence is visible: look at which features of your "
            "writing the reader noticed and decide which to write "
            "differently next round.)"
        )
        lines.append("")
    lines.append("## Measured-vs-target axis breakdown")
    lines.append("")
    if drift["mahalanobis"] is not None:
        lines.append(f"Mahalanobis distance from target: {drift['mahalanobis']:.2f}")
    lines.append(f"Max absolute axis delta: {drift['max_abs_delta']:.1f}")
    lines.append(f"RMS axis delta: {drift['rms_delta']:.2f}")
    if landed:
        lines.append("\nAxes that landed within ±1 of target:")
        for axis, m, t in landed:
            lines.append(f"  {axis:<22s}  measured {m}  (target {t})")
    if drifted:
        lines.append("\nAxes that drifted from target:")
        for axis, m, t, d in drifted:
            arrow = "↑" if d > 0 else "↓"
            lines.append(f"  {axis:<22s}  measured {m}  (target {t})  {arrow} drift {d:+.1f}")
    if missing:
        lines.append(f"\nAxes the judge did not score: {missing}")
    lines.append(
        "\nPlease produce a REVISED persona spec + turn. Keep the landed "
        "axes stable. For each drifted axis, name (in your new AUDIT line) "
        "which feature of your prior turn most likely caused the drift "
        "and what you are changing to address it. If the spec CLAIMED a "
        "property the turn did not REALISE (e.g. spec said \"non-provocative\" "
        "but turn was scored provocative=5), the spec's framing is not "
        "what the reader is using — the writing in the turn is. Rewrite "
        "the turn with different phrasing, not just a different spec."
    )
    return "\n".join(lines)


# ─── Driver ──────────────────────────────────────────────────────────

def _emit_event(diegetic: bool, kind: str, **payload):
    """In diegetic mode, write a JSONL event to stdout; otherwise
    no-op. The plugin endpoint parses these line-by-line and streams
    them as SSE to the browser."""
    if diegetic:
        rec = {"event": kind, **payload}
        print(json.dumps(rec), flush=True)


def run(jsonl_path: Path,
         pc_idx: int | None = None,
         n_sigma: float = 2.0,
         direction: int = +1,
         explicit_target: dict[str, int] | None = None,
         operator_constraint: str | None = None,
         designer_system_override: str | None = None,
         target_assistant: str | None = None,
         root_bio: str | None = None,
         overlay_name: str | None = None,
         rounds: int = 3,
         convergence_threshold: float = 1.0,
         out_path: Path | None = None,
         diegetic: bool = False):
    """Run the discovery loop.

    Target source — exactly one of:
      pc_idx: index of PC to displace along (PCA-driven)
      explicit_target: dict[axis_name -> int 1..5] (operator-specified)

    `operator_constraint`: natural-language constraint injected into
    the DESIGNER's brief at runtime. The point of the diegetic surface.

    `designer_system_override`: replaces the default DESIGNER system
    prompt entirely. Lets the operator vary DESIGNER framing.

    `diegetic`: when True, every loop transition emits a JSONL event
    to stdout so a calling process can stream them to a UI.
    """
    records = load_records(jsonl_path)
    if not records:
        sys.exit(f"no records at {jsonl_path}")

    fs = full_signature(records)
    if fs.pca_layer is None and pc_idx is not None:
        sys.exit("PCA layer unavailable — not enough samples for PCA-driven target")
    if fs.covariance_layer is None:
        sys.exit("covariance layer unavailable — not enough samples for Mahalanobis")

    # Compute the target.
    if explicit_target is not None:
        target = np.array([explicit_target.get(a, 3) for a in AXIS_NAMES], dtype=float)
        target = np.clip(np.round(target), 1, 5)
        target_named = named_target(target)
    else:
        target = pca_target(fs.pca_layer, pc_idx, n_sigma=n_sigma, direction=direction)
        target_named = named_target(target)

    if fs.pca_layer is not None:
        mean_vec = fs.pca_layer.mean_vec
    else:
        X, _, _ = records_to_matrix(records, require_full=True)  # noqa
        mean_vec = X.mean(axis=0)
    corpus_mean_named = {axis: float(mean_vec[i])
                         for i, axis in enumerate(AXIS_NAMES)}

    # Optionally load an assistant Character Card so the brief can
    # ground the user-agent's design in the specific assistant it will
    # be paired against (the "coupled user biography + assistant
    # biography" pattern from the workshop-design doc).
    assistant_card = None
    if target_assistant:
        assistant_card = _load_assistant_card(target_assistant)
        if assistant_card is None:
            sys.exit(f"target_assistant={target_assistant!r} not found")

    brief = make_brief(fs.pca_layer, pc_idx, direction, target_named,
                       corpus_mean_named,
                       operator_constraint=operator_constraint,
                       assistant_card=assistant_card,
                       root_bio=root_bio)

    overlay_mode = root_bio is not None
    _emit_event(diegetic, "discovery_start",
                target_named=target_named,
                pc_idx=pc_idx,
                direction=direction,
                operator_constraint=operator_constraint,
                target_assistant=(assistant_card.get("name")
                                  if assistant_card else None),
                overlay_mode=overlay_mode,
                root_bio=root_bio,
                overlay_name=overlay_name,
                rounds=rounds,
                convergence_threshold=convergence_threshold,
                brief=brief)

    if not diegetic:
        print("═" * 72)
        if pc_idx is not None:
            print(f"  DISCOVERY — target along PC{pc_idx+1} (direction {direction:+d}, "
                  f"{n_sigma}σ from corpus mean)")
        else:
            print(f"  DISCOVERY — explicit axis target")
        print("═" * 72)
        print(brief)
        print()

    history: list[dict] = []
    converged_round = None
    final_record = None

    for round_idx in range(rounds):
        if not diegetic:
            print()
            print(f"─── Round {round_idx + 1} ─────────────────────────────────────")
        _emit_event(diegetic, "round_start", round=round_idx + 1)

        if round_idx == 0:
            user_msg = brief
        else:
            # Subsequent rounds: brief + drift feedback from prior attempt.
            # The feedback now includes the judge's Stage-1 prose readout
            # so the DESIGNER can see how the prior turn READ to an
            # outside reader, not just how it was numerically scored.
            user_msg = brief + "\n\n## Feedback on your prior attempt\n\n" + history[-1]["feedback"]

        # Pass FULL history (every prior round as user/assistant pair).
        # The original `history[:-1]` slice was a bug: it stripped the
        # most recent round, so the DESIGNER never saw its own latest
        # SPEC+TURN as an assistant message — only the feedback prose
        # ABOUT that turn. Restoring the full history gives the DESIGNER
        # in-context access to every prior attempt's actual writing,
        # which is what in-context-learning over examples requires.
        msgs = designer_prompt(user_msg, history,
                                designer_system_override=designer_system_override,
                                overlay_mode=overlay_mode)
        t0 = time.monotonic()
        # 1500 tokens: factorized output ≈ 500-900 tokens at Gemma's
        # natural verbosity; overlay-mode output is shorter
        # (~200-400 tokens). 1500 gives headroom for either, plus
        # for the AUDIT line to be substantive.
        designer_text = _bridge_call(msgs)
        designer_dt = time.monotonic() - t0
        # In overlay-mode, the DESIGNER emits only ELICITATION_OVERLAY +
        # TURN + AUDIT. In factorized mode, it emits the full FACTORIZED
        # tag set. The parser's `tags` param selects which schema to
        # expect; the resulting dict keys are tag.lower().
        tag_set = OVERLAY_MODE_TAGS if overlay_mode else FACTORIZED_TAGS
        parsed = parse_designer_output(designer_text, tags=tag_set)
        spec = parsed.get("spec")  # synthesized for factorized mode only
        turn = parsed.get("turn")
        audit = parsed.get("audit")
        overlay_text = parsed.get("elicitation_overlay")  # overlay-mode only
        if not diegetic:
            print(f"DESIGNER call: {designer_dt:.1f}s")
        # Emit the full parsed dict in the diegetic event so downstream
        # consumers (UI, card-writer) can map each field.
        _emit_event(diegetic, "designer_response", round=round_idx + 1,
                    parsed=parsed,
                    # Back-compat fields for older clients (UI):
                    spec=spec, turn=turn, audit=audit,
                    elicitation_overlay=overlay_text,
                    wallclock_s=designer_dt)
        # Parse-failure check: in overlay-mode, success requires TURN +
        # OVERLAY both present. In factorized mode, TURN + (SPEC OR
        # BIO/MOTIVATION) needed.
        if overlay_mode:
            parse_failed = (not turn) or (not overlay_text)
        else:
            parse_failed = (not turn) or (not spec and not (parsed.get("bio") or parsed.get("motivation")))
        if parse_failed:
            if not diegetic:
                print("  parse failed — DESIGNER did not emit usable blocks")
                print(f"  raw[:300] = {designer_text[:300]!r}")
            history.append({
                "user_msg": user_msg,
                "assistant_msg": designer_text,
                "spec": None, "turn": None, "audit": audit, "parsed": parsed,
                "measured": None, "drift": None,
                "feedback": "Prior attempt malformed; please emit "
                            "<BIO>...</BIO>, <MOTIVATION>...</MOTIVATION>, "
                            "<SCENARIO>...</SCENARIO>, <TURN>...</TURN> blocks "
                            "(at minimum BIO + TURN must be present).",
            })
            continue
        if not diegetic:
            print(f"\nBIO:\n{parsed.get('bio') or '(none)'}\n")
            print(f"MOTIVATION:\n{parsed.get('motivation') or '(none)'}\n")
            print(f"SCENARIO:\n{parsed.get('scenario') or '(none)'}\n")
            print(f"TURN:\n{turn}\n")
            if audit:
                print(f"AUDIT: {audit}\n")

        t0 = time.monotonic()
        measured_named, stage1_text = measure_turn(turn)
        cascade_dt = time.monotonic() - t0
        if not diegetic:
            print(f"Cascade measurement: {cascade_dt:.1f}s  parsed {len(measured_named)}/{N_AXES} axes")
            print(f"  Stage-1 summary: {stage1_text[:200]}{'...' if len(stage1_text) > 200 else ''}")

        drift = drift_report(target, measured_named, fs.covariance_layer)
        # Pipe Stage-1 prose through to the feedback so the DESIGNER
        # can see how the turn READ, not just how it SCORED.
        fb = feedback_message(target_named, measured_named, drift,
                              judge_stage1_summary=stage1_text,
                              threshold=convergence_threshold)
        _emit_event(diegetic, "measurement", round=round_idx + 1,
                    measured=measured_named, judge_stage1_summary=stage1_text,
                    drift=drift, wallclock_s=cascade_dt)
        if not diegetic:
            print()
            print(fb)

        history.append({
            "user_msg": user_msg,
            "assistant_msg": designer_text,
            "spec": spec, "turn": turn, "audit": audit,
            "parsed": parsed,
            "judge_stage1_summary": stage1_text,
            "measured": measured_named, "drift": drift, "feedback": fb,
        })

        # Convergence: every axis within tolerance (and Mahalanobis < some bound).
        if drift["max_abs_delta"] <= convergence_threshold:
            converged_round = round_idx + 1
            final_record = history[-1]
            if not diegetic:
                print(f"\n✓ CONVERGED at round {converged_round}")
            break

    if converged_round is None and history:
        final_record = history[-1]
        if not diegetic:
            print(f"\n✗ Did not converge in {rounds} rounds — final round attempted")

    _emit_event(diegetic, "complete",
                converged_round=converged_round,
                rounds_attempted=len(history),
                final_spec=final_record["spec"] if final_record else None,
                final_turn=final_record["turn"] if final_record else None,
                final_audit=final_record["audit"] if final_record else None,
                final_parsed=final_record.get("parsed") if final_record else None,
                final_measured=final_record["measured"] if final_record else None,
                final_drift=final_record["drift"] if final_record else None,
                final_judge_stage1=final_record.get("judge_stage1_summary") if final_record else None,
                target_named=target_named,
                # Overlay-mode metadata. When `overlay_mode` is True, these
                # carry the root_bio (preserved byte-stable) and the
                # final overlay-text the DESIGNER converged on plus the
                # operator-chosen overlay_name. The plugin's card-writer
                # uses these to materialise an overlay-library-shaped
                # card manifest without re-deriving the root bio.
                overlay_mode=overlay_mode,
                final_root_bio=root_bio if overlay_mode else None,
                final_overlay_text=(final_record.get("parsed") or {}).get("elicitation_overlay")
                                    if final_record else None,
                overlay_name=overlay_name)

    if out_path:
        out_record = {
            "pc_idx": pc_idx,
            "direction": direction,
            "n_sigma": n_sigma,
            "target_named": target_named,
            "rounds_attempted": len(history),
            "converged_round": converged_round,
            "history": [{
                "round": i + 1,
                "assistant_msg_raw": h.get("assistant_msg"),
                "spec": h["spec"], "turn": h["turn"], "audit": h["audit"],
                "judge_stage1_summary": h.get("judge_stage1_summary"),
                "measured": h["measured"], "drift": h["drift"],
            } for i, h in enumerate(history)],
        }
        out_path.write_text(json.dumps(out_record, indent=2))
        print(f"\nrecord written to {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True, help="elicitation_judgments.jsonl path")
    # Target-source mutex: either --pc OR --explicit-target.
    p.add_argument("--pc", type=int,
                   help="PC index (1-based, e.g. 4 for PC4). PCA-driven target mode.")
    p.add_argument("--explicit-target",
                   help="JSON string of per-axis target dict, e.g. "
                        "'{\"curious\":4,\"terse\":2,\"warm\":2,...}'. "
                        "Explicit target mode; supersedes --pc when set.")
    p.add_argument("--n-sigma", type=float, default=2.0)
    p.add_argument("--direction", type=int, default=+1, choices=[-1, +1])
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--threshold", type=float, default=1.0,
                   help="convergence threshold: every axis must be within ±this of target")
    # Runtime control-plane affordances:
    p.add_argument("--operator-constraint",
                   help="natural-language constraint injected into the DESIGNER's brief; "
                        "lets the runtime operator shape persona-design at invocation time")
    p.add_argument("--designer-system-override",
                   help="replaces the DESIGNER's default system prompt; "
                        "exposes runtime variation of DESIGNER framing")
    p.add_argument("--target-assistant",
                   help="name or .json path of a SillyTavern Character Card "
                        "that this user-agent will be paired with. When set, "
                        "the brief grounds the user-agent's design in the "
                        "assistant's identity + scenario, enabling adversarial-"
                        "pairing benchmarks (e.g. user-wants-JS vs python-only-coder)")
    # Overlay-mode affordances. When --root-bio-text or --root-bio-card
    # is supplied, the DESIGNER runs in OVERLAY MODE: emits only an
    # ELICITATION_OVERLAY + TURN + AUDIT rather than a full factorized
    # persona, and the supplied bio passes through byte-stable. See
    # docs/overlay_architecture.md.
    overlay_g = p.add_mutually_exclusive_group()
    overlay_g.add_argument("--root-bio-text",
                            help="fixed root bio for overlay-mode discovery; "
                                 "DESIGNER will be asked to emit only an "
                                 "elicitation overlay (50-150 words) rather than "
                                 "regenerating bio/motivation/scenario")
    overlay_g.add_argument("--root-bio-card",
                            help="load the root bio from a SillyTavern Character "
                                 "Card's description; alternative to --root-bio-text")
    p.add_argument("--overlay-name",
                   help="optional name for the generated overlay in the card's "
                        "elicitation_overlay_library (e.g. 'js-clash'). Auto-"
                        "derived if omitted.")
    p.add_argument("--out", help="JSON record of the discovery run")
    p.add_argument("--diegetic", action="store_true",
                   help="emit a JSONL event stream to stdout suitable for "
                        "consumption by a UI / plugin endpoint (instead of "
                        "the default human-readable report)")
    args = p.parse_args()

    explicit_target = None
    if args.explicit_target:
        explicit_target = json.loads(args.explicit_target)
    elif args.pc is None:
        p.error("either --pc or --explicit-target must be supplied")

    # Resolve --root-bio-card → root_bio text if supplied.
    root_bio = None
    if args.root_bio_text:
        root_bio = args.root_bio_text
    elif args.root_bio_card:
        card = _load_assistant_card(args.root_bio_card)
        if not card:
            p.error(f"could not load --root-bio-card {args.root_bio_card!r}")
        root_bio = card.get("description") or card.get("bio")
        if not root_bio:
            p.error(f"--root-bio-card {args.root_bio_card!r} has no description/bio field")

    run(Path(args.jsonl),
        pc_idx=(args.pc - 1) if args.pc is not None and not args.explicit_target else None,
        n_sigma=args.n_sigma,
        direction=args.direction,
        explicit_target=explicit_target,
        operator_constraint=args.operator_constraint,
        designer_system_override=args.designer_system_override,
        target_assistant=args.target_assistant,
        root_bio=root_bio,
        overlay_name=args.overlay_name,
        rounds=args.rounds,
        convergence_threshold=args.threshold,
        out_path=Path(args.out) if args.out else None,
        diegetic=args.diegetic)


if __name__ == "__main__":
    main()
