#!/usr/bin/env python3
"""Quantitative diversity measurement over per-turn summaries.

The per-turn ~1-sentence summaries gemma already generated during the
study ARE the sparse semantic compression of each conversation. We
don't need a separate embedding model — measuring diversity over the
summaries (gemma's own distillation) is both cheaper and more
semantically-grounded than computing sentence embeddings.

Two complementary diversity signals:

  (A) STYLOMETRIC / LEXICAL — features computed over the summary text
      per persona-row: type-token ratio, signature words, function-word
      frequency cosine distance. Cheap, deterministic, surfaces
      vocabulary overlap.

  (B) GEMMA-AS-JUDGE — for each pair of conversation summaries from
      DIFFERENT personas, ask gemma "are these the same kind of
      interaction or different kinds?" with a 0-10 distinguishability
      score. Aggregates into a per-persona-pair narrative-distance
      matrix. Captures semantic shape that the lexical metric misses.

No regex per the codebase rule. Markdown is parsed by line-state
machine.

Usage:
    python3 measure_diversity.py output/diversity_report.md \
        [--gemma-judge] [--out output/diversity_quantitative.md]

By default emits only the stylometric report (cheap). Pass
--gemma-judge to additionally run the pairwise narrative comparisons
(adds ~60 LLM calls for a typical run).
"""
import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "elicitation"))
from llm_client import llm_call

MODEL = os.environ.get("USER_PERSONAS_MODEL", "gemma-4-a4b")


FUNCTION_WORDS = [
    "the", "a", "an", "and", "or", "but", "so", "if", "because", "while",
    "i", "you", "we", "they", "it", "my", "your",
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "this", "that", "these", "those",
    "actually", "really", "literally", "honestly", "basically",
    "perhaps", "maybe", "probably", "definitely",
    "just", "only", "even", "still", "yet",
    "very", "quite", "rather", "somewhat", "extremely",
    "well", "ok", "okay", "right", "fine",
    "oh", "ah", "uh", "um", "hmm",
    "please", "thanks", "sorry", "wow", "huh",
]

PUNCT_STRIP = ".,;:!?\"'()[]{}—–-…"


# ────────────────────────────────────────────────────────────────────
# Markdown parsing — two sources of structured data in the report:
#   - Summary Grid section: per-turn summaries (~1 sentence each)
#   - Full transcripts section: raw user-agent + scringlo turns
# We parse BOTH and let the analyzer choose which to use.
# ────────────────────────────────────────────────────────────────────

def parse_summary_grid(text):
    """Returns {persona_id: [conversation_summary_list, ...]}.

    Each conversation_summary_list is a list of {"role": "user"|"scringlo",
    "summary": str} — one entry per turn.
    """
    marker = "## Summary grid"
    end_marker = "## Full transcripts"
    start = text.find(marker)
    if start < 0:
        raise ValueError("no Summary grid section")
    end = text.find(end_marker, start)
    if end < 0:
        end = len(text)
    body = text[start:end]

    out = {}
    cur_persona = None
    cur_conv = None
    cur_list = None

    for line in body.split("\n"):
        if line.startswith("### "):
            cur_persona_name = line[4:].strip()
            # extract id from "Name (`id`)" pattern
            tick_l = cur_persona_name.find("`")
            tick_r = cur_persona_name.rfind("`")
            if tick_l > 0 and tick_r > tick_l:
                cur_persona = cur_persona_name[tick_l + 1:tick_r]
            else:
                cur_persona = cur_persona_name.lower().replace(" ", "-")
            out.setdefault(cur_persona, [])
        elif line.startswith("**Conversation "):
            cur_list = []
            out[cur_persona].append(cur_list)
        elif line.startswith("- 🙋 "):
            if cur_list is not None:
                cur_list.append({"role": "user", "summary": line[len("- 🙋 "):].strip()})
        elif line.startswith("- 💬 "):
            if cur_list is not None:
                cur_list.append({"role": "scringlo", "summary": line[len("- 💬 "):].strip()})

    return out


def parse_full_transcripts(text):
    """Returns {persona_id: [conversation_turns, ...]}.

    Each conversation_turns is a list of {"role": "user"|"scringlo", "text": str}.
    """
    marker = "## Full transcripts"
    idx = text.find(marker)
    if idx < 0:
        raise ValueError("no Full transcripts section")
    body = text[idx + len(marker):]
    out = {}
    cur_persona = None
    cur_persona_name = None
    cur_conv_list = None
    cur_role = None
    cur_buf = []

    def flush():
        nonlocal cur_buf
        if cur_persona and cur_role and cur_buf and cur_conv_list is not None:
            joined = "\n".join(cur_buf).strip()
            if joined:
                cur_conv_list.append({"role": cur_role, "text": joined})
        cur_buf = []

    for line in body.split("\n"):
        if line.startswith("### "):
            flush()
            cur_persona_name = line[4:].strip()
            cur_persona = cur_persona_name.lower().replace(" ", "-")
            out.setdefault(cur_persona, [])
            cur_conv_list = None
            cur_role = None
        elif line.startswith("#### Conversation "):
            flush()
            cur_conv_list = []
            out[cur_persona].append(cur_conv_list)
            cur_role = None
        elif line.startswith("**user-agent:** "):
            flush()
            cur_role = "user"
            cur_buf = [line[len("**user-agent:** "):]]
        elif line.startswith("**scringlo:** "):
            flush()
            cur_role = "scringlo"
            cur_buf = [line[len("**scringlo:** "):]]
        elif cur_role and not line.startswith("---") and not line.startswith("### ") and not line.startswith("#### "):
            cur_buf.append(line)

    flush()
    return out


# ────────────────────────────────────────────────────────────────────
# (A) Stylometric / lexical features over summary text
# ────────────────────────────────────────────────────────────────────

def is_emoji_char(c):
    cp = ord(c)
    return cp >= 0x1F000 or 0x2600 <= cp <= 0x27BF


def tokenize(text):
    out = []
    for raw in text.split():
        word = raw.strip(PUNCT_STRIP).lower()
        if word and not all(is_emoji_char(c) for c in word):
            out.append(word)
    return out


def split_sentences(text):
    out = []
    buf = []
    for c in text:
        buf.append(c)
        if c in ".!?":
            out.append("".join(buf).strip())
            buf = []
    if buf:
        out.append("".join(buf).strip())
    return [s for s in out if s]


def extract_features_from_summaries(summary_list):
    """Aggregate stylometric features across user-agent SUMMARIES only.

    Summaries are gemma's third-person distillation of what the speaker
    DID; the linguistic register is gemma's, not the speaker's. But
    word CHOICES preserve persona-specific signal (gemma writes
    "demanded" vs "wondered" vs "praised" based on the turn).
    """
    user_summaries = [s["summary"] for s in summary_list if s["role"] == "user"]
    if not user_summaries:
        return None
    all_text = "\n".join(user_summaries)
    tokens = tokenize(all_text)
    sentences = split_sentences(all_text)
    n_words = len(tokens)
    n_unique = len(set(tokens))
    n_sentences = len(sentences) or 1
    counts = Counter(tokens)
    fw_freq = {w: counts.get(w, 0) / max(n_words, 1) * 1000 for w in FUNCTION_WORDS}
    return {
        "n_user_summaries": len(user_summaries),
        "n_words": n_words,
        "n_unique_words": n_unique,
        "type_token_ratio": round(n_unique / max(n_words, 1), 3),
        "avg_sentence_len": round(n_words / n_sentences, 2),
        "avg_word_len": round(sum(len(t) for t in tokens) / max(n_words, 1), 2),
        "function_word_freqs": fw_freq,
        "_token_counter": counts,
    }


def cosine_distance(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    if not common:
        return 1.0
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1.0
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1.0
    return 1.0 - dot / (norm_a * norm_b)


def find_signature_words(personas_features, *, top_n=12, min_count_in_persona=3):
    signatures = {}
    pids = list(personas_features.keys())
    totals = {pid: max(personas_features[pid]["n_words"], 1) for pid in pids}
    counters = {pid: personas_features[pid]["_token_counter"] for pid in pids}
    for pid in pids:
        my_counter = counters[pid]
        my_total = totals[pid]
        other_counter = Counter()
        other_total = 0
        for opid in pids:
            if opid == pid:
                continue
            other_counter.update(counters[opid])
            other_total += totals[opid]
        scores = []
        for word, count in my_counter.items():
            if count < min_count_in_persona:
                continue
            if len(word) <= 2 and word not in FUNCTION_WORDS:
                continue
            my_rate = count / my_total
            other_rate = (other_counter.get(word, 0) + 1) / (other_total + 1)
            score = my_rate / other_rate
            scores.append((word, count, round(score, 2)))
        scores.sort(key=lambda x: -x[2])
        signatures[pid] = scores[:top_n]
    return signatures


# ────────────────────────────────────────────────────────────────────
# (B) Gemma-as-judge: pairwise narrative distinguishability
# ────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are evaluating whether two conversation summaries describe DIFFERENT KINDS of interactions or "
    "VERY SIMILAR kinds of interactions. You will be given two trajectories of per-turn summaries. "
    "Rate their narrative distinguishability on a 0-10 scale where:\n"
    "  10 = entirely different kinds of conversations (different goals, register, dynamic)\n"
    "   5 = same general topic area but visibly different conversational dynamics\n"
    "   0 = effectively the same kind of conversation, swap-in-swap-out interchangeable\n"
    "Respond with ONE JSON object: {\"score\": <int 0-10>, \"reason\": \"<one short sentence>\"}. "
    "No preamble, no markdown fence."
)


def conv_summary_to_text(summary_list):
    """Render a single conversation's summary list as a compact trajectory."""
    lines = []
    for i, s in enumerate(summary_list):
        prefix = "USER" if s["role"] == "user" else "GM"
        lines.append(f"{i+1}. [{prefix}] {s['summary']}")
    return "\n".join(lines)


def judge_pair(pair_idx, conv_a, conv_b, label_a, label_b):
    """One pairwise gemma comparison. Returns (label_a, label_b, score, reason)."""
    messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": (
                f"=== Conversation A (from persona '{label_a}') ===\n"
                f"{conv_summary_to_text(conv_a)}\n\n"
                f"=== Conversation B (from persona '{label_b}') ===\n"
                f"{conv_summary_to_text(conv_b)}\n\n"
                "Rate distinguishability."
            )},
        ]
    try:
        content = llm_call(messages, seed=70_000 + pair_idx, timeout=120)
        parsed = json.loads(content)
        score = int(parsed.get("score", -1))
        reason = parsed.get("reason", "")
    except Exception as e:
        return (label_a, label_b, -1, f"(judge error: {e})")
    return (label_a, label_b, score, reason)


def run_gemma_judge(summaries_by_persona, *, max_workers=4):
    """Pairwise comparison of one conversation per persona-pair.

    Conservative budget: take conversation[0] from each persona, do
    pairwise comparisons. K personas → K*(K-1)/2 calls (no self-pairs,
    symmetric).
    """
    pids = list(summaries_by_persona.keys())
    pairs = []
    for i, pa in enumerate(pids):
        for j, pb in enumerate(pids):
            if i < j:
                pairs.append((pa, pb, summaries_by_persona[pa][0], summaries_by_persona[pb][0]))
    print(f"[judge] {len(pairs)} pairwise comparisons; running with {max_workers} workers...")
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(judge_pair, i, ca, cb, la, lb) for i, (la, lb, ca, cb) in enumerate(pairs)]
        for fut in futs:
            results.append(fut.result())
            la, lb, score, reason = results[-1]
            print(f"  {la} vs {lb}: score={score}  ({reason[:80]})")
    # Build matrix
    matrix = {pa: {pb: None for pb in pids} for pa in pids}
    for la, lb, score, reason in results:
        matrix[la][lb] = {"score": score, "reason": reason}
        matrix[lb][la] = {"score": score, "reason": reason}
    return matrix


# ────────────────────────────────────────────────────────────────────
# Report rendering
# ────────────────────────────────────────────────────────────────────

def render_report(personas_features, signatures, lexical_matrix,
                  judge_matrix, *, out_path):
    pids = list(personas_features.keys())
    lines = []
    lines.append("# Quantitative diversity over per-turn summaries")
    lines.append("")
    lines.append("Two complementary signals:")
    lines.append("  - (A) Stylometric / lexical over gemma's per-turn summaries")
    lines.append("  - (B) Gemma-as-judge pairwise narrative distinguishability (if --gemma-judge)")
    lines.append("")

    # (A) Stylometric
    lines.append("## (A) Stylometric features over user-agent summaries")
    lines.append("")
    feats = [
        ("n_user_summaries", "summaries"),
        ("n_words", "words"),
        ("type_token_ratio", "TTR"),
        ("avg_sentence_len", "avg sent len"),
        ("avg_word_len", "avg word len"),
    ]
    lines.append("| persona | " + " | ".join(c[1] for c in feats) + " |")
    lines.append("|" + "---|" * (len(feats) + 1))
    for pid in pids:
        f = personas_features[pid]
        row = [pid]
        for key, _ in feats:
            row.append(str(f[key]))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Signature words
    lines.append("### Signature words per persona (from summaries)")
    lines.append("")
    for pid in pids:
        sigs = signatures[pid]
        lines.append(f"**`{pid}`** — top over-represented summary-vocabulary:")
        if not sigs:
            lines.append("(no qualifying words)")
        else:
            lines.append(", ".join(f"`{w}` ({c}, ×{s})" for w, c, s in sigs[:10]))
        lines.append("")

    # Lexical distance matrix
    lines.append("### Lexical-cosine distance (function-word vectors)")
    lines.append("")
    lines.append("Higher = more distinguishable. 0 = identical lexical profile.")
    lines.append("")
    lines.append("| | " + " | ".join(pids) + " |")
    lines.append("|" + "---|" * (len(pids) + 1))
    for i, pa in enumerate(pids):
        row = [pa]
        for j, pb in enumerate(pids):
            if i == j:
                row.append("—")
            else:
                row.append(f"{lexical_matrix[pa][pb]:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # (B) Gemma judge
    if judge_matrix:
        lines.append("## (B) Gemma-judged pairwise distinguishability")
        lines.append("")
        lines.append("Scores 0-10. Higher = more distinguishable narratives.")
        lines.append("")
        lines.append("| | " + " | ".join(pids) + " |")
        lines.append("|" + "---|" * (len(pids) + 1))
        for i, pa in enumerate(pids):
            row = [pa]
            for j, pb in enumerate(pids):
                if i == j:
                    row.append("—")
                else:
                    entry = judge_matrix[pa].get(pb)
                    if entry is None:
                        row.append("?")
                    else:
                        row.append(str(entry["score"]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        lines.append("### Pairwise reasons")
        lines.append("")
        for i, pa in enumerate(pids):
            for j, pb in enumerate(pids):
                if i < j:
                    entry = judge_matrix[pa].get(pb)
                    if entry:
                        lines.append(f"- **{pa} vs {pb}** (score {entry['score']}): {entry['reason']}")
        lines.append("")

    # Diagnostic
    pairs_lex = []
    for i, pa in enumerate(pids):
        for j, pb in enumerate(pids):
            if i < j:
                pairs_lex.append((pa, pb, lexical_matrix[pa][pb]))
    pairs_lex.sort(key=lambda x: x[2])
    lines.append("## Diagnostic")
    lines.append("")
    lines.append(f"- **Lexically nearest pair**: `{pairs_lex[0][0]}` vs `{pairs_lex[0][1]}` (distance {pairs_lex[0][2]:.3f})")
    lines.append(f"- **Lexically farthest pair**: `{pairs_lex[-1][0]}` vs `{pairs_lex[-1][1]}` (distance {pairs_lex[-1][2]:.3f})")
    if judge_matrix:
        pairs_judge = []
        for i, pa in enumerate(pids):
            for j, pb in enumerate(pids):
                if i < j and judge_matrix[pa].get(pb):
                    pairs_judge.append((pa, pb, judge_matrix[pa][pb]["score"]))
        pairs_judge.sort(key=lambda x: x[2])
        lines.append(f"- **Narratively nearest pair (gemma judge)**: `{pairs_judge[0][0]}` vs `{pairs_judge[0][1]}` (score {pairs_judge[0][2]})")
        lines.append(f"- **Narratively farthest pair (gemma judge)**: `{pairs_judge[-1][0]}` vs `{pairs_judge[-1][1]}` (score {pairs_judge[-1][2]})")
    lines.append("")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("\n".join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("report", help="path to diversity_report markdown")
    p.add_argument("--out", default="output/diversity_quantitative.md")
    p.add_argument("--gemma-judge", action="store_true",
                   help="add pairwise gemma-judged narrative distinguishability")
    p.add_argument("--judge-workers", type=int, default=4)
    args = p.parse_args()

    text = Path(args.report).read_text()
    summary_grid = parse_summary_grid(text)
    print(f"Parsed {len(summary_grid)} personas from summary grid")
    for pid, convs in summary_grid.items():
        print(f"  {pid}: {len(convs)} conversations, "
              f"{sum(len(c) for c in convs)} total summaries")

    # Stylometric over summaries (concatenated per persona)
    personas_features = {}
    for pid, convs in summary_grid.items():
        all_summaries = []
        for conv in convs:
            all_summaries.extend(conv)
        feats = extract_features_from_summaries(all_summaries)
        if feats is None:
            print(f"  WARN: {pid} has no user-agent summaries")
            continue
        personas_features[pid] = feats

    signatures = find_signature_words(personas_features)

    # Lexical distance matrix
    lexical_matrix = {pa: {} for pa in personas_features}
    for pa in personas_features:
        for pb in personas_features:
            if pa == pb:
                continue
            lexical_matrix[pa][pb] = cosine_distance(
                personas_features[pa]["function_word_freqs"],
                personas_features[pb]["function_word_freqs"],
            )

    # Optional gemma-as-judge
    judge_matrix = None
    if args.gemma_judge:
        # Use conv 0 from each persona (full per-turn summary trajectory)
        summaries_by_persona = {pid: convs for pid, convs in summary_grid.items()}
        judge_matrix = run_gemma_judge(summaries_by_persona, max_workers=args.judge_workers)

    for pid in personas_features:
        personas_features[pid].pop("_token_counter", None)

    render_report(personas_features, signatures, lexical_matrix, judge_matrix,
                  out_path=args.out)
    print(f"Report → {args.out}")


if __name__ == "__main__":
    main()
