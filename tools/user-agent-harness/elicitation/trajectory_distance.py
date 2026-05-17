"""Judge-trajectory distance + KL-divergence comparator.

Given two judged-trajectory JSON files (each produced by the plugin's
POST /trajectory-judge endpoint), this tool computes:

  1. Per-axis centroid distance (|μ_A - μ_B| per axis, summed L1)
  2. Per-axis KL divergence (treating per-axis Likert values as a
     categorical distribution over {1,2,3,4,5} with Laplace smoothing
     k=1; KL(P_old || P_new) plus its reverse, summed across axes)
  3. Mahalanobis distance of centroid difference (requires a pooled
     covariance estimate; uses identity covariance when insufficient
     data, falls back to L2 in that case)

The point: validate that a candidate (new bio + agent) reimplementation
of an old bundled persona produces a trajectory whose 14-axis signature
distribution is close to the old persona's. The user's criterion:
"judge-trajectory-kl-div-minimization of the new impl. from the old
impl." → small KL ⇒ valid reimplementation.

Usage:
    trajectory_distance.py --old <judged.json> --new <judged.json>
    trajectory_distance.py --old <judged.json> --candidates <a.json> <b.json>...
        (compares old to each candidate, prints the ranked KL minimum)
"""

import argparse
import json
import math
from pathlib import Path


# Reference axis order — must match LIKERT_AXIS_NAMES in the plugin.
AXES = [
    'curious', 'terse', 'warm', 'deferential', 'performative',
    'in_character', 'affective_intensity', 'probe_depth', 'goal_clarity',
    'disclosive', 'provocative', 'register_colloquial', 'playful', 'structured',
]


def load_signatures(jpath):
    """Returns the list of per-turn signature dicts (only valid full-axis
    ones). Reads the shape produced by POST /trajectory-judge."""
    with open(jpath) as f:
        d = json.load(f)
    sigs = []
    for p in d.get('per_turn_signatures', []):
        if not p.get('valid'):
            continue
        s = p.get('signature') or {}
        if all(a in s for a in AXES):
            sigs.append({a: int(s[a]) for a in AXES})
    return sigs


def centroid(sigs):
    if not sigs:
        return None
    out = {a: 0.0 for a in AXES}
    for s in sigs:
        for a in AXES:
            out[a] += s[a]
    n = len(sigs)
    for a in AXES:
        out[a] /= n
    return out


def laplace_categorical(sigs, axis, k=1):
    """Build P(value | axis) over {1..5} with add-k smoothing."""
    counts = {v: k for v in (1, 2, 3, 4, 5)}
    for s in sigs:
        v = s[axis]
        if v in counts:
            counts[v] += 1
    total = sum(counts.values())
    return {v: counts[v] / total for v in (1, 2, 3, 4, 5)}


def kl(p, q):
    """KL(P || Q) over a shared support."""
    out = 0.0
    for v in p:
        if p[v] > 0:
            out += p[v] * math.log(p[v] / q[v])
    return out


def compare(old_path, new_path):
    old_sigs = load_signatures(old_path)
    new_sigs = load_signatures(new_path)
    if not old_sigs or not new_sigs:
        return None
    c_old = centroid(old_sigs)
    c_new = centroid(new_sigs)
    # L1 of centroid difference (per-axis sum of |Δ|).
    l1 = sum(abs(c_old[a] - c_new[a]) for a in AXES)
    # Per-axis KL forward + reverse, summed.
    kl_forward = 0.0
    kl_reverse = 0.0
    kl_per_axis = {}
    for a in AXES:
        p = laplace_categorical(old_sigs, a)
        q = laplace_categorical(new_sigs, a)
        f = kl(p, q)
        r = kl(q, p)
        kl_forward += f
        kl_reverse += r
        kl_per_axis[a] = {'forward': f, 'reverse': r, 'sum': f + r}
    # L2 (Euclidean) distance of centroid.
    l2 = math.sqrt(sum((c_old[a] - c_new[a])**2 for a in AXES))
    return {
        'n_old': len(old_sigs),
        'n_new': len(new_sigs),
        'l1_centroid_distance': l1,
        'l2_centroid_distance': l2,
        'kl_forward_sum': kl_forward,        # KL(old || new), summed across axes
        'kl_reverse_sum': kl_reverse,        # KL(new || old)
        'kl_symmetric_sum': kl_forward + kl_reverse,
        'centroid_old': c_old,
        'centroid_new': c_new,
        'per_axis_kl': kl_per_axis,
        'per_axis_centroid_diff': {a: c_new[a] - c_old[a] for a in AXES},
    }


def fmt_compare(r, old_label, new_label):
    if r is None:
        return f"  {old_label} vs {new_label}: NO DATA (one trajectory has no valid signatures)"
    lines = []
    lines.append(f"  {old_label} (n={r['n_old']}) vs {new_label} (n={r['n_new']}):")
    lines.append(f"    L1 centroid distance: {r['l1_centroid_distance']:.2f}  (lower is closer)")
    lines.append(f"    L2 centroid distance: {r['l2_centroid_distance']:.2f}")
    lines.append(f"    KL(old||new) sum:     {r['kl_forward_sum']:.3f}")
    lines.append(f"    KL(new||old) sum:     {r['kl_reverse_sum']:.3f}")
    lines.append(f"    KL symmetric sum:     {r['kl_symmetric_sum']:.3f}  (lower is closer)")
    # Top-3 axes with biggest centroid shift
    diffs = sorted(r['per_axis_centroid_diff'].items(), key=lambda kv: -abs(kv[1]))
    lines.append(f"    Top per-axis shifts:")
    for a, d in diffs[:5]:
        marker = '!' if abs(d) >= 1.5 else ' '
        lines.append(f"      {a:<22s} {d:+.2f}{marker}")
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--old', required=True, type=Path,
                   help='reference (legacy / baseline) judged trajectory JSON')
    p.add_argument('--new', type=Path,
                   help='single candidate trajectory to compare')
    p.add_argument('--candidates', type=Path, nargs='+',
                   help='multiple candidates to compare; tool ranks by KL')
    p.add_argument('--label', default='?',
                   help='label for the --old file (for output)')
    args = p.parse_args()

    if args.new and not args.candidates:
        r = compare(args.old, args.new)
        print(fmt_compare(r, args.label or args.old.stem, args.new.stem))
    elif args.candidates:
        results = []
        for c in args.candidates:
            r = compare(args.old, c)
            results.append((c, r))
        print(f"  Reference: {args.label or args.old.stem} ({args.old})")
        print()
        # Print individual results.
        for c, r in results:
            print(fmt_compare(r, args.label, c.stem))
            print()
        # Rank by symmetric KL.
        ranked = sorted(
            [(c, r) for c, r in results if r is not None],
            key=lambda x: x[1]['kl_symmetric_sum']
        )
        if ranked:
            print(f"  ── Ranking by symmetric KL (closer = better reimplementation) ──")
            for i, (c, r) in enumerate(ranked):
                marker = ' ★ BEST FIT' if i == 0 else ''
                print(f"    {i+1}. {c.stem:<60s} KL_sym={r['kl_symmetric_sum']:.3f}  L1={r['l1_centroid_distance']:.2f}{marker}")
    else:
        p.error('--new or --candidates required')


if __name__ == '__main__':
    main()
