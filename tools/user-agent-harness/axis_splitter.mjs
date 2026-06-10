#!/usr/bin/env node
// axis_splitter.mjs
//
// Entanglement detector + splitter (item #4 in feature_factorization_design.md).
//
// Given (a) a trajectory file produced by lock_in_iterative.mjs and
// (b) the name of a parent axis whose measurement diverged across
// chat contexts within the same bio, this script:
//
//   1. Bucket user turns by original chat context (here: by agent_target.slug).
//   2. Confirm a context-dependent gap on the parent axis (sanity check).
//   3. Ask DESIGNER_S to propose 2-3 candidate split hypotheses
//      — each a pair of new sub-axes with rubrics.
//   4. For each hypothesis, re-judge every historical user turn under
//      the proposed axis pair (one bridge call per turn per hypothesis).
//   5. Compute Cohen's d for each new axis between contexts.
//   6. Accept the winning hypothesis iff max |d| ≥ SEPARATION_THRESHOLD.
//      On accept: register the derived axes via axis_registry.
//      On no-accept: record "axis appears genuinely entangled at this
//      resolution" with all evidence.
//
// Output: data/axis_splits/<parent>-<ts>.json containing full evidence
// (proposed hypotheses, per-turn re-judgments, per-axis means/stds,
// Cohen's d, and the verdict).
//
// CLI: node axis_splitter.mjs <traj.json> <parent_axis_name>

import fs from 'node:fs';
import path from 'node:path';
// All HTTP / bridge plumbing + shared stats live in harness_lib so the
// contract surface has ONE source of truth (the 2026-05-18 dedup).
// Axes are queried from the plugin (axes/*.json cards) — single source
// of truth, durable across runs. The splitter REGISTERS derived axes
// by POSTing new axis cards (axis_splitter's contribution to the
// growing schema is durable from one run to the next).
import * as L from './harness_lib.mjs';
import { bridgeCall, meanStd } from './harness_lib.mjs';

function toThreeSigFigs(n) {
    if (!Number.isFinite(n) || n === 0) return n;
    return Number(n.toPrecision(3));
}

function normalizeLikertNumber(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return Math.max(1, Math.min(5, toThreeSigFigs(n)));
}

// Project id → name to keep compatibility with code that uses `.name`.
const _AXES_CACHE = (await L.fetchAxes()).map(a => ({
    name: a.id, def: a.def, kind: a.kind,
}));
const _AXES_BY_ID = new Map(_AXES_CACHE.map(a => [a.name, a]));
function allAxes() { return _AXES_CACHE; }
function axisByName(name) { return _AXES_BY_ID.get(name) || null; }
async function registerDerivedAxis({ name, kind, def, derived_from }) {
    const card = await L.http('POST', `${L.ENDPOINTS.PLUGIN}/axes/${encodeURIComponent(name)}`, {
        name, kind, def, derived_from,
    });
    return card.axis || card;
}

const OUT_DIR = process.env.USER_PERSONAS_AXIS_SPLITS_DIR
    || path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'data', 'axis_splits');

const SEPARATION_THRESHOLD = 0.8;   // Cohen's d ≥ 0.8 = "large effect"
const N_HYPOTHESES = 3;

// Acceptance criteria (in addition to the d threshold):
//   (a) the winning sub-axis must recover the SIGN of the parent gap
//       (otherwise the split has measured something orthogonal — possibly
//       real, but not a factorization of the parent's entanglement);
//   (b) the winning sub-axis's |Cohen's d| must EXCEED the parent's own
//       |Cohen's d| on the same per-turn data — i.e. the split improves
//       discrimination, not just re-scores it.

// ── trajectory traversal ─────────────────────────────────────────────

/**
 * Walk a trajectory file and return:
 *   { contextLabel: [ { turn, agentTargetSig, outerIter, innerIter } ] }
 * where contextLabel is `agentTarget.slug` (e.g. "steals" / "romances-and-steals").
 */
function bucketTurnsByContext(traj, parentAxisRecord = null) {
    // Inner-loop turnJudgments carry AGENT-axis signatures only (the
    // inner loop judges agent behavior). Outer-loop bioTurnJudgments
    // carry BIO-axis signatures (judged in a separate pass on the best
    // inner chats). For a bio-kind parent axis we MUST read from
    // bioTurnJudgments — innerResults' turnJudgments don't carry that
    // axis at all and the gap detector returns -∞. Each
    // bioTurnJudgments entry is tagged with `context` (the source
    // agent_target.slug) by lock_in_iterative so bucketing works.
    const isBioParent = parentAxisRecord?.kind === 'bio';
    const byCtx = new Map();
    for (let oi = 0; oi < traj.result.attempts.length; oi++) {
        const outer = traj.result.attempts[oi];
        if (isBioParent && Array.isArray(outer.bioTurnJudgments)) {
            for (const tj of outer.bioTurnJudgments) {
                const ctx = tj.context || 'unknown';
                if (!byCtx.has(ctx)) byCtx.set(ctx, []);
                byCtx.get(ctx).push({
                    turn: tj.turn,
                    original_sig: tj.sig,
                    agent_target: null,
                    outer_iter: oi,
                    inner_iter: null,
                });
            }
        } else {
            for (const ir of outer.innerResults) {
                const ctx = ir.agentTarget.slug;
                if (!byCtx.has(ctx)) byCtx.set(ctx, []);
                for (let ii = 0; ii < ir.attempts.length; ii++) {
                    const ia = ir.attempts[ii];
                    for (const tj of ia.turnJudgments) {
                        byCtx.get(ctx).push({
                            turn: tj.turn,
                            original_sig: tj.sig,
                            agent_target: ir.agentTarget.target_agent,
                            outer_iter: oi,
                            inner_iter: ii,
                        });
                    }
                }
            }
        }
    }
    return byCtx;
}

// ── parent-axis context-gap diagnostic ───────────────────────────────

function meanOnAxis(turns, axisName) {
    const vs = turns.map(t => t.original_sig[axisName]).filter(v => Number.isFinite(v));
    if (!vs.length) return null;
    return vs.reduce((a, b) => a + b, 0) / vs.length;
}

function diagnoseGap(byCtx, parentAxis) {
    const summary = {};
    for (const [ctx, turns] of byCtx) {
        const vs = turns.map(t => t.original_sig[parentAxis]).filter(Number.isFinite);
        summary[ctx] = { n: turns.length, ...meanStd(vs) };
    }
    // The gap is defined over ALL context means (max − min). The contrast
    // PAIR that realizes it is the evaluation yardstick downstream: with
    // >2 contexts (e.g. 2 agent overlays × 3 counterparties = 6), taking
    // "the first two" instead compared two counterparties under the SAME
    // overlay — orthogonal to the detected entanglement — and produced
    // parent d≈0 false NO_SPLITs on the strongest candidates (observed:
    // star_sign gap 3.50 → parent d 0.00, 2026-06-10 sweep).
    let ctxMax = null, ctxMin = null;
    for (const [ctx, s] of Object.entries(summary)) {
        if (!Number.isFinite(s.mean)) continue;
        if (ctxMax === null || s.mean > summary[ctxMax].mean) ctxMax = ctx;
        if (ctxMin === null || s.mean < summary[ctxMin].mean) ctxMin = ctx;
    }
    const means = Object.values(summary).map(s => s.mean).filter(Number.isFinite);
    const gap = Math.max(...means) - Math.min(...means);
    // gap_pair = [high-mean ctx, low-mean ctx]; parent d on that pair is
    // the directly comparable yardstick for the children to beat.
    const gap_pair = (ctxMax !== null && ctxMin !== null && ctxMax !== ctxMin)
        ? [ctxMax, ctxMin]
        : null;
    let parent_cohens_d = null;
    let parent_sign = null;
    if (gap_pair) {
        const [cA, cB] = gap_pair;
        const vA = byCtx.get(cA).map(t => t.original_sig[parentAxis]).filter(Number.isFinite);
        const vB = byCtx.get(cB).map(t => t.original_sig[parentAxis]).filter(Number.isFinite);
        parent_cohens_d = cohensD(vA, vB);
        parent_sign = parent_cohens_d == null ? null : Math.sign(parent_cohens_d);
    }
    return { summary, gap, gap_pair, parent_cohens_d, parent_sign };
}

// ── DESIGNER_S: propose split hypotheses ─────────────────────────────

function fmtCtxSamples(turns, parentAxis, k = 3) {
    const sample = turns.slice(0, k);
    return sample.map((t, i) =>
        `(${i+1}) [${parentAxis}=${t.original_sig[parentAxis] ?? '?'}] ${t.turn.slice(0, 350).replace(/\n+/g, ' ')}…`
    ).join('\n\n');
}

async function proposeSplits(parentAxisRecord, byCtx, gapSummary) {
    const sys =
        'You are an axis-decomposition designer. The operator has been ' +
        'measuring user-side chat turns on a single behavioral axis, and ' +
        'the measurement keeps coming out differently depending on the ' +
        'surrounding chat context — even when the design target on that ' +
        'axis was identical. Your job is to propose 2-3 candidate ways to ' +
        'split the parent axis into a PAIR of new sub-axes such that:\n' +
        '  (a) the pair together captures the behavior the parent was meant to,\n' +
        '  (b) the pair separates the contexts where the parent measured differently.\n' +
        'Each new axis must be a 1-5 Likert with a one-sentence rubric in ' +
        'the same "1: low pole · 5: high pole" form. Output ONLY a JSON ' +
        'object: {"hypotheses": [{"id": int, "name1": str, "def1": str, ' +
        '"name2": str, "def2": str, "rationale": str}, ...]}. No preamble.';

    const ctxBlocks = [];
    for (const [ctx, turns] of byCtx) {
        const m = gapSummary.summary[ctx].mean;
        ctxBlocks.push(`### Context "${ctx}" (n=${turns.length} turns, mean ${parentAxisRecord.name}=${m?.toFixed(2)})\n\n${fmtCtxSamples(turns, parentAxisRecord.name, 3)}`);
    }

    const usr =
        '## Parent axis (entangled — measurement varies by context)\n\n' +
        `name: ${parentAxisRecord.name}\n` +
        `rubric: ${parentAxisRecord.def}\n` +
        `kind: ${parentAxisRecord.kind}\n\n` +
        `## Observed context-dependent measurements\n\n` +
        `Gap across contexts: ${gapSummary.gap.toFixed(2)} (1-5 Likert range).\n\n` +
        ctxBlocks.join('\n\n') + '\n\n' +
        `## Emit\n\nPropose ${N_HYPOTHESES} candidate splits as JSON. Each hypothesis names two new sub-axes (with rubrics) and a one-sentence rationale for why this split would separate the contexts.`;

    // Per moratorium (lint_generation_config.mjs): no max_tokens at caller.
    // Bridge default + natural EOS apply.
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    // Tolerant JSON extraction (designer-LLM may emit prose around the JSON block)
    const m = raw.match(/\{[\s\S]*\}/);
    if (!m) throw new Error(`could not find JSON object in DESIGNER_S output:\n${raw}`);
    const parsed = JSON.parse(m[0]);
    if (!Array.isArray(parsed.hypotheses)) throw new Error('DESIGNER_S output missing hypotheses[]');
    return { hypotheses: parsed.hypotheses, raw };
}

// ── JUDGE_S: re-score a turn on a candidate axis pair ────────────────

async function judgeOnPair(turn, pair) {
    const rubric = `- **${pair.name1}** — ${pair.def1}\n- **${pair.name2}** — ${pair.def2}`;
    const template = `${pair.name1}: ?\n${pair.name2}: ?`;
    const sys =
        'You are a behavioural-axis judge. You read ONE user-side chat ' +
        'turn and score it on the listed axes (number 1-5 each). The ' +
        'turn is the only ground truth. Be willing to score 1 (absence) ' +
        'when the turn genuinely shows no expression. Output ONLY the axis ' +
        'lines below — one per line as "axis_name: <number 1-5>". No ' +
        'preamble, no commentary, no markdown.';
    const usr =
        '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
        '## Turn to score\n\n' +
        '> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
        '## Emit\n\n' + template + '\n';
    // Per moratorium: no max_tokens at caller. Bridge default + EOS apply.
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const sig = { [pair.name1]: null, [pair.name2]: null };
    for (const line of raw.split('\n')) {
        for (const aname of [pair.name1, pair.name2]) {
            const re = new RegExp(
                '^\\s*[-*]?\\s*\\**["\']?' + aname.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '["\']?\\**\\s*[:=]\\s*([+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+))',
                'i');
            const m = line.match(re);
            const parsed = m ? normalizeLikertNumber(m[1]) : null;
            if (parsed != null) sig[aname] = parsed;
        }
    }
    return { sig, raw };
}

// ── statistics ───────────────────────────────────────────────────────
// `meanStd` is imported from harness_lib at the top of this file. Local
// helpers below build on it (Cohen's d, etc.).

function cohensD(arrA, arrB) {
    const a = meanStd(arrA);
    const b = meanStd(arrB);
    if (a.n < 2 || b.n < 2 || a.mean == null || b.mean == null) return null;
    const pooledStd = Math.sqrt(((a.n - 1) * a.std ** 2 + (b.n - 1) * b.std ** 2) / (a.n + b.n - 2));
    if (pooledStd === 0) return Math.sign(a.mean - b.mean) * Infinity;
    return (a.mean - b.mean) / pooledStd;
}

// ── evaluate a single hypothesis: re-judge every turn under the pair ─

async function evaluateSplit(hypothesis, byCtx, contextLabels, contrastPair = null) {
    const evals = {};  // ctxLabel → { name1: [scores], name2: [scores] }
    for (const ctx of contextLabels) evals[ctx] = { [hypothesis.name1]: [], [hypothesis.name2]: [] };
    // Re-judge every turn (one bridge call each), bounded to the engine kernel
    // width via saturatedMap — the old pre-started Promise.all fired all at once.
    const work = [];
    for (const ctx of contextLabels) {
        for (const t of byCtx.get(ctx)) work.push({ ctx, t });
    }
    const results = await L.saturatedMap(work,
        ({ ctx, t }) => judgeOnPair(t.turn, hypothesis).then(j => ({ ctx, t, j })));
    const perTurn = [];
    for (const { ctx, t, j } of results) {
        evals[ctx][hypothesis.name1].push(j.sig[hypothesis.name1]);
        evals[ctx][hypothesis.name2].push(j.sig[hypothesis.name2]);
        perTurn.push({ ctx, turn_excerpt: t.turn.slice(0, 120), sig: j.sig, raw: j.raw });
    }
    // Cohen's d is computed on the CONTRAST PAIR — the two contexts whose
    // parent-axis means realize the detected gap (diagnoseGap.gap_pair).
    // With >2 contexts, defaulting to "the first two" compared contexts
    // orthogonal to the entanglement (same overlay, different counterparty)
    // and yielded d≈0 false NO_SPLITs. Fallback to first two only when no
    // pair was derivable (e.g. <2 finite context means).
    let [cA, cB] = contrastPair || [];
    if (!cA || !cB) {
        console.warn(`[axis_splitter] no contrast pair derived; falling back to first two of ${contextLabels.length} contexts.`);
        [cA, cB] = contextLabels;
    }
    const stats = {
        [hypothesis.name1]: {
            [cA]: meanStd(evals[cA][hypothesis.name1]),
            [cB]: meanStd(evals[cB][hypothesis.name1]),
            cohens_d: cohensD(evals[cA][hypothesis.name1], evals[cB][hypothesis.name1]),
        },
        [hypothesis.name2]: {
            [cA]: meanStd(evals[cA][hypothesis.name2]),
            [cB]: meanStd(evals[cB][hypothesis.name2]),
            cohens_d: cohensD(evals[cA][hypothesis.name2], evals[cB][hypothesis.name2]),
        },
    };
    const d1 = stats[hypothesis.name1].cohens_d;
    const d2 = stats[hypothesis.name2].cohens_d;
    const maxAbsD = Math.max(Math.abs(d1 || 0), Math.abs(d2 || 0));
    // Per-sub-axis sign and magnitude for the acceptance criteria
    const subAxisD = {
        [hypothesis.name1]: { d: d1, sign: d1 == null ? null : Math.sign(d1), absd: Math.abs(d1 || 0) },
        [hypothesis.name2]: { d: d2, sign: d2 == null ? null : Math.sign(d2), absd: Math.abs(d2 || 0) },
    };
    return { hypothesis, stats, perTurn, max_abs_cohens_d: maxAbsD, sub_axis_d: subAxisD,
             contrast_pair: [cA, cB] };
}

// ── main orchestrator ────────────────────────────────────────────────

async function runSplitter(trajFile, parentAxisName) {
    const traj = JSON.parse(fs.readFileSync(trajFile, 'utf8'));
    const parentAxis = axisByName(parentAxisName);
    if (!parentAxis) throw new Error(`axis '${parentAxisName}' not in registry`);

    console.log(`[splitter] parent axis: ${parentAxisName} (${parentAxis.kind})`);
    console.log(`[splitter] trajectory:  ${trajFile}`);

    const byCtx = bucketTurnsByContext(traj, parentAxis);
    const contexts = [...byCtx.keys()];
    console.log(`[splitter] contexts: ${contexts.map(c => `${c} (n=${byCtx.get(c).length})`).join(', ')}`);

    const gapSummary = diagnoseGap(byCtx, parentAxisName);
    console.log(`[splitter] parent-axis context means:`);
    for (const [ctx, s] of Object.entries(gapSummary.summary)) {
        console.log(`            ${ctx}: mean ${parentAxisName}=${s.mean?.toFixed(2)} (n=${s.n})`);
    }
    console.log(`[splitter] gap: ${gapSummary.gap.toFixed(2)}`);
    if (gapSummary.gap_pair) {
        console.log(`[splitter] contrast pair (gap-defining): ${gapSummary.gap_pair[0]} vs ${gapSummary.gap_pair[1]}`);
    }

    if (gapSummary.gap < 0.5) {
        console.log(`[splitter] gap too small (< 0.5) — no entanglement to split. Done.`);
        return null;
    }

    console.log(`\n[splitter] DESIGNER_S proposing ${N_HYPOTHESES} split hypotheses…`);
    const { hypotheses, raw: designerRaw } = await proposeSplits(parentAxis, byCtx, gapSummary);
    for (const h of hypotheses) {
        console.log(`  H${h.id}: ${h.name1} / ${h.name2}`);
        console.log(`    rationale: ${h.rationale}`);
    }

    console.log(`\n[splitter] JUDGE_S re-scoring all turns under each hypothesis…`);
    const evaluations = [];
    for (const h of hypotheses) {
        process.stdout.write(`  H${h.id}: `);
        const ev = await evaluateSplit(h, byCtx, contexts, gapSummary.gap_pair);
        evaluations.push(ev);
        const [cA, cB] = ev.contrast_pair;
        const s = ev.stats;
        console.log(`max|d|=${ev.max_abs_cohens_d.toFixed(2)}`);
        console.log(`    ${h.name1}: ${cA}=${s[h.name1][cA].mean?.toFixed(2)}±${s[h.name1][cA].std?.toFixed(2)} ${cB}=${s[h.name1][cB].mean?.toFixed(2)}±${s[h.name1][cB].std?.toFixed(2)} d=${s[h.name1].cohens_d?.toFixed(2)}`);
        console.log(`    ${h.name2}: ${cA}=${s[h.name2][cA].mean?.toFixed(2)}±${s[h.name2][cA].std?.toFixed(2)} ${cB}=${s[h.name2][cB].mean?.toFixed(2)}±${s[h.name2][cB].std?.toFixed(2)} d=${s[h.name2].cohens_d?.toFixed(2)}`);
    }

    // ── Pick winner under STRICT criteria ─────────────────────────────
    // For each hypothesis, the "qualified" sub-axes are those that:
    //   (a) recover the parent's gap SIGN (Cohen's d sign matches parent's), AND
    //   (b) match-or-exceed the parent's |Cohen's d| on the same per-turn data.
    // Hypotheses are ranked by the MAX qualified sub-axis |d|.
    // Unqualified hypotheses (no qualified sub-axis) are ranked last.
    const parentAbsD = Math.abs(gapSummary.parent_cohens_d || 0);
    const parentSign = gapSummary.parent_sign;
    console.log(`\n[splitter] parent reference: cohens_d=${gapSummary.parent_cohens_d?.toFixed(2)} (sign=${parentSign}, |d|=${parentAbsD.toFixed(2)})`);
    for (const ev of evaluations) {
        const h = ev.hypothesis;
        ev.qualified = [];
        for (const name of [h.name1, h.name2]) {
            const sd = ev.sub_axis_d[name];
            const signOK = sd.sign != null && parentSign != null && sd.sign === parentSign;
            const magOK  = sd.absd >= parentAbsD;
            if (signOK && magOK) ev.qualified.push({ name, ...sd });
        }
        ev.qualified_max_d = ev.qualified.length
            ? Math.max(...ev.qualified.map(q => q.absd))
            : 0;
    }
    evaluations.sort((a, b) => b.qualified_max_d - a.qualified_max_d || b.max_abs_cohens_d - a.max_abs_cohens_d);
    const top = evaluations[0];
    const verdict = top.qualified.length > 0 && top.qualified_max_d >= SEPARATION_THRESHOLD
        ? 'SPLIT_ACCEPTED'
        : 'NO_SPLIT_FOUND';
    console.log(`[splitter] verdict: ${verdict}`);
    console.log(`            top qualified |d|=${top.qualified_max_d.toFixed(2)}, threshold=${SEPARATION_THRESHOLD}`);
    console.log(`            top qualified sub-axes: ${top.qualified.map(q => `${q.name} (d=${q.d.toFixed(2)})`).join(', ') || '(none — every candidate had wrong sign or insufficient magnitude)'}`);

    let registered = null;
    if (verdict === 'SPLIT_ACCEPTED') {
        const h = top.hypothesis;
        try {
            const r1 = await registerDerivedAxis({
                name: h.name1, kind: parentAxis.kind, def: h.def1,
                derived_from: { parent: parentAxisName, contexts: (gapSummary.gap_pair || contexts).join(' vs '), hypothesis_id: h.id, sibling: h.name2 },
            });
            const r2 = await registerDerivedAxis({
                name: h.name2, kind: parentAxis.kind, def: h.def2,
                derived_from: { parent: parentAxisName, contexts: (gapSummary.gap_pair || contexts).join(' vs '), hypothesis_id: h.id, sibling: h.name1 },
            });
            registered = [r1, r2];
            console.log(`[splitter] registered derived axes: ${h.name1}, ${h.name2}`);
        } catch (e) {
            console.warn(`[splitter] could not register derived axes: ${e.message}`);
        }
    } else {
        console.log(`[splitter] no hypothesis separated contexts ≥ ${SEPARATION_THRESHOLD}; parent axis stays as-is.`);
    }

    fs.mkdirSync(OUT_DIR, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const outFile = path.join(OUT_DIR, `${parentAxisName}-${ts}.json`);
    fs.writeFileSync(outFile, JSON.stringify({
        parent_axis: parentAxisName,
        parent_axis_record: parentAxis,
        source_trajectory: trajFile,
        bio_slug: traj.bio?.slug,
        contexts,
        gap_summary: gapSummary,
        designer_raw_output: designerRaw,
        evaluations,                          // sorted descending by max|d|
        verdict,
        threshold: SEPARATION_THRESHOLD,
        registered,
    }, null, 2));
    console.log(`[splitter] full evidence: ${outFile}`);
    return { verdict, outFile, top, registered };
}

// ── CLI ──────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
if (args.length !== 2) {
    console.error('usage: node axis_splitter.mjs <trajectory.json> <parent_axis_name>');
    process.exit(2);
}
await runSplitter(args[0], args[1]);
