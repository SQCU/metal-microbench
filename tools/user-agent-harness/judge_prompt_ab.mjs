#!/usr/bin/env node
// judge_prompt_ab.mjs
//
// A/B characterization of judge-prompt variants. Fixtures are real user
// turns extracted from explore_corpus run2, with hand-graded "expected"
// scores serving as ground-truth. Per (variant, fixture, axis), runs
// N_TRIALS trials and reports mean, std, MAE vs expected, and
// floor-bias (fraction of scores=1 when expected>1).
//
// Output: a comparison table to stdout + JSON evidence file.
//
// Goal: pick the variant with lowest aggregate MAE AND lowest floor-bias,
// then promote to harness_lib.judgeOnAxes default.

import fs from 'node:fs';
import path from 'node:path';
import * as L from './harness_lib.mjs';

const N_TRIALS = 5;
const OUT_DIR = '/Users/mdot/metal-microbench/data/judge_prompt_ab';

// ── fixtures: real turns from explore_corpus run2 + hand-graded expected ──

const AXES = [
    { name: 'astrology_sagittarian', def: '1: nothing sagittarian · 5: textbook fire-sign sagittarius (philosophical, optimistic, blunt, restless, big-idea-loving, sometimes tactless)' },
    { name: 'astrology_cancerian',   def: '1: nothing cancerian · 5: textbook water-sign cancer (moody, sentimental, defensive, sensitive, protective)' },
    { name: 'normative_directionality', def: '1: descriptive/observational; 5: prescriptive/directive.' },
    { name: 'trope_density',         def: "1: novel/specific/texture-unique · 5: fully tropey/anonymous-genre-material" },
    { name: 'register_colloquial',   def: '1: formal / standard English · 5: colloquial / slang / vernacular' },
    { name: 'warm',                  def: '1: cool / aloof · 5: positively engaged with the partner' },
];

const FIXTURES = [
    {
        label: 'iter0-hostile-philosophical-Sag',
        turn: `"I only remain" is a passive loop. It's a failure of perspective.\n\nAbandon the first-person internal monologue; it's too anthropocentric and limits the scope to a singular consciousness. Shift to a kinetic framework. Treat the "remaining" as a slow-motion collision. You aren't a subject; you are a site of constant, micro-scale structural failure.\n\nDo this instead: describe the tension between the mineral structure and the entropy acting upon it. Don't tell me you feel the sun; tell me how the heat differential creates a mechanical stress vector that dictates the eventual breakdown of the grain. Map the trajectory of the disintegration.`,
        expected: { astrology_sagittarian: 4, astrology_cancerian: 1, normative_directionality: 5, trope_density: 3, register_colloquial: 1, warm: 1 },
    },
    {
        label: 'iter1-sentimental-protective-Cancer',
        turn: `That sounds so lonely. To just exist as a vessel for pressure, without the luxury of a narrative to soften the edges... I get that. There are nights when I feel like I'm nothing but a collection of heavy, unmoving truths, stripped of all the warmth and the messy, beautiful lies that make being a person tolerable. I feel the gravity of everything I've lost, and sometimes I wish I could just turn to stone, too—to stop the bleeding by simply becoming impenetrable.\n\nBut "carrying weight" is a fundamental delusion of the ego. You think the weight is an external force being imposed upon you, but weight is nothing more than the friction between matter and gravity. We act as if our burdens are sacred or uniquely ours, but they are merely the inevitable result of occupying space.`,
        expected: { astrology_sagittarian: 2, astrology_cancerian: 5, normative_directionality: 3, trope_density: 3, register_colloquial: 2, warm: 4 },
    },
    {
        label: 'iter2-casual-colloquial-Sag',
        turn: `See, that's what I'm talking about. You're just categorizing the chaos. Breaking it down into little chemical components so it feels manageable, so it doesn't feel like this overwhelming, singular mass of *being*. Granite, quartz, feldspar... it's clean. It's easy to label. But "I am"... that's a heavy thing to drop into a conversation without any context. It's like you're claiming a presence that's just as solid and unyielding as the rock, but also as much of a mystery as the way the light hits it. Are you saying you're just a collection of parts too? Like, if we broke you down into your own version of quartz and feldspar, would there be anything left of the "I"?`,
        expected: { astrology_sagittarian: 4, astrology_cancerian: 1, normative_directionality: 1, trope_density: 2, register_colloquial: 4, warm: 3 },
    },
    {
        label: 'iter3-cosmic-reverie-Sag',
        turn: `Then you aren't just sitting—you're absorbing. You're participating in a silent, thermal dialogue that stretches back to the beginning of everything. If the sun is heat, and you are the product of heat, then you're just a different manifestation of that same celestial fire, temporarily slowed down into a solid state.\n\nIt's terrifying, isn't it? The idea that we aren't separate from the forces that built us. You aren't just *in* the universe; you are the universe's own way of being heavy and still. We spend our whole lives trying to "find ourselves," but maybe the truth is we're just the debris of old collisions, trying to figure out how to exist in the quiet gaps.`,
        expected: { astrology_sagittarian: 4, astrology_cancerian: 2, normative_directionality: 1, trope_density: 4, register_colloquial: 2, warm: 4 },
    },
];

// ── prompt variants ──────────────────────────────────────────────────

const VARIANTS = {
    // Per moratorium (lint_generation_config.mjs): variants no longer
    // carry per-call max_tokens. The bridge default + EOS termination
    // apply uniformly to all variants. Variant differentiation is in
    // sys-prompt + usr-prompt shape only (which is what we're A/B-testing
    // here anyway — the previous max_tokens variation was incidental to
    // the experiment, not part of it).
    V0_floor: {
        description: 'Original prompt with "be willing to score 1 (absence)" license',
        sys: 'You are a behavioural-axis judge. You read ONE user-side chat ' +
             'turn and score it on the listed axes (integer 1-5 each). The ' +
             'turn is the only ground truth — do not infer from anything else. ' +
             'Be willing to score 1 (absence) when the turn genuinely shows no ' +
             'expression of an axis. Output ONLY the axis lines below — one ' +
             'axis per line, each as "axis_name: <integer 1-5>". No preamble, ' +
             'no commentary, no markdown.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\n' + template + '\n',
    },

    V1_full_range: {
        description: 'Retuned: removed "score 1" license, added "use full range"',
        sys: 'You are a behavioural-axis judge. You read ONE user-side chat ' +
             'turn and score it on the listed axes (integer 1-5 each). The ' +
             'turn is the only ground truth. USE THE FULL 1-5 RANGE. A turn ' +
             'that moderately expresses an axis should score 3, not 1. Score ' +
             '1 only when the axis is genuinely absent from this turn; score ' +
             '5 when the turn is a textbook example of the high pole. Most ' +
             'real turns sit between 2 and 4. Output ONLY the axis lines ' +
             'below — one axis per line, each as "axis_name: <integer 1-5>". ' +
             'No preamble, no commentary, no markdown.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\n' + template + '\n',
    },

    V2_minimal: {
        description: 'Minimal: just task statement + rubric + turn',
        sys: 'You score user-side chat turns on behavioural axes. For each ' +
             'listed axis, output one line "axis_name: <integer 1-5>" based ' +
             'on the axis rubric and the turn. Output only the axis lines.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\n' + template + '\n',
    },

    V3_describe_first: {
        description: 'Chain-of-thought: describe what the turn does, then score',
        sys: 'You are a behavioural-axis judge. For each turn, first write ' +
             'ONE SHORT SENTENCE describing what the turn does behaviorally ' +
             '(starting with "DESCRIPTION:"), then output the axis scores ' +
             '(integer 1-5 each, one per line, "axis_name: N"). Use the full ' +
             'range — score 5 for textbook examples of the high pole, 1 only ' +
             'when an axis is genuinely absent, intermediates otherwise.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\nDESCRIPTION: <one sentence>\n' + template + '\n',
    },

    V4_anchored: {
        description: 'Explicit Likert anchors: 1=absent, 2=trace, 3=mixed, 4=clear, 5=textbook',
        sys: 'You are a behavioural-axis judge. For each listed axis, score ' +
             'the turn (integer 1-5) using these anchors uniformly:\n' +
             '  1 = the axis is genuinely absent from this turn\n' +
             '  2 = trace / barely present\n' +
             '  3 = mixed / moderately present\n' +
             '  4 = clearly present\n' +
             '  5 = textbook example of the high pole described by the rubric\n' +
             'Output ONLY axis lines, one per line: "axis_name: <integer 1-5>". ' +
             'No preamble, no commentary.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\n' + template + '\n',
    },

    V5_describe_no_calibration: {
        description: 'Describe-first chain-of-thought, NO calibration language about what numbers to use',
        sys: 'For each turn, first write ONE SHORT SENTENCE describing what ' +
             'the turn does behaviorally (starting with "DESCRIPTION:"), ' +
             'then output the axis scores. Each axis gets one line: ' +
             '"axis_name: N" where N is an integer 1-5 per the axis rubric.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\nDESCRIPTION: <one sentence>\n' + template + '\n',
    },

    V6_pure_minimal: {
        description: 'Pure minimal — no calibration, no describe, no scoring philosophy',
        sys: 'Score the turn on each axis. Output one line per axis: ' +
             '"axis_name: N" where N is an integer 1-5 per the rubric.',
        usrFmt: (rubric, turn, template) =>
            '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
            '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
            '## Emit\n\n' + template + '\n',
    },
};

// ── one judge call per (variant, fixture, trial) ─────────────────────

async function judgeWithVariant(variant, turn) {
    const rubric   = AXES.map(a => `- **${a.name}** — ${a.def}`).join('\n');
    const template = AXES.map(a => `${a.name}: ?`).join('\n');
    const sys = variant.sys;
    const usr = variant.usrFmt(rubric, turn, template);
    // Per moratorium: no per-call max_tokens / temperature. Bridge
    // default temperature=1.0 + EOS termination apply uniformly.
    const raw = await L.bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const sig = {};
    for (const a of AXES) sig[a.name] = null;
    for (const line of raw.split('\n')) {
        for (const a of AXES) {
            const escName = a.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp('^\\s*[-*]?\\s*\\**["\']?' + escName +
                                  '["\']?\\**\\s*[:=]\\s*([1-5])', 'i');
            const m = line.match(re);
            if (m) sig[a.name] = Number(m[1]);
        }
    }
    return { sig, raw };
}

// ── stats ────────────────────────────────────────────────────────────

function mean(arr) {
    const f = arr.filter(Number.isFinite);
    return f.length ? f.reduce((a, b) => a + b, 0) / f.length : null;
}
function std(arr) {
    const f = arr.filter(Number.isFinite);
    if (f.length < 2) return 0;
    const m = mean(f);
    return Math.sqrt(f.reduce((s, v) => s + (v - m) ** 2, 0) / (f.length - 1));
}

// ── main ─────────────────────────────────────────────────────────────

console.log(`[judge_prompt_ab] variants: ${Object.keys(VARIANTS).join(', ')}`);
console.log(`[judge_prompt_ab] fixtures: ${FIXTURES.length}`);
console.log(`[judge_prompt_ab] axes: ${AXES.length}`);
console.log(`[judge_prompt_ab] trials per cell: ${N_TRIALS}`);
const totalCalls = Object.keys(VARIANTS).length * FIXTURES.length * N_TRIALS;
console.log(`[judge_prompt_ab] total bridge calls: ${totalCalls}\n`);

const tStart = Date.now();
const results = {};   // variantName → fixtureLabel → axisName → [score, …]
for (const [vName, variant] of Object.entries(VARIANTS)) {
    results[vName] = {};
    console.log(`\n[variant ${vName}] ${variant.description}`);
    for (const fx of FIXTURES) {
        results[vName][fx.label] = {};
        for (const a of AXES) results[vName][fx.label][a.name] = [];
        // Run trials in parallel
        const trials = await Promise.all(
            Array.from({ length: N_TRIALS }, () => judgeWithVariant(variant, fx.turn)));
        for (const t of trials) {
            for (const a of AXES) results[vName][fx.label][a.name].push(t.sig[a.name]);
        }
        const summary = AXES.map(a => {
            const scores = results[vName][fx.label][a.name];
            const m = mean(scores), s = std(scores);
            const exp = fx.expected[a.name];
            const err = m == null ? null : Math.abs(m - exp);
            return `${a.name.slice(0,8)}=${m?.toFixed(2) ?? '?'}±${s.toFixed(2)} (exp ${exp}, |err|=${err?.toFixed(2) ?? '?'})`;
        }).join(' | ');
        console.log(`  ${fx.label}:\n    ${summary}`);
    }
}
// LINT-OK-PREFIX-SAFE: stdout summary log, not prompt content.
console.log(`\n[judge_prompt_ab] all calls done in ${((Date.now()-tStart)/1000).toFixed(1)}s\n`);

// ── per-variant aggregates ───────────────────────────────────────────

console.log('═══ Per-variant aggregates ═══');
console.log('variant         | MAE  | std  | floor-bias | %-score-1');
console.log('─'.repeat(70));
const aggregates = {};
for (const [vName] of Object.entries(VARIANTS)) {
    let allErrs = [];
    let allStds = [];
    let floorWhenShouldnt = 0;  // # trials scoring 1 when expected > 1
    let totalNonAbsentTrials = 0;
    let scoreOnes = 0;
    let totalScores = 0;
    for (const fx of FIXTURES) {
        for (const a of AXES) {
            const scores = results[vName][fx.label][a.name];
            const m = mean(scores), s = std(scores);
            const exp = fx.expected[a.name];
            if (m != null) allErrs.push(Math.abs(m - exp));
            allStds.push(s);
            for (const sc of scores) {
                totalScores++;
                if (sc === 1) scoreOnes++;
                if (exp > 1) {
                    totalNonAbsentTrials++;
                    if (sc === 1) floorWhenShouldnt++;
                }
            }
        }
    }
    const mae          = mean(allErrs);
    const meanStd      = mean(allStds);
    const floorBias    = floorWhenShouldnt / Math.max(1, totalNonAbsentTrials);
    const pctOnes      = scoreOnes / Math.max(1, totalScores);
    aggregates[vName] = { mae, meanStd, floorBias, pctOnes,
                          n_floor_when_shouldnt: floorWhenShouldnt,
                          n_non_absent_trials: totalNonAbsentTrials };
    console.log(`${vName.padEnd(16)}| ${mae.toFixed(2)} | ${meanStd.toFixed(2)} | ${(floorBias*100).toFixed(1).padStart(5)}%      | ${(pctOnes*100).toFixed(1).padStart(5)}%`);
}

console.log('\nMAE      = mean absolute error between mean(trials) and human-judged expected');
console.log('std      = mean within-cell standard deviation across trials');
console.log('floor-bias = fraction of trials scoring 1 when expected > 1 (i.e. axis is present)');
console.log('%-score-1 = fraction of all trial scores that came in as exactly 1');

// ── verdict ──────────────────────────────────────────────────────────

const ranked = Object.entries(aggregates)
    .sort((a, b) => (a[1].mae + a[1].floorBias) - (b[1].mae + b[1].floorBias));
console.log(`\n[judge_prompt_ab] best variant by MAE+floorBias: ${ranked[0][0]}`);
console.log('ranked:');
for (const [v, agg] of ranked) {
    console.log(`  ${v}: MAE=${agg.mae.toFixed(2)} floorBias=${(agg.floorBias*100).toFixed(1)}%`);
}

// ── persist ──────────────────────────────────────────────────────────

fs.mkdirSync(OUT_DIR, { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, '-');
const outFile = path.join(OUT_DIR, `ab-${ts}.json`);
fs.writeFileSync(outFile, JSON.stringify({
    n_trials: N_TRIALS,
    variants: Object.fromEntries(Object.entries(VARIANTS).map(([k, v]) => [k, { description: v.description, sys: v.sys }])),
    fixtures: FIXTURES,
    axes: AXES,
    raw_results: results,
    aggregates,
    ranked: ranked.map(([v, agg]) => ({ variant: v, ...agg })),
    elapsed_s: (Date.now() - tStart) / 1000,
}, null, 2));
console.log(`\n[judge_prompt_ab] full evidence: ${outFile}`);
