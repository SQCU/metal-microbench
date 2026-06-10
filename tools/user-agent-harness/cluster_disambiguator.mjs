#!/usr/bin/env node
// cluster_disambiguator.mjs
//
// Item 6 in docs/feature_factorization_design.md.
//
// Given a "tight cluster" of bios that all measure similarly on the
// existing axis registry, attempt to find a NEW behavioral axis that
// spreads the cluster. If none exists, distinguish the failure mode:
//   - ParaphraseDegenerate    (bios are reworded versions of each other)
//   - BehaviorallyDegenerate  (bios are prose-different but project to
//                              identical behavior under current vocab)
//
// Every cluster bio is paired with a CHEAPLY-ITERATED agent (K=1 single
// pass) — we never measure bios in the asking-the-LM-to-play-both-roles
// degenerate mode.
//
// Turn-depth matters more than trajectory-count: N_TURNS=4 catches
// trailing-off / boring-fixed-point signals; N_TRAJ=2 keeps wallclock
// down. (See §6 for rationale.)
//
// CLI: node cluster_disambiguator.mjs <cluster_spec.json>

import fs from 'node:fs';
import path from 'node:path';
// All HTTP / bridge / persistence / chat / judge / statistics live in
// harness_lib so the contract surface has ONE source of truth (the
// 2026-05-18 dedup). Axes are queried from the plugin (axes/*.json
// cards). registerDerivedAxis POSTs a new axis card so disambiguator
// findings are durable across runs. This file owns ONLY the
// cluster-disambiguation algorithm — proposing spread axes, ANOVA
// F-ratio acceptance, the
// paraphrase-vs-behaviorally-degenerate fork.
import {
    ENDPOINTS,
    http, bridgeCall,
    saveBio, saveAgent,
    designCheapAgent,
    fetchCounterparty, runChat, userTurns,
    judgeOnAxis,
    meanStd,
    fetchAxes,
    saturatedMap,
} from './harness_lib.mjs';
const { ST, PLUGIN } = ENDPOINTS;

// Project id → name to keep compatibility with code that uses `.name`.
// derived_from retained: the membership idempotency gate below needs it.
const _AXES_CACHE = (await fetchAxes()).map(a => ({
    name: a.id, def: a.def, kind: a.kind, derived_from: a.derived_from || null,
}));
function allAxes() { return _AXES_CACHE; }

// Membership idempotency gate — same family as the splitter's
// children-gate. A cluster that already produced a spread axis for the
// SAME member set stays disambiguated: re-running on unchanged
// membership registers a paraphrase spread axis per dispatch (observed
// 2026-06-10: sensory_specificity then discursive_velocity for one
// identical 4-bio cluster across two outer_outer passes). Keyed on the
// sorted member list recorded in derived_from.cluster_members.
// CLUSTER_DISAMBIG_FORCE=1 overrides for deliberate re-runs.
function priorSpreadAxisForMembers(memberKeys) {
    const key = [...memberKeys].sort().join('|');
    return _AXES_CACHE.find(a =>
        a.derived_from?.reason === 'spread_axis'
        && Array.isArray(a.derived_from?.cluster_members)
        && [...a.derived_from.cluster_members].sort().join('|') === key) || null;
}
async function registerDerivedAxis({ name, kind, def, derived_from }) {
    const card = await http('POST', `${PLUGIN}/axes/${encodeURIComponent(name)}`, {
        name, kind, def, derived_from,
    });
    return card.axis || card;
}

const OUT_DIR = process.env.USER_PERSONAS_CLUSTER_DISAMBIG_DIR
    || path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'data', 'cluster_disambig');

// ── tunables (see §6 Acceptance Thresholds) ──────────────────────────
const N_TRAJ_PER_BIO     = 2;
const N_TURNS_PER_TRAJ   = 4;
const N_HYPOTHESES       = 3;
const SPREAD_THRESHOLD   = 1.5;     // Likert points (max - min of per-bio means)
const F_RATIO_THRESHOLD  = 3.5;     // ≈ F_crit(2,21) at α=0.05 for k=3, N_T=2, n_t=4
const PARAPHRASE_THRESHOLD = 4.0;   // pairwise prose-sim ≥ 4/5 → paraphrase verdict
const TIGHTNESS_THRESHOLD  = 1.0;   // max pairwise existing-axis distance for "tight"
const EPSILON = 1e-6;

/**
 * One-way ANOVA F-ratio: between-group / within-group variance.
 * `perGroup`: array of arrays of scalars. Returns { F, between_var, within_var, df_between, df_within }.
 */
function fRatio(perGroup) {
    const groups = perGroup.filter(g => g.filter(Number.isFinite).length > 0);
    const k = groups.length;
    const grandAll = groups.flat().filter(Number.isFinite);
    const N = grandAll.length;
    if (k < 2 || N < k + 1) return { F: null, between_var: null, within_var: null };
    const grandMean = grandAll.reduce((a, b) => a + b, 0) / N;
    let ssBetween = 0, ssWithin = 0;
    for (const g of groups) {
        const vals = g.filter(Number.isFinite);
        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
        ssBetween += vals.length * (m - grandMean) ** 2;
        for (const v of vals) ssWithin += (v - m) ** 2;
    }
    const dfB = k - 1, dfW = N - k;
    const msB = ssBetween / dfB;
    const msW = ssWithin / dfW;
    return { F: msW > EPSILON ? msB / msW : Infinity,
             between_var: msB, within_var: msW,
             df_between: dfB, df_within: dfW };
}

// ── DESIGNER_C: propose candidate spread axes ────────────────────────

function fmtBioBlock(bio, sampleTurns) {
    return `### "${bio.name}" (id: ${bio.canonical_key})\n\n` +
           `prose: ${bio.prose}\n\n` +
           `sample turns:\n` +
           sampleTurns.map((t, i) => `  (${i+1}) ${t.slice(0, 250).replace(/\n+/g, ' ')}…`).join('\n');
}

async function proposeSpreadAxes(bios, trajs, tightnessReport) {
    const sys =
        'You are a behavioral-axis disambiguator. The operator has a ' +
        `cluster of ${bios.length} user-persona bios that all measure ` +
        'identically on the existing axis registry. Your job is to ' +
        'propose NEW Likert axes (1-5, with a one-sentence rubric in the ' +
        'usual "1: low pole · 5: high pole" form) such that AT LEAST ONE ' +
        'of them spreads these bios as widely as possible when ' +
        'each bio is scored on it. The axes must be:\n' +
        '  - behaviorally observable in user-side chat turns (not "what ' +
        '    the prose says about the user", but "what the user actually does"),\n' +
        '  - not redundant with the existing registry (listed below),\n' +
        '  - capable of producing meaningfully different scores across the bios.\n' +
        'Output ONLY a JSON object: ' +
        '{"hypotheses": [{"id": int, "name": str, "def": str, "rationale": str}, ...]}. ' +
        'No preamble, no markdown.';
    const registryListing = allAxes()
        .map(a => `  - ${a.name} (${a.kind}): ${a.def.slice(0, 120)}…`).join('\n');
    const bioBlocks = bios.map(b => fmtBioBlock(b, userTurns(trajs[b.canonical_key][0]).slice(0, 2))).join('\n\n');
    const usr =
        `## Tightness report (existing-axis signature)\n\n${tightnessReport}\n\n` +
        `## The cluster\n\n${bioBlocks}\n\n` +
        `## Existing axis registry (do NOT propose duplicates of these)\n\n${registryListing}\n\n` +
        `## Emit\n\nPropose ${N_HYPOTHESES} candidate spread axes as JSON. Each must include a one-sentence rationale for why this axis would distinguish at least 2 of these bios.`;
    // Per moratorium (lint_generation_config.mjs): no max_tokens at caller.
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const m = raw.match(/\{[\s\S]*\}/);
    if (!m) throw new Error(`could not parse JSON from DESIGNER_C:\n${raw.slice(0, 500)}`);
    const parsed = JSON.parse(m[0]);
    if (!Array.isArray(parsed.hypotheses)) throw new Error('DESIGNER_C output missing hypotheses[]');
    return { hypotheses: parsed.hypotheses, raw };
}

// ── pre-flight tightness check on nominal-tight axis ─────────────────

async function tightnessPreflight(bios, trajs, nominalAxisName) {
    const axis = allAxes().find(a => a.name === nominalAxisName);
    if (!axis) {
        return { skipped: true, reason: `nominal_tight_axis '${nominalAxisName}' not in registry`,
                 tight: true, distances: {}, summary: 'skipped' };
    }
    const perBio = {};
    for (const bio of bios) {
        const turns = trajs[bio.canonical_key].flatMap(userTurns);
        const scores = await saturatedMap(turns, t => judgeOnAxis(t, axis.name, axis.def));
        perBio[bio.canonical_key] = meanStd(scores);
    }
    const means = Object.values(perBio).map(s => s.mean).filter(Number.isFinite);
    const distance = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    const summaryLines = bios.map(b => {
        const s = perBio[b.canonical_key];
        return `  ${b.canonical_key}: ${nominalAxisName}=${s.mean?.toFixed(2)}±${s.std?.toFixed(2)} (n=${s.n})`;
    });
    return {
        skipped: false,
        nominal_axis: nominalAxisName,
        per_bio: perBio,
        spread: distance,
        tight: distance <= TIGHTNESS_THRESHOLD,
        summary: `existing-axis tightness on ${nominalAxisName}: max-min spread=${distance.toFixed(2)} (threshold ${TIGHTNESS_THRESHOLD})\n${summaryLines.join('\n')}`,
    };
}

// ── pairwise similarity (LLM-as-judge) ───────────────────────────────

function pairs(arr) {
    const out = [];
    for (let i = 0; i < arr.length; i++)
        for (let j = i+1; j < arr.length; j++) out.push([arr[i], arr[j]]);
    return out;
}

async function judgeSimilarity(promptLabel, defText, blockA, blockB) {
    const sys =
        'You score similarity between two blocks of text on a 1-5 Likert ' +
        'scale. Output ONLY the line "similarity: <number 1-5>". No ' +
        'preamble, no markdown.';
    const usr =
        `## What "similarity" means here\n\n${defText}\n\n` +
        `## Block A\n\n${blockA}\n\n## Block B\n\n${blockB}\n\n` +
        `## Emit\n\nsimilarity: ?\n`;
    // Per moratorium: no max_tokens at caller. Bridge default + EOS apply.
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const m = raw.match(/similarity\s*:\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))/i);
    if (!m) return null;
    const n = Number(m[1]);
    if (!Number.isFinite(n)) return null;
    return Math.max(1, Math.min(5, Number(n.toPrecision(3))));
}

async function pairwiseProseSim(bios) {
    const ps = pairs(bios);
    const scores = await saturatedMap(ps, ([a, b]) => judgeSimilarity(
        'prose-similarity',
        '1: completely different (unrelated topics, registers, or content); 3: same topic, different presentation; 5: paraphrases of each other (same content, different wording)',
        a.prose, b.prose));
    return { mean: scores.filter(Number.isFinite).reduce((x, y) => x + y, 0) / scores.length,
             pairs: ps.map(([a, b], i) => ({ a: a.name, b: b.name, score: scores[i] })) };
}

async function pairwiseBehaviorSim(bios, trajs) {
    const ps = pairs(bios);
    const sample = bio => userTurns(trajs[bio.canonical_key][0]).slice(0, 2)
        .map((t, i) => `  (${i+1}) ${t.slice(0, 250)}`).join('\n');
    const scores = await saturatedMap(ps, ([a, b]) => judgeSimilarity(
        'behavioral-similarity',
        '1: completely different behaviors / moves / strategies; 3: overlapping but distinguishable behavioral repertoires; 5: behaviorally interchangeable (same moves, same strategies, same dispositional expression)',
        sample(a), sample(b)));
    return { mean: scores.filter(Number.isFinite).reduce((x, y) => x + y, 0) / scores.length,
             pairs: ps.map(([a, b], i) => ({ a: a.name, b: b.name, score: scores[i] })) };
}

// ── evaluate one spread-axis hypothesis ──────────────────────────────

async function evaluateSpread(hypothesis, bios, trajs) {
    const perBioRaw = {};   // bio_id → [score per turn]
    const work = [];        // flat (bio, turn, idx) list, judged kernel-width-bounded
    for (const bio of bios) {
        const turns = trajs[bio.canonical_key].flatMap(userTurns);
        perBioRaw[bio.canonical_key] = new Array(turns.length).fill(null);
        for (let i = 0; i < turns.length; i++) {
            work.push({ key: bio.canonical_key, turn: turns[i], idx: i });
        }
    }
    await saturatedMap(work, ({ key, turn, idx }) =>
        judgeOnAxis(turn, hypothesis.name, hypothesis.def)
            .then(score => { perBioRaw[key][idx] = score; }));
    const perBio = {};
    for (const bio of bios) {
        perBio[bio.canonical_key] = meanStd(perBioRaw[bio.canonical_key]);
    }
    const means = Object.values(perBio).map(s => s.mean).filter(Number.isFinite);
    const spread = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    const fr = fRatio(bios.map(b => perBioRaw[b.canonical_key]));
    return { hypothesis, perBio, perBioRaw, spread, ...fr };
}

// ── orchestrator ─────────────────────────────────────────────────────

async function runDisambiguator(specPath) {
    const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
    console.log(`[disambig] cluster: ${spec.cluster_id} (${spec.bios.length} bios)`);
    console.log(`[disambig] counterparty: ${spec.counterparty_avatar}`);
    console.log(`[disambig] nominal tight axis: ${spec.nominal_tight_axis}`);

    // Idempotency: this exact membership already has a registered spread
    // axis → done (the new axis needs corpus SCORES, not a re-derivation;
    // membership changes → new key → fresh disambiguation).
    const prior = priorSpreadAxisForMembers(spec.bios.map(b => b.canonical_key));
    if (prior && process.env.CLUSTER_DISAMBIG_FORCE !== '1') {
        console.log(`[disambig] membership already disambiguated → spread axis '${prior.name}' — skipping (idempotent; CLUSTER_DISAMBIG_FORCE=1 to re-run). Done.`);
        return null;
    }

    // 1. Install bios + design cheap agents
    console.log(`\n[disambig] installing bios + designing cheap agents (K=1 each)…`);
    const cp = await fetchCounterparty(spec.counterparty_avatar);
    const agentIds = {};
    for (const bio of spec.bios) {
        await saveBio(bio);
        const agentText = await designCheapAgent(bio);
        // Plugin agent ID_RE is /^[a-z0-9_-]+$/ (lowercase only) — bio
        // canonical_keys carry case + dots ("1778…-DespoticMiscreant.png"),
        // so sanitize the whole id or POST /agents 400s (observed live
        // 2026-06-10, second disambiguator dispatch ever).
        const agentId = `${spec.cluster_id}-${bio.canonical_key.replace(/\.png$/, '')}-cheap`
            .toLowerCase().replace(/[^a-z0-9_-]/g, '-');
        await saveAgent(agentId, `${bio.name} (cheap)`, agentText, bio.canonical_key);
        agentIds[bio.canonical_key] = { id: agentId, text: agentText };
        console.log(`  ${bio.canonical_key}: cheap_agent="${agentText.slice(0, 80).replace(/\n/g, ' ')}…"`);
    }

    // 2. Run chats per bio
    console.log(`\n[disambig] running ${N_TRAJ_PER_BIO} trajectories × ${N_TURNS_PER_TRAJ} turns per bio…`);
    const trajs = {};
    const tChat0 = Date.now();
    const chatWork = [];
    for (const bio of spec.bios) {
        trajs[bio.canonical_key] = [];
        for (let r = 0; r < N_TRAJ_PER_BIO; r++) {
            const taskIdx = trajs[bio.canonical_key].push(null) - 1;
            chatWork.push({ bio, taskIdx });
        }
    }
    await saturatedMap(chatWork, ({ bio, taskIdx }) =>
        runChat(bio, agentIds[bio.canonical_key].id, cp, N_TURNS_PER_TRAJ)
            .then(chat => { trajs[bio.canonical_key][taskIdx] = chat; }));
    // LINT-OK-PREFIX-SAFE: stderr-style timing log, not prompt content.
    console.log(`[disambig] chats done in ${((Date.now()-tChat0)/1000).toFixed(1)}s`);

    // 3. Pre-flight tightness on nominal axis
    console.log(`\n[disambig] tightness pre-flight on ${spec.nominal_tight_axis}…`);
    const tightness = await tightnessPreflight(spec.bios, trajs, spec.nominal_tight_axis);
    console.log(`[disambig] ${tightness.summary}`);
    if (!tightness.skipped && !tightness.tight) {
        console.warn(`[disambig] WARNING: cluster is NOT tight on ${spec.nominal_tight_axis} (spread=${tightness.spread.toFixed(2)} > ${TIGHTNESS_THRESHOLD}). Proceeding anyway but verdict may be misleading.`);
    }

    // 4. DESIGNER_C proposes candidate spread axes
    console.log(`\n[disambig] DESIGNER_C proposing ${N_HYPOTHESES} candidate spread axes…`);
    const { hypotheses, raw: designerRaw } = await proposeSpreadAxes(spec.bios, trajs, tightness.summary);
    for (const h of hypotheses) {
        console.log(`  H${h.id}: ${h.name}\n     def: ${h.def}\n     rationale: ${h.rationale}`);
    }

    // 5. JUDGE_C evaluates each hypothesis
    console.log(`\n[disambig] JUDGE_C scoring all turns under each hypothesis…`);
    const evaluations = [];
    for (const h of hypotheses) {
        process.stdout.write(`  H${h.id}: `);
        const ev = await evaluateSpread(h, spec.bios, trajs);
        evaluations.push(ev);
        const meansStr = spec.bios.map(b => {
            const s = ev.perBio[b.canonical_key];
            return `${b.name.split(/\s+/).pop()}=${s.mean?.toFixed(2)}±${s.std?.toFixed(2)}`;
        }).join(' ');
        console.log(`spread=${ev.spread.toFixed(2)} F=${ev.F?.toFixed(2)} (msB=${ev.between_var?.toFixed(2)}, msW=${ev.within_var?.toFixed(2)})`);
        console.log(`     per-bio means: ${meansStr}`);
    }

    // 6. Pick winner under spread + F thresholds
    const qualified = evaluations.filter(e =>
        Number.isFinite(e.F) && e.F >= F_RATIO_THRESHOLD && e.spread >= SPREAD_THRESHOLD);
    qualified.sort((a, b) => b.F - a.F);

    let verdict, registered = null, similarityReport = null;
    if (qualified.length > 0) {
        const top = qualified[0];
        verdict = 'SPREAD_AXIS_FOUND';
        console.log(`\n[disambig] verdict: SPREAD_AXIS_FOUND`);
        console.log(`            winning axis: ${top.hypothesis.name}`);
        console.log(`            F=${top.F.toFixed(2)} (threshold ${F_RATIO_THRESHOLD}), spread=${top.spread.toFixed(2)} (threshold ${SPREAD_THRESHOLD})`);
        try {
            registered = await registerDerivedAxis({
                name: top.hypothesis.name, kind: 'bio', def: top.hypothesis.def,
                derived_from: { contexts: `cluster:${spec.cluster_id}`,
                                cluster_members: spec.bios.map(b => b.canonical_key),
                                reason: 'spread_axis' },
            });
            console.log(`            registered as derived axis (kind=bio)`);
        } catch (e) {
            console.warn(`            could not register: ${e.message}`);
        }
    } else {
        console.log(`\n[disambig] no axis qualified (F≥${F_RATIO_THRESHOLD} ∧ spread≥${SPREAD_THRESHOLD}); checking paraphrase vs behavioral degeneracy…`);
        const [proseSim, behaviorSim] = await Promise.all([
            pairwiseProseSim(spec.bios),
            pairwiseBehaviorSim(spec.bios, trajs),
        ]);
        similarityReport = { prose: proseSim, behavior: behaviorSim };
        console.log(`            pairwise prose-similarity:    mean=${proseSim.mean.toFixed(2)}`);
        for (const p of proseSim.pairs) console.log(`              ${p.a} ⇔ ${p.b}: ${p.score}`);
        console.log(`            pairwise behavior-similarity: mean=${behaviorSim.mean.toFixed(2)}`);
        for (const p of behaviorSim.pairs) console.log(`              ${p.a} ⇔ ${p.b}: ${p.score}`);
        verdict = proseSim.mean >= PARAPHRASE_THRESHOLD
            ? 'CLUSTER_IS_PARAPHRASE_DEGENERATE'
            : 'CLUSTER_IS_BEHAVIORALLY_DEGENERATE';
        console.log(`[disambig] verdict: ${verdict}`);
    }

    // 7. Persist evidence
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const outFile = path.join(OUT_DIR, `${spec.cluster_id}-${ts}.json`);
    fs.writeFileSync(outFile, JSON.stringify({
        cluster_spec: spec,
        cheap_agents: agentIds,
        trajectories: trajs,
        tightness,
        designer_raw_output: designerRaw,
        hypotheses,
        evaluations,
        qualified: qualified.map(e => e.hypothesis.name),
        verdict,
        registered,
        similarity_report: similarityReport,
        thresholds: { F_RATIO_THRESHOLD, SPREAD_THRESHOLD, PARAPHRASE_THRESHOLD, TIGHTNESS_THRESHOLD },
    }, null, 2));
    console.log(`[disambig] full evidence: ${outFile}`);
    return { verdict, outFile, qualified, registered, similarityReport };
}

// ── CLI ──────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
if (args.length !== 1) {
    console.error('usage: node cluster_disambiguator.mjs <cluster_spec.json>');
    process.exit(2);
}
await runDisambiguator(args[0]);
