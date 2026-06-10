#!/usr/bin/env node
// explore_corpus.mjs
//
// Item 3 in docs/feature_factorization_design.md: outer-outer loop
// with effective-dimensionality objective.
//
// MVP scope: pick the next bio target by ΔPR (participation-ratio
// proxy for PCA effective dimensionality), design bio prose against
// it (single-pass), pair with a cheap agent, run one chat, measure
// the resulting bio's signature on the bio-axis subset, append to
// the corpus. Repeat N times. Persist state.
//
// Deliberately omits (for this MVP):
//   - the full inner loop (designAgent over multiple agent_targets);
//     we use one cheap "be yourself" agent per bio
//   - automatic dispatch to splitter / disambiguator when entanglements
//     or tight clusters are detected (next iteration of this script)
//   - sparse-sampling controller (we judge on ALL bio_axes per turn;
//     pickSubset integration deferred)
//
// CLI:
//   node explore_corpus.mjs <state.json> [--max-iter N] [--k-candidates K]
//
// On first run, if state.json doesn't exist or has empty bios[], the
// initial iterations pick random targets (no cloud to be far from);
// later iterations switch to ΔPR-driven selection.

import fs from 'node:fs';
import path from 'node:path';
import * as L from './harness_lib.mjs';

// Project id → name to keep compatibility with code that uses `.name`.
const _AXES_CACHE = (await L.fetchAxes()).map(a => ({
    name: a.id, def: a.def, kind: a.kind, derived_from: a.derived_from || null,
}));
function allAxes() { return _AXES_CACHE; }
// Lineage weights so the ΔPR objective counts each latent direction once
// regardless of how many descendant/duplicate axes the splitter registered.
const _LINEAGE_W = L.lineageWeightsFromCards(
    _AXES_CACHE.map(a => ({ id: a.name, derived_from: a.derived_from })));

// ── config ───────────────────────────────────────────────────────────

const DEFAULT_MAX_ITER     = 3;
const DEFAULT_K_CANDIDATES = 8;
const N_TURNS_PER_CHAT     = 4;
// TEMP_BIO_DESIGN removed: per moratorium, no per-call temperature override.
// Bridge default = 1.0 (the value this constant carried anyway).

// Bio-side axes are discovered from the plugin's axis cards. Older
// versions hardcoded a stale six-axis list from a prior experiment
// (`normative_directionality`, `warm`, etc.); that made the supposedly
// self-contained cloned repo depend on an axis registry it no longer
// shipped. The current source of truth is axes/*.json.
const BIO_AXIS_NAMES = allAxes()
    .filter(a => a.kind === 'bio' || a.kind === 'either')
    .map(a => a.name);
if (BIO_AXIS_NAMES.length === 0) {
    throw new Error('explore_corpus: no bio/either axis cards loaded from plugin');
}

// ── state I/O ────────────────────────────────────────────────────────

function loadState(statePath) {
    if (fs.existsSync(statePath)) {
        const s = JSON.parse(fs.readFileSync(statePath, 'utf8'));
        if (!Array.isArray(s.bios))       s.bios = [];
        if (!Array.isArray(s.iterations)) s.iterations = [];
        return s;
    }
    return {
        version: 1,
        created_at: new Date().toISOString(),
        counterparty_avatar: 'the-rock.png',
        bio_axes: BIO_AXIS_NAMES,
        bios: [],
        iterations: [],
    };
}

function saveState(statePath, state) {
    fs.mkdirSync(path.dirname(statePath), { recursive: true });
    fs.writeFileSync(statePath, JSON.stringify(state, null, 2));
}

// ── target proposal + scoring ────────────────────────────────────────

function randomTarget(rng) {
    const t = {};
    // Continuous in [1.5, 4.5] so we don't always sit at corners
    for (const a of BIO_AXIS_NAMES) t[a] = 1.5 + rng() * 3.0;
    return t;
}

function sigsByBio(state) {
    const out = {};
    for (const b of state.bios) {
        out[b.canonical_key] = b.measured_sig || {};
    }
    return out;
}

function effDim(state) {
    return L.effDimParticipationRatio(sigsByBio(state), BIO_AXIS_NAMES, _LINEAGE_W);
}

function effDimWithCandidate(state, candidate) {
    const sigs = sigsByBio(state);
    sigs.__candidate__ = candidate;
    return L.effDimParticipationRatio(sigs, BIO_AXIS_NAMES, _LINEAGE_W);
}

function pickNextTarget(state, kCandidates, rng) {
    const baseline = effDim(state);
    const candidates = Array.from({ length: kCandidates }, () => randomTarget(rng));
    if (state.bios.length < 2) {
        // No baseline — pick random target. ΔPR undefined for k<2.
        return {
            target: candidates[0],
            mode: 'random_warmup',
            baseline_eff_dim: baseline.effDim,
            candidates_evaluated: 1,
        };
    }
    const scored = candidates.map(c => {
        const after = effDimWithCandidate(state, c);
        const delta = (after.effDim ?? 0) - (baseline.effDim ?? 0);
        return { candidate: c, after_eff_dim: after.effDim, delta };
    });
    scored.sort((a, b) => b.delta - a.delta);
    return {
        target: scored[0].candidate,
        mode: 'delta_pr_argmax',
        baseline_eff_dim: baseline.effDim,
        all_scored: scored,
        expected_delta_eff_dim: scored[0].delta,
        expected_after_eff_dim: scored[0].after_eff_dim,
    };
}

// ── single-pass bio designer ─────────────────────────────────────────

function fmtTargetForPrompt(target) {
    return BIO_AXIS_NAMES.map(a => {
        const axis = allAxes().find(x => x.name === a);
        return `  - ${a} = ${target[a].toFixed(2)} (rubric: ${axis.def})`;
    }).join('\n');
}

async function designBioOnce(target, designBrief = '') {
    const sys =
        'You design a user-side biography for a behavioral measurement ' +
        'study. Given a target per-axis signature (Likert 1-5 floats), ' +
        'write a 3-5 sentence first-person prose biography of a user ' +
        'whose CHAT BEHAVIOR (not whose self-description) would, when ' +
        'aggregated across several user turns, hit each axis near its ' +
        'target value. Show-don\'t-tell: describe what the user does, ' +
        'how they speak, what kinds of moves they make — not what they ' +
        'claim about themselves. Output ONLY the bio prose, no preamble.';
    const usr =
        '## Target signature on the bio-axis subset\n\n' +
        fmtTargetForPrompt(target) + '\n\n' +
        (designBrief ? `## Design brief\n\n${designBrief}\n\n` : '') +
        'Write the bio prose now.';
    // Per moratorium (lint_generation_config.mjs): no max_tokens / no
    // temperature at the caller layer. Bridge default temperature=1.0 +
    // EOS termination apply. (TEMP_BIO_DESIGN was already 1.0; the cap
    // on max_tokens was a soft "bio prose should be short" — but bio
    // length is controlled by the system-prompt "3-5 sentences" instruction,
    // not by truncation.)
    return await L.bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
}

// ── one explore iteration ────────────────────────────────────────────

async function exploreOneIteration(state, iter, rng, cp, kCandidates) {
    const t0 = Date.now();

    // 1. Pick target by eff-dim
    const pick = pickNextTarget(state, kCandidates, rng);
    console.log(`\n[iter ${iter}] mode=${pick.mode} baseline_eff_dim=${pick.baseline_eff_dim?.toFixed(3) ?? 'n/a'}`);
    if (pick.expected_delta_eff_dim != null) {
        console.log(`            expected ΔPR=${pick.expected_delta_eff_dim.toFixed(3)} (after=${pick.expected_after_eff_dim.toFixed(3)})`);
    }
    console.log(`            target: ${BIO_AXIS_NAMES.map(a => `${a.slice(0,4)}=${pick.target[a].toFixed(2)}`).join(' ')}`);

    // 2. Design bio prose
    const prose = await designBioOnce(pick.target);
    console.log(`            bio prose: ${prose.slice(0, 150).replace(/\n/g, ' ')}…`);

    // 3. Install bio + cheap agent
    // LINT-OK-PREFIX-SAFE: slug for bio canonical_key (filesystem-ish ID), not prompt content.
    const slug = `explore-iter${iter}-${Date.now().toString(36)}`;
    const canonical_key = `${slug}.png`;
    const name = `Explore Iter ${iter}`;
    const bio = { canonical_key, name, prose };
    await L.saveBio(bio);
    const cheapAgentText = await L.designCheapAgent(bio);
    const agentId = `${slug}-cheap`;
    await L.saveAgent(agentId, `${name} (cheap)`, cheapAgentText, canonical_key);
    console.log(`            cheap_agent: ${cheapAgentText.slice(0, 100).replace(/\n/g, ' ')}…`);

    // 4. Run one chat × N_TURNS
    const chat = await L.runChat(bio, agentId, cp, N_TURNS_PER_CHAT);
    const turns = L.userTurns(chat);
    console.log(`            ran ${turns.length} user turns vs ${cp.name}`);

    // 5. Judge all turns on all bio axes (kernel-width-bounded, one call per
    //    turn, multi-axis in one prompt via judgeOnAxes).
    const axisRecords = BIO_AXIS_NAMES.map(n => {
        const r = allAxes().find(a => a.name === n);
        return { name: n, def: r.def };
    });
    const judged = await L.saturatedMap(turns, t => L.judgeOnAxes(t, axisRecords));
    const perAxis = {};
    for (const a of BIO_AXIS_NAMES) {
        const vs = judged.map(j => j.sig[a]).filter(Number.isFinite);
        perAxis[a] = vs.length ? vs.reduce((x, y) => x + y, 0) / vs.length : null;
    }
    const dist = {};
    for (const a of BIO_AXIS_NAMES) {
        dist[a] = perAxis[a] == null ? null : Math.abs(perAxis[a] - pick.target[a]);
    }
    const meanDist = Object.values(dist).filter(Number.isFinite)
        .reduce((s, d, _, arr) => s + d / arr.length, 0);
    console.log(`            measured: ${BIO_AXIS_NAMES.map(a => `${a.slice(0,4)}=${perAxis[a]?.toFixed(2) ?? '?'}`).join(' ')}`);
    console.log(`            mean axis distance from target: ${meanDist.toFixed(2)}`);

    // 6. Append to corpus
    const bioRecord = {
        canonical_key, name, prose,
        target_sig: pick.target,
        measured_sig: perAxis,
        dist_per_axis: dist,
        cheap_agent_id: agentId,
        cheap_agent_text: cheapAgentText,
        chat,
        source: `explore_corpus iter ${iter}`,
        added_at: new Date().toISOString(),
    };
    state.bios.push(bioRecord);

    // 7. Compute achieved eff-dim
    const after = effDim(state);
    const achievedDelta = (after.effDim ?? 0) - (pick.baseline_eff_dim ?? 0);
    console.log(`            ACHIEVED eff_dim: ${pick.baseline_eff_dim?.toFixed(3) ?? 'n/a'} → ${after.effDim?.toFixed(3) ?? 'n/a'} (Δ=${achievedDelta.toFixed(3)})`);

    state.iterations.push({
        iter,
        mode: pick.mode,
        baseline_eff_dim: pick.baseline_eff_dim,
        expected_delta_eff_dim: pick.expected_delta_eff_dim ?? null,
        expected_after_eff_dim: pick.expected_after_eff_dim ?? null,
        target: pick.target,
        all_candidates_scored: pick.all_scored ?? null,
        bio_canonical_key: canonical_key,
        measured_sig: perAxis,
        mean_axis_dist: meanDist,
        achieved_eff_dim: after.effDim,
        achieved_delta_eff_dim: achievedDelta,
        elapsed_ms: Date.now() - t0,
    });
    return bioRecord;
}

// ── main ─────────────────────────────────────────────────────────────

function parseArgs(argv) {
    const args = { statePath: null, maxIter: DEFAULT_MAX_ITER, k: DEFAULT_K_CANDIDATES };
    const rest = argv.slice(2);
    for (let i = 0; i < rest.length; i++) {
        if (rest[i] === '--max-iter')        args.maxIter = Number(rest[++i]);
        else if (rest[i] === '--k-candidates') args.k = Number(rest[++i]);
        else if (!args.statePath)            args.statePath = rest[i];
    }
    return args;
}

const args = parseArgs(process.argv);
if (!args.statePath) {
    console.error('usage: node explore_corpus.mjs <state.json> [--max-iter N] [--k-candidates K]');
    process.exit(2);
}

const state = loadState(args.statePath);
console.log(`[explore_corpus] state: ${args.statePath}`);
console.log(`[explore_corpus] corpus size: ${state.bios.length} bios`);
console.log(`[explore_corpus] bio_axes (${BIO_AXIS_NAMES.length}): ${BIO_AXIS_NAMES.join(', ')}`);
console.log(`[explore_corpus] will run ${args.maxIter} iter(s); K=${args.k} candidates per iter`);

const cp = await L.fetchCounterparty(state.counterparty_avatar);
console.log(`[explore_corpus] counterparty: ${cp.name} (sys_prompt ${cp.system_prompt.length} chars)`);

// Cheap deterministic RNG so reruns are reproducible per state size.
let _rngSeed = (state.bios.length + 1) * 0x9e3779b1;
const rng = () => {
    _rngSeed = ((_rngSeed * 1664525) + 1013904223) >>> 0;
    return _rngSeed / 0x100000000;
};

const tAll = Date.now();
const startIter = state.iterations.length;
for (let i = 0; i < args.maxIter; i++) {
    const iter = startIter + i;
    try {
        await exploreOneIteration(state, iter, rng, cp, args.k);
    } catch (e) {
        console.error(`[iter ${iter}] FAILED: ${e.message}`);
        state.iterations.push({ iter, mode: 'failed', error: e.message,
                                elapsed_ms: 0, achieved_eff_dim: null });
    }
    saveState(args.statePath, state);
    console.log(`[iter ${iter}] state persisted`);
}

const finalEff = effDim(state);
// LINT-OK-PREFIX-SAFE: stdout summary log, not prompt content.
console.log(`\n[explore_corpus] done. ${args.maxIter} iter(s) in ${((Date.now()-tAll)/1000).toFixed(1)}s`);
console.log(`[explore_corpus] FINAL corpus size: ${state.bios.length} bios`);
console.log(`[explore_corpus] FINAL eff_dim: ${finalEff.effDim?.toFixed(3) ?? 'n/a'} (${finalEff.note ?? ''})`);
if (finalEff.perAxisVar) {
    console.log(`[explore_corpus] per-axis variance contribution:`);
    for (const a of BIO_AXIS_NAMES) {
        const v = finalEff.perAxisVar[a] ?? 0;
        const p = finalEff.normalized?.[a] ?? 0;
        console.log(`    ${a}: var=${v.toFixed(3)} p=${(p*100).toFixed(1)}%`);
    }
}
