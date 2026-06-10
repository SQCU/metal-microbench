#!/usr/bin/env node
// outer_outer.mjs — iterated effective-dimensionality-driven outer-outer
//
// Composes the existing working algorithms — no internal rewrites:
//
//   pass 0: spawn lock_in_iterative.mjs <experiment_id>
//             (runs the inner pipeline over the operator's predeclared
//              bios + agent_targets; produces multi-context trajectories)
//   for pass k in 1..K_OUTER_OUTER:
//     1. pick a new bio target by ΔPR (eff-dim argmax over K candidates)
//        — using explore_corpus's algorithm
//     2. write a transient one-bio experiment-spec card at the picked
//        target, paired with the original spec's agent_targets
//     3. spawn lock_in_iterative.mjs against the transient spec
//        (so axis_splitter has multi-context data for the new bio)
//     4. for every bio_axis in the registry, spawn axis_splitter.mjs
//        on every bio's trajectory — splitter self-gates on gap ≥ 0.5
//        and Cohen's d ≥ separation threshold
//     5. spawn cluster_disambiguator.mjs on the corpus
//     6. delete the transient experiment-spec card; the produced
//        bio + agents remain in the corpus
//   final: persist a summary, log final eff-dim + derived-axis count
//
// CLI: node outer_outer.mjs <experiment_id>
// Env: LOCK_IN_RUN_ID, EXPERIMENT_ID — same contract as lock_in_iterative
//
// This is the iterated demonstration of feature factorization: each
// pass grows the corpus by one ΔPR-driven bio, then the auto-dispatch
// probes for axes the harness can split. The seed `lock_in_tetrad`
// was designed with deliberate axis collapse (kiss/steal+steal motives
// both include theft) to force the harness to expose its decomposition
// machinery here.

import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import * as L from './harness_lib.mjs';

const EXPERIMENT_ID = process.env.EXPERIMENT_ID || process.argv[2];
if (!EXPERIMENT_ID) {
    console.error('usage: node outer_outer.mjs <experiment_id>');
    process.exit(2);
}

const RUN_ID = process.env.LOCK_IN_RUN_ID
    // LINT-OK-PREFIX-SAFE: RUN_ID stamped into filesystem output dirs, not prompt content.
    || `${EXPERIMENT_ID}-${new Date().toISOString().replace(/[:.]/g, '-')}-oo`;

const HARNESS_DIR = path.dirname(new URL(import.meta.url).pathname);
const LOCK_IN_OUT = process.env.USER_PERSONAS_LOCK_IN_OUT
    || process.env.USER_PERSONAS_LOCK_IN_DATA_DIR
    || path.resolve(HARNESS_DIR, '..', 'data', 'lock_in_iterative');
const OUTER_OUTER_OUT_BASE = process.env.USER_PERSONAS_OUTER_OUTER_DIR
    || path.resolve(HARNESS_DIR, '..', 'data', 'outer_outer');
const OUT_DIR = path.join(OUTER_OUTER_OUT_BASE, EXPERIMENT_ID);
fs.mkdirSync(OUT_DIR, { recursive: true });

// Defense-in-depth k-ceiling (mirrors the validator + lock_in_iterative
// self-clamp added in task #191). Even an operator-typo of 50 outer-outer
// iterations gets bounded.
const K_MAX_CEILING = 5;
function clampK(label, value, fallback) {
    const n = Number.isInteger(value) ? value : fallback;
    if (n > K_MAX_CEILING) {
        console.warn(`[outer_outer] ${label}=${n} exceeds ceiling ${K_MAX_CEILING}; clamping`);
        return K_MAX_CEILING;
    }
    return Math.max(1, n);
}

// Auto-dispatch heuristic constants (mirror what's in axis_splitter +
// cluster_disambiguator; we just decide which (bio, axis) pairs to spawn
// for and let the spawned scripts self-gate on their own thresholds).
const K_CANDIDATES         = 6;     // ΔPR-argmax over this many random targets per pass
const CLUSTER_DISTANCE_EPS = 1.5;   // bios closer than this in B-space → disambiguator gets called

// ── spawn helper ─────────────────────────────────────────────────────

function spawnAndWait(scriptName, args, label, extraEnv = {}) {
    return new Promise((resolve) => {
        const child = spawn('node', [path.join(HARNESS_DIR, scriptName), ...args], {
            stdio: 'inherit',
            env: { ...process.env, ...extraEnv },
        });
        const t0 = Date.now();
        child.on('exit', (code, signal) => {
            const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
            console.log(`[outer_outer] ${label} exit=${code} signal=${signal} elapsed=${elapsed}s`);
            resolve({ code, signal, elapsed_s: Number(elapsed) });
        });
    });
}

// ── ΔPR target picker (ported from explore_corpus.mjs:108) ───────────
//
// Pick the bio target that, if achieved as a new bio in the corpus,
// maximizes the participation-ratio effective-dimensionality. K random
// candidates are scored against the current corpus + each candidate as
// a hypothetical new entry; the highest ΔPR wins. For tiny corpora
// (n < 2) ΔPR is undefined so the first candidate is picked random.

function _rngFromSeed(seed) {
    let s = seed >>> 0;
    return () => {
        s = ((s * 1664525) + 1013904223) >>> 0;
        return s / 0x100000000;
    };
}

function randomTargetForAxes(axisNames, rng) {
    const t = {};
    for (const a of axisNames) t[a] = 1 + Math.round(rng() * 4);  // integer 1..5
    return t;
}

function sigsByBio(bios) {
    const out = {};
    for (const b of bios) out[b.id] = b.signature || {};
    return out;
}

function pickNextTarget(bios, axisNames, kCandidates, rng, lineageWeights = null) {
    // lineageWeights (L.lineageWeightsFromCards): keeps the ΔPR objective
    // honest as the registry grows — duplicated/descendant axes share one
    // unit of variance weight, so the picker maximizes BEHAVIORAL
    // diversity, not bookkeeping multiplicity.
    const baseline = L.effDimParticipationRatio(sigsByBio(bios), axisNames, lineageWeights);
    const candidates = Array.from({ length: kCandidates }, () => randomTargetForAxes(axisNames, rng));
    if (bios.length < 2) {
        return {
            target: candidates[0],
            mode: 'random_warmup',
            baseline_eff_dim: baseline.effDim,
        };
    }
    const scored = candidates.map(c => {
        const sigs = sigsByBio(bios);
        sigs.__candidate__ = c;
        const after = L.effDimParticipationRatio(sigs, axisNames, lineageWeights);
        const delta = (after.effDim ?? 0) - (baseline.effDim ?? 0);
        return { candidate: c, after_eff_dim: after.effDim, delta };
    });
    scored.sort((a, b) => b.delta - a.delta);
    return {
        target: scored[0].candidate,
        mode: 'delta_pr_argmax',
        baseline_eff_dim: baseline.effDim,
        expected_after_eff_dim: scored[0].after_eff_dim,
        expected_delta_eff_dim: scored[0].delta,
        all_scored: scored,
    };
}

// ── transient one-bio spec construction ──────────────────────────────
//
// To feed a new ΔPR-picked bio target through lock_in_iterative's
// full inner pipeline (so the new bio gets multi-context trajectory
// data that axis_splitter can work with), we materialize a transient
// experiment-spec card. It uses the operator's original agent_targets +
// bio_axes + agent_axes + counterparty + loop_control, but with ONE
// bio whose target_bio is the freshly picked vector.

async function postTransientSpec(transientId, originalSpec, biosArray) {
    const card = {
        ...originalSpec,
        id: transientId,
        name: `${originalSpec.name || originalSpec.id} (outer_outer iter)`,
        description: `Transient outer_outer iteration card derived from ${originalSpec.id}.`,
        bios: biosArray,
    };
    delete card.experiment_schema;
    delete card.created_at;
    const body = JSON.stringify(card);
    await L.http('POST', `${L.ENDPOINTS.PLUGIN}/experiments/${transientId}`, JSON.parse(body));
}

async function deleteSpec(specId) {
    try { await L.http('DELETE', `${L.ENDPOINTS.PLUGIN}/experiments/${specId}`); }
    catch (e) { console.warn(`[outer_outer] DELETE ${specId}: ${e.message}`); }
}

// ── auto-dispatch sweep over the current corpus ──────────────────────
//
// After each pass:
//   - For each bio whose trajectory exists on disk, for each bio-kind
//     axis in the current registry, spawn axis_splitter. Splitter
//     self-gates: gap < 0.5 → no-op; gap ≥ 0.5 and best-hypothesis
//     Cohen's d ≥ SEPARATION_THRESHOLD → register derived axes.
//   - Probe pairwise bio L2 in B-space; if any pair within
//     CLUSTER_DISTANCE_EPS → cluster_disambiguator on the original
//     spec.

function loadTrajectory(experimentId, bioSlug) {
    const p = path.join(LOCK_IN_OUT, experimentId, `${bioSlug}.json`);
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

async function autoDispatchSweep(originalSpec, completedTrajectories, axesNow) {
    // Idempotency: parents that already have registered children are not
    // re-split (the splitter self-gates too, but skipping here saves a
    // spawn per candidate and keeps the sweep arithmetic honest). The
    // children themselves remain candidates — they accrue per-turn scores
    // in later trajectories and split further when evidence justifies it.
    const factored = new Set(axesNow
        .filter(a => a.derived_from?.parent)
        .map(a => a.derived_from.parent));
    const bioKindAxes = axesNow.filter(a =>
        (a.kind === 'bio' || a.kind === 'either') && !factored.has(a.id));
    const skipped = axesNow.filter(a =>
        (a.kind === 'bio' || a.kind === 'either') && factored.has(a.id));
    if (skipped.length > 0) {
        console.log(`[outer_outer]   sweep skip (already factored): ${skipped.map(a => a.id).join(', ')}`);
    }
    console.log(`[outer_outer]   sweep: ${completedTrajectories.length} bios × ${bioKindAxes.length} bio-kind axes = ${completedTrajectories.length * bioKindAxes.length} splitter candidates`);

    const splitterResults = [];
    for (const traj of completedTrajectories) {
        for (const ax of bioKindAxes) {
            const r = await spawnAndWait(
                'axis_splitter.mjs', [traj.path, ax.id],
                `axis_splitter[${traj.bio_slug}/${ax.id}]`,
                { LOCK_IN_RUN_ID: RUN_ID });
            splitterResults.push({ bio_slug: traj.bio_slug, axis: ax.id, exit_code: r.code, elapsed_s: r.elapsed_s });
        }
    }

    // cluster collapse: pairwise L2 over current corpus bios
    const allBios = (await L.http('GET', `${L.ENDPOINTS.PLUGIN}/personas`)).personas || [];
    const collapsedPairs = [];
    for (let i = 0; i < allBios.length; i++) {
        for (let j = i + 1; j < allBios.length; j++) {
            const a = allBios[i], b = allBios[j];
            if (!a.signature || !b.signature) continue;
            const keys = new Set([...Object.keys(a.signature), ...Object.keys(b.signature)]);
            let s = 0;
            for (const k of keys) {
                const d = (a.signature[k] ?? 3) - (b.signature[k] ?? 3);
                s += d * d;
            }
            const dist = Math.sqrt(s);
            if (dist < CLUSTER_DISTANCE_EPS) collapsedPairs.push({ a: a.id, b: b.id, distance: dist });
        }
    }

    let disambResult = null;
    if (collapsedPairs.length > 0) {
        console.log(`[outer_outer]   ${collapsedPairs.length} bio pairs within ε=${CLUSTER_DISTANCE_EPS} → dispatching cluster_disambiguator`);
        // The disambiguator consumes a CLUSTER spec ({cluster_id,
        // nominal_tight_axis, counterparty_avatar, bios:[{canonical_key,
        // name, prose}]}), NOT an experiment spec. Passing the experiment
        // card here produced cluster_id=undefined → "undefined-…-cheap"
        // agent ids → 400 → child exit 1 on every dispatch (observed
        // 2026-06-10). Synthesize a transient cluster spec from the
        // collapsed pairs instead.
        const memberIds = [...new Set(collapsedPairs.flatMap(p => [p.a, p.b]))];
        const members = allBios.filter(b => memberIds.includes(b.id));
        // nominal_tight_axis = the axis the cluster is TIGHTEST on,
        // measured from the members' own signatures (empirical, not
        // authored): smallest max−min spread across members, requiring
        // the axis to be present on every member.
        let tightAxis = null, tightSpread = Infinity;
        const axisIds = [...new Set(members.flatMap(b => Object.keys(b.signature || {})))];
        for (const ax of axisIds) {
            const vs = members.map(b => b.signature?.[ax]).filter(Number.isFinite);
            if (vs.length !== members.length) continue;
            const spread = Math.max(...vs) - Math.min(...vs);
            if (spread < tightSpread) { tightSpread = spread; tightAxis = ax; }
        }
        // Agent ids derived from this (disambiguator: `${cluster_id}-${bio}-cheap`)
        // must satisfy the plugin's ID_RE = /^[a-z0-9_-]+$/ — LOWERCASE only —
        // and stay short enough to read. Use the run id's unique tail.
        const clusterId = `oo-collapse-${RUN_ID.slice(-8)}`
            .toLowerCase().replace(/[^a-z0-9_-]/g, '-');
        const clusterSpec = {
            cluster_id: clusterId,
            label: `outer-outer collapse cluster (${originalSpec.id}, ε=${CLUSTER_DISTANCE_EPS})`,
            nominal_tight_axis: tightAxis,
            counterparty_avatar: originalSpec.counterparty_avatar || 'the-rock.png',
            bios: members.map(b => ({
                canonical_key: b.id,
                name: b.name || b.id,
                prose: b.bio || '',
            })),
            collapsed_pairs: collapsedPairs,
        };
        const pluginDir = process.env.USER_PERSONAS_PLUGIN_DIR
            || path.resolve(HARNESS_DIR, '..');
        const clustersDir = process.env.USER_PERSONAS_CLUSTERS_DATA_DIR
            || path.join(pluginDir, 'data', 'clusters');
        fs.mkdirSync(clustersDir, { recursive: true });
        const specPath = path.join(clustersDir, `${clusterId}.json`);
        fs.writeFileSync(specPath, JSON.stringify(clusterSpec, null, 2));
        console.log(`[outer_outer]   cluster spec: ${specPath} (${members.length} bios, tight axis: ${tightAxis})`);
        disambResult = await spawnAndWait(
            'cluster_disambiguator.mjs', [specPath],
            'cluster_disambiguator', { LOCK_IN_RUN_ID: RUN_ID });
    }

    return { splitterResults, collapsedPairs, disambResult };
}

// ── main ─────────────────────────────────────────────────────────────

const originalSpec = await L.fetchExperiment(EXPERIMENT_ID);
// K_OUTER_OUTER_OVERRIDE env var allows CLI override without editing the spec card.
// Pass K_OUTER_OUTER_OVERRIDE=2 to run exactly pass 0 + one ΔPR spur pass.
const _kFromEnv = process.env.K_OUTER_OUTER_OVERRIDE ? Number(process.env.K_OUTER_OUTER_OVERRIDE) : NaN;
const K_OUTER_OUTER = clampK('k_max_outer_outer',
    Number.isInteger(_kFromEnv) ? _kFromEnv : originalSpec.loop_control?.k_max_outer_outer, 3);
console.log(`[outer_outer] experiment=${EXPERIMENT_ID} run_id=${RUN_ID}`);
console.log(`[outer_outer] K_OUTER_OUTER=${K_OUTER_OUTER} (predeclared bios pass 0 + ${K_OUTER_OUTER - 1} ΔPR passes)`);

const passRecords = [];
const transientSpecs = [];   // track for cleanup

// Pass 0: predeclared bios from the original spec
console.log(`\n[outer_outer] ═══ pass 0 / ${K_OUTER_OUTER - 1} : predeclared bios ═══`);
const pass0Result = await spawnAndWait(
    'lock_in_iterative.mjs', [EXPERIMENT_ID], 'lock_in_iterative[pass 0]',
    { LOCK_IN_RUN_ID: RUN_ID, EXPERIMENT_ID });

const pass0Trajectories = originalSpec.bios.map(b => ({
    bio_slug: b.slug,
    path: path.join(LOCK_IN_OUT, EXPERIMENT_ID, `${b.slug}.json`),
})).filter(t => fs.existsSync(t.path));

const pass0Axes = await L.fetchAxes();
const pass0Sweep = await autoDispatchSweep(originalSpec, pass0Trajectories, pass0Axes);
passRecords.push({ pass: 0, inner_result: pass0Result, sweep: pass0Sweep });

// Subsequent passes: ΔPR-picked bio targets, one new bio per pass
const bioAxisNames = originalSpec.bio_axes || [];
const rng = _rngFromSeed(Date.now() ^ 0x9e3779b9);

for (let k = 1; k < K_OUTER_OUTER; k++) {
    console.log(`\n[outer_outer] ═══ pass ${k} / ${K_OUTER_OUTER - 1} : ΔPR-driven new bio ═══`);

    // Refresh axis registry — splits from prior pass may have grown it
    const axesNow = await L.fetchAxes();
    const bioKindNames = axesNow.filter(a => a.kind === 'bio' || a.kind === 'either').map(a => a.id);
    console.log(`[outer_outer]   axes registry: ${axesNow.length} total (bio-kind: ${bioKindNames.length})`);

    // Fetch current corpus bios for ΔPR baseline
    const allBios = (await L.http('GET', `${L.ENDPOINTS.PLUGIN}/personas`)).personas || [];
    const lineageW = L.lineageWeightsFromCards(axesNow);
    const pick = pickNextTarget(allBios, bioAxisNames, K_CANDIDATES, rng, lineageW);
    console.log(`[outer_outer]   pick: mode=${pick.mode} baseline_eff_dim=${pick.baseline_eff_dim?.toFixed(3) ?? 'n/a'} expected_after=${pick.expected_after_eff_dim?.toFixed(3) ?? 'n/a'} ΔPR=${pick.expected_delta_eff_dim?.toFixed(3) ?? 'n/a'}`);
    console.log(`[outer_outer]   target: ${bioAxisNames.map(a => `${a.slice(0,8)}=${pick.target[a]}`).join(' ')}`);

    // Materialize a transient one-bio spec at the picked target
    // LINT-OK-PREFIX-SAFE: transient experiment id (filesystem + DB key), not prompt content.
    // Prefix with "outer-outer-" so bio provenance.experiment_id is unambiguously
    // traceable to this outer-outer run (acceptance check: startsWith("outer-outer-")).
    const transientId = `outer-outer-${EXPERIMENT_ID}-pass${k}-${Date.now().toString(36)}`; // LINT-OK-PREFIX-SAFE: transient experiment id, not prompt content.
    const newBioSlug = `oo-${EXPERIMENT_ID}-pass${k}`;
    const newBio = {
        canonical_key: `user-personas-${newBioSlug}.png`,
        slug: newBioSlug,
        name: `outer_outer ${EXPERIMENT_ID} pass ${k}`,
        target_bio: pick.target,
        design_brief: `ΔPR-driven target for ${EXPERIMENT_ID} corpus expansion. ` +
            `Show-don't-tell prose for a user whose chat behavior would land at the target signature.`,
        seed_phrase: 'outer_outer iteration',
    };
    await postTransientSpec(transientId, originalSpec, [newBio]);
    transientSpecs.push(transientId);

    // Run lock_in_iterative against the transient spec
    const innerResult = await spawnAndWait(
        'lock_in_iterative.mjs', [transientId], `lock_in_iterative[pass ${k}]`,
        { LOCK_IN_RUN_ID: RUN_ID, EXPERIMENT_ID: transientId });

    // Auto-dispatch sweep: probe all bios (including the new one) on all bio-kind axes
    const passTrajectories = [
        ...pass0Trajectories,
        { bio_slug: newBio.slug, path: path.join(LOCK_IN_OUT, transientId, `${newBio.slug}.json`) },
    ].filter(t => fs.existsSync(t.path));
    const sweep = await autoDispatchSweep(originalSpec, passTrajectories, await L.fetchAxes());
    passRecords.push({ pass: k, target: pick.target, inner_result: innerResult, sweep });
}

// Cleanup transient specs
for (const id of transientSpecs) {
    await deleteSpec(id);
}

// Summary
const finalAxes = await L.fetchAxes();
const finalDerived = finalAxes.filter(a => a.derived_from);
const summary = {
    experiment_id: EXPERIMENT_ID,
    run_id: RUN_ID,
    started_at: new Date().toISOString(),
    K_OUTER_OUTER,
    passes: passRecords,
    final_axes: finalAxes.length,
    derived_axes: finalDerived.map(a => ({ id: a.id, derived_from: a.derived_from })),
};
const summaryPath = path.join(OUT_DIR, `${RUN_ID}.json`);
fs.writeFileSync(summaryPath, JSON.stringify(summary, null, 2));
console.log(`\n[outer_outer] summary → ${summaryPath}`);
console.log(`[outer_outer] axes at end: ${finalAxes.length} (derived: ${finalDerived.length})`);
if (finalDerived.length > 0) {
    console.log(`[outer_outer] derived axes:`);
    for (const a of finalDerived) console.log(`  ${a.id} ← ${JSON.stringify(a.derived_from)}`);
}
console.log(`[outer_outer] DONE`);
