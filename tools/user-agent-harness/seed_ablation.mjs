#!/usr/bin/env node
// Seed paraphrase × convergence ablation.
//
// Counterfactual question: does Claude's at-edit-time paraphrase
// expansion of operator seed phrases (e.g. "rpg wizard but he a
// sagittarius" → "An RPG wizard whose communication style is textbook
// Sagittarius: fire-sign, philosophical, blunt, restless,
// big-idea-loving. References spells, planes, alignment, the weave.")
// IMPROVE the fixed-point loop's ability to converge bios+agents to
// their target axis signatures? Or is it noise the harness would
// reach equivalent (or better) outcomes without?
//
// Method: run two pre-registered experiment-spec cards against the
// existing /experiments/:id/run plugin endpoint.
//
//   - lock_in_tetrad          → Claude-paraphrased design_briefs +
//                               motive_hints (current state, baseline)
//   - lock_in_tetrad_verbatim → operator-verbatim seed phrases as the
//                               design_briefs + motive_hints; same
//                               axes / targets / counterparty /
//                               loop_control
//
// Harvest the per-bio output JSON each run writes to
// data/lock_in_iterative/<id>/<bio_slug>.json — these carry the full
// per-iteration trajectory (target signature, measured signature, L2
// per axis, inner-loop attempts, outer-loop attempts, wall_ms).
//
// Per-cell × per-bio metrics:
//   - outer_k_converged (lower better; null if never converged)
//   - final_max_off (final iteration's max-off-axis distance)
//   - inner_iterations_total (sum across outer-k inner-loops)
//   - wall_ms_total
//   - per_axis_final {axis: distance}
//
// Aggregate:
//   - convergence_rate (bios that converged within k_max_outer / total bios)
//   - mean_final_max_off
//   - mean_wall_ms
//
// We run each cell ONCE per bio (4 compositions × 2 cells). That's a
// noisy sample but the per-iteration trajectory is itself the signal —
// we can see whether one cell needed retries the other didn't.

import * as L from './harness_lib.mjs';
import * as fs from 'fs';
import * as path from 'path';

const PLUGIN = L.ENDPOINTS.PLUGIN;
const CELLS = [
    { id: 'lock_in_tetrad_verbatim', label: 'verbatim'  },
    { id: 'lock_in_tetrad',          label: 'expanded'  },
];
const ITER_OUT_DIR = process.env.USER_PERSONAS_LOCK_IN_OUT
    || process.env.USER_PERSONAS_LOCK_IN_DATA_DIR
    || '/Users/mdot/metal-microbench/data/lock_in_iterative';
const ABLATION_OUT_DIR = process.env.USER_PERSONAS_SEED_ABLATION_DIR
    || '/Users/mdot/metal-microbench/data/seed_ablation';

// ── helpers ────────────────────────────────────────────────────────────

async function dispatchRun(experimentId) {
    const r = await L.http('POST', `${PLUGIN}/experiments/${experimentId}/run`, {});
    return r.run_id;
}

async function waitForLogfileDone(runId, expectedExperimentId) {
    // Follow the active plugin instance instead of assuming root
    // sillytavern-fork's on-disk run directory. st-debug and root ST have
    // distinct plugin data roots, while /experiments/runs/:run_id is the
    // canonical status/log facade for whichever instance we dispatched to.
    const deadline = Date.now() + 30 * 60 * 1000;
    while (Date.now() < deadline) {
        try {
            const status = await L.http('GET', `${PLUGIN}/experiments/runs/${encodeURIComponent(runId)}`);
            if (status?.run?.experiment_id && status.run.experiment_id !== expectedExperimentId) {
                throw new Error(`run ${runId} belongs to ${status.run.experiment_id}, expected ${expectedExperimentId}`);
            }
            const buf = status?.log || '';
            if (buf.includes('CHILD EXIT')) {
                const elapsed = (Date.now() - parseDispatchTs(runId)) / 1000;
                return { ok: !buf.includes('Uncaught'), tailLines: buf.split('\n').slice(-30), wall_s: elapsed };
            }
        } catch (_) {
            // Status record/log not visible yet — keep polling.
        }
        await sleep(15_000);
    }
    return { ok: false, tailLines: ['(timed out waiting 30 min)'], wall_s: 1800 };
}

function parseDispatchTs(runId) {
    // run_id format: <expid>-YYYY-MM-DDTHH-MM-SS-mmmZ-suffix
    const m = runId.match(/(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d+Z)/);
    if (!m) return Date.now();
    const iso = m[1].replace(/-(\d{2})-(\d{2})-(\d+Z)/, ':$1:$2.$3');
    return Date.parse(iso);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function harvestBioOutput(experimentId, bioSlug) {
    // Output path written by lock_in_iterative.mjs.
    const p = path.join(ITER_OUT_DIR, experimentId, `${bioSlug}.json`);
    if (!fs.existsSync(p)) return null;
    return JSON.parse(fs.readFileSync(p, 'utf8'));
}

function summarizeBio(bioOutput, targetBio) {
    if (!bioOutput) return null;
    // Shape (from lock_in_iterative.mjs writes):
    //   { bio, agent_targets, bio_axes, agent_axes, elapsed_ms_total,
    //     result: { stop_reason, best, attempts: [...] } }
    // Each result.attempts[i] = { iter, prose, measured, dist_per_axis,
    //   max_off_axis, innerResults: [{ agentTarget, attempts: [{iter, ...}] }], elapsed_ms }
    const result = bioOutput.result || {};
    const attempts = result.attempts || [];
    const best = result.best || attempts[attempts.length - 1] || null;
    const converged = result.stop_reason === 'converged';
    const outer_k_converged = converged ? (best?.iter ?? null) : null;
    const final_max_off = best?.max_off_axis ?? null;
    const per_axis_final = best?.dist_per_axis || {};
    // Per-axis distance fallback if best didn't carry it (older runs).
    if (Object.keys(per_axis_final).length === 0 && best?.measured) {
        for (const ax of Object.keys(targetBio)) {
            per_axis_final[ax] = Math.abs((best.measured[ax] ?? 3) - targetBio[ax]);
        }
    }
    // Total inner iterations across all outer attempts and all
    // agent_targets within each outer.
    let inner_iterations_total = 0;
    for (const a of attempts) {
        for (const ir of (a.innerResults || [])) {
            inner_iterations_total += (ir.attempts || []).length;
        }
    }
    const wall_ms_total = bioOutput.elapsed_ms_total || 0;
    return {
        outer_k_converged,
        stop_reason: result.stop_reason,
        final_max_off,
        inner_iterations_total,
        wall_ms_total,
        per_axis_final,
        outer_attempts_count: attempts.length,
    };
}

// ── main ───────────────────────────────────────────────────────────────

console.log(`[seed_ablation] cells: ${CELLS.map(c => c.label).join(', ')}`);

const cellResults = {};
for (const cell of CELLS) {
    console.log(`\n[seed_ablation] === cell ${cell.label} (experiment=${cell.id}) ===`);
    const t0 = Date.now();
    const runId = await dispatchRun(cell.id);
    console.log(`[seed_ablation]   dispatched run_id=${runId}`);
    const { ok, tailLines, wall_s } = await waitForLogfileDone(runId, cell.id);
    console.log(`[seed_ablation]   run ${ok ? 'OK' : 'FAILED'} after ${wall_s.toFixed(1)}s`);
    if (!ok) {
        console.log('[seed_ablation]   tail of log:');
        for (const line of tailLines.slice(-15)) console.log(`     ${line}`);
    }
    // Harvest both bios' output JSON
    const spec = await L.fetchExperiment(cell.id);
    const bios = {};
    for (const b of spec.bios) {
        const out = harvestBioOutput(cell.id, b.slug);
        const summary = summarizeBio(out, b.target_bio);
        bios[b.slug] = {
            target_bio: b.target_bio,
            design_brief: b.design_brief,
            summary,
            raw_outer_count: out?.outer_iterations?.length ?? 0,
        };
        if (summary) {
            console.log(`     [${b.slug}] outer_k=${summary.outer_k_converged ?? 'NOT-CONV'} max_off=${summary.final_max_off.toFixed(2)} inner_total=${summary.inner_iterations_total} wall=${(summary.wall_ms_total/1000).toFixed(1)}s`);
        } else {
            console.log(`     [${b.slug}] NO OUTPUT FILE`);
        }
    }
    cellResults[cell.label] = {
        experiment_id: cell.id,
        run_id: runId,
        wall_s_total: wall_s,
        bios,
    };
}

// ── comparison table ───────────────────────────────────────────────────

console.log('\n[seed_ablation] === COMPARISON ===');
function fmt(v, w = 8) {
    return (v == null ? 'null' : v.toString()).padEnd(w);
}
function fmtNum(v, dp = 2, w = 8) {
    return (v == null ? 'null' : v.toFixed(dp)).padEnd(w);
}

const cells = Object.keys(cellResults);
console.log('\nbio'.padEnd(28) + 'cell'.padEnd(12) + 'outer_k'.padEnd(10) + 'max_off'.padEnd(10) + 'inner#'.padEnd(8) + 'wall_s'.padEnd(8) + 'design_brief (truncated)');
console.log('─'.repeat(120));
const allBioSlugs = new Set();
for (const c of cells) for (const b of Object.keys(cellResults[c].bios)) allBioSlugs.add(b);
for (const slug of allBioSlugs) {
    for (const c of cells) {
        const b = cellResults[c].bios[slug];
        if (!b || !b.summary) {
            console.log(`${slug.padEnd(28)}${c.padEnd(12)}${'NO-OUT'.padEnd(10)}`);
            continue;
        }
        const s = b.summary;
        const brief = b.design_brief.slice(0, 50);
        console.log(
            `${slug.padEnd(28)}${c.padEnd(12)}${fmt(s.outer_k_converged, 10)}${fmtNum(s.final_max_off, 2, 10)}${fmt(s.inner_iterations_total, 8)}${fmtNum(s.wall_ms_total/1000, 1, 8)}${brief}`);
    }
}

// Aggregate by cell.
console.log('\nAGGREGATE per cell:');
for (const c of cells) {
    const sums = Object.values(cellResults[c].bios).map(b => b.summary).filter(Boolean);
    const n = sums.length;
    const conv = sums.filter(s => s.outer_k_converged != null).length;
    const meanMaxOff = sums.reduce((s, x) => s + x.final_max_off, 0) / Math.max(1, n);
    const meanWall = sums.reduce((s, x) => s + x.wall_ms_total, 0) / Math.max(1, n);
    const meanInner = sums.reduce((s, x) => s + x.inner_iterations_total, 0) / Math.max(1, n);
    console.log(`  ${c}: n=${n} convergence_rate=${(conv/n*100).toFixed(0)}% mean_max_off=${meanMaxOff.toFixed(2)} mean_inner=${meanInner.toFixed(1)} mean_wall=${(meanWall/1000).toFixed(1)}s`);
}

// Persist.
fs.mkdirSync(ABLATION_OUT_DIR, { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = path.join(ABLATION_OUT_DIR, `run_${ts}.json`);
fs.writeFileSync(outPath, JSON.stringify(cellResults, null, 2));
console.log(`\n[seed_ablation] cell results → ${outPath}`);
