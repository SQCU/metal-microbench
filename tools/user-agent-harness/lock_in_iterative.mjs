#!/usr/bin/env node
// lock_in_iterative.mjs
//
// Demonstrates THE LOOP: user bios + user-agents produced by fixed-point
// iteration against a judge that scores on this experiment's axes.
//
// Scope:
//   - Inner (user-agent) designer iterates K_max_inner times. Each pass:
//       DESIGNER_A → agent_text → /poll → N user turns vs counterparty
//       → judgeOnAxes(agent_axes) per turn → mean per axis → compare
//       to target on agent axes → either convergence (early exit) or
//       feedback to designer for next iteration.
//   - Outer (bio) designer iterates K_max_outer times. Each pass:
//       DESIGNER_B → bio_prose → run inner for every agent_target →
//       aggregate per-turn scores on bio_axes across all agent chats →
//       compare to bio target → convergence or feedback.
//
// EVERYTHING experiment-specific (which bios, which agent_targets,
// which axes, which counterparty, loop control values) lives on the
// experiment-spec card at experiments/<id>.json. This script is the
// pure algorithm; experiments/lock_in_tetrad.json is the canonical
// demo configuration (RPG Wizard/Rogue × Steals/Romances-and-Steals).
//
// Usage:
//   node lock_in_iterative.mjs [experiment_id]
//   # default: experiment_id = 'lock_in_tetrad'
//
// Output: trajectories per (bio, agent) including ALL design iterations
// the loop took to converge, with per-iteration measurements visible.

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
// All HTTP / bridge / persistence / chat / judge live in harness_lib so
// the contract surface has ONE source of truth.
import {
    ENDPOINTS,
    saveBio, saveAgent,
    fetchCounterparty, runChat, userTurns,
    judgeOnAxes,
    fetchAxes, fetchExperiment,
    bridgeCall,
} from './harness_lib.mjs';
const { ST, BRIDGE, PLUGIN } = ENDPOINTS;

const EXPERIMENT_ID = process.argv[2] || 'lock_in_tetrad';

// Load the experiment spec from the plugin. Bios, agent_targets, axis
// subsets, counterparty, and loop-control all come from there.
const spec = await fetchExperiment(EXPERIMENT_ID);

// Resolve axis cards (with rubrics) for the axis-id lists the spec
// declares. The plugin already has full rubrics in axes/*.json; we
// project to {name, def} for judgeOnAxes / designer prompts.
const allAxesCards = await fetchAxes();
const axesById = new Map(allAxesCards.map(a => [a.id, a]));
function axisRef(id) {
    const a = axesById.get(id);
    if (!a) throw new Error(`experiment ${EXPERIMENT_ID} references axis '${id}' but no card at axes/${id}.json`);
    return { name: a.id, def: a.def };
}
const AGENT_AXES = spec.agent_axes.map(axisRef);
const BIO_AXES   = spec.bio_axes.map(axisRef);
const ALL_AXES   = [...AGENT_AXES, ...BIO_AXES];

const BIOS          = spec.bios;
const AGENT_TARGETS = spec.agent_targets;

// Loop control values from the spec (with defaults baked into the
// plugin-side validateExperimentCard so omitted keys behave sanely).
const K_MAX_INNER     = spec.loop_control.k_max_inner;
const K_MAX_OUTER     = spec.loop_control.k_max_outer;
const N_TURNS_PER_CHAT = spec.loop_control.n_turns_per_chat;
const EPS_PER_AXIS    = spec.loop_control.eps_per_axis;
const STALL_WINDOW    = spec.loop_control.stall_window;
const STALL_THRESHOLD = spec.loop_control.stall_threshold;

const OUT_DIR = `/Users/mdot/metal-microbench/data/lock_in_iterative/${EXPERIMENT_ID}`;

// ── judge ────────────────────────────────────────────────────────────
//
// Judge calls go through harness_lib.judgeOnAxes (V6_pure_minimal prompt,
// selected by the prior A/B run as best on MAE + floor-bias). Same
// {sig, raw} return shape this script's call sites expect.

const judgeOneTurn = judgeOnAxes;   // alias for call-site readability

function meanSig(sigs, axes) {
    const out = {};
    for (const a of axes) {
        const vs = sigs.map(s => s[a.name]).filter(v => Number.isFinite(v));
        out[a.name] = vs.length ? vs.reduce((x, y) => x + y, 0) / vs.length : null;
    }
    return out;
}

function distancePerAxis(sig, target, axes) {
    const d = {};
    for (const a of axes) {
        const m = sig[a.name];
        const t = target[a.name];
        d[a.name] = (m == null || t == null) ? null : Math.abs(m - t);
    }
    return d;
}

function maxOffAxes(distancePerAxis) {
    return Math.max(...Object.values(distancePerAxis).filter(v => v != null));
}

function fmtSig(sig) {
    return Object.entries(sig).map(([k, v]) => `${k.split('_').map(p => p.slice(0,3)).join('_')}=${v == null ? '?' : v.toFixed(2)}`).join(' ');
}

/**
 * Velocity-stall check. Returns true iff we've accumulated at least
 * STALL_WINDOW attempts AND the mean per-iteration improvement on
 * max_off_axis over the last STALL_WINDOW attempts is ≤ STALL_THRESHOLD.
 * "Improvement" = previous_max_off - current_max_off (positive = improving).
 */
function isStalled(attempts) {
    if (attempts.length < STALL_WINDOW) return false;
    const recent = attempts.slice(-STALL_WINDOW).map(a => a.max_off_axis);
    let totalImprovement = 0;
    for (let i = 1; i < recent.length; i++) totalImprovement += (recent[i-1] - recent[i]);
    const meanImprovementPerStep = totalImprovement / (recent.length - 1);
    return meanImprovementPerStep <= STALL_THRESHOLD;
}

// Persistence (saveBio / saveAgent) is imported from harness_lib at
// the top. The previously-vendored saveAgent here was missing the
// `signature` field that the plugin's POST /agents now requires; the
// import inherits the current contract.

// ── DESIGNER calls ────────────────────────────────────────────────────

async function designBioPass(bio, prior_attempts) {
    const sys =
        'You design user-side biographies for an iterative measurement loop. ' +
        'The operator gives you a bio target (per-axis values 1-5) + a brief. ' +
        'You produce a 3-5 sentence prose biography describing WHO THE USER IS. ' +
        'You may also see prior attempts and the measurements they achieved when ' +
        'the bio was run with diverse user-agents and a counterparty character. ' +
        'If prior attempts drift off-target on an axis, rewrite the bio to push ' +
        'that axis closer to target. Output ONLY the new bio prose — no preamble.';
    const priorBlock = prior_attempts.length
        ? '## Prior attempts and their measured bio-axis signatures\n\n' +
          prior_attempts.map((p, i) =>
              `### Attempt ${i+1}\nbio prose: ${p.prose}\nmeasured: ${fmtSig(p.measured)}`
          ).join('\n\n') + '\n\n'
        : '';
    const usr =
        '## Target bio-axis signature\n\n' +
        BIO_AXES.map(a => `- ${a.name}: ${bio.target_bio[a.name]} (rubric: ${a.def})`).join('\n') + '\n\n' +
        '## Design brief\n\n' + bio.design_brief + '\n\n' +
        priorBlock +
        'Write the bio prose now.';
    // No max_tokens cap: "3-5 sentence prose biography" + "Output ONLY
    // the new bio prose" gives the model a clear emit-and-stop shape.
    // Trust EOS; the generation-config moratorium forbids hidden
    // mid-turn truncators (they leave the next prefill staring at an
    // unfinished assistant turn, off the chat-template manifold).
    return await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
}

async function designAgentPass(bio, agentTarget, prior_attempts) {
    const sys =
        'You design user-agent overlays for an iterative measurement loop. ' +
        'You\'re given a bio (the user-persona) + an agent target (per-axis ' +
        'values 1-5 on agent axes) + axis rubrics + any prior attempts with ' +
        'their measured agent-axis signatures. You produce an agent_text — a ' +
        '2-4 sentence author\'s-note-style snippet (second person "You will…") ' +
        'that when injected at depth-1 makes the user-persona enact the target ' +
        'on each agent axis. If a prior attempt under-shot or over-shot an axis, ' +
        'rewrite to push that axis closer to target. Output ONLY agent_text.';
    const priorBlock = prior_attempts.length
        ? '## Prior agent_text attempts and their measured signatures\n\n' +
          prior_attempts.map((p, i) =>
              `### Attempt ${i+1}\nagent_text: ${p.agent_text}\nmeasured (mean over ${N_TURNS_PER_CHAT} user turns): ${fmtSig(p.measured)}`
          ).join('\n\n') + '\n\n'
        : '';
    const usr =
        '## Bio (the user-persona this agent will overlay)\n\n' + bio.prose + '\n\n' +
        '## Target agent-axis signature\n\n' +
        AGENT_AXES.map(a => `- ${a.name}: ${agentTarget.target_agent[a.name]} (rubric: ${a.def})`).join('\n') + '\n\n' +
        '## Hint\n\n' + agentTarget.motive_hint + '\n\n' +
        priorBlock +
        'Write the agent_text now.';
    // No max_tokens cap: "2-4 sentence author's-note-style snippet" +
    // "Output ONLY agent_text" gives the model a deterministic stop
    // shape. Trust EOS; moratorium forbids inline caps.
    return await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
}

// ── chat runner ──────────────────────────────────────────────────────

// Counterparty fetch + chat-running + user-turn filter are imported
// from harness_lib at the top. The specific counterparty for this
// run is named in the experiment spec (spec.counterparty_avatar).

// ── INNER LOOP: user-agent designer with judge feedback ──────────────

async function designAgentToConvergence(bio, agentTarget, rock) {
    const attempts = [];
    let bestAttempt = null;
    let bestMaxDist = Infinity;
    for (let k = 0; k < K_MAX_INNER; k++) {
        const t0 = Date.now();
        const agent_text = await designAgentPass(bio, agentTarget, attempts);
        const agent_id = `${bio.slug}-${agentTarget.slug}-iter${k}`;
        await saveAgent(agent_id, `${bio.name} — ${agentTarget.slug} (iter ${k})`,
                        agent_text, bio.canonical_key);
        const chat = await runChat(bio, agent_id, rock, N_TURNS_PER_CHAT);
        // Judge each user turn on the agent axes ONLY (the inner loop's target is in A)
        const turnJudgments = [];
        for (const turn of userTurns(chat)) {
            const { sig, raw } = await judgeOneTurn(turn, AGENT_AXES);
            turnJudgments.push({ turn, sig, raw });
        }
        const measured_agent = meanSig(turnJudgments.map(j => j.sig), AGENT_AXES);
        const dist = distancePerAxis(measured_agent, agentTarget.target_agent, AGENT_AXES);
        const maxDist = maxOffAxes(dist);
        const attempt = {
            iter: k, agent_text, agent_id,
            chat, turnJudgments, measured: measured_agent, dist_per_axis: dist, max_off_axis: maxDist,
            elapsed_ms: Date.now() - t0,
        };
        attempts.push(attempt);
        if (maxDist < bestMaxDist) { bestAttempt = attempt; bestMaxDist = maxDist; }
        console.log(`    [inner k=${k}] measured ${fmtSig(measured_agent)} | dist ${fmtSig(dist)} | max_off=${maxDist.toFixed(2)} | ${attempt.elapsed_ms}ms`);
        if (maxDist <= EPS_PER_AXIS) {
            attempt.stop_reason = 'converged';
            console.log(`    [inner k=${k}] CONVERGED (max_off=${maxDist.toFixed(2)} ≤ ${EPS_PER_AXIS})`);
            break;
        }
        if (isStalled(attempts)) {
            attempt.stop_reason = 'stalled';
            console.log(`    [inner k=${k}] STALLED (mean improvement over last ${STALL_WINDOW} ≤ ${STALL_THRESHOLD}); accepting best so far (max_off=${bestMaxDist.toFixed(2)})`);
            break;
        }
    }
    return { attempts, best: bestAttempt, stop_reason: bestAttempt?.stop_reason || 'k_max' };
}

// ── OUTER LOOP: bio designer with judge feedback aggregated across agents ──

async function designBioToConvergence(bio, rock) {
    const attempts = [];
    let bestAttempt = null;
    let bestMaxDist = Infinity;
    for (let k = 0; k < K_MAX_OUTER; k++) {
        const t0 = Date.now();
        const bioProse = await designBioPass(bio, attempts);
        bio.prose = bioProse;
        await saveBio(bio);
        console.log(`  [outer k=${k}] bio prose: ${bioProse.slice(0,120).replace(/\n/g,' ')}…`);
        // Run inner for each agent target
        const innerResults = [];
        for (const agentTarget of AGENT_TARGETS) {
            console.log(`    → inner: agent target = ${agentTarget.slug} (${fmtSig(agentTarget.target_agent)})`);
            const innerR = await designAgentToConvergence(bio, agentTarget, rock);
            innerResults.push({ agentTarget, ...innerR });
        }
        // Aggregate BIO-axis measurements across all user turns of all best
        // inner runs. Bio axes are scored per-turn but in a separate judge
        // call (so the inner loop's judge isn't muddled with bio-axis noise).
        const allUserTurns = [];
        for (const r of innerResults) {
            allUserTurns.push(...userTurns(r.best.chat));
        }
        const bioTurnJudgments = [];
        for (const turn of allUserTurns) {
            const { sig, raw } = await judgeOneTurn(turn, BIO_AXES);
            bioTurnJudgments.push({ turn, sig, raw });
        }
        const measured_bio = meanSig(bioTurnJudgments.map(j => j.sig), BIO_AXES);
        const dist = distancePerAxis(measured_bio, bio.target_bio, BIO_AXES);
        const maxDist = maxOffAxes(dist);
        const attempt = {
            iter: k, prose: bioProse, measured: measured_bio, dist_per_axis: dist, max_off_axis: maxDist,
            innerResults, bioTurnJudgments,
            elapsed_ms: Date.now() - t0,
        };
        attempts.push(attempt);
        if (maxDist < bestMaxDist) { bestAttempt = attempt; bestMaxDist = maxDist; }
        console.log(`  [outer k=${k}] BIO measured ${fmtSig(measured_bio)} | dist ${fmtSig(dist)} | max_off=${maxDist.toFixed(2)} | ${attempt.elapsed_ms}ms`);
        if (maxDist <= EPS_PER_AXIS) {
            attempt.stop_reason = 'converged';
            console.log(`  [outer k=${k}] CONVERGED`);
            break;
        }
        if (isStalled(attempts)) {
            attempt.stop_reason = 'stalled';
            console.log(`  [outer k=${k}] STALLED (mean improvement over last ${STALL_WINDOW} ≤ ${STALL_THRESHOLD}); accepting best so far (max_off=${bestMaxDist.toFixed(2)})`);
            break;
        }
    }
    return { attempts, best: bestAttempt, stop_reason: bestAttempt?.stop_reason || 'k_max' };
}

// ── main ─────────────────────────────────────────────────────────────

fs.mkdirSync(OUT_DIR, { recursive: true });
console.log(`[lock_in_iterative] K_max_inner=${K_MAX_INNER}, K_max_outer=${K_MAX_OUTER}, n_turns=${N_TURNS_PER_CHAT}, eps=${EPS_PER_AXIS}`);
console.log(`[lock_in_iterative] output: ${OUT_DIR}`);

const rock = await fetchCounterparty(spec.counterparty_avatar);
console.log(`[rock] system_prompt length=${rock.system_prompt.length}`);

const tAll = Date.now();
for (const bio of BIOS) {
    console.log(`\n[bio ${bio.slug}] target_bio ${fmtSig(bio.target_bio)} — running outer loop`);
    const result = await designBioToConvergence(bio, rock);
    const outFile = path.join(OUT_DIR, `${bio.slug}.json`);
    fs.writeFileSync(outFile, JSON.stringify({
        bio: { slug: bio.slug, canonical_key: bio.canonical_key, name: bio.name,
               target_bio: bio.target_bio, design_brief: bio.design_brief },
        agent_targets: AGENT_TARGETS,
        bio_axes: BIO_AXES, agent_axes: AGENT_AXES,
        result,
        elapsed_ms_total: Date.now() - tAll,
    }, null, 2));
    console.log(`[bio ${bio.slug}] saved: ${outFile}`);
}

console.log(`\n[lock_in_iterative] done. elapsed: ${((Date.now()-tAll)/1000).toFixed(1)}s`);
