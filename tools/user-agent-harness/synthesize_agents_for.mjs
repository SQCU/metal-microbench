#!/usr/bin/env node
// synthesize_agents_for.mjs
//
// Works through the "pending synthesis" queue: bios that exist in the
// inventory but have no agent yet. For each, designs an agent_text
// (the depth-1 author's-note overlay) via the cheap agent designer
// and POSTs it. saveAgent computes the composition signature via
// /signature-extract using axes from the axes/*.json cards — same
// path the on-policy synthesis harness uses, so vectors are
// commensurable.
//
// The ontological closure: a bio is usable iff it has at least one
// synthesized agent. Synthesizing an agent is the ONLY operation that
// creates a signature for that (bio, agent) composition. So this
// script's act of making bios usable IS the act of creating their
// signatures — no separate "sign the bio" step exists.
//
// Designed to be interruptible/resumable: re-running picks up where
// the last run left off (bios that already have any agent are
// skipped). Failures on individual bios don't kill the run.
//
// Usage:
//   node synthesize_agents_for.mjs [--st-url=http://127.0.0.1:8002]
//                                  [--all]   # process bios that already
//                                            # have agents too (force a
//                                            # second agent per bio)

import process from 'node:process';
import * as L from './harness_lib.mjs';

const FORCE_ALL = process.argv.includes('--all');
// All HTTP + persistence goes through L.http / L.saveBio / L.saveAgent
// so the contract with the plugin lives in ONE place (harness_lib.mjs).
// Axes are queried from the plugin (axes/*.json cards) — single source
// of truth, durable across script restarts, includes any derived axes
// from prior splitter/disambiguator runs.

const axesAvailable = await L.fetchAxes();
console.error(`[synthesize_agents_for] ST=${L.ENDPOINTS.ST} force_all=${FORCE_ALL} axes_available=${axesAvailable.length}`);

const { personas } = await L.http('GET', `${L.ENDPOINTS.PLUGIN}/personas`);
console.error(`[synthesize_agents_for] ${personas.length} bios loaded`);

let synthesized = 0, skipped = 0, failed = 0;

for (const p of personas) {
    const hasAgent = Array.isArray(p.agents_for_bio) && p.agents_for_bio.length > 0;
    if (hasAgent && !FORCE_ALL) {
        console.error(`  [${p.id}] SKIP — has ${p.agents_for_bio.length} agent(s)`);
        skipped++; continue;
    }
    try {
        const t0 = Date.now();
        // Step 1: design the agent_text. Cheap K=1 single-pass — a
        // neutral "be vividly yourself" overlay. The labeling-queue UI
        // could later swap this for an interactive design loop.
        const agent_text = await L.designCheapAgent({ prose: p.bio || '', name: p.name });
        // Step 2: POST the agent. saveAgent computes the composition
        // signature via /signature-extract (axes default to the plugin's
        // axes/*.json card set) and includes it in the body.
        const slug = p.id.replace(/\.png$/, '').toLowerCase().replace(/[^a-z0-9_-]+/g, '-');
        const agent_id = `${slug}-default`;
        await L.saveAgent(agent_id, `${p.name} — default agent`, agent_text, p.id);
        const wall = Date.now() - t0;
        console.error(`  [${p.id}] OK agent_id=${agent_id} (${wall}ms)`);
        synthesized++;
    } catch (e) {
        console.warn(`  [${p.id}] FAIL: ${e.message.slice(0, 200)}`);
        failed++;
    }
}

console.error(`\n[synthesize_agents_for] done. synthesized=${synthesized} skipped=${skipped} failed=${failed}`);
