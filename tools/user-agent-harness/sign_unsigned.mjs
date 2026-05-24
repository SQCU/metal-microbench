#!/usr/bin/env node
// sign_unsigned.mjs
//
// One-time administrative tool: extract a signature for any bio/agent
// that lacks one, using the EXACT same /signature-extract path the
// synthesis harness uses (commensurable vectors), and POST it back
// so the result lives on the card. After this runs, /yapper-seed
// stops failing on "inventory contains unsigned candidates".
//
// New bios/agents created via L.saveBio / L.saveAgent are born signed —
// this script only exists for legacy items (pre-existing canonical
// bios that were never synthesized through the harness).
//
// All HTTP + persistence goes through harness_lib.mjs so the contract
// with the plugin has ONE source of truth. See the 2026-05-18 dedup
// commit for the audit that motivated this.
//
// Usage:
//   node sign_unsigned.mjs [--force]

import process from 'node:process';
import * as L from './harness_lib.mjs';

const FORCE = process.argv.includes('--force');
// Axes come from the plugin's axes/*.json card set (single source of
// truth). The plugin's /signature-extract also defaults to this set
// when caller omits body.axes, so we could pass nothing — we pass
// explicitly here only for the operator-log line below.
const axes = await L.fetchAxes();
console.log(`[sign_unsigned] ST=${L.ENDPOINTS.ST} force=${FORCE} axes=${axes.length}`);

const { personas } = await L.http('GET', `${L.ENDPOINTS.PLUGIN}/personas`);
console.log(`[sign_unsigned] ${personas.length} bios loaded`);

let signed = 0, skipped = 0, failed = 0;

for (const p of personas) {
    const hasSig = p.signature && Object.keys(p.signature).length > 0;
    if (hasSig && !FORCE) {
        console.log(`  [bio ${p.id}] SKIP (already has ${Object.keys(p.signature).length}-axis signature)`);
        skipped++; continue;
    }
    const prose = `Bio (user-side persona):\n${p.bio || ''}\n\nVoice clauses:\n${p.system_prompt || ''}`;
    try {
        const t0 = Date.now();
        const sigResp = await L.httpRetrying('POST', `${L.ENDPOINTS.PLUGIN}/signature-extract`,
            { prose, axes });
        if (!sigResp.signature || Object.keys(sigResp.signature).length === 0) {
            throw new Error(`extract returned empty signature`);
        }
        await L.http('POST', `${L.ENDPOINTS.PLUGIN}/personas/${encodeURIComponent(p.id)}`, {
            name: p.name, bio: p.bio, system_prompt: p.system_prompt,
            signature: sigResp.signature,
        });
        // LINT-OK-PREFIX-SAFE: wall-clock timing for operator log; not a prompt.
        const wall = Date.now() - t0;
        const scoresStr = Object.entries(sigResp.signature)
            .slice(0, 6).map(([k, v]) => `${k.slice(0,4)}=${v}`).join(' ');
        console.log(`  [bio ${p.id}] OK ${Object.keys(sigResp.signature).length} axes in ${wall}ms — ${scoresStr}…`);
        signed++;
    } catch (e) {
        console.warn(`  [bio ${p.id}] FAIL: ${e.message}`);
        failed++;
    }
}

const { agents } = await L.http('GET', `${L.ENDPOINTS.PLUGIN}/agents`);
console.log(`\n[sign_unsigned] ${agents.length} agents loaded`);
let agentsSigned = 0, agentsSkipped = 0, agentsFailed = 0;
const bioById = new Map(personas.map(p => [p.id, p]));
for (const a of agents) {
    const hasSig = a.signature && Object.keys(a.signature).length > 0;
    if (hasSig && !FORCE) {
        console.log(`  [agent ${a.id}] SKIP`);
        agentsSkipped++; continue;
    }
    const bio = bioById.get(a.designed_for_bio_id);
    if (!bio) {
        console.warn(`  [agent ${a.id}] FAIL: designed_for_bio_id=${a.designed_for_bio_id} not loaded`);
        agentsFailed++; continue;
    }
    const prose =
        `Bio (user-side persona):\n${bio.bio || ''}\n\n` +
        `Bio voice clauses:\n${bio.system_prompt || ''}\n\n` +
        `Agent voice clauses (injected at depth ${a.injection_depth || 1}):\n${a.agent_text || ''}`;
    try {
        const t0 = Date.now();
        const sigResp = await L.httpRetrying('POST', `${L.ENDPOINTS.PLUGIN}/signature-extract`,
            { prose, axes });
        await L.http('POST', `${L.ENDPOINTS.PLUGIN}/agents/${encodeURIComponent(a.id)}`, {
            name: a.name, agent_text: a.agent_text,
            designed_for_bio_id: a.designed_for_bio_id,
            injection_mode: a.injection_mode || 'authors_note',
            injection_depth: a.injection_depth || 1,
            signature: sigResp.signature,
        });
        // LINT-OK-PREFIX-SAFE: wall-clock timing for operator log; not a prompt.
        const wall = Date.now() - t0;
        console.log(`  [agent ${a.id}] OK ${Object.keys(sigResp.signature).length} axes in ${wall}ms`);
        agentsSigned++;
    } catch (e) {
        console.warn(`  [agent ${a.id}] FAIL: ${e.message}`);
        agentsFailed++;
    }
}

console.log(`\n[sign_unsigned] done. bios: signed=${signed} skipped=${skipped} failed=${failed} ; agents: signed=${agentsSigned} skipped=${agentsSkipped} failed=${agentsFailed}`);
