#!/usr/bin/env node
// profile_seed_corpus.mjs
//
// Corpus-profile probe for the committed seed personas. This is the
// cheap, scaling-friendly measurement layer that sits before expensive
// fixed-point synthesis:
//   - sample at most min(--limit, N) existing bios (default limit 8),
//   - extract bio signatures only for sampled bios that do not already
//     carry one,
//   - read stored agent signatures for their existing compositions,
//   - compute effective dimensionality + per-axis variance,
//   - persist a JSON artifact under data/corpus_profiles/.
//
// It intentionally does NOT exhaustively judge every persona x agent x
// counterparty. That belongs to fixed-point trajectories. This probe is
// the "what axes already separate the seed corpus?" check.

import fs from 'node:fs';
import path from 'node:path';
import * as L from './harness_lib.mjs';

const DEFAULT_LIMIT = 8;
const DEFAULT_OUT_DIR = path.resolve(
    path.dirname(new URL(import.meta.url).pathname),
    '..', 'data', 'corpus_profiles');

function parseArgs(argv) {
    const args = {
        limit: DEFAULT_LIMIT,
        outDir: process.env.USER_PERSONAS_CORPUS_PROFILE_DIR || DEFAULT_OUT_DIR,
        seedIds: [],
        label: 'seed-corpus',
        write: true,
    };
    const rest = argv.slice(2);
    for (let i = 0; i < rest.length; i++) {
        const a = rest[i];
        if (a === '--limit') args.limit = Math.max(1, Number(rest[++i]) || DEFAULT_LIMIT);
        else if (a === '--out-dir') args.outDir = rest[++i];
        else if (a === '--label') args.label = rest[++i] || args.label;
        else if (a === '--seed-ids') args.seedIds = rest[++i].split(',').map(s => s.trim()).filter(Boolean);
        else if (a === '--no-write') args.write = false;
        else throw new Error(`unknown arg ${a}`);
    }
    return args;
}

function stableHash(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) {
        h ^= s.charCodeAt(i);
        h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
}

function chooseProbeBios(personas, limit, preferredIds = []) {
    const byId = new Map(personas.map(p => [p.id, p]));
    const chosen = [];
    const seen = new Set();
    for (const id of preferredIds) {
        const p = byId.get(id);
        if (p && !seen.has(p.id)) {
            chosen.push(p);
            seen.add(p.id);
        }
    }
    const remaining = personas
        .filter(p => !seen.has(p.id))
        .sort((a, b) => stableHash(a.id) - stableHash(b.id));
    for (const p of remaining) {
        if (chosen.length >= Math.min(limit, personas.length)) break;
        chosen.push(p);
    }
    return chosen;
}

function signatureAxes(sig) {
    return sig && typeof sig === 'object' ? Object.keys(sig) : [];
}

function meanStd(arr) {
    return L.meanStd(arr);
}

function topVarianceRows(sigsById, axisNames, k = 8) {
    const rows = axisNames.map(axis => {
        const vals = Object.values(sigsById)
            .map(sig => sig?.[axis])
            .filter(Number.isFinite);
        const stats = meanStd(vals);
        return { axis, n: stats.n, mean: stats.mean, variance: stats.var || 0 };
    });
    rows.sort((a, b) => b.variance - a.variance || a.axis.localeCompare(b.axis));
    return rows.slice(0, k);
}

function fmtPR(pr) {
    return pr.effDim == null ? `n/a (${pr.note || 'no PR'})` : pr.effDim.toFixed(3);
}

async function extractBioSignature(bio) {
    const prose = [
        `Bio name: ${bio.name || bio.id}`,
        '',
        'Bio prose:',
        bio.bio || bio.system_prompt || '',
    ].join('\n');
    const r = await L.httpRetrying('POST', `${L.ENDPOINTS.PLUGIN}/signature-extract`, { prose }, { attempts: 3 });
    return r.signature || {};
}

async function mapLimit(items, limit, fn) {
    const out = new Array(items.length);
    let next = 0;
    const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
        while (next < items.length) {
            const i = next++;
            out[i] = await fn(items[i], i);
        }
    });
    await Promise.all(workers);
    return out;
}

const args = parseArgs(process.argv);
const [personasResp, agentsResp, axesResp] = await Promise.all([
    L.http('GET', `${L.ENDPOINTS.PLUGIN}/personas`),
    L.http('GET', `${L.ENDPOINTS.PLUGIN}/agents`),
    L.http('GET', `${L.ENDPOINTS.PLUGIN}/axes`),
]);

const personas = personasResp.personas || [];
const agents = agentsResp.agents || [];
const axes = axesResp.axes || [];
const axisNames = axes.map(a => a.id);
const probeLimit = Math.min(args.limit, personas.length);
const sampledBios = chooseProbeBios(personas, probeLimit, args.seedIds);

console.log(`[profile_seed_corpus] personas=${personas.length} agents=${agents.length} axes=${axisNames.length}`);
console.log(`[profile_seed_corpus] probing ${sampledBios.length}/${personas.length} bios (limit=${args.limit})`);

const bioSigsById = {};
const bioRows = await mapLimit(sampledBios, 4, async bio => {
    let sig = bio.signature || {};
    let source = 'stored';
    if (signatureAxes(sig).length === 0) {
        source = 'extracted';
        sig = await extractBioSignature(bio);
    }
    bioSigsById[bio.id] = sig;
    return {
        id: bio.id,
        name: bio.name || bio.id,
        signature_source: source,
        n_axes: signatureAxes(sig).length,
        agents_for_bio: (bio.agents_for_bio || []).map(a => a.id),
    };
});

const sampledBioIds = new Set(sampledBios.map(b => b.id));
const sampledAgents = agents.filter(a => sampledBioIds.has(a.designed_for_bio_id));
const agentSigsById = {};
for (const a of sampledAgents) {
    if (signatureAxes(a.signature).length > 0) agentSigsById[a.id] = a.signature;
}

const bioPR = L.effDimParticipationRatio(bioSigsById, axisNames);
const agentPR = L.effDimParticipationRatio(agentSigsById, axisNames);
const artifact = {
    profile_schema: 'seed-corpus-profile-v1',
    label: args.label,
    created_at: new Date().toISOString(),
    plugin_url: L.ENDPOINTS.PLUGIN,
    sample_policy: {
        strategy: 'preferred ids first, then stable hash order',
        limit: args.limit,
        sampled_bios: sampledBios.length,
        corpus_bios: personas.length,
        rationale: 'probe min(8,N) by default; avoid exhaustive corpus judge fanout as N grows',
    },
    axes: axisNames,
    bios: bioRows,
    agents: sampledAgents.map(a => ({
        id: a.id,
        name: a.name,
        designed_for_bio_id: a.designed_for_bio_id,
        n_axes: signatureAxes(a.signature).length,
    })),
    bio_effective_dim: bioPR,
    agent_effective_dim: agentPR,
    top_bio_variance_axes: topVarianceRows(bioSigsById, axisNames),
    top_agent_variance_axes: topVarianceRows(agentSigsById, axisNames),
};

console.log(`[profile_seed_corpus] bio eff_dim=${fmtPR(bioPR)}; agent eff_dim=${fmtPR(agentPR)}`);
console.log('[profile_seed_corpus] top bio variance axes:');
for (const r of artifact.top_bio_variance_axes.slice(0, 6)) {
    console.log(`  ${r.axis}: var=${r.variance.toFixed(3)} n=${r.n} mean=${r.mean == null ? '?' : r.mean.toFixed(2)}`);
}
console.log('[profile_seed_corpus] top agent variance axes:');
for (const r of artifact.top_agent_variance_axes.slice(0, 6)) {
    console.log(`  ${r.axis}: var=${r.variance.toFixed(3)} n=${r.n} mean=${r.mean == null ? '?' : r.mean.toFixed(2)}`);
}

if (args.write) {
    fs.mkdirSync(args.outDir, { recursive: true });
    // LINT-OK-PREFIX-SAFE: timestamp is only an output artifact filename, never prompt content.
    const outPath = path.join(args.outDir, `${args.label}-${new Date().toISOString().replace(/[:.]/g, '-')}.json`);
    fs.writeFileSync(outPath, JSON.stringify(artifact, null, 2));
    console.log(`[profile_seed_corpus] wrote ${outPath}`);
}
