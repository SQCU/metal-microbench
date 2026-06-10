#!/usr/bin/env node
// context_suggester.mjs
//
// Phase A of the diegetic-UX shift to handle large procedurally-generated
// persona corpora. See feature_factorization_design.md §8 (to be written
// once Phase B lands).
//
// The flow:
//   1. Build a persona-signature manifest by scanning data/ run files.
//      The manifest is a sparse table: persona_id → measured axis values
//      where each persona may have measurements on a different subset
//      of the axis registry.
//   2. ContextJudge: one bridge call given (axis registry + chat
//      context). NEVER includes any bio prose or agent_text. The judge
//      proposes K_1+K_2 SPARSE TARGET VECTORS — each one a partial
//      signature describing what kind of persona would fit this context.
//   3. Cheap nearest-neighbor: for each target, compute sparse Euclidean
//      distance to every persona; pick the nearest. K_1 picks would
//      auto-trigger /poll in the UI; K_2 picks would render as
//      click-to-enable.
//   4. Output: pick list with target labels + distances. Phase B will
//      port this to a plugin endpoint + sidebar refactor.
//
// CLI:
//   node context_suggester.mjs <chat.json> [--k-active=2] [--k-disabled=2]
//   chat.json: { messages: [{ role, content }, ...] }

import fs from 'node:fs';
import path from 'node:path';
import * as L from './harness_lib.mjs';

// Fetch the axis card set ONCE at module load — axes don't change
// mid-script. The plugin returns full cards; downstream call sites in
// this script use `.name` so we project id → name to keep them
// compatible without touching every reference.
const _AXES_CACHE = (await L.fetchAxes()).map(a => ({
    name: a.id, def: a.def, kind: a.kind,
}));
function allAxes() { return _AXES_CACHE; }

const DEFAULT_K_ACTIVE = 2;
const DEFAULT_K_DISABLED = 2;
const DATA_ROOT = process.env.USER_PERSONAS_DATA_DIR
    || path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'data');
const OUT_DIR = path.join(DATA_ROOT, 'context_suggester');

// ── manifest loader: scan data/ files for measured signatures ────────

function safeReadJsons(dir) {
    if (!fs.existsSync(dir)) return [];
    return fs.readdirSync(dir)
        .filter(f => f.endsWith('.json'))
        .map(f => {
            try { return { fname: f, data: JSON.parse(fs.readFileSync(path.join(dir, f), 'utf8')) }; }
            catch (e) { console.warn(`[manifest] skipping ${f}: ${e.message}`); return null; }
        })
        .filter(Boolean);
}

function loadManifest() {
    const manifest = new Map();  // canonical_key → { name, sig, source[], n_axes }

    function upsert(canonical_key, partial, source) {
        const existing = manifest.get(canonical_key);
        if (!existing) {
            manifest.set(canonical_key, {
                canonical_key,
                name: partial.name || canonical_key,
                sig: { ...partial.sig },
                sources: [source],
            });
        } else {
            // Merge sigs (last writer wins per axis; could also average)
            for (const [a, v] of Object.entries(partial.sig)) {
                if (Number.isFinite(v)) existing.sig[a] = v;
            }
            existing.sources.push(source);
            if (!existing.name && partial.name) existing.name = partial.name;
        }
    }

    // explore_corpus runs: per-iter bios with measured_sig on bio_axes
    for (const { fname, data } of safeReadJsons(path.join(DATA_ROOT, 'explore_corpus'))) {
        for (const b of (data.bios || [])) {
            if (!b.canonical_key || !b.measured_sig) continue;
            upsert(b.canonical_key, { name: b.name, sig: b.measured_sig },
                   `explore_corpus/${fname}`);
        }
    }

    // cluster_disambig: tightness pre-flight (1 axis) + per-hypothesis
    // evaluations (1 axis each from each hypothesis's primary name).
    for (const { fname, data } of safeReadJsons(path.join(DATA_ROOT, 'cluster_disambig'))) {
        const cluster = data.cluster_spec || {};
        const bios = cluster.bios || [];
        const tightness = data.tightness || {};
        const evals = data.evaluations || [];
        for (const b of bios) {
            const sig = {};
            // From tightness pre-flight (one axis)
            if (tightness.per_bio?.[b.canonical_key]?.mean != null) {
                sig[tightness.nominal_axis] = tightness.per_bio[b.canonical_key].mean;
            }
            // From each evaluation (hypothesis name = axis name)
            for (const ev of evals) {
                const m = ev.perBio?.[b.canonical_key]?.mean;
                if (m != null) sig[ev.hypothesis.name] = m;
            }
            if (Object.keys(sig).length > 0) {
                upsert(b.canonical_key, { name: b.name, sig },
                       `cluster_disambig/${fname}`);
            }
        }
    }

    // lock_in_iterative: best bio sig on 2 bio axes
    for (const { fname, data } of safeReadJsons(path.join(DATA_ROOT, 'lock_in_iterative'))) {
        if (!data.bio?.canonical_key || !data.result?.best?.measured) continue;
        upsert(data.bio.canonical_key,
               { name: data.bio.name, sig: data.result.best.measured },
               `lock_in_iterative/${fname}`);
    }

    return manifest;
}

// ── context judge: propose K sparse target vectors ────────────────────

/**
 * Compute per-axis coverage across the manifest: (axis_name → # personas
 * with a measured value on that axis). The judge uses this to prefer
 * well-measured axes — otherwise it proposes targets the manifest can't
 * meaningfully match against.
 */
function axisCoverage(manifest) {
    const counts = {};
    for (const a of allAxes()) counts[a.name] = 0;
    for (const [, rec] of manifest) {
        for (const a of Object.keys(rec.sig || {})) {
            if (Number.isFinite(rec.sig[a]) && counts[a] !== undefined) counts[a]++;
        }
    }
    return counts;
}

async function judgeContext(messages, kTotal, manifest) {
    const axes = allAxes();
    const coverage = axisCoverage(manifest);
    const N = manifest.size;
    // Sort axes by coverage descending so the registry listing shows
    // best-covered first, and inject the coverage % into each line.
    const sortedAxes = [...axes].sort((a, b) => (coverage[b.name] || 0) - (coverage[a.name] || 0));
    const registry = sortedAxes.map(a => {
        const c = coverage[a.name] || 0;
        const pct = N > 0 ? Math.round(100 * c / N) : 0;
        const tag = c === 0 ? ' [⚠ NOT MEASURED]' : ` [${c}/${N} = ${pct}%]`;
        return `- **${a.name}** (${a.kind})${tag}: ${a.def}`;
    }).join('\n');
    const wellCovered = sortedAxes.filter(a => (coverage[a.name] || 0) >= Math.max(1, Math.ceil(N * 0.3)));
    const wellCoveredList = wellCovered.map(a => a.name).join(', ');
    const chatText = messages
        .map(m => `[${m.role || (m.is_user ? 'user' : 'assistant')}] ${m.content || m.mes || ''}`)
        .join('\n\n');
    const sys =
        'Given a chat context and a registry of behavioral axes, propose ' +
        `${kTotal} target persona vectors. Each target describes ONE KIND ` +
        'of user-persona who would be appropriate for this conversation — ' +
        'targets should be diverse from one another, not minor variants. ' +
        'Each target is a sparse map of axis_name → number 1-5 — only ' +
        'set axes that matter for that target, leave others out. Each ' +
        'target also has a short human-readable label (5-10 words) ' +
        'naming what kind of persona it represents. You do not see the ' +
        'actual personas — you propose what would fit the context, the ' +
        'system picks the nearest existing persona to each target.\n\n' +
        'IMPORTANT — axis coverage: each axis in the registry is tagged ' +
        'with its coverage across the persona manifest (how many personas ' +
        'have a measured value on that axis). STRONGLY PREFER well-covered ' +
        'axes in your targets — a target that uses only [⚠ NOT MEASURED] ' +
        'axes will produce no match. Include at LEAST one well-covered ' +
        '(≥30%) axis per target, ideally 2-4. Output ONLY a JSON object: ' +
        '{"targets": [{"label": str, "rationale": str, ' +
        '"axes": {axis_name: int, ...}}, ...]}.';
    const usr =
        `## Axis registry (sorted by coverage; ${N} personas total)\n\n${registry}\n\n` +
        `## Well-covered axes (use these preferentially)\n\n${wellCoveredList || '(none — manifest too sparse)'}\n\n` +
        `## Chat context\n\n${chatText || '(empty — propose generally-useful targets)'}\n\n` +
        `## Emit\n\n${kTotal} target vectors as JSON. Each target MUST include at least one well-covered axis.`;
    // Per moratorium (lint_generation_config.mjs): no max_tokens at caller.
    const raw = await L.bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const m = raw.match(/\{[\s\S]*\}/);
    if (!m) throw new Error(`could not parse JSON from ContextJudge:\n${raw.slice(0, 500)}`);
    const parsed = JSON.parse(m[0]);
    if (!Array.isArray(parsed.targets)) throw new Error('ContextJudge output missing targets[]');
    return { targets: parsed.targets, raw, coverage };
}

// ── nearest-neighbor on sparse targets ───────────────────────────────

// Penalty per missing axis: equivalent to "we have no information about
// this persona on this axis, so assume it's at neutral Likert mid (=3),
// expected sqdiff to any target value is bounded by (5-1)/2 = 2".
// Implementing as a fixed contribution of MISSING_AXIS_PENALTY² to the
// summed squared distance per missing axis means:
//   - If a target requests N axes and a persona has 0 of them:
//     distance = sqrt(N · P²/N) = P  (uniform baseline)
//   - If the persona has some of them: distance drops below baseline
//     proportional to how well-matched the covered axes are.
//   - Personas are never EXCLUDED from picks — they all get a finite
//     distance. Selectivity comes from the persona that actually
//     matches target axes well having the lowest distance.
//
// Sets uniform-prior fallback (no selectivity → ~equal distances →
// argmin essentially uniform-random) and biased-sampling-when-selective
// (real matches outscore the missing-axis penalty floor).
const MISSING_AXIS_PENALTY = 2.0;  // half the Likert range, in "axis units"

/**
 * Sparse Euclidean distance with uniform-prior fallback for missing axes.
 * Every persona gets a finite distance; none are excluded.
 */
function sparseDistance(target, sig) {
    let sumSq = 0, compared = 0, missing = 0;
    for (const [axis, t] of Object.entries(target)) {
        if (sig[axis] != null && Number.isFinite(sig[axis])) {
            sumSq += (t - sig[axis]) ** 2;
            compared++;
        } else {
            sumSq += MISSING_AXIS_PENALTY ** 2;
            missing++;
        }
    }
    const totalAxes = compared + missing;
    if (totalAxes === 0) return { distance: MISSING_AXIS_PENALTY, n_axes_compared: 0, n_axes_missing: 0 };
    return { distance: Math.sqrt(sumSq / totalAxes), n_axes_compared: compared, n_axes_missing: missing };
}

function pickNearestForTarget(target, manifest, exclude = new Set()) {
    const scored = [];
    for (const [pid, rec] of manifest) {
        if (exclude.has(pid)) continue;
        const d = sparseDistance(target.axes, rec.sig);
        scored.push({ persona_id: pid, name: rec.name, sig: rec.sig, ...d });
    }
    scored.sort((a, b) => a.distance - b.distance);
    return scored[0] || null;
}

// ── main orchestrator ────────────────────────────────────────────────

async function runSuggester(chatPath, kActive, kDisabled) {
    const kTotal = kActive + kDisabled;
    const chat = JSON.parse(fs.readFileSync(chatPath, 'utf8'));
    const messages = chat.messages || chat.chat || chat;

    console.log(`[suggester] chat: ${chatPath}`);
    console.log(`[suggester] messages: ${messages.length}, K_active=${kActive}, K_disabled=${kDisabled}`);

    const manifest = loadManifest();
    console.log(`[suggester] manifest: ${manifest.size} personas`);
    if (manifest.size === 0) {
        throw new Error('persona manifest is empty — run lock_in_iterative / cluster_disambiguator / explore_corpus first');
    }

    console.log(`\n[suggester] calling ContextJudge…`);
    const { targets, raw: judgeRaw, coverage } = await judgeContext(messages, kTotal, manifest);
    const N = manifest.size;
    console.log(`[suggester] axis coverage across manifest (axes with ≥1 persona measured):`);
    for (const a of Object.keys(coverage).filter(a => coverage[a] > 0).sort((x, y) => coverage[y] - coverage[x])) {
        console.log(`    ${a}: ${coverage[a]}/${N} (${Math.round(100*coverage[a]/N)}%)`);
    }
    console.log(`[suggester] judge proposed ${targets.length} targets:`);
    for (let i = 0; i < targets.length; i++) {
        const t = targets[i];
        const axisStr = Object.entries(t.axes).map(([a, v]) => `${a}=${v}`).join(' ');
        console.log(`  T${i+1}: "${t.label}"`);
        console.log(`      rationale: ${t.rationale}`);
        console.log(`      axes: ${axisStr}`);
    }

    console.log(`\n[suggester] picking nearest persona per target…`);
    const picks = [];
    const used = new Set();  // don't double-pick the same persona
    for (let i = 0; i < targets.length; i++) {
        const t = targets[i];
        const pick = pickNearestForTarget(t, manifest, used);
        if (!pick) {
            console.log(`  T${i+1} → NO PERSONAS AVAILABLE`);
            picks.push({ target: t, persona: null });
            continue;
        }
        used.add(pick.persona_id);
        picks.push({ target: t, persona: pick });
        console.log(`  T${i+1} "${t.label}" → ${pick.name}`);
        console.log(`      dist=${pick.distance.toFixed(2)} (compared ${pick.n_axes_compared} axes, ${pick.n_axes_missing} target axes missing from persona)`);
        const sigStr = Object.entries(pick.sig).map(([a, v]) => `${a}=${typeof v === 'number' ? v.toFixed(2) : v}`).join(' ');
        console.log(`      persona sig: ${sigStr || '(empty)'}`);
    }

    const active = picks.slice(0, kActive);
    const disabled = picks.slice(kActive, kTotal);

    console.log(`\n═══ ACTIVE picks (K_1=${kActive}, would auto-/poll) ═══`);
    for (const p of active) {
        if (p.persona) console.log(`  ${p.persona.name}  (target: "${p.target.label}")`);
        else            console.log(`  (no match for target "${p.target.label}")`);
    }
    console.log(`\n═══ DISABLED picks (K_2=${kDisabled}, shown but click-to-enable) ═══`);
    for (const p of disabled) {
        if (p.persona) console.log(`  ${p.persona.name}  (target: "${p.target.label}")`);
        else            console.log(`  (no match for target "${p.target.label}")`);
    }

    fs.mkdirSync(OUT_DIR, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const chatLabel = path.basename(chatPath, '.json');
    const outFile = path.join(OUT_DIR, `${chatLabel}-${ts}.json`);
    fs.writeFileSync(outFile, JSON.stringify({
        chat_path: chatPath,
        chat_messages: messages,
        k_active: kActive,
        k_disabled: kDisabled,
        manifest_size: manifest.size,
        judge_raw: judgeRaw,
        targets,
        picks,
        active: active.map(p => ({ target_label: p.target.label, persona_id: p.persona?.persona_id, persona_name: p.persona?.name })),
        disabled: disabled.map(p => ({ target_label: p.target.label, persona_id: p.persona?.persona_id, persona_name: p.persona?.name })),
    }, null, 2));
    console.log(`\n[suggester] full evidence: ${outFile}`);
    return { active, disabled, manifest_size: manifest.size };
}

// ── CLI ──────────────────────────────────────────────────────────────

const args = (() => {
    const a = { chatPath: null, kActive: DEFAULT_K_ACTIVE, kDisabled: DEFAULT_K_DISABLED };
    const r = process.argv.slice(2);
    for (let i = 0; i < r.length; i++) {
        if (r[i] === '--k-active') a.kActive = Number(r[++i]);
        else if (r[i] === '--k-disabled') a.kDisabled = Number(r[++i]);
        else if (!a.chatPath) a.chatPath = r[i];
    }
    return a;
})();

if (!args.chatPath) {
    console.error('usage: node context_suggester.mjs <chat.json> [--k-active N] [--k-disabled N]');
    process.exit(2);
}
await runSuggester(args.chatPath, args.kActive, args.kDisabled);
