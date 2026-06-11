#!/usr/bin/env node
// coverage_probe.mjs — held-out ontology recovery meter (EVALUATION-ONLY).
//
// The variegation loop's internal meter (lineage-normalized eff-dim) says
// "the spread grew"; this probe says "it grew toward the KNOWN holes."
// A closed categorical ontology that the seeds undersample (2 of 12
// zodiac signs; 2-ish of 5 MtG colors) is held OUT of all generation:
// it appears in NO synthesis prompt, NO scene schema, and is NEVER
// registered as axes (registering 12 sign-axes would be authoring axes —
// the forbidden move; see empirical_posterior_not_claude_prior). It is
// used purely to CLASSIFY what the loop produced. The supervisory signal
// is the coverage curve across iterations: histogram entropy / distinct
// labels evinced. Recovery (2 → 5 → 9 signs) = iterated measurement-
// driven self-play genuinely reaches under-sampled regions of persona-
// space; plateau = mode collapse, detected without any human ever
// writing a capricorn.
//
// Material per bio = the bio's own prose + (when available) its most
// recent trajectory user-turns — all on-policy Gemma text.
//
// CLI:
//   node coverage_probe.mjs [--labels zodiac|mtg|both] [--traj-dir <dir>]
// Env: ST_URL (required, like every harness script), PLUGIN_URL optional,
//      USER_PERSONAS_DATA_DIR for artifact placement.
// Output: data/coverage_probe/<labelset>-<ts>.json (full evidence) +
//         data/coverage_probe/series.jsonl (one summary line per run —
//         the recovery curve's time series).

import fs from 'node:fs';
import path from 'node:path';
import * as L from './harness_lib.mjs';

const LABEL_SETS = {
    zodiac: ['aries', 'taurus', 'gemini', 'cancer', 'leo', 'virgo',
             'libra', 'scorpio', 'sagittarius', 'capricorn', 'aquarius', 'pisces'],
    mtg: ['white', 'blue', 'black', 'red', 'green'],
};

const args = process.argv.slice(2);
function argVal(flag, dflt) {
    const i = args.indexOf(flag);
    return i >= 0 && args[i + 1] ? args[i + 1] : dflt;
}
const labelsArg = argVal('--labels', 'both');
const SETS = labelsArg === 'both' ? ['zodiac', 'mtg'] : [labelsArg];
for (const s of SETS) {
    if (!LABEL_SETS[s]) throw new Error(`unknown label set '${s}' (zodiac|mtg|both)`);
}

const DATA_DIR = process.env.USER_PERSONAS_DATA_DIR
    || path.resolve(path.dirname(new URL(import.meta.url).pathname), '..', 'data');
const OUT_DIR = path.join(DATA_DIR, 'coverage_probe');
const TRAJ_DIR = argVal('--traj-dir', null);

// ── trajectory turn harvest (on-policy user text) ────────────────────
// Walk any lock_in trajectory JSON collecting user-turn strings from
// turnJudgments / bioTurnJudgments without caring about shape details.
function harvestTurns(node, out) {
    if (!node || typeof node !== 'object') return;
    if (Array.isArray(node)) { for (const v of node) harvestTurns(v, out); return; }
    if (typeof node.turn === 'string' && node.turn.trim()) out.push(node.turn.trim());
    for (const v of Object.values(node)) harvestTurns(v, out);
}

function latestTrajDirFor(lockInBase) {
    // Newest experiment subdir under the lock_in data dir (by mtime).
    if (!fs.existsSync(lockInBase)) return null;
    const dirs = fs.readdirSync(lockInBase)
        .map(d => path.join(lockInBase, d))
        .filter(p => { try { return fs.statSync(p).isDirectory(); } catch { return false; } })
        .sort((a, b) => fs.statSync(b).mtimeMs - fs.statSync(a).mtimeMs);
    return dirs[0] || null;
}

function turnsByBioSlug(trajDir) {
    const out = new Map();
    if (!trajDir || !fs.existsSync(trajDir)) return out;
    for (const f of fs.readdirSync(trajDir).filter(f => f.endsWith('.json'))) {
        try {
            const traj = JSON.parse(fs.readFileSync(path.join(trajDir, f), 'utf8'));
            const turns = [];
            harvestTurns(traj, turns);
            out.set(f.replace(/\.json$/, ''), [...new Set(turns)]);
        } catch { /* unreadable file → skip; disclosed in artifact */ }
    }
    return out;
}

// ── closed-set classification judge ──────────────────────────────────
// K-shot consumer pattern: tolerant parse + in-loop feedback retry
// (k_shot_consumer_pattern). The judge sees ONLY the label list and the
// material — no rubrics, no sign/color lore from us (the model's own
// prior of the categories is exactly what we're measuring against).
async function classifyOne(material, labels, attempts = 3) {
    const sys =
        'You are a closed-set behavioral classifier. Read the persona ' +
        'material and answer with EXACTLY ONE label from the list — the ' +
        'one the persona\'s behavior and voice most evince. Output only ' +
        'the line "label: <label>". No prose, no reasoning, no markdown.';
    let feedback = '';
    for (let i = 0; i < attempts; i++) {
        const usr =
            `## Labels\n\n${labels.join(', ')}\n\n` +
            `## Persona material\n\n${material}\n\n` +
            (feedback ? `## Note\n\n${feedback}\n\n` : '') +
            `## Emit\n\nlabel: ?\n`;
        const raw = await L.bridgeCall(
            [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
        const lower = raw.toLowerCase();
        const m = lower.match(/label\s*[:=]\s*([a-z]+)/);
        if (m && labels.includes(m[1])) return { label: m[1], raw };
        // Tolerant fallback: exactly one label word present anywhere.
        const present = labels.filter(l => lower.includes(l));
        if (present.length === 1) return { label: present[0], raw };
        feedback = `Your previous answer was not exactly one label from the list. Answer with one of: ${labels.join(', ')}.`;
    }
    return { label: null, raw: '(unparseable after retries)' };
}

function entropyBits(hist, n) {
    if (n === 0) return 0;
    let h = 0;
    for (const c of Object.values(hist)) {
        if (c > 0) { const p = c / n; h -= p * Math.log2(p); }
    }
    return h;
}

// ── main ─────────────────────────────────────────────────────────────
const personas = (await L.http('GET', `${L.ENDPOINTS.PLUGIN}/personas`)).personas || [];
const items = personas.filter(p => (p.bio || '').trim() || (p.system_prompt || '').trim());
if (items.length === 0) throw new Error('coverage_probe: no personas with prose in corpus');

const lockInBase = process.env.USER_PERSONAS_LOCK_IN_DATA_DIR
    || path.join(DATA_DIR, 'lock_in_iterative');
const trajDir = TRAJ_DIR || latestTrajDirFor(lockInBase);
const turnsBySlug = turnsByBioSlug(trajDir);
console.log(`[coverage_probe] ${items.length} personas; trajectory turns from: ${trajDir || '(none found)'}`);

function materialFor(p) {
    const slugGuesses = [
        (p.name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-'),
        (p.id || '').replace(/\.png$/, '').toLowerCase(),
    ];
    let turns = [];
    for (const [slug, ts] of turnsBySlug) {
        if (slugGuesses.some(g => g && (slug.includes(g) || g.includes(slug)))) { turns = ts; break; }
    }
    const sampled = turns.slice(0, 4).map((t, i) => `(${i + 1}) ${t.slice(0, 300)}`).join('\n');
    return [
        `Name: ${p.name || p.id}`,
        p.bio ? `Bio: ${p.bio.slice(0, 800)}` : '',
        p.system_prompt ? `Voice: ${p.system_prompt.slice(0, 400)}` : '',
        sampled ? `Recent in-character turns:\n${sampled}` : '',
    ].filter(Boolean).join('\n\n');
}

const ts = new Date().toISOString().replace(/[:.]/g, '-');
fs.mkdirSync(OUT_DIR, { recursive: true });

for (const setName of SETS) {
    const labels = LABEL_SETS[setName];
    const results = await L.saturatedMap(items, async p => {
        const { label, raw } = await classifyOne(materialFor(p), labels);
        return { id: p.id, name: p.name || p.id, label, raw_tail: raw.slice(-120) };
    });
    const hist = Object.fromEntries(labels.map(l => [l, 0]));
    let unparseable = 0;
    for (const r of results) {
        if (r.label) hist[r.label]++;
        else unparseable++;
    }
    const n = results.length - unparseable;
    const distinct = Object.values(hist).filter(c => c > 0).length;
    const h = entropyBits(hist, n);
    const summary = {
        ts: new Date().toISOString(),
        label_set: setName,
        n_items: results.length,
        n_classified: n,
        unparseable,
        distinct_labels: distinct,
        of_possible: labels.length,
        entropy_bits: Number(h.toFixed(3)),
        max_entropy_bits: Number(Math.log2(labels.length).toFixed(3)),
        histogram: hist,
        traj_dir: trajDir || null,
    };
    const evidence = { ...summary, results };
    const outFile = path.join(OUT_DIR, `${setName}-${ts}.json`);
    fs.writeFileSync(outFile, JSON.stringify(evidence, null, 2));
    fs.appendFileSync(path.join(OUT_DIR, 'series.jsonl'), JSON.stringify(summary) + '\n');
    console.log(`\n[coverage_probe:${setName}] coverage ${distinct}/${labels.length} labels, entropy ${h.toFixed(2)}/${Math.log2(labels.length).toFixed(2)} bits (n=${n}${unparseable ? `, ${unparseable} unparseable` : ''})`);
    for (const [l, c] of Object.entries(hist)) {
        if (c > 0) console.log(`    ${l.padEnd(12)} ${'█'.repeat(c)} ${c}`);
    }
    console.log(`[coverage_probe:${setName}] evidence: ${outFile}`);
}
