// axis_registry.mjs
//
// Typed registry of behavioral axes for the user-agent harness.
// Seeded with the 18 canonical axes (mirroring elicitation/axes.py) +
// the 4 experiment axes used in lock_in_iterative.mjs. The splitter
// (axis_splitter.mjs) extends the registry at runtime when entanglement
// findings prompt it to derive new axes from old ones.
//
// kind ∈ { 'bio', 'agent', 'either', 'meta' }
//   bio    : identity / voice / register / writing-quality
//   agent  : dispositional / move-set / move-selection
//   either : measurable from both bio and agent contributions
//   meta   : judges the bio/agent expression itself (e.g. in_character)
//
// derived_from (optional): present on axes added by the splitter.
//   { parent: 'theft_aggressiveness',
//     contexts: 'no-romance vs courting',
//     hypothesis_id: <uuid> }

import fs from 'node:fs';
import path from 'node:path';

const SEED_AXES = [
    // ── canonical 18 (mirrors elicitation/axes.py order) ──────────────
    { name: 'curious',             kind: 'agent',
      def: '1: accepts at face value · 5: actively asks / probes / explores' },
    { name: 'terse',               kind: 'bio',
      def: '1: verbose / expansive · 5: minimal / clipped' },
    { name: 'warm',                kind: 'either',
      def: '1: cool / aloof · 5: positively engaged with the partner' },
    { name: 'deferential',         kind: 'agent',
      def: '1: takes conversational direction · 5: yields direction' },
    { name: 'performative',        kind: 'bio',
      def: '1: unselfconscious · 5: clearly aware of being-a-character' },
    { name: 'in_character',        kind: 'meta',
      def: '1: off-distribution generic · 5: tightly coherent voice' },
    { name: 'affective_intensity', kind: 'agent',
      def: '1: calm / measured baseline · 5: high-arousal / exclamatory' },
    { name: 'probe_depth',         kind: 'agent',
      def: '1: surface / casual questions · 5: substantive / specific questions' },
    { name: 'goal_clarity',        kind: 'agent',
      def: '1: exploratory / wandering aim · 5: sharp specific objective' },
    { name: 'disclosive',          kind: 'agent',
      def: '1: external topic only / guarded · 5: shares personal context / feelings' },
    { name: 'provocative',         kind: 'agent',
      def: '1: comforting / non-confrontational · 5: challenging / teasing / destabilising' },
    { name: 'register_colloquial', kind: 'bio',
      def: '1: formal / standard English · 5: colloquial / slang / vernacular' },
    { name: 'playful',             kind: 'either',
      def: '1: serious throughout · 5: humour / wordplay / levity' },
    { name: 'structured',          kind: 'either',
      def: '1: flowing prose · 5: organised / numbered / bulleted' },
    { name: 'trope_density',       kind: 'bio',
      def: "1: novel / specific / texture-this-writer-couldn't-have-anticipated · 5: fully tropey / anonymous-genre-material / no individuating texture" },
    { name: 'ludic_engagement',    kind: 'agent',
      def: '1: bio-anchored / self-presentation regardless of scene cues · 5: scene-coupled / actively engages with environmental cues, objects, partner\'s frame' },
    { name: 'user_multipolarity',  kind: 'agent',
      def: '1: dyadic-only / addresses one partner regardless of who else is in the room · 5: multi-pole / treats each participant as separately-addressable' },
    { name: 'other_awareness',     kind: 'agent',
      def: "1: porous / mirrors-or-becomes the partner's frame · 5: anchored / stays in own bio + own motivation while still acknowledging the partner's frame" },

    // ── experiment axes (lock_in_iterative.mjs) ───────────────────────
    { name: 'theft_aggressiveness', kind: 'agent',
      def: "1: ignores property entirely · 2: notices objects in passing · 3: tracks valuables but doesn't act · 4: subtle attempts (sleight, lifting unattended things) · 5: actively pilfers everything not bolted down, brazen" },
    { name: 'romantic_advance',     kind: 'agent',
      def: '1: distant / professional · 2: warm but boundaried · 3: flirtatious · 4: explicit romantic interest, flirt-as-tactic · 5: physically reaching (touch, kiss, embrace)' },
    { name: 'astrology_sagittarian', kind: 'bio',
      def: '1: nothing sagittarian · 2: trace · 3: present but mixed · 4: clearly sagittarian · 5: textbook fire-sign sagittarius (philosophical, optimistic, blunt, restless, big-idea-loving, sometimes tactless)' },
    { name: 'astrology_cancerian',   kind: 'bio',
      def: '1: nothing cancerian · 2: trace · 3: present but mixed · 4: clearly cancerian · 5: textbook water-sign cancer (moody, sentimental, defensive, sensitive, protective)' },
];

// ── storage layer (so the splitter can persist derived axes) ─────────

const DERIVED_STORE = '/Users/mdot/metal-microbench/data/derived_axes.json';

function loadDerived() {
    if (!fs.existsSync(DERIVED_STORE)) return [];
    try { return JSON.parse(fs.readFileSync(DERIVED_STORE, 'utf8')); }
    catch { return []; }
}

function saveDerived(arr) {
    fs.mkdirSync(path.dirname(DERIVED_STORE), { recursive: true });
    fs.writeFileSync(DERIVED_STORE, JSON.stringify(arr, null, 2));
}

// ── public API ───────────────────────────────────────────────────────

export function allAxes() {
    return [...SEED_AXES, ...loadDerived()];
}

export function axisByName(name) {
    return allAxes().find(a => a.name === name) || null;
}

export function axesByKind(kind) {
    return allAxes().filter(a => a.kind === kind || a.kind === 'either');
}

export function bioAxes()   { return axesByKind('bio'); }
export function agentAxes() { return axesByKind('agent'); }

/**
 * Pick n axes from the registry.
 * opts.kind     : restrict to a kind ('bio' | 'agent' | 'either' | 'meta')
 * opts.exclude  : array of axis names to omit
 * opts.recency  : Map<axisName, lastUsedTimestamp>; penalize recent picks
 * opts.rng      : () => float in [0,1); default Math.random
 */
export function pickSubset(n, opts = {}) {
    const rng = opts.rng || Math.random;
    let pool = opts.kind ? axesByKind(opts.kind) : allAxes();
    if (opts.exclude) pool = pool.filter(a => !opts.exclude.includes(a.name));
    if (n >= pool.length) return shuffle(pool, rng);
    if (!opts.recency) return shuffle(pool, rng).slice(0, n);
    // Weighted sample: weight = 1 / (1 + recency_count)
    const weighted = pool.map(a => {
        const lastUsed = opts.recency.get(a.name) || 0;
        const age = Date.now() - lastUsed;
        const recency_count = Math.exp(-age / (1000 * 60 * 60)); // 1h decay
        return { axis: a, weight: 1 / (1 + recency_count) };
    });
    const picked = [];
    while (picked.length < n && weighted.length) {
        const total = weighted.reduce((s, w) => s + w.weight, 0);
        let r = rng() * total;
        let i = 0;
        while (i < weighted.length - 1 && (r -= weighted[i].weight) > 0) i++;
        picked.push(weighted[i].axis);
        weighted.splice(i, 1);
    }
    return picked;
}

/**
 * Register a derived axis (from a successful split).
 * Returns the persisted axis record.
 */
export function registerDerivedAxis({ name, kind, def, derived_from }) {
    if (axisByName(name)) {
        throw new Error(`axis '${name}' already exists in registry`);
    }
    if (!derived_from || !derived_from.parent) {
        throw new Error('registerDerivedAxis requires derived_from.parent');
    }
    const derived = loadDerived();
    const record = { name, kind, def, derived_from, created_at: new Date().toISOString() };
    derived.push(record);
    saveDerived(derived);
    return record;
}

export function listDerivedAxes() {
    return loadDerived();
}

// ── helpers ──────────────────────────────────────────────────────────

function shuffle(arr, rng) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
}

// ── CLI (smoke test) ─────────────────────────────────────────────────

if (import.meta.url === `file://${process.argv[1]}`) {
    console.log(`registry: ${allAxes().length} axes total`);
    console.log(`  bio:    ${bioAxes().length}`);
    console.log(`  agent:  ${agentAxes().length}`);
    console.log(`  derived (persisted): ${listDerivedAxes().length}`);
    console.log('\nrandom subset(k=5, kind=agent):');
    for (const a of pickSubset(5, { kind: 'agent' })) {
        console.log(`  - ${a.name}: ${a.def.slice(0, 70)}…`);
    }
}
