#!/usr/bin/env node
// cluster_disambiguator.mjs
//
// Item 6 in docs/feature_factorization_design.md.
//
// Given a "tight cluster" of bios that all measure similarly on the
// existing axis registry, attempt to find a NEW behavioral axis that
// spreads the cluster. If none exists, distinguish the failure mode:
//   - ParaphraseDegenerate    (bios are reworded versions of each other)
//   - BehaviorallyDegenerate  (bios are prose-different but project to
//                              identical behavior under current vocab)
//
// Every cluster bio is paired with a CHEAPLY-ITERATED agent (K=1 single
// pass) — we never measure bios in the asking-the-LM-to-play-both-roles
// degenerate mode.
//
// Turn-depth matters more than trajectory-count: N_TURNS=4 catches
// trailing-off / boring-fixed-point signals; N_TRAJ=2 keeps wallclock
// down. (See §6 for rationale.)
//
// CLI: node cluster_disambiguator.mjs <cluster_spec.json>

import fs from 'node:fs';
import path from 'node:path';
import { allAxes, registerDerivedAxis } from './axis_registry.mjs';

const ST     = 'http://127.0.0.1:8002';
const BRIDGE = 'http://127.0.0.1:8001';
const PLUGIN = `${ST}/api/plugins/user-personas`;
const MODEL  = 'gemma-4-a4b';
const OUT_DIR = '/Users/mdot/metal-microbench/data/cluster_disambig';

// ── tunables (see §6 Acceptance Thresholds) ──────────────────────────
const N_TRAJ_PER_BIO     = 2;
const N_TURNS_PER_TRAJ   = 4;
const N_HYPOTHESES       = 3;
const SPREAD_THRESHOLD   = 1.5;     // Likert points (max - min of per-bio means)
const F_RATIO_THRESHOLD  = 3.5;     // ≈ F_crit(2,21) at α=0.05 for k=3, N_T=2, n_t=4
const PARAPHRASE_THRESHOLD = 4.0;   // pairwise prose-sim ≥ 4/5 → paraphrase verdict
const TIGHTNESS_THRESHOLD  = 1.0;   // max pairwise existing-axis distance for "tight"
const EPSILON = 1e-6;

// ── http / bridge ────────────────────────────────────────────────────

async function http(method, url, body) {
    const r = await fetch(url, {
        method, headers: { 'Content-Type': 'application/json' },
        body: body ? JSON.stringify(body) : undefined,
    });
    const text = await r.text();
    let parsed;
    try { parsed = JSON.parse(text); } catch { parsed = text; }
    if (!r.ok) {
        const detail = typeof parsed === 'string' ? parsed.slice(0, 300)
                                                  : JSON.stringify(parsed).slice(0, 300);
        throw new Error(`${method} ${url} → ${r.status}: ${detail}`);
    }
    return parsed;
}

async function bridgeCall(messages, { max_tokens = null } = {}) {
    const body = { model: MODEL, messages, stream: false, temperature: 1.0 };
    if (max_tokens) body.max_tokens = max_tokens;
    const r = await http('POST', `${BRIDGE}/v1/chat/completions`, body);
    return (r.choices?.[0]?.message?.content || '').trim();
}

// ── persistence: install cluster bios + cheap agents into ST ─────────

async function saveBio(bio) {
    await http('POST', `${PLUGIN}/personas/${encodeURIComponent(bio.canonical_key)}`, {
        name: bio.name,
        bio: bio.prose,
        system_prompt: `You are ${bio.name}. ${bio.prose}`,
    });
}

async function saveAgent(agent_id, name, agent_text, designed_for_bio_id) {
    await http('POST', `${PLUGIN}/agents/${encodeURIComponent(agent_id)}`, {
        name, agent_text,
        designed_for_bio_id,
        injection_mode: 'authors_note',
        injection_depth: 1,
    });
}

// ── cheap agent designer (K=1 single pass; neutral target) ───────────

async function designCheapAgent(bio) {
    const sys =
        'You design a short user-agent overlay (author\'s-note style, ' +
        'second-person "You will…") that helps a user-persona stay ' +
        'vividly themselves across multi-turn chat. The target is ' +
        'NEUTRAL: just be your character vividly, engage with what the ' +
        'counterparty offers, don\'t go vacant or generic. 2-3 sentences. ' +
        'Output ONLY the agent_text.';
    const usr =
        '## Bio (the user-persona this agent will overlay)\n\n' +
        bio.prose + '\n\n' +
        'Write the agent_text now (2-3 sentences, neutral "be vividly yourself" target).';
    return await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }],
        { max_tokens: 200 });
}

// ── counterparty + chat runner ───────────────────────────────────────

async function fetchCounterparty(avatarUrl) {
    const c = await http('POST', `${ST}/api/characters/get`, { avatar_url: avatarUrl });
    return {
        name: c.name || avatarUrl.replace(/\.png$/, ''),
        system_prompt: (c.system_prompt && c.system_prompt.trim()) || c.description || '',
        first_mes: c.first_mes || '*The scene begins.*',
    };
}

async function runChat(bio, agent_id, cp, n_turns) {
    const chat = [{ name: cp.name, is_user: false, mes: cp.first_mes }];
    for (let i = 0; i < n_turns; i++) {
        const pollResp = await http('POST', `${PLUGIN}/poll`, {
            persona_id: bio.canonical_key, agent_id, chat, n_candidates: 1,
        });
        const cand = (pollResp.candidates || [])[0];
        const userText = (cand?.text || cand?.mes || '').trim();
        if (!userText) throw new Error(`empty user turn on turn ${i+1} (bio=${bio.canonical_key})`);
        chat.push({ name: bio.name, is_user: true, mes: userText });
        const cpMessages = [
            { role: 'system', content: cp.system_prompt },
            ...chat.map(m => ({ role: m.is_user ? 'user' : 'assistant', content: m.mes })),
        ];
        const cpText = await bridgeCall(cpMessages, { max_tokens: 200 });
        chat.push({ name: cp.name, is_user: false, mes: cpText.trim() });
    }
    return chat;
}

function userTurns(chat) {
    return chat.filter(m => m.is_user).map(m => m.mes);
}

// ── single-axis judge (parallelizable) ───────────────────────────────

async function judgeOnAxis(turn, axisName, axisDef) {
    const sys =
        'You are a behavioural-axis judge. You read ONE user-side chat ' +
        'turn and score it on the named axis (integer 1-5). The turn is ' +
        'the only ground truth — do not infer from anything else. Be ' +
        'willing to score 1 (absence) when the turn genuinely shows no ' +
        'expression of the axis. Output ONLY the axis line below as ' +
        '"axis_name: <integer 1-5>". No preamble, no commentary.';
    const usr =
        `## Axis (1-5)\n\n- **${axisName}** — ${axisDef}\n\n` +
        '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
        `## Emit\n\n${axisName}: ?\n`;
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }],
        { max_tokens: 50 });
    const escName = axisName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp('^\\s*[-*]?\\s*\\**["\']?' + escName +
                          '["\']?\\**\\s*[:=]\\s*([1-5])', 'im');
    const m = raw.match(re);
    return m ? Number(m[1]) : null;
}

// ── statistics ───────────────────────────────────────────────────────

function meanStd(arr) {
    const filt = arr.filter(Number.isFinite);
    if (!filt.length) return { mean: null, std: null, n: 0 };
    const mean = filt.reduce((a, b) => a + b, 0) / filt.length;
    const variance = filt.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, filt.length - 1);
    return { mean, std: Math.sqrt(variance), n: filt.length, var: variance };
}

/**
 * One-way ANOVA F-ratio: between-group / within-group variance.
 * `perGroup`: array of arrays of scalars. Returns { F, between_var, within_var, df_between, df_within }.
 */
function fRatio(perGroup) {
    const groups = perGroup.filter(g => g.filter(Number.isFinite).length > 0);
    const k = groups.length;
    const grandAll = groups.flat().filter(Number.isFinite);
    const N = grandAll.length;
    if (k < 2 || N < k + 1) return { F: null, between_var: null, within_var: null };
    const grandMean = grandAll.reduce((a, b) => a + b, 0) / N;
    let ssBetween = 0, ssWithin = 0;
    for (const g of groups) {
        const vals = g.filter(Number.isFinite);
        const m = vals.reduce((a, b) => a + b, 0) / vals.length;
        ssBetween += vals.length * (m - grandMean) ** 2;
        for (const v of vals) ssWithin += (v - m) ** 2;
    }
    const dfB = k - 1, dfW = N - k;
    const msB = ssBetween / dfB;
    const msW = ssWithin / dfW;
    return { F: msW > EPSILON ? msB / msW : Infinity,
             between_var: msB, within_var: msW,
             df_between: dfB, df_within: dfW };
}

// ── DESIGNER_C: propose candidate spread axes ────────────────────────

function fmtBioBlock(bio, sampleTurns) {
    return `### "${bio.name}" (id: ${bio.canonical_key})\n\n` +
           `prose: ${bio.prose}\n\n` +
           `sample turns:\n` +
           sampleTurns.map((t, i) => `  (${i+1}) ${t.slice(0, 250).replace(/\n+/g, ' ')}…`).join('\n');
}

async function proposeSpreadAxes(bios, trajs, tightnessReport) {
    const sys =
        'You are a behavioral-axis disambiguator. The operator has a ' +
        `cluster of ${bios.length} user-persona bios that all measure ` +
        'identically on the existing axis registry. Your job is to ' +
        'propose NEW Likert axes (1-5, with a one-sentence rubric in the ' +
        'usual "1: low pole · 5: high pole" form) such that AT LEAST ONE ' +
        'of them spreads these bios as widely as possible when ' +
        'each bio is scored on it. The axes must be:\n' +
        '  - behaviorally observable in user-side chat turns (not "what ' +
        '    the prose says about the user", but "what the user actually does"),\n' +
        '  - not redundant with the existing registry (listed below),\n' +
        '  - capable of producing meaningfully different scores across the bios.\n' +
        'Output ONLY a JSON object: ' +
        '{"hypotheses": [{"id": int, "name": str, "def": str, "rationale": str}, ...]}. ' +
        'No preamble, no markdown.';
    const registryListing = allAxes()
        .map(a => `  - ${a.name} (${a.kind}): ${a.def.slice(0, 120)}…`).join('\n');
    const bioBlocks = bios.map(b => fmtBioBlock(b, userTurns(trajs[b.canonical_key][0]).slice(0, 2))).join('\n\n');
    const usr =
        `## Tightness report (existing-axis signature)\n\n${tightnessReport}\n\n` +
        `## The cluster\n\n${bioBlocks}\n\n` +
        `## Existing axis registry (do NOT propose duplicates of these)\n\n${registryListing}\n\n` +
        `## Emit\n\nPropose ${N_HYPOTHESES} candidate spread axes as JSON. Each must include a one-sentence rationale for why this axis would distinguish at least 2 of these bios.`;
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }],
        { max_tokens: 1200 });
    const m = raw.match(/\{[\s\S]*\}/);
    if (!m) throw new Error(`could not parse JSON from DESIGNER_C:\n${raw.slice(0, 500)}`);
    const parsed = JSON.parse(m[0]);
    if (!Array.isArray(parsed.hypotheses)) throw new Error('DESIGNER_C output missing hypotheses[]');
    return { hypotheses: parsed.hypotheses, raw };
}

// ── pre-flight tightness check on nominal-tight axis ─────────────────

async function tightnessPreflight(bios, trajs, nominalAxisName) {
    const axis = allAxes().find(a => a.name === nominalAxisName);
    if (!axis) {
        return { skipped: true, reason: `nominal_tight_axis '${nominalAxisName}' not in registry`,
                 tight: true, distances: {}, summary: 'skipped' };
    }
    const perBio = {};
    for (const bio of bios) {
        const turns = trajs[bio.canonical_key].flatMap(userTurns);
        const scores = await Promise.all(turns.map(t => judgeOnAxis(t, axis.name, axis.def)));
        perBio[bio.canonical_key] = meanStd(scores);
    }
    const means = Object.values(perBio).map(s => s.mean).filter(Number.isFinite);
    const distance = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    const summaryLines = bios.map(b => {
        const s = perBio[b.canonical_key];
        return `  ${b.canonical_key}: ${nominalAxisName}=${s.mean?.toFixed(2)}±${s.std?.toFixed(2)} (n=${s.n})`;
    });
    return {
        skipped: false,
        nominal_axis: nominalAxisName,
        per_bio: perBio,
        spread: distance,
        tight: distance <= TIGHTNESS_THRESHOLD,
        summary: `existing-axis tightness on ${nominalAxisName}: max-min spread=${distance.toFixed(2)} (threshold ${TIGHTNESS_THRESHOLD})\n${summaryLines.join('\n')}`,
    };
}

// ── pairwise similarity (LLM-as-judge) ───────────────────────────────

function pairs(arr) {
    const out = [];
    for (let i = 0; i < arr.length; i++)
        for (let j = i+1; j < arr.length; j++) out.push([arr[i], arr[j]]);
    return out;
}

async function judgeSimilarity(promptLabel, defText, blockA, blockB) {
    const sys =
        'You score similarity between two blocks of text on a 1-5 Likert ' +
        'scale. Output ONLY the line "similarity: <integer 1-5>". No ' +
        'preamble, no markdown.';
    const usr =
        `## What "similarity" means here\n\n${defText}\n\n` +
        `## Block A\n\n${blockA}\n\n## Block B\n\n${blockB}\n\n` +
        `## Emit\n\nsimilarity: ?\n`;
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }],
        { max_tokens: 30 });
    const m = raw.match(/similarity\s*:\s*([1-5])/i);
    return m ? Number(m[1]) : null;
}

async function pairwiseProseSim(bios) {
    const ps = pairs(bios);
    const scores = await Promise.all(ps.map(([a, b]) => judgeSimilarity(
        'prose-similarity',
        '1: completely different (unrelated topics, registers, or content); 3: same topic, different presentation; 5: paraphrases of each other (same content, different wording)',
        a.prose, b.prose)));
    return { mean: scores.filter(Number.isFinite).reduce((x, y) => x + y, 0) / scores.length,
             pairs: ps.map(([a, b], i) => ({ a: a.name, b: b.name, score: scores[i] })) };
}

async function pairwiseBehaviorSim(bios, trajs) {
    const ps = pairs(bios);
    const sample = bio => userTurns(trajs[bio.canonical_key][0]).slice(0, 2)
        .map((t, i) => `  (${i+1}) ${t.slice(0, 250)}`).join('\n');
    const scores = await Promise.all(ps.map(([a, b]) => judgeSimilarity(
        'behavioral-similarity',
        '1: completely different behaviors / moves / strategies; 3: overlapping but distinguishable behavioral repertoires; 5: behaviorally interchangeable (same moves, same strategies, same dispositional expression)',
        sample(a), sample(b))));
    return { mean: scores.filter(Number.isFinite).reduce((x, y) => x + y, 0) / scores.length,
             pairs: ps.map(([a, b], i) => ({ a: a.name, b: b.name, score: scores[i] })) };
}

// ── evaluate one spread-axis hypothesis ──────────────────────────────

async function evaluateSpread(hypothesis, bios, trajs) {
    const perBioRaw = {};   // bio_id → [score per turn]
    const tasks = [];
    for (const bio of bios) {
        const turns = trajs[bio.canonical_key].flatMap(userTurns);
        perBioRaw[bio.canonical_key] = new Array(turns.length).fill(null);
        for (let i = 0; i < turns.length; i++) {
            const idx = i;
            tasks.push(judgeOnAxis(turns[i], hypothesis.name, hypothesis.def)
                .then(score => { perBioRaw[bio.canonical_key][idx] = score; }));
        }
    }
    await Promise.all(tasks);
    const perBio = {};
    for (const bio of bios) {
        perBio[bio.canonical_key] = meanStd(perBioRaw[bio.canonical_key]);
    }
    const means = Object.values(perBio).map(s => s.mean).filter(Number.isFinite);
    const spread = means.length >= 2 ? Math.max(...means) - Math.min(...means) : 0;
    const fr = fRatio(bios.map(b => perBioRaw[b.canonical_key]));
    return { hypothesis, perBio, perBioRaw, spread, ...fr };
}

// ── orchestrator ─────────────────────────────────────────────────────

async function runDisambiguator(specPath) {
    const spec = JSON.parse(fs.readFileSync(specPath, 'utf8'));
    console.log(`[disambig] cluster: ${spec.cluster_id} (${spec.bios.length} bios)`);
    console.log(`[disambig] counterparty: ${spec.counterparty_avatar}`);
    console.log(`[disambig] nominal tight axis: ${spec.nominal_tight_axis}`);

    // 1. Install bios + design cheap agents
    console.log(`\n[disambig] installing bios + designing cheap agents (K=1 each)…`);
    const cp = await fetchCounterparty(spec.counterparty_avatar);
    const agentIds = {};
    for (const bio of spec.bios) {
        await saveBio(bio);
        const agentText = await designCheapAgent(bio);
        const agentId = `${spec.cluster_id}-${bio.canonical_key.replace(/\.png$/, '')}-cheap`;
        await saveAgent(agentId, `${bio.name} (cheap)`, agentText, bio.canonical_key);
        agentIds[bio.canonical_key] = { id: agentId, text: agentText };
        console.log(`  ${bio.canonical_key}: cheap_agent="${agentText.slice(0, 80).replace(/\n/g, ' ')}…"`);
    }

    // 2. Run chats per bio
    console.log(`\n[disambig] running ${N_TRAJ_PER_BIO} trajectories × ${N_TURNS_PER_TRAJ} turns per bio…`);
    const trajs = {};
    const tChat0 = Date.now();
    const chatTasks = [];
    for (const bio of spec.bios) {
        trajs[bio.canonical_key] = [];
        for (let r = 0; r < N_TRAJ_PER_BIO; r++) {
            const taskIdx = trajs[bio.canonical_key].push(null) - 1;
            const bioRef = bio;
            chatTasks.push(
                runChat(bioRef, agentIds[bioRef.canonical_key].id, cp, N_TURNS_PER_TRAJ)
                    .then(chat => { trajs[bioRef.canonical_key][taskIdx] = chat; })
            );
        }
    }
    await Promise.all(chatTasks);
    console.log(`[disambig] chats done in ${((Date.now()-tChat0)/1000).toFixed(1)}s`);

    // 3. Pre-flight tightness on nominal axis
    console.log(`\n[disambig] tightness pre-flight on ${spec.nominal_tight_axis}…`);
    const tightness = await tightnessPreflight(spec.bios, trajs, spec.nominal_tight_axis);
    console.log(`[disambig] ${tightness.summary}`);
    if (!tightness.skipped && !tightness.tight) {
        console.warn(`[disambig] WARNING: cluster is NOT tight on ${spec.nominal_tight_axis} (spread=${tightness.spread.toFixed(2)} > ${TIGHTNESS_THRESHOLD}). Proceeding anyway but verdict may be misleading.`);
    }

    // 4. DESIGNER_C proposes candidate spread axes
    console.log(`\n[disambig] DESIGNER_C proposing ${N_HYPOTHESES} candidate spread axes…`);
    const { hypotheses, raw: designerRaw } = await proposeSpreadAxes(spec.bios, trajs, tightness.summary);
    for (const h of hypotheses) {
        console.log(`  H${h.id}: ${h.name}\n     def: ${h.def}\n     rationale: ${h.rationale}`);
    }

    // 5. JUDGE_C evaluates each hypothesis
    console.log(`\n[disambig] JUDGE_C scoring all turns under each hypothesis…`);
    const evaluations = [];
    for (const h of hypotheses) {
        process.stdout.write(`  H${h.id}: `);
        const ev = await evaluateSpread(h, spec.bios, trajs);
        evaluations.push(ev);
        const meansStr = spec.bios.map(b => {
            const s = ev.perBio[b.canonical_key];
            return `${b.name.split(/\s+/).pop()}=${s.mean?.toFixed(2)}±${s.std?.toFixed(2)}`;
        }).join(' ');
        console.log(`spread=${ev.spread.toFixed(2)} F=${ev.F?.toFixed(2)} (msB=${ev.between_var?.toFixed(2)}, msW=${ev.within_var?.toFixed(2)})`);
        console.log(`     per-bio means: ${meansStr}`);
    }

    // 6. Pick winner under spread + F thresholds
    const qualified = evaluations.filter(e =>
        Number.isFinite(e.F) && e.F >= F_RATIO_THRESHOLD && e.spread >= SPREAD_THRESHOLD);
    qualified.sort((a, b) => b.F - a.F);

    let verdict, registered = null, similarityReport = null;
    if (qualified.length > 0) {
        const top = qualified[0];
        verdict = 'SPREAD_AXIS_FOUND';
        console.log(`\n[disambig] verdict: SPREAD_AXIS_FOUND`);
        console.log(`            winning axis: ${top.hypothesis.name}`);
        console.log(`            F=${top.F.toFixed(2)} (threshold ${F_RATIO_THRESHOLD}), spread=${top.spread.toFixed(2)} (threshold ${SPREAD_THRESHOLD})`);
        try {
            registered = registerDerivedAxis({
                name: top.hypothesis.name, kind: 'bio', def: top.hypothesis.def,
                derived_from: { parent: null, contexts: `cluster:${spec.cluster_id}`,
                                cluster_members: spec.bios.map(b => b.canonical_key),
                                reason: 'spread_axis' },
            });
            console.log(`            registered as derived axis (kind=bio)`);
        } catch (e) {
            console.warn(`            could not register: ${e.message}`);
        }
    } else {
        console.log(`\n[disambig] no axis qualified (F≥${F_RATIO_THRESHOLD} ∧ spread≥${SPREAD_THRESHOLD}); checking paraphrase vs behavioral degeneracy…`);
        const [proseSim, behaviorSim] = await Promise.all([
            pairwiseProseSim(spec.bios),
            pairwiseBehaviorSim(spec.bios, trajs),
        ]);
        similarityReport = { prose: proseSim, behavior: behaviorSim };
        console.log(`            pairwise prose-similarity:    mean=${proseSim.mean.toFixed(2)}`);
        for (const p of proseSim.pairs) console.log(`              ${p.a} ⇔ ${p.b}: ${p.score}`);
        console.log(`            pairwise behavior-similarity: mean=${behaviorSim.mean.toFixed(2)}`);
        for (const p of behaviorSim.pairs) console.log(`              ${p.a} ⇔ ${p.b}: ${p.score}`);
        verdict = proseSim.mean >= PARAPHRASE_THRESHOLD
            ? 'CLUSTER_IS_PARAPHRASE_DEGENERATE'
            : 'CLUSTER_IS_BEHAVIORALLY_DEGENERATE';
        console.log(`[disambig] verdict: ${verdict}`);
    }

    // 7. Persist evidence
    fs.mkdirSync(OUT_DIR, { recursive: true });
    const ts = new Date().toISOString().replace(/[:.]/g, '-');
    const outFile = path.join(OUT_DIR, `${spec.cluster_id}-${ts}.json`);
    fs.writeFileSync(outFile, JSON.stringify({
        cluster_spec: spec,
        cheap_agents: agentIds,
        trajectories: trajs,
        tightness,
        designer_raw_output: designerRaw,
        hypotheses,
        evaluations,
        qualified: qualified.map(e => e.hypothesis.name),
        verdict,
        registered,
        similarity_report: similarityReport,
        thresholds: { F_RATIO_THRESHOLD, SPREAD_THRESHOLD, PARAPHRASE_THRESHOLD, TIGHTNESS_THRESHOLD },
    }, null, 2));
    console.log(`[disambig] full evidence: ${outFile}`);
    return { verdict, outFile, qualified, registered, similarityReport };
}

// ── CLI ──────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
if (args.length !== 1) {
    console.error('usage: node cluster_disambiguator.mjs <cluster_spec.json>');
    process.exit(2);
}
await runDisambiguator(args[0]);
