// harness_lib.mjs
//
// Shared primitives across the user-agent harness scripts.
// Extracted (initially) for explore_corpus.mjs; lock_in_iterative.mjs
// and cluster_disambiguator.mjs each currently vendor their own copies.
// Future refactor opportunity: migrate them to import from here.

const ST     = 'http://127.0.0.1:8002';
const BRIDGE = 'http://127.0.0.1:8001';
const PLUGIN = `${ST}/api/plugins/user-personas`;
const MODEL  = 'gemma-4-a4b';

export const ENDPOINTS = { ST, BRIDGE, PLUGIN, MODEL };

// ── http / bridge ────────────────────────────────────────────────────

export async function http(method, url, body) {
    const r = await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
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

export async function bridgeCall(messages, { max_tokens = null, temperature = 1.0 } = {}) {
    const body = { model: MODEL, messages, stream: false, temperature };
    if (max_tokens) body.max_tokens = max_tokens;
    const r = await http('POST', `${BRIDGE}/v1/chat/completions`, body);
    return (r.choices?.[0]?.message?.content || '').trim();
}

// ── persistence ──────────────────────────────────────────────────────

export async function saveBio({ canonical_key, name, prose }) {
    await http('POST', `${PLUGIN}/personas/${encodeURIComponent(canonical_key)}`, {
        name, bio: prose,
        system_prompt: `You are ${name}. ${prose}`,
    });
}

export async function saveAgent(agent_id, name, agent_text, designed_for_bio_id) {
    await http('POST', `${PLUGIN}/agents/${encodeURIComponent(agent_id)}`, {
        name, agent_text,
        designed_for_bio_id,
        injection_mode: 'authors_note',
        injection_depth: 1,
    });
}

// ── counterparty + chat ──────────────────────────────────────────────

export async function fetchCounterparty(avatarUrl) {
    const c = await http('POST', `${ST}/api/characters/get`, { avatar_url: avatarUrl });
    return {
        name: c.name || avatarUrl.replace(/\.png$/, ''),
        system_prompt: (c.system_prompt && c.system_prompt.trim()) || c.description || '',
        first_mes: c.first_mes || '*The scene begins.*',
    };
}

export async function runChat(bio, agent_id, cp, n_turns) {
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

export function userTurns(chat) {
    return chat.filter(m => m.is_user).map(m => m.mes);
}

// ── cheap agent designer (K=1 single-pass, neutral target) ──────────

export async function designCheapAgent(bio) {
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

// ── judge: single axis on one turn ───────────────────────────────────

export async function judgeOnAxis(turn, axisName, axisDef) {
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

// ── judge: multiple axes in one prompt (cheaper, prose-line emission) ──
//
// Same pattern as judgeTurnMerged: rubric up front, turn at end, prose
// lines emitted, tolerant per-line parse. Used for sparse-sampled
// multi-axis judging when ≤ ~14 axes fit comfortably in one prompt.

export async function judgeOnAxes(turn, axes) {
    // V6_pure_minimal prompt: best variant from the A/B characterization
    // in judge_prompt_ab.mjs (run-2). Aggregate MAE=0.61, std=0.08, the
    // best on both axes among 7 variants. Every variant that TOLD the
    // judge what numbers to favor (V0 "score 1 when absent", V1 "use full
    // range / most turns 2-4", V3 "describe + use full range", V4 explicit
    // anchors) sat in the middle or back of the pack. Stripping calibration
    // nudges entirely beats prompt-engineering them. The lesson: tell the
    // judge what to score, not how to score.
    const rubric = axes.map(a => `- **${a.name}** — ${a.def}`).join('\n');
    const template = axes.map(a => `${a.name}: ?`).join('\n');
    const sys =
        'Score the turn on each axis. Output one line per axis: ' +
        '"axis_name: N" where N is an integer 1-5 per the rubric.';
    const usr =
        '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
        '## Turn to score\n\n' +
        '> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
        '## Emit\n\n' + template + '\n';
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }],
        { max_tokens: 50 + axes.length * 20 });
    const sig = {};
    for (const a of axes) sig[a.name] = null;
    for (const line of raw.split('\n')) {
        for (const a of axes) {
            const escName = a.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp(
                '^\\s*[-*]?\\s*\\**["\']?' + escName + '["\']?\\**\\s*[:=]\\s*([1-5])',
                'i');
            const m = line.match(re);
            if (m) sig[a.name] = Number(m[1]);
        }
    }
    return { sig, raw };
}

// ── statistics ───────────────────────────────────────────────────────

export function meanStd(arr) {
    const filt = arr.filter(Number.isFinite);
    if (!filt.length) return { mean: null, std: null, n: 0, var: null };
    const mean = filt.reduce((a, b) => a + b, 0) / filt.length;
    const variance = filt.reduce((s, v) => s + (v - mean) ** 2, 0) / Math.max(1, filt.length - 1);
    return { mean, std: Math.sqrt(variance), n: filt.length, var: variance };
}

/**
 * Per-axis participation-ratio "effective dimensionality" proxy.
 *
 * Given a matrix of signatures (rows = bios, cols = axes; missing =
 * null), compute per-axis variance across bios, normalize to a
 * probability distribution, then:
 *     PR = 1 / Σ p_i²
 *
 * Ranges from 1 (one axis dominates the spread) to N_axes (all axes
 * carry equal spread). Returns null if fewer than 2 bios or zero total
 * variance.
 *
 * NOTE: this is the per-axis-marginal proxy, NOT real PCA. It ignores
 * cross-axis correlations. For small k (< ~10 bios) on a low-dimension
 * axis set with mostly-uncorrelated axes, it's a reasonable monotonic
 * proxy. Swap for real PCA effective-dim (e.g. participation ratio over
 * singular values of the centered data matrix) once k or correlation
 * structure makes this misleading.
 */
export function effDimParticipationRatio(sigsByBio, axisNames) {
    const bios = Object.keys(sigsByBio);
    if (bios.length < 2) return { effDim: null, perAxisVar: {}, n: bios.length, note: 'need ≥2 bios' };
    const perAxisVar = {};
    for (const a of axisNames) {
        const vals = bios.map(b => sigsByBio[b][a]).filter(Number.isFinite);
        perAxisVar[a] = vals.length >= 2 ? meanStd(vals).var : 0;
    }
    const totalVar = Object.values(perAxisVar).reduce((a, b) => a + b, 0);
    if (totalVar <= 0) return { effDim: null, perAxisVar, n: bios.length, note: 'zero total variance' };
    const p = Object.fromEntries(Object.entries(perAxisVar).map(([k, v]) => [k, v / totalVar]));
    const effDim = 1 / Object.values(p).reduce((s, pi) => s + pi * pi, 0);
    return { effDim, perAxisVar, normalized: p, totalVar, n: bios.length };
}
