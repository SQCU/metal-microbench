// harness_lib.mjs
//
// Shared primitives across the user-agent harness scripts.
// Extracted (initially) for explore_corpus.mjs; lock_in_iterative.mjs
// and cluster_disambiguator.mjs each currently vendor their own copies.
// Future refactor opportunity: migrate them to import from here.

// 2026-05-24: env-driven URLs. Hardcoded host:port literals here would
// (and did) silently break every deployment that isn't the local-canonical
// setup. The plugin that spawns this harness exports its own resolved URLs
// as env vars so the child inherits them. There are intentionally no default
// ports here: ad-hoc shell invocations must pass ST_URL or PLUGIN_URL.
//
// Lint rule (scripts/lint_port_hardcodes.mjs) forbids new literal
// `127.0.0.1:80\d+` / `localhost:80\d+` in any source file under
// /plugins/user-personas/ or /tools/user-agent-harness/. Add new URL
// surfaces here, not at the literal-use site.
function requireEnv(name) {
    const value = process.env[name];
    if (!value) throw new Error(`[harness_lib] missing required ${name}`);
    return value.replace(/\/+$/, '');
}

const ST     = requireEnv('ST_URL');
const PLUGIN = process.env.PLUGIN_URL || `${ST}/api/plugins/user-personas`;
const MODEL  = process.env.GEMMA_MODEL_NAME || 'gemma-4-a4b';

export const ENDPOINTS = { ST, PLUGIN, MODEL };

function toThreeSigFigs(n) {
    if (!Number.isFinite(n) || n === 0) return n;
    return Number(n.toPrecision(3));
}

function normalizeLikertNumber(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return Math.max(1, Math.min(5, toThreeSigFigs(n)));
}

// ── Auth (loopback into an AUTHENTICATED ST deploy) ──────────────────
//
// The plugin that spawns this harness runs INSIDE ST; the harness fetches
// BACK into ST's HTTP surface — plugin routes (/api/plugins/user-personas/*)
// plus a few core routes (e.g. /api/characters/get). A real deployment gates
// those behind basicAuth (Authorization: Basic) AND CSRF (csrf-sync: an
// X-CSRF-Token header bound to a session cookie, required on writes). A
// spawned child has no inbound request to forward, so it must authenticate on
// its own — otherwise the very first GET /experiments/:id 401s and the
// synthesis run dies code=1 before writing any agent (the "synthesis never
// concludes / personas never get agents" bug). The spawning plugin passes the
// basicAuth creds in USER_PERSONAS_ST_BASIC_AUTH ("user:pass"); we attach the
// Basic header to every call and lazily run the GET /csrf-token handshake to
// obtain a token + session cookie for write methods. Unauthenticated ST (e.g.
// the st-debug instance with --disableCsrf and no basicAuth) leaves the env
// unset and both layers are no-ops.
const _basicAuthCreds = process.env.USER_PERSONAS_ST_BASIC_AUTH || '';
const _basicAuthHeader = _basicAuthCreds
    ? 'Basic ' + Buffer.from(_basicAuthCreds).toString('base64')
    : '';
let _csrfCache = null;   // { token, cookie } once handshaked (or sentinel)

function _isWriteMethod(method) {
    return !['GET', 'HEAD', 'OPTIONS'].includes(String(method || 'GET').toUpperCase());
}

async function _ensureCsrf() {
    if (_csrfCache) return _csrfCache;
    try {
        const headers = {};
        if (_basicAuthHeader) headers.Authorization = _basicAuthHeader;
        const r = await fetch(`${ST}/csrf-token`, { headers });
        if (!r.ok) { _csrfCache = { token: '', cookie: '' }; return _csrfCache; }
        const setCookies = typeof r.headers.getSetCookie === 'function'
            ? r.headers.getSetCookie()
            : [r.headers.get('set-cookie')].filter(Boolean);
        const cookie = setCookies.map(c => c.split(';')[0]).join('; ');
        let token = '';
        try { token = (await r.json()).token || ''; } catch { /* no token field */ }
        _csrfCache = { token, cookie };
    } catch {
        // CSRF endpoint unreachable (unauthenticated dev ST) — sentinel so we
        // don't re-handshake on every write.
        _csrfCache = { token: '', cookie: '' };
    }
    return _csrfCache;
}

async function _authedHeaders(method) {
    const headers = { 'Content-Type': 'application/json' };
    if (_basicAuthHeader) headers.Authorization = _basicAuthHeader;
    if (_isWriteMethod(method)) {
        const { token, cookie } = await _ensureCsrf();
        if (token) headers['X-CSRF-Token'] = token;
        if (cookie) headers.Cookie = cookie;
    }
    return headers;
}

// ── http / bridge ────────────────────────────────────────────────────

// http(): fail-fast HTTP. Throws on any non-2xx. Use for control-plane
// writes (POST /agents, POST /experiments, DELETE /experiments/:id) where
// idempotency is the caller's job and a 4xx/5xx genuinely means the call
// did not succeed.
//
// For STOCHASTIC sampling endpoints (anything that maps to a model
// inference behind ST's configured provider), use httpRetrying() — these are inherently
// allowed to fail in expectation and the right response to a transient
// 5xx is to sample again with a fresh seed, not to crash the loop.
export async function http(method, url, body) {
    const r = await fetch(url, {
        method,
        headers: await _authedHeaders(method),
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

// httpRetrying(): the K-shot consumer pattern for stochastic backends.
// The bridge serves model inference (judges, signature extraction,
// drafter calls) that is ~90-95% reliable per call — model truncations,
// JSON shape misses, stream-lifecycle reaps, transient backend pressure.
// A single failure is NOT terminal; it's a sample we sample again.
//
// Same shape as http() but retries on 5xx and on network errors with
// exponential backoff (250ms × 2^attempt up to 4s). 4xx errors don't
// retry — those are caller bugs that more samples won't fix. After
// `attempts` total tries the last error is thrown (still surfaces the
// failure; just gives the backend a fair number of chances first).
//
// Default attempts=4 means the failure probability compounds favourably:
// at 90% per-call reliability, 4 attempts ≈ 99.99% terminal success.
// /signature-extract itself does per-group K-shot retries internally,
// so this wrapper handles the OUTER layer (transient transport failure,
// 500s from upstream restart, etc.).
export async function httpRetrying(method, url, body, { attempts = 4 } = {}) {
    let lastError = null;
    for (let i = 0; i < attempts; i++) {
        try {
            return await http(method, url, body);
        } catch (e) {
            const msg = e.message || String(e);
            // Don't retry 4xx — those are caller-input issues. Look for
            // "→ 4" in the error string from http()'s formatter.
            if (/→ 4\d\d:/.test(msg)) throw e;
            lastError = e;
            if (i < attempts - 1) {
                const backoff = Math.min(4000, 250 * Math.pow(2, i));
                console.warn(`[httpRetrying] ${method} ${url} attempt ${i + 1}/${attempts} failed: ${msg.slice(0, 200)} — sleeping ${backoff}ms`);
                await new Promise(r => setTimeout(r, backoff));
            }
        }
    }
    throw lastError;
}

export async function bridgeCall(messages, { max_tokens = null, temperature = 1.0, seed = null } = {}) {
    void temperature;
    const body = { messages };
    if (max_tokens) body.max_tokens = max_tokens;
    if (seed != null) body.seed = seed;  // OpenAI-compatible field, threads through ST to the provider when supported.
    const r = await http('POST', `${PLUGIN}/llm-call`, body);
    return (r.text || '').trim();
}

// ── batch saturation ─────────────────────────────────────────────────
// The local engine decodes through a fixed-width kernel (B=8 streams/step)
// behind an M=64 session cap. Running fewer than B concurrent calls wastes
// the kernel; an UNBOUNDED Promise.all (one call per turn/bio, dozens at
// once) oversubscribes it. Both are the same "guessed a concurrency number"
// bug tools/batch_scaler.py kills — this is its JS twin (ported from
// tools/user-agent-harness/harness_lib.mjs, adapted: this copy routes
// inference via PLUGIN /llm-call so it has no BRIDGE endpoint; the width
// comes from GEMMA_KERNEL_BATCH env, else the optional bridge-diagnostics
// /health probe (USER_PERSONAS_BRIDGE_DIAGNOSTICS_URL, exported by the
// spawning plugin when configured), else the engine defaults. When ST's
// provider is NOT the local bridge the defaults are still a sane bound —
// the point is never to fire 50 calls at once).
let _batchShape = null;
async function _resolveBatchShape() {
    if (_batchShape) return _batchShape;
    let kb = 8, ms = 64, source = 'default';
    const env = process.env.GEMMA_KERNEL_BATCH;
    const diagUrl = (process.env.USER_PERSONAS_BRIDGE_DIAGNOSTICS_URL || '').replace(/\/+$/, '');
    if (env && Number(env) > 0) { kb = Number(env); source = 'env'; }
    else if (diagUrl) {
        try {
            const r = await fetch(`${diagUrl}/health`);
            const caps = (r.ok ? (await r.json())?.capabilities : null) || {};
            if (Number(caps.kernel_batch) > 0) { kb = Number(caps.kernel_batch); source = 'health'; }
            if (Number(caps.max_sessions) > 0) ms = Number(caps.max_sessions);
        } catch { /* unreachable diagnostics → keep defaults */ }
    }
    ms = Math.max(ms, kb);
    _batchShape = { kernelBatch: kb, maxSessions: ms, source };
    process.stderr.write(`[batch_scaler] saturating at kernel_batch=${kb} `
        + `(max_sessions=${ms}, source=${source})\n`);
    return _batchShape;
}

export async function kernelBatch() { return (await _resolveBatchShape()).kernelBatch; }
export async function maxSessions() { return (await _resolveBatchShape()).maxSessions; }

// How many concurrent calls to run for nItems of work: clamp(kernelBatch*fill,
// 1, maxSessions), then down to nItems. The ONE place JS clients get a
// concurrency number.
export async function targetWorkers(nItems = null, { fill = 1 } = {}) {
    const s = await _resolveBatchShape();
    let w = Math.min(s.kernelBatch * Math.max(1, fill), s.maxSessions);
    if (nItems != null) w = Math.min(w, Math.max(1, nItems));
    return Math.max(1, w);
}

// Drop-in replacement for `Promise.all(items.map(fn))` that bounds concurrency
// to the engine's kernel width instead of firing all at once. Preserves input
// order in the result array; fn is (item, index) => Promise. Rejections
// propagate (like Promise.all) — wrap fn in try/catch for per-item capture.
export async function saturatedMap(items, fn, { fill = 1 } = {}) {
    const arr = [...items];
    const out = new Array(arr.length);
    const width = await targetWorkers(arr.length, { fill });
    let next = 0;
    async function worker() {
        while (true) {
            const i = next++;
            if (i >= arr.length) return;
            out[i] = await fn(arr[i], i);
        }
    }
    await Promise.all(Array.from({ length: width }, worker));
    return out;
}

// ── persistence ──────────────────────────────────────────────────────

// Fetch the current axis card set from the plugin. Axes used to be
// inlined as JS source in axis_registry.mjs; they now live as cards
// under plugins/user-personas/axes/ — same storage paradigm as bios,
// agents, characters, tools. The plugin returns whatever axes are
// currently on disk, including any that synthesis added via POST /axes
// (e.g. axis_splitter's derived axes). One source of truth, durable
// across script runs.
//
// Returns the full axis cards as the plugin stores them:
//   [{ axis_schema, id, name, def, kind, scale_min, scale_max,
//      derived_from, created_at }, ...]
// Optionally a kind filter restricts to bio / agent / either / meta —
// 'either' is included whenever a specific kind is requested.
//
// Callers that need the {name, def} pair for /signature-extract should
// map down themselves: `(await fetchAxes()).map(a => ({name: a.id, def: a.def}))`
export async function fetchAxes(kind = null) {
    const r = await http('GET', `${PLUGIN}/axes`);
    let list = (r.axes || []);
    if (kind) list = list.filter(a => a.kind === kind || a.kind === 'either');
    return list;
}

// Fetch one experiment-spec card by id. Returns the full card as
// validated/stored by the plugin (bios, agent_targets, axis lists,
// counterparty, loop_control). Orchestration scripts use this to
// configure themselves from disk rather than inlining the tetrad
// (or whatever N-tuple their experiment defines) as JS source.
export async function fetchExperiment(id) {
    return await http('GET', `${PLUGIN}/experiments/${encodeURIComponent(id)}`);
}

// List every experiment-spec card. Used by the "Fixed-point iteration"
// client tab to populate its picker.
export async function fetchExperiments() {
    const r = await http('GET', `${PLUGIN}/experiments`);
    return r.experiments || [];
}

/**
 * Save a bio. Bios alone aren't signed — the ontological closure says
 * only compositions (bio+agent) carry signatures. This helper just
 * persists the prose + name + system_prompt; the signature lands on
 * the agent that gets designed for this bio.
 *
 * F16 provenance: callers pass `provenance` (optional) describing how
 * this bio came to exist. The plugin's POST /personas/:id validates
 * the kind and persists it onto the player PNG card so views (suggester
 * filter, corpus dashboard) can later decide what to surface vs hide
 * without scanning fragile filename patterns.
 */
export async function saveBio({ canonical_key, name, prose, provenance, signature }) {
    // Optional `signature`: the bio's measured behavioral signature on
    // bio_axes (judged across user turns of the converged inner runs).
    // When supplied, it's persisted to the bio card so downstream surfaces
    // (corpus dashboard PR, outer_outer's ΔPR target picker, suggester's
    // L2 distance, axis registry's scored-on counts) can read it without
    // having to re-judge from the trajectory. Earlier saveBio omitted it,
    // which left bio cards unsigned and made ΔPR-driven target picking
    // fall back to first-random-candidate.
    const system_prompt = `You are ${name}. ${prose}`;
    const body = { name, bio: prose, system_prompt };
    if (provenance) body.provenance = provenance;
    if (signature) body.signature = signature;
    await http('POST', `${PLUGIN}/personas/${encodeURIComponent(canonical_key)}`, body);
}

/**
 * Save an agent. Same hard rule + same axis source as saveBio.
 *
 * F16 provenance: callers pass `provenance` (optional) — for the
 * fixed-point loop this is `{kind:'experiment_output', experiment_id,
 * run_id, iter:{outer,inner}}`. The plugin's POST /agents/:id validates
 * and persists.
 */
export async function saveAgent(agent_id, name, agent_text, designed_for_bio_id, provenance) {
    const personasResp = await http('GET', `${PLUGIN}/personas`);
    const bio = (personasResp.personas || []).find(p => p.id === designed_for_bio_id);
    const bioProse = bio?.bio || '';
    const bioVoice = bio?.system_prompt || '';
    const compositionProse =
        `Bio (user-side persona):\n${bioProse}\n\n` +
        `Bio voice clauses:\n${bioVoice}\n\n` +
        `Agent voice clauses (injected at depth 1):\n${agent_text}`;
    // axes omitted: the plugin's /signature-extract defaults to the
    // current axes/*.json card set, which IS the source of truth.
    // Passing axes explicitly would override that — used to be necessary
    // when axes lived in axis_registry.mjs, no longer is. The signature
    // produced uses the same rubrics /yapper-seed will use to extract
    // target signatures at query time, so the candidate and target
    // vectors share the exact same metric space by construction.
    // K-shot consumer pattern: /signature-extract is a stochastic judge
    // call behind the bridge. The endpoint already retries each judge
    // group internally up to MAX_GROUP_RETRIES; this outer wrapper
    // handles the rarer transport-level failures (bridge stream reap,
    // 500 from upstream restart, network blip). See httpRetrying for
    // the policy rationale.
    const sig = await httpRetrying('POST', `${PLUGIN}/signature-extract`, { prose: compositionProse });
    if (!sig || !sig.signature || typeof sig.signature !== 'object') {
        throw new Error(`saveAgent: /signature-extract returned no signature for ${agent_id}`);
    }
    const body = {
        name, agent_text,
        designed_for_bio_id,
        injection_mode: 'authors_note',
        injection_depth: 1,
        signature: sig.signature,
    };
    if (provenance) body.provenance = provenance;
    await http('POST', `${PLUGIN}/agents/${encodeURIComponent(agent_id)}`, body);
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
    const counterpartySystem = [
        cp.system_prompt,
        'Measurement harness turn contract: reply with one short in-character chat turn only. No markdown fences, no code, no dice tables, no analysis, no extended scene writeup. End with at most one direct prompt back to the user.',
    ].filter(Boolean).join('\n\n');
    for (let i = 0; i < n_turns; i++) {
        const pollResp = await http('POST', `${PLUGIN}/poll`, {
            persona_id: bio.canonical_key, agent_id, chat, n_candidates: 1,
            shareable_prefix: true,
        });
        const cand = (pollResp.candidates || [])[0];
        const userText = (cand?.text || cand?.mes || '').trim();
        if (!userText) throw new Error(`empty user turn on turn ${i+1} (bio=${bio.canonical_key})`);
        chat.push({ name: bio.name, is_user: true, mes: userText });
        const cpMessages = [
            { role: 'system', content: counterpartySystem },
            ...chat.map(m => ({ role: m.is_user ? 'user' : 'assistant', content: m.mes })),
        ];
        // No max_tokens cap: the counterparty plays a chat turn ending
        // at EOS / chat-template <turn|> stop-sequence. A 200-token cap
        // here used to truncate mid-utterance, leaving the next prefill
        // looking at an unfinished assistant turn — off-manifold.
        const cpText = await bridgeCall(cpMessages);
        chat.push({ name: cp.name, is_user: false, mes: cpText.trim() });
    }
    return chat;
}

export function userTurns(chat) {
    return chat.filter(m => m.is_user).map(m => m.mes);
}

// designCheapAgent: produces a neutral "be vividly yourself" agent overlay
// for a bio in a single bridge call. Used by explore_corpus.mjs (outer-
// outer ΔPR-driven corpus expansion) and cluster_disambiguator.mjs as the
// quick "give me an agent so I can elicit chat from this bio" path. It is
// NOT a substitute for the fixed-point iterative agent design in
// lock_in_iterative.mjs — those two paths serve different layers of the
// pipeline. Earlier in this session this function was deleted under the
// "no single-iteration paths" rule; that was a mistaken read — explore_corpus
// and cluster_disambiguator depend on it as their elicitation vehicle.
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
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
}

// ── judge: single axis on one turn ───────────────────────────────────

export async function judgeOnAxis(turn, axisName, axisDef) {
    const sys =
        'You are a behavioural-axis judge. You read ONE user-side chat ' +
        'turn and score it on the named axis (number 1-5). The turn is ' +
        'the only ground truth — do not infer from anything else. Be ' +
        'willing to score 1 (absence) when the turn genuinely shows no ' +
        'expression of the axis. Output ONLY the axis line below as ' +
        '"axis_name: <number 1-5>". No preamble, no commentary.';
    const usr =
        `## Axis (1-5)\n\n- **${axisName}** — ${axisDef}\n\n` +
        '## Turn to score\n\n> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
        `## Emit\n\n${axisName}: ?\n`;
    // No max_tokens cap: prompt shape ("Output ONLY the axis line")
    // gives a deterministic single-line emit + EOS. A 50-token cap
    // would truncate any judge that emitted any preamble before the
    // line — silent failure rather than visible parse error.
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const escName = axisName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const re = new RegExp('^\\s*[-*]?\\s*\\**["\']?' + escName +
                          '["\']?\\**\\s*[:=]\\s*([+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+))', 'im');
    const m = raw.match(re);
    return m ? normalizeLikertNumber(m[1]) : null;
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
        '"axis_name: N" where N is a number from 1 to 5 per the rubric.';
    const usr =
        '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
        '## Turn to score\n\n' +
        '> ' + turn.replace(/\n/g, '\n> ') + '\n\n' +
        '## Emit\n\n' + template + '\n';
    // No max_tokens cap: the rubric + template emit-shape constrains
    // the model to one line per axis with explicit termination. A
    // formula-derived cap was still a cap and could truncate when the
    // model emitted any unexpected preamble — silent half-signature
    // rather than visible parse error.
    const raw = await bridgeCall(
        [{ role: 'system', content: sys }, { role: 'user', content: usr }]);
    const sig = {};
    for (const a of axes) sig[a.name] = null;
    for (const line of raw.split('\n')) {
        for (const a of axes) {
            const escName = a.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const re = new RegExp(
                '^\\s*[-*]?\\s*\\**["\']?' + escName + '["\']?\\**\\s*[:=]\\s*([+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+))',
                'i');
            const m = line.match(re);
            const parsed = m ? normalizeLikertNumber(m[1]) : null;
            if (parsed != null) sig[a.name] = parsed;
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
// Lineage weights from full axis cards: every axis reachable from the
// same root via derived_from.parent links shares ONE unit of metric
// weight (w = 1/|lineage|). Without this, every accepted split (and
// every duplicate registration) multiplies the metric weight of its
// latent direction by the number of registered descendants — the
// comparison metric then measures registry bookkeeping (which axis got
// split/re-split most) instead of behavior, and eff-dim objectives are
// inflated by phantom near-collinear coordinates. Cycle-safe; axes with
// dangling parents are treated as their own roots.
export function lineageWeightsFromCards(cards) {
    const byId = new Map(cards.map(c => [c.id ?? c.name, c]));
    const rootOf = new Map();
    function rootFor(id, seen = new Set()) {
        if (rootOf.has(id)) return rootOf.get(id);
        if (seen.has(id)) return id;            // cycle guard → treat as root
        seen.add(id);
        const parent = byId.get(id)?.derived_from?.parent;
        const root = (parent && parent !== id && byId.has(parent)) ? rootFor(parent, seen) : id;
        rootOf.set(id, root);
        return root;
    }
    const sizes = new Map();
    for (const id of byId.keys()) {
        const r = rootFor(id);
        sizes.set(r, (sizes.get(r) || 0) + 1);
    }
    const weights = {};
    for (const id of byId.keys()) weights[id] = 1 / sizes.get(rootOf.get(id));
    return weights;
}

export function effDimParticipationRatio(sigsByBio, axisNames, weights = null) {
    const bios = Object.keys(sigsByBio);
    if (bios.length < 2) return { effDim: null, perAxisVar: {}, n: bios.length, note: 'need ≥2 bios' };
    const perAxisVar = {};
    for (const a of axisNames) {
        const vals = bios.map(b => sigsByBio[b][a]).filter(Number.isFinite);
        // Optional lineage weighting (lineageWeightsFromCards): variance
        // along duplicated/descendant axes is down-scaled so one latent
        // direction never counts more than once however many times the
        // splitter has factored it.
        perAxisVar[a] = (vals.length >= 2 ? meanStd(vals).var : 0) * (weights?.[a] ?? 1);
    }
    const totalVar = Object.values(perAxisVar).reduce((a, b) => a + b, 0);
    if (totalVar <= 0) return { effDim: null, perAxisVar, n: bios.length, note: 'zero total variance' };
    const p = Object.fromEntries(Object.entries(perAxisVar).map(([k, v]) => [k, v / totalVar]));
    const effDim = 1 / Object.values(p).reduce((s, pi) => s + pi * pi, 0);
    return { effDim, perAxisVar, normalized: p, totalVar, n: bios.length };
}
