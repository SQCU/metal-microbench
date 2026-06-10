// Format-agnostic schema rendering + parsing + agent-in-the-loop fixup.
//
// A typed schema is the abstract thing we want from the model. Rendering
// it as a particular output format (JSON / YAML / prose-lines / TOML) is
// a CHOICE we can ablate. Different formats produce different per-call
// reliability rates from the same model — closing `}` tokens are harder
// for some models, indentation rules are harder for others, etc. The
// only way to know which is the bottleneck is to swap the format and
// measure.
//
// Two reliability layers live here:
//
//   1. Format-agnostic schema with pluggable render+parse per format.
//      Same schema, different surface syntax, parsers all tolerant
//      (handles preamble, decoration, partial emission).
//
//   2. Agent-in-the-loop fixup retry. When the parsed output is
//      incomplete or unparseable, we append the failed model output to
//      the conversation and ask the model to fix it — same context,
//      same prefix-maxx prefix, just a small fixup turn. This is
//      what a human reviewer does: "your column 3 is malformed,
//      please re-emit." It's separate from fresh-seed retry, which
//      gives the model a clean slate but throws away the in-context
//      hint about what went wrong.
//
// The /signature-extract server endpoint already converged on prose-
// lines (after the #169 judge-prompt A/B). This module lets us:
//   (a) run an ablation to confirm/refute that choice quantitatively
//   (b) hand client harnesses a tested fixup-retry policy for any
//       format they end up needing
//   (c) attribute future failures correctly — "the model can't do
//       this task at any format" vs "the model can do this task in
//       YAML but not in JSON"

// ── Format renderers ────────────────────────────────────────────────────

export const FORMATS = ['prose-lines', 'json', 'yaml', 'toml'];

/**
 * Render the OUTPUT INSTRUCTION the model receives, telling it what
 * shape we want back. The instruction is appended to the prompt suffix
 * (after the rubric + the text-to-score). Prefix-maxx stays intact:
 * the rubric + text live in a shared prefix, the format-specific
 * instruction is the per-call suffix.
 *
 * `axisNames` is an array of strings. The schema is implicitly:
 * every axis is a number 1..5.
 */
export function renderInstruction(format, axisNames) {
    switch (format) {
        case 'prose-lines': {
            const template = axisNames.map(a => `${a}: ?`).join('\n');
            return (
                'Emit ONE LINE per axis as "axis_name: N" where N is a number from 1 to 5. ' +
                'No preamble, no commentary, no fences.\n\n' +
                '## Emit\n\n' + template
            );
        }
        case 'json': {
            const obj = '{ ' + axisNames.map(a => `"${a}": <number 1-5>`).join(', ') + ' }';
            return (
                'Emit ONLY a JSON object with exactly these keys (no preamble, no markdown fences):\n' +
                obj
            );
        }
        case 'yaml': {
            const lines = axisNames.map(a => `${a}: <number 1-5>`).join('\n');
            return (
                'Emit ONLY YAML key:value pairs, one per line, no fences:\n\n' +
                lines
            );
        }
        case 'toml': {
            const lines = axisNames.map(a => `${a} = <number 1-5>`).join('\n');
            return (
                'Emit ONLY TOML key = value pairs, one per line, no fences, no section headers:\n\n' +
                lines
            );
        }
        default:
            throw new Error(`renderInstruction: unknown format ${JSON.stringify(format)}`);
    }
}

// ── Tolerant parsers ────────────────────────────────────────────────────
//
// All parsers obey the same contract:
//   parse(axisNames, raw) → { scores, missing, error? }
// scores is a partial object axis → number 1-5; missing lists axis names
// that couldn't be extracted; error is set only on whole-output parse
// failure (e.g. JSON.parse throws).
//
// All parsers tolerate:
//   - preamble before the actual data
//   - markdown fences (```json ... ```)
//   - extra prose after the data
//   - bullet/asterisk/quote decoration on keys
//   - case-insensitive key match
//   - missing keys (return them in `missing`, don't throw)

const KEYED_LINE_FORMATS = new Set(['prose-lines', 'yaml', 'toml']);

export function parse(format, axisNames, raw) {
    if (typeof raw !== 'string') return { scores: {}, missing: [...axisNames], error: 'non-string output' };
    const text = stripFences(raw);

    if (format === 'json') return parseJson(axisNames, text);
    if (KEYED_LINE_FORMATS.has(format)) {
        // TOML uses `=`, the others use `:`. JSON has its own parser.
        const sep = format === 'toml' ? '=' : ':';
        return parseKeyedLines(axisNames, text, sep);
    }
    return { scores: {}, missing: [...axisNames], error: `parse: unknown format ${JSON.stringify(format)}` };
}

function stripFences(raw) {
    // ```json ... ```  or  ```yaml ... ```  or just ``` ... ```
    const m = raw.match(/```[a-zA-Z]*\n?([\s\S]*?)```/);
    if (m) return m[1];
    return raw;
}

function parseJson(axisNames, text) {
    const scores = {};
    const missing = [];
    // 1) Try a true JSON.parse of the largest brace-balanced span.
    const obj = extractBraceObject(text);
    if (obj) {
        for (const a of axisNames) {
            const v = Number(obj[a]);
            if (Number.isFinite(v)) scores[a] = clamp(v);
            else missing.push(a);
        }
        return { scores, missing };
    }
    // 2) JSON failed — fall back to keyed-line parse on `"key": N`
    //    pairs. This salvages truncated JSON (the most common failure
    //    mode: model emits keys+values but never closes the brace).
    //    The keyed-line regex matches the same key:value pairs that
    //    YAML/prose-lines do, just with quotes that we tolerate.
    return parseKeyedLines(axisNames, text, ':', { error: 'JSON parse fell back to line scan' });
}

function extractBraceObject(text) {
    // Find first `{` and walk to its matching `}` with depth counting.
    // Falls through to null if no closing brace exists.
    const start = text.indexOf('{');
    if (start === -1) return null;
    let depth = 0;
    for (let i = start; i < text.length; i++) {
        if (text[i] === '{') depth++;
        else if (text[i] === '}') {
            depth--;
            if (depth === 0) {
                const span = text.slice(start, i + 1);
                try { return JSON.parse(span); }
                catch (_) { return null; }
            }
        }
    }
    return null;  // unterminated
}

function parseKeyedLines(axisNames, text, sep, extra = {}) {
    const scores = {};
    const missing = [];
    const lines = text.split('\n');
    for (const a of axisNames) {
        const esc = a.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        // ^ ... `[-*]?` bullet ... `**` md emphasis ... `"` or `'` quote
        // ... key ... close quote/emphasis ... sep ... value
        // sep is interpolated literal; sanitize the only allowed chars (`:` or `=`).
        const sepEsc = sep === '=' ? '=' : ':';
        const re = new RegExp(
            '^\\s*[-*]?\\s*\\**["\']?' + esc + '["\']?\\**\\s*' + sepEsc + '\\s*"?([+-]?(?:\\d+(?:\\.\\d+)?|\\.\\d+))"?',
            'im');
        let v = null;
        for (const line of lines) {
            const m = line.match(re);
            if (m) { v = Number(m[1]); break; }
        }
        if (v == null) missing.push(a);
        else scores[a] = clamp(v);
    }
    return { scores, missing, ...extra };
}

function clamp(v) {
    const n = Number(v);
    if (!Number.isFinite(n)) return null;
    return Math.max(1, Math.min(5, Number(n.toPrecision(3))));
}

// ── Agent-in-the-loop fixup retry ───────────────────────────────────────
//
// Same prefix (rubric + prose + format instruction), but if parsing
// fails or returns partial, we APPEND a fixup turn to the conversation
// and let the model see its own broken output. The model gets actual
// information about what went wrong, in-context. This is different
// from `fresh-seed` retry (which just changes RNG seed and re-rolls)
// because it's sample-efficient: the model isn't blind to the issue.
//
// Trade-off vs fresh-seed: feedback retries are more expensive per call
// (longer prompt) but converge in fewer attempts. The ablation script
// measures the trade-off directly.

export function feedbackTurnFor(format, missing, parseError) {
    const parts = [];
    if (parseError) {
        parts.push(`Your previous output didn't parse as ${format} (${parseError}).`);
    }
    if (missing && missing.length > 0) {
        parts.push(`These axes were not parseable: ${missing.join(', ')}.`);
    }
    if (parts.length === 0) return null;
    parts.push(`Please re-emit the COMPLETE output for ALL axes in the EXACT format specified above. All values are numbers 1-5. No preamble.`);
    return parts.join(' ');
}

/**
 * judgeWithFeedback({ axes, prose, format, bridgeCall, maxAttempts, mode })
 *
 *   axes: [{ name, def }, ...]
 *   prose: text to score
 *   format: 'prose-lines' | 'json' | 'yaml' | 'toml'
 *   bridgeCall: async (messages) → text   (the harness's bridgeCall)
 *   maxAttempts: int, default 3
 *   mode: 'feedback' | 'fresh-seed' | 'one-shot'
 *       'feedback'   = append fixup turn to conversation on each retry
 *       'fresh-seed' = new conversation with bumped seed on each retry
 *       'one-shot'   = no retry; whatever the first call returned is final
 *
 * Returns:
 *   { scores, missing, attempts, format, mode, raws[] }
 */
export async function judgeWithFeedback({
    axes, prose, format, bridgeCall,
    maxAttempts = 3, mode = 'feedback',
    seedBase = 50000,
}) {
    if (!Array.isArray(axes) || axes.length === 0) throw new Error('judgeWithFeedback: axes required');
    if (!FORMATS.includes(format)) throw new Error(`judgeWithFeedback: unknown format ${format}`);
    if (!['feedback', 'fresh-seed', 'one-shot'].includes(mode)) {
        throw new Error(`judgeWithFeedback: unknown mode ${mode}`);
    }
    if (mode === 'one-shot') maxAttempts = 1;

    const axisNames = axes.map(a => a.name);
    const rubric = axes.map(a => `- **${a.name}** — ${a.def}`).join('\n');
    const instruction = renderInstruction(format, axisNames);

    const sys =
        `You are a behavioural-axis scorer. ${instruction}\n` +
        'Score the text below on each axis (number 1-5).';
    const userPrimary =
        '## Axes (each 1-5)\n\n' + rubric + '\n\n' +
        '## Text to score\n\n' + prose + '\n\n' +
        instruction;

    let lastResult = { scores: {}, missing: [...axisNames] };
    const raws = [];

    if (mode === 'feedback' || mode === 'one-shot') {
        // Single conversation, append turns on each retry.
        const messages = [
            { role: 'system', content: sys },
            { role: 'user', content: userPrimary },
        ];
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const text = await bridgeCall(messages, { seed: seedBase + attempt });
            raws.push(text);
            const result = parse(format, axisNames, text);
            lastResult = result;
            const complete = Object.keys(result.scores).length === axisNames.length;
            if (complete) return { ...result, attempts: attempt, format, mode, raws };
            if (attempt === maxAttempts) break;
            const fb = feedbackTurnFor(format, result.missing, result.error);
            if (!fb) break;  // unreachable since we know it's incomplete, but guard anyway
            messages.push({ role: 'assistant', content: text });
            messages.push({ role: 'user', content: fb });
        }
    } else {
        // fresh-seed: independent calls, no shared context.
        for (let attempt = 1; attempt <= maxAttempts; attempt++) {
            const messages = [
                { role: 'system', content: sys },
                { role: 'user', content: userPrimary },
            ];
            const text = await bridgeCall(messages, { seed: seedBase + attempt * 1000 });
            raws.push(text);
            const result = parse(format, axisNames, text);
            // Pick best-so-far (most-axes-extracted).
            if (Object.keys(result.scores).length > Object.keys(lastResult.scores).length) {
                lastResult = result;
            }
            if (Object.keys(result.scores).length === axisNames.length) {
                return { ...result, attempts: attempt, format, mode, raws };
            }
        }
    }

    return { ...lastResult, attempts: maxAttempts, format, mode, raws };
}
