// Spec 79 — ST openai-CUSTOM mid-stream reconnect via X-Stream-Id +
// append-log offset replay (2026-05-23 refactor).
//
// Validates the wiring added to `src/endpoints/backends/chat-completions.js`
// that auto-reconnects via `GET <custom_url>/v1/streams/{id}/sse?since={N}`
// when the upstream SSE socket drops mid-stream and the upstream emitted
// an `X-Stream-Id` header on the initial POST (the metal-microbench
// bridge's reconnect contract).
//
// 2026-05-23 APPEND-LOG REFACTOR: the bridge now keeps a per-stream
// append-only log of every StreamUpdate and prefixes each SSE event
// with a `: offset=N\n` comment line. This spec asserts that:
//   1. ST's reconnect request includes `?since={lastSeenOffset+1}`
//      based on the offsets it saw pre-disconnect.
//   2. The stitched body contains NO gap in token content — every
//      pre-disconnect chunk AND every post-reconnect chunk is delivered.
//
// Strategy (no browser): we stand up a small Node HTTP "flaky upstream"
// on a free port that mimics the bridge contract. It emits SSE events
// with `: offset=N\n` comment prefixes, drops mid-stream, and then on
// reconnect honors the `?since=N` query parameter by emitting the
// remaining tokens starting at the requested offset.
//
// Falls back to "fixture not viable" skip if ST is not reachable on :8002.

import { test, expect } from '@playwright/test';
import http from 'node:http';
import { setTimeout as wait } from 'node:timers/promises';
import { URL } from 'node:url';

const ST_BASE = process.env.ST_BASE || 'http://127.0.0.1:8002';

// SSE chunk template (OAI chat.completion.chunk) with optional offset prefix.
function sseChunk(text, finish = null, offset = null) {
    const payload = {
        id: 'chatcmpl-test-stub',
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: 'flaky-stub',
        choices: [{ index: 0, delta: { content: text }, finish_reason: finish }],
    };
    const data = `data: ${JSON.stringify(payload)}\n\n`;
    return offset !== null ? `: offset=${offset}\n${data}` : data;
}

// Build the flaky-upstream server.
async function startFlakyUpstream() {
    const STREAM_ID = '424242';
    // Full token sequence; emitted in order across the initial POST +
    // any reconnect. Offsets are 0..N-1 in the canonical log.
    const FULL_TOKENS = ['hello ', 'from ', 'flaky ', 'upstream ', 'and ', 'finally '];
    const PRE_DROP_COUNT = 3; // first 3 (offsets 0,1,2) go pre-drop
    // The POST_RECONNECT slice = FULL_TOKENS.slice(PRE_DROP_COUNT)
    // = ['upstream ', 'and ', 'finally '] at offsets 3, 4, 5.

    let postCount = 0;
    let reconnectCount = 0;
    let lastReconnectSince = null;

    const server = http.createServer(async (req, res) => {
        // POST /v1/chat/completions — initial streaming request.
        if (req.method === 'POST' && req.url === '/v1/chat/completions') {
            postCount += 1;
            for await (const _chunk of req) { /* noop */ }
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Stream-Id': STREAM_ID,
            });
            // Emit PRE_DROP_COUNT chunks WITH offsets, then RST the socket.
            for (let i = 0; i < PRE_DROP_COUNT; i++) {
                res.write(sseChunk(FULL_TOKENS[i], null, i));
                await wait(20);
            }
            // Force a TCP RST so node-fetch sees a premature body end.
            req.socket.destroy();
            return;
        }

        // GET /v1/streams/{id}/sse?since={N} — reconnect endpoint.
        const parsed = new URL(req.url, `http://${req.headers.host}`);
        const m = parsed.pathname.match(/^\/v1\/streams\/([^\/]+)\/sse$/);
        if (req.method === 'GET' && m) {
            const sid = decodeURIComponent(m[1]);
            if (sid !== STREAM_ID) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'unknown stream_id' }));
                return;
            }
            reconnectCount += 1;
            const sinceParam = parseInt(parsed.searchParams.get('since') || '0', 10);
            lastReconnectSince = sinceParam;
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Stream-Id': STREAM_ID,
            });
            // Replay starting at sinceParam. Honors the append-log
            // contract: send every offset >= sinceParam.
            for (let i = sinceParam; i < FULL_TOKENS.length; i++) {
                res.write(sseChunk(FULL_TOKENS[i], null, i));
                await wait(20);
            }
            // Terminal frame + DONE. Same offset as the last token,
            // since the bridge convention is "one offset per StreamUpdate
            // can produce multiple SSE frames".
            const terminalOffset = FULL_TOKENS.length - 1;
            res.write(sseChunk('', 'stop', terminalOffset));
            res.write(`: offset=${terminalOffset}\ndata: [DONE]\n\n`);
            res.end();
            return;
        }

        res.writeHead(404).end();
    });

    await new Promise((resolve, reject) => {
        server.on('error', reject);
        server.listen(0, '127.0.0.1', resolve);
    });
    const { port } = server.address();
    return {
        url: `http://127.0.0.1:${port}/v1`,
        close: () => new Promise((resolve) => server.close(() => resolve())),
        getStats: () => ({ postCount, reconnectCount, lastReconnectSince }),
    };
}

// Issue a streaming request through ST's CUSTOM-source generate proxy.
async function streamThroughST(customUrl) {
    const body = JSON.stringify({
        chat_completion_source: 'custom',
        custom_url: customUrl,
        model: 'flaky-stub',
        stream: true,
        messages: [{ role: 'user', content: 'hi' }],
        max_tokens: 32,
        temperature: 0.7,
    });
    const resp = await fetch(`${ST_BASE}/api/backends/chat-completions/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        text += decoder.decode(value, { stream: true });
    }
    return { status: resp.status, text };
}

test.describe('ST openai-CUSTOM reconnect via X-Stream-Id + append-log (spec 79)', () => {
    test.setTimeout(60_000);

    test('mid-stream upstream RST → ST reconnects with ?since=N and stitched SSE has no gap', async () => {
        let stHealthy = false;
        try {
            const r = await fetch(`${ST_BASE}/`, { method: 'GET' });
            stHealthy = r.status < 500;
        } catch (_) { /* unreachable */ }
        test.skip(!stHealthy, `ST not reachable at ${ST_BASE} — skipping`);

        const upstream = await startFlakyUpstream();
        try {
            const { status, text } = await streamThroughST(upstream.url);
            console.log(`ST response status=${status}`);
            console.log(`ST response body (${text.length} chars):\n${text.slice(0, 1200)}`);

            const stats = upstream.getStats();
            console.log(`upstream stats: ${JSON.stringify(stats)}`);
            expect(stats.postCount, 'flaky upstream saw initial POST').toBeGreaterThanOrEqual(1);

            // Reconnect MUST have been issued.
            expect(stats.reconnectCount,
                'ST issued GET /v1/streams/{id}/sse after mid-stream drop'
            ).toBeGreaterThanOrEqual(1);

            // 2026-05-23 APPEND-LOG ASSERTION: reconnect MUST include
            // ?since=N where N = lastSeenOffset + 1. The flaky upstream
            // emitted offsets 0, 1, 2 pre-drop, so the reconnect must
            // ask for since=3 (= 2 + 1). This is the load-bearing
            // "no mid-stream gap" guarantee.
            expect(stats.lastReconnectSince,
                'ST reconnect request includes ?since={lastSeenOffset+1}'
            ).toBe(3);

            // Stitched body MUST contain EVERY token from the full
            // sequence. This is the "no mid-stream gap" assertion:
            // tokens 'hello from flaky ' pre-drop + 'upstream and
            // finally ' post-reconnect.
            for (const tok of ['hello', 'from', 'flaky', 'upstream', 'and', 'finally']) {
                expect(text, `stitched response contains "${tok}"`).toContain(tok);
            }
            expect(text, 'response body terminates with [DONE]').toContain('data: [DONE]');
        } finally {
            await upstream.close();
        }
    });

    test('upstream that does NOT emit X-Stream-Id → no reconnect attempt', async () => {
        let stHealthy = false;
        try {
            const r = await fetch(`${ST_BASE}/`, { method: 'GET' });
            stHealthy = r.status < 500;
        } catch (_) { /* unreachable */ }
        test.skip(!stHealthy, `ST not reachable at ${ST_BASE} — skipping`);

        let postCount = 0;
        let reconnectCount = 0;
        const server = http.createServer(async (req, res) => {
            if (req.method === 'POST' && req.url === '/v1/chat/completions') {
                postCount += 1;
                for await (const _ of req) { /* noop */ }
                // NOTE: deliberately NO X-Stream-Id header.
                res.writeHead(200, { 'Content-Type': 'text/event-stream' });
                res.write(sseChunk('partial '));
                await wait(30);
                req.socket.destroy();
                return;
            }
            if (req.method === 'GET' && /\/v1\/streams\//.test(req.url)) {
                reconnectCount += 1;
                res.writeHead(404).end();
                return;
            }
            res.writeHead(404).end();
        });
        await new Promise((r) => server.listen(0, '127.0.0.1', r));
        const port = server.address().port;
        try {
            const ac = new AbortController();
            const reqPromise = fetch(`${ST_BASE}/api/backends/chat-completions/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    chat_completion_source: 'custom',
                    custom_url: `http://127.0.0.1:${port}/v1`,
                    model: 'flaky-stub',
                    stream: true,
                    messages: [{ role: 'user', content: 'hi' }],
                    max_tokens: 32,
                    temperature: 0.7,
                }),
                signal: ac.signal,
            });
            await wait(2500);
            ac.abort();
            try { await reqPromise; } catch (_) { /* aborted */ }

            expect(postCount, 'upstream POST reached').toBeGreaterThanOrEqual(1);
            expect(reconnectCount,
                'no reconnect attempt when X-Stream-Id absent'
            ).toBe(0);
        } finally {
            await new Promise((r) => server.close(r));
        }
    });
});
