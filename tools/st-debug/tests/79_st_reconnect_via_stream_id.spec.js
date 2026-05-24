// Spec 79 — ST openai-CUSTOM mid-stream reconnect via X-Stream-Id.
//
// Validates the wiring added to `src/endpoints/backends/chat-completions.js`
// that auto-reconnects via `GET <custom_url>/v1/streams/{id}/sse` when the
// upstream SSE socket drops mid-stream and the upstream emitted an
// `X-Stream-Id` header on the initial POST (the metal-microbench bridge's
// reconnect contract).
//
// Strategy (no browser): we stand up a small Node HTTP "flaky upstream"
// on a free port. It mimics the bridge contract:
//   - POST /v1/chat/completions stream=true: respond 200 with X-Stream-Id
//     header, emit a few SSE chunks, then RST the socket without sending
//     `data: [DONE]`. (Simulates a mid-stream drop.)
//   - GET /v1/streams/{stream_id}/sse: respond 200 with the same
//     X-Stream-Id, emit the REMAINING tokens plus `data: [DONE]`.
// We then drive ST's `/api/backends/chat-completions/generate` proxy with
// `chat_completion_source=custom` + `custom_url=<our flaky upstream>` and
// assert ST returns a stitched SSE stream containing both pre-drop and
// post-reconnect chunks, ending with `[DONE]`.
//
// This proves the Node-side reconnect wiring works end-to-end without
// needing a real bridge. A separate validation against the real bridge
// is in `/Users/mdot/metal-microbench/server/test_reconnect.py`.
//
// Falls back to "fixture not viable" skip if ST is not reachable on :8002.

import { test, expect } from '@playwright/test';
import http from 'node:http';
import { setTimeout as wait } from 'node:timers/promises';

const ST_BASE = process.env.ST_BASE || 'http://127.0.0.1:8002';

// SSE chunk template (OAI chat.completion.chunk).
function sseChunk(text, finish = null) {
    const payload = {
        id: 'chatcmpl-test-stub',
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now() / 1000),
        model: 'flaky-stub',
        choices: [{ index: 0, delta: { content: text }, finish_reason: finish }],
    };
    return `data: ${JSON.stringify(payload)}\n\n`;
}

// Build the flaky-upstream server. Returns { url, close, postCount,
// reconnectCount } so the test can drive + introspect it.
async function startFlakyUpstream() {
    const STREAM_ID = '424242';
    const PRE_DROP_CHUNKS = ['hello ', 'from ', 'flaky '];
    const POST_RECONNECT_CHUNKS = ['upstream ', 'and ', 'finally '];

    let postCount = 0;
    let reconnectCount = 0;

    const server = http.createServer(async (req, res) => {
        // POST /v1/chat/completions — initial streaming request.
        if (req.method === 'POST' && req.url === '/v1/chat/completions') {
            postCount += 1;
            // Read + discard the request body.
            for await (const _chunk of req) { /* noop */ }
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Stream-Id': STREAM_ID,
            });
            // Emit a few chunks, then deliberately rip the socket.
            for (const t of PRE_DROP_CHUNKS) {
                res.write(sseChunk(t));
                await wait(20);
            }
            // Force a TCP RST without HTTP-clean close. node-fetch will
            // see this as a premature end of the body Readable.
            req.socket.destroy();
            return;
        }

        // GET /v1/streams/{id}/sse — reconnect endpoint.
        const m = req.url && req.url.match(/^\/v1\/streams\/([^\/]+)\/sse$/);
        if (req.method === 'GET' && m) {
            const sid = decodeURIComponent(m[1]);
            if (sid !== STREAM_ID) {
                res.writeHead(404, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: 'unknown stream_id' }));
                return;
            }
            reconnectCount += 1;
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'X-Stream-Id': STREAM_ID,
            });
            for (const t of POST_RECONNECT_CHUNKS) {
                res.write(sseChunk(t));
                await wait(20);
            }
            // Terminal frame with finish_reason + DONE sentinel.
            res.write(sseChunk('', 'stop'));
            res.write('data: [DONE]\n\n');
            res.end();
            return;
        }

        // Everything else: 404.
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
        getStats: () => ({ postCount, reconnectCount }),
    };
}

// Issue a streaming request through ST's CUSTOM-source generate proxy.
// Returns the aggregated SSE body text plus the response status.
async function streamThroughST(customUrl) {
    const body = JSON.stringify({
        chat_completion_source: 'custom',
        custom_url: customUrl,
        // Required by the proxy even though we never read the model.
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

test.describe('ST openai-CUSTOM reconnect via X-Stream-Id (spec 79)', () => {
    test.setTimeout(60_000);

    test('mid-stream upstream RST → ST reconnects and surfaces stitched SSE', async () => {
        // Skip if ST isn't reachable.
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
            console.log(`ST response body (${text.length} chars): ${text.slice(0, 800)}`);

            // ST must reach the upstream at least once for the POST.
            const stats = upstream.getStats();
            console.log(`upstream stats: ${JSON.stringify(stats)}`);
            expect(stats.postCount, 'flaky upstream saw initial POST').toBeGreaterThanOrEqual(1);

            // The load-bearing assertion: ST hit the reconnect endpoint
            // after the upstream RST'd mid-stream.
            expect(stats.reconnectCount,
                'ST issued GET /v1/streams/{id}/sse after mid-stream drop'
            ).toBeGreaterThanOrEqual(1);

            // The body must contain content from BOTH halves of the
            // stream (pre-drop + post-reconnect), plus [DONE].
            expect(text, 'response body contains pre-drop content').toMatch(/hello/);
            expect(text, 'response body contains post-reconnect content').toMatch(/finally/);
            expect(text, 'response body terminates with [DONE]').toContain('data: [DONE]');
        } finally {
            await upstream.close();
        }
    });

    test('upstream that does NOT emit X-Stream-Id → no reconnect attempt', async () => {
        // Sanity: a non-bridge upstream that simply drops mid-stream
        // without emitting X-Stream-Id should NOT trigger any reconnect
        // attempts. We don't care whether ST cleanly ends the stream
        // (plain forwardFetchResponse may hang on upstream RST without
        // DONE — pre-existing behavior, out of scope for this spec).
        // We only assert that the reconnect endpoint was never hit.
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
            // Fire the request with an AbortController so we don't hang
            // on the post-drop unfinished stream.
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
            // Let the upstream POST + RST cycle complete, plus a generous
            // window during which a (mis)wired reconnect would have fired.
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
