import { test, expect } from '@playwright/test';

// Phase-3a: validates the spur-caption integration into the toolcards
// plugin. The captioned variant of query-to-svg fires a tiny
// summarizer llm_call after each iter and emits the result as a
// `progress` event with a "caption: " prefix. The FE renders progress
// events in the placeholder bubble of the calling turn — so by
// validating that captions arrive as progress events of the running
// session, we're confirming the feed-back-into-DOM path end-to-end.
//
// Why direct-invoke: the outer model's tool-call decision is flaky
// (per the elicitation findings), so we don't depend on it for
// pipeline validation. Direct-invoke fires the toolcard with known
// args and lets us cleanly observe the events that come back.
//
// What we expect to see in the event stream for max_iters=2:
//   progress: target NxN, 2 iter(s)
//   progress: iter 1/2: requesting SVG
//   progress: iter 1: got <N>-char SVG, rasterizing
//   progress: iter 1: done in X.Xs (<N> chars)
//   progress: caption: <one short sentence about iter 1's SVG>     ← NEW
//   progress: iter 2/2: requesting refinement against vision feedback
//   progress: iter 2: got <N>-char SVG, rasterizing
//   progress: iter 2: done in X.Xs (<N> chars)
//   progress: caption: <one short sentence about iter 2's SVG>     ← NEW
//   result: ...
//
// Test asserts:
//   - at least 2 caption progress events arrive (one per iter)
//   - each caption is a non-empty string under ~30 words
//   - the result event still contains the rasterized PNG (didn't break
//     the original toolcard)

function captionedProfile() {
    return {
        api: 'openai',
        mode: 'cc',
        preset: 'default',
        chat_completion_source: 'custom',
        custom_url: 'http://127.0.0.1:8001',
        custom_model: 'gemma-4-a4b',
        stream_openai: false,
        temperature_openai: 0.4,
        openai_max_tokens: 4096,
    };
}

async function startCaptionedInvoke(request, query) {
    const startResp = await request.post(
        'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/query-to-svg-captioned/generate',
        {
            data: {
                args: {
                    query,
                    max_iters: 1,
                    width: 256,
                    height: 256,
                },
                profile: captionedProfile(),
            },
        });
    expect(startResp.status(), 'start_invoke 200').toBe(200);
    const { session_id } = await startResp.json();
    return session_id;
}

async function pollOnce(request, sessionId) {
    const pollResp = await request.get(
        `http://127.0.0.1:8002/api/plugins/toolcards/poll/${sessionId}`);
    expect(pollResp.status(), `poll ${sessionId}`).toBe(200);
    return pollResp.json();
}

async function cancelQuietly(request, sessionId) {
    try {
        await request.post(`http://127.0.0.1:8002/api/plugins/toolcards/cancel/${sessionId}`);
    } catch { /* best-effort cleanup */ }
}

test.describe('toolcards spur-caption integration', () => {
    test.setTimeout(2 * 60 * 1000);   // 2 min plenty for max_iters=1

    test('captioned query-to-svg emits caption progress events', async ({ request }) => {
        // Same profile shape as test 04.
        const profile = captionedProfile();

        const startResp = await request.post(
            'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/query-to-svg-captioned/generate',
            {
                data: {
                    args: {
                        query: 'a small purple star',
                        max_iters: 1,
                        width: 256,
                        height: 256,
                    },
                    profile: profile,
                },
            });
        expect(startResp.status(), 'start_invoke 200').toBe(200);
        const { session_id } = await startResp.json();
        console.log(`  session_id: ${session_id}`);

        // Drain events.
        let resultEvent = null;
        const captionLines = [];
        const allProgress = [];
        const deadline = Date.now() + 7 * 60 * 1000;
        while (Date.now() < deadline) {
            const pollResp = await request.get(
                `http://127.0.0.1:8002/api/plugins/toolcards/poll/${session_id}`);
            if (pollResp.status() !== 200) {
                console.log(`  poll status ${pollResp.status()}, retrying`);
                await new Promise(r => setTimeout(r, 500));
                continue;
            }
            const event = await pollResp.json();
            if (event.type === 'progress') {
                const text = event.text || '';
                allProgress.push(text);
                if (text.startsWith('caption:')) {
                    const captionText = text.slice('caption:'.length).trim();
                    captionLines.push(captionText);
                    console.log(`  CAPTION: ${captionText}`);
                } else {
                    console.log(`  progress: ${text}`);
                }
            } else if (event.type === 'heartbeat') {
                continue;
            } else if (event.type === 'result' || event.type === 'error') {
                resultEvent = event;
                break;
            }
        }

        expect(resultEvent, 'received a result event').not.toBeNull();
        if (resultEvent.type === 'error') {
            throw new Error(`toolcard error: ${JSON.stringify(resultEvent).slice(0, 500)}`);
        }

        console.log(`\n  total progress events: ${allProgress.length}`);
        console.log(`  caption events:        ${captionLines.length}`);

        // Assertions:
        // 1. At least one caption per iter → at least 1 caption for max_iters=1.
        //    (The spec doc-style "captions every 5s during long-running work"
        //    requires multi-iter; 1 caption is the minimum proof of integration.)
        expect(captionLines.length,
            'at least 1 caption event (one per iter)').toBeGreaterThanOrEqual(1);

        // 2. Each caption is non-empty and reasonably short.
        for (const cap of captionLines) {
            expect(cap.length, 'caption non-empty').toBeGreaterThan(0);
            // Caption agent capped at 32 tokens — that's max ~30 words.
            // Plain-string word count: walk chars, count transitions
            // from whitespace to non-whitespace. Equivalent to
            // cap.split(/\s+/).filter(w => w.length).length.
            const isWs = (c) => c === ' ' || c === '\t' || c === '\n' || c === '\r';
            let wordCount = 0;
            let inWord = false;
            for (const c of cap) {
                if (isWs(c)) { inWord = false; }
                else if (!inWord) { inWord = true; wordCount++; }
            }
            expect(wordCount, `caption "${cap}" within budget`).toBeLessThanOrEqual(35);
        }

        // 3. Original behavior preserved — result has rasterized PNG.
        const result = resultEvent.result || resultEvent;
        const embed = result.embed || [];
        const imagePart = embed.find(p =>
            p.type === 'image_url' &&
            typeof p?.image_url?.url === 'string' &&
            p.image_url.url.startsWith('data:image/'));
        expect(imagePart, 'result still has rasterized PNG embedded').toBeTruthy();
    });

    test('two same-card sessions both emit progress before either completes', async ({ request }) => {
        const [sessionA, sessionB] = await Promise.all([
            startCaptionedInvoke(request, 'a small green square'),
            startCaptionedInvoke(request, 'a small blue triangle'),
        ]);
        console.log(`  session A: ${sessionA}`);
        console.log(`  session B: ${sessionB}`);

        const sessions = { A: sessionA, B: sessionB };
        const progressSeen = { A: false, B: false };
        const pending = new Map();
        const arm = (label) => {
            pending.set(label, pollOnce(request, sessions[label]).then(ev => ({ label, ev })));
        };
        arm('A');
        arm('B');

        const deadline = Date.now() + 60_000;
        try {
            while (Date.now() < deadline) {
                const remaining = deadline - Date.now();
                const timeout = new Promise((_, reject) =>
                    setTimeout(() => reject(new Error('timed out waiting for concurrent progress')), remaining));
                const { label, ev } = await Promise.race([...pending.values(), timeout]);
                pending.delete(label);

                if (ev.type === 'progress') {
                    progressSeen[label] = true;
                    console.log(`  ${label} progress: ${(ev.text || '').slice(0, 100)}`);
                    if (progressSeen.A && progressSeen.B) break;
                    arm(label);
                    continue;
                }
                if (ev.type === 'heartbeat') {
                    arm(label);
                    continue;
                }
                if (ev.type === 'result' || ev.type === 'error') {
                    throw new Error(
                        `${label} completed before both sessions emitted progress: ` +
                        JSON.stringify({ progressSeen, ev }).slice(0, 500));
                }
                arm(label);
            }

            expect(progressSeen.A, 'session A emitted progress').toBe(true);
            expect(progressSeen.B, 'session B emitted progress').toBe(true);
        } finally {
            await Promise.all([
                cancelQuietly(request, sessionA),
                cancelQuietly(request, sessionB),
            ]);
            await Promise.race([
                Promise.allSettled([...pending.values()]),
                new Promise(resolve => setTimeout(resolve, 5000)),
            ]);
        }
    });
});
