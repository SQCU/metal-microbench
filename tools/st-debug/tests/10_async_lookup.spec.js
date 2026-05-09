import { test, expect } from '@playwright/test';

function directInvokeProfile() {
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

async function startAsyncLookup(request, topic) {
    const startResp = await request.post(
        'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/async-lookup/lookup',
        {
            data: {
                args: { topic },
                profile: directInvokeProfile(),
            },
        },
    );
    expect(startResp.status(), 'start_invoke 200').toBe(200);

    const body = await startResp.json();
    expect(body.session_id, 'session_id is returned').toEqual(expect.any(String));
    expect(body.session_id.length, 'session_id is non-empty').toBeGreaterThan(0);
    return body.session_id;
}

async function pollForResult(request, sessionId, timeoutMs = 75_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        const pollResp = await request.get(
            `http://127.0.0.1:8002/api/plugins/toolcards/poll/${sessionId}`,
        );
        if (pollResp.status() === 404) {
            throw new Error(`session ${sessionId} not found at poll`);
        }
        expect(pollResp.status(), `poll ${sessionId}`).toBe(200);

        const event = await pollResp.json();
        if (event.type === 'heartbeat') continue;
        if (event.type === 'progress') continue;
        if (event.type === 'result') {
            expect(event.ok, `tool result ok: ${JSON.stringify(event).slice(0, 500)}`)
                .toBe(true);
            return event.result;
        }
        if (event.type === 'error') {
            throw new Error(`toolcard returned error: ${JSON.stringify(event).slice(0, 500)}`);
        }
    }

    throw new Error(`timed out waiting for async-lookup result ${sessionId}`);
}

function expectAsyncLookupResult(result, topic) {
    expect(result.topic, 'result preserves topic').toBe(topic);
    expect(result.simulated_lookup_delay_s, 'simulated delay marker').toBe(6);
    expect(result.elapsed_s, 'elapsed_s includes simulated delay').toBeGreaterThanOrEqual(6);
    expect(result.answer, 'answer is a string').toEqual(expect.any(String));
    expect(result.answer.trim().length, 'answer is non-empty').toBeGreaterThan(0);
}

test.describe('async-lookup toolcard direct invoke', () => {
    test.setTimeout(90_000);

    test('fire-and-forget start returns before slow descendant result', async ({ request }) => {
        const topic = 'current bird species spotted in Hyde Park, London this week';

        const t0 = Date.now();
        const sessionId = await startAsyncLookup(request, topic);
        const t1 = Date.now();

        expect(t1 - t0, 'start_invoke returns session_id without blocking on 6s work')
            .toBeLessThan(1000);

        const result = await pollForResult(request, sessionId);
        const t2 = Date.now();

        expect(t2 - t1, 'result arrives after descendant simulated lookup delay')
            .toBeGreaterThan(5000);
        expectAsyncLookupResult(result, topic);
    });

    test('multiple concurrent fire-and-forget invocations complete concurrently', async ({ request }) => {
        const topics = [
            'alpha-quartz-271 lookup status for a warehouse cache refresh',
            'beta-cobalt-582 lookup status for a museum label audit',
            'gamma-moss-943 lookup status for a rooftop irrigation check',
        ];

        const t0 = Date.now();
        const sessionIds = await Promise.all(
            topics.map(topic => startAsyncLookup(request, topic)),
        );
        const t1 = Date.now();

        expect(t1 - t0, 'three start_invoke calls return session_ids without waiting')
            .toBeLessThan(1000);
        expect(new Set(sessionIds).size, 'each invocation gets its own session')
            .toBe(sessionIds.length);

        const results = await Promise.all(
            sessionIds.map(sessionId => pollForResult(request, sessionId, 15_000)),
        );
        const t2 = Date.now();

        expect(t2 - t0, 'all concurrent results land within the 15s wall budget')
            .toBeLessThan(15_000);

        for (let i = 0; i < results.length; i++) {
            expectAsyncLookupResult(results[i], topics[i]);
        }

        const returnedTopics = results.map(result => result.topic);
        expect(new Set(returnedTopics).size, 'results keep distinct topics')
            .toBe(topics.length);

        const answers = results.map(result => result.answer.trim());
        expect(new Set(answers).size, 'different topics produce distinct answers')
            .toBe(topics.length);
    });
});
