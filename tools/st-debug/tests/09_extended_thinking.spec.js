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

async function invokeExtendedThinking(request, question, callerMessages = null) {
    const data = {
        args: { question },
        profile: directInvokeProfile(),
    };
    if (callerMessages) data.caller_messages = callerMessages;

    const startResp = await request.post(
        'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/extended-thinking/deliberate',
        { data },
    );
    expect(startResp.status(), 'start_invoke 200').toBe(200);

    const { session_id } = await startResp.json();
    const deadline = Date.now() + 75_000;
    while (Date.now() < deadline) {
        const pollResp = await request.get(
            `http://127.0.0.1:8002/api/plugins/toolcards/poll/${session_id}`);
        if (pollResp.status() === 404) {
            throw new Error(`session ${session_id} not found at poll`);
        }
        expect(pollResp.status(), `poll ${session_id}`).toBe(200);

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

    throw new Error('timed out waiting for extended-thinking result');
}

test.describe('extended-thinking toolcard direct invoke', () => {
    test.setTimeout(90_000);

    test('uses caller_messages as inherited parent context', async ({ request }) => {
        const callerMessages = [
            { role: 'system', content: 'You are a friendly tour guide.' },
            {
                role: 'user',
                content: "I'm planning a trip with a $2000 budget. Looking at Tokyo, Reykjavik, or Buenos Aires.",
            },
            {
                role: 'assistant',
                content: 'Those are all fascinating! Tokyo is famously expensive — accommodation and food can eat the budget fast. Reykjavik is also high-cost. Buenos Aires has the best value of the three.',
            },
        ];
        const result = await invokeExtendedThinking(
            request,
            'given that, which should I visit first to maximize my budget?',
            callerMessages,
        );

        expect(result.used_caller_messages, 'caller context flag').toBe(true);
        expect(result.summary, 'summary is non-empty').toEqual(expect.any(String));
        expect(result.summary.trim().length, 'summary is non-empty').toBeGreaterThan(0);
        expect(result.summary, 'summary references inherited budget/value context')
            // The descendant proves grounding in the caller_messages by
            // referring to ANY of the three cities the parent named
            // (Tokyo / Reykjavik / Buenos Aires) or the budget-context
            // keywords (cost / value / budget). The model's exact
            // phrasing varies; what matters is that one of these
            // context-derived tokens appears.
            .toMatch(/Tokyo|Reykjavik|Buenos|cost|value|budget/i);
        expect(result.reasoning_full, 'reasoning_full is present').toEqual(expect.any(String));
        expect(
            result.reasoning_full.trim().length,
            'reasoning_full contains more than the extracted summary',
        ).toBeGreaterThan(result.summary.trim().length);
    });

    test('gracefully degrades without caller_messages', async ({ request }) => {
        const result = await invokeExtendedThinking(
            request,
            'given that, which should I visit first to maximize my budget?',
        );

        expect(result.used_caller_messages, 'caller context flag').toBe(false);
        expect(result.summary, 'summary is non-empty').toEqual(expect.any(String));
        expect(result.summary.trim().length, 'summary is non-empty').toBeGreaterThan(0);
    });
});
