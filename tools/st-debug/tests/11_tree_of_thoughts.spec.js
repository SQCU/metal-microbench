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

async function startTreeOfThoughts(request, question, callerMessages = null) {
    const data = {
        args: { question },
        profile: directInvokeProfile(),
    };
    if (callerMessages) data.caller_messages = callerMessages;

    const startResp = await request.post(
        'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/tree-of-thoughts/explore',
        { data },
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

    throw new Error(`timed out waiting for tree-of-thoughts result ${sessionId}`);
}

function expectNonEmptyString(value, label) {
    expect(value, label).toEqual(expect.any(String));
    expect(value.trim().length, label).toBeGreaterThan(0);
}

test.describe('tree-of-thoughts toolcard direct invoke', () => {
    test.setTimeout(90_000);

    test('fire-and-forget start, concurrent branches, and context grounding', async ({ request }) => {
        const callerMessages = [
            { role: 'system', content: 'You are a friendly tour guide.' },
            {
                role: 'user',
                content: "I'm planning a trip with a $2000 budget. Looking at Tokyo, Reykjavik, or Buenos Aires.",
            },
            {
                role: 'assistant',
                content: 'Those are all fascinating! Tokyo is famously expensive - accommodation and food can eat the budget fast. Reykjavik is also high-cost. Buenos Aires has the best value of the three.',
            },
        ];
        const question = 'given that, which should I visit first to maximize my budget?';

        const t0 = Date.now();
        const sessionId = await startTreeOfThoughts(request, question, callerMessages);
        const t1 = Date.now();

        expect(t1 - t0, 'start_invoke returned without waiting for branches or synthesis')
            .toBeLessThan(2000);

        const result = await pollForResult(request, sessionId);
        const t2 = Date.now();

        expect(result.used_caller_messages, 'caller context flag').toBe(true);
        expect(result.branches, 'branches array').toEqual(expect.any(Array));
        expect(result.branches.length, 'default branch count').toBe(3);
        expect(result.branches.map(branch => branch.label), 'default branch labels')
            .toEqual(['practical', 'creative', 'skeptical']);

        for (const branch of result.branches) {
            expect(branch.label, 'branch label').toEqual(expect.any(String));
            expectNonEmptyString(branch.reasoning, `reasoning for ${branch.label}`);
            expectNonEmptyString(branch.summary, `summary for ${branch.label}`);
        }

        const contextRe = /Tokyo|Reykjavik|Buenos|cost|value|budget/i;
        const groundedBranches = result.branches.filter(branch =>
            contextRe.test(`${branch.reasoning}\n${branch.summary}`),
        );
        expect(groundedBranches.length, 'at least two branches reference inherited context')
            .toBeGreaterThanOrEqual(2);

        expectNonEmptyString(result.synthesis, 'synthesis is non-empty');
        expect(result.synthesis, 'synthesis references inherited context').toMatch(contextRe);

        const reasonings = result.branches.map(branch => branch.reasoning.trim());
        expect(new Set(reasonings).size, 'branch reasonings are distinct')
            .toBe(reasonings.length);

        expect(t2 - t1, 'parallel branches plus synthesis stay under concurrency budget')
            .toBeLessThan(25_000);
    });

    test('gracefully degrades without caller_messages', async ({ request }) => {
        const question = 'given that, which should I visit first to maximize my budget?';
        const sessionId = await startTreeOfThoughts(request, question);
        const result = await pollForResult(request, sessionId);

        expect(result.used_caller_messages, 'caller context flag').toBe(false);
        expect(result.branches, 'branches array').toEqual(expect.any(Array));
        expect(result.branches.length, 'default branch count').toBe(3);
        expectNonEmptyString(result.synthesis, 'synthesis is non-empty');
    });
});
