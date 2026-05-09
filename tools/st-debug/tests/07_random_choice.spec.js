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

async function invokeRandomChoice(request, args) {
    const startResp = await request.post(
        'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/random-choice/uniform',
        {
            data: {
                args,
                profile: directInvokeProfile(),
            },
        });
    expect(startResp.status(), 'start_invoke 200').toBe(200);

    const { session_id } = await startResp.json();
    const deadline = Date.now() + 20_000;
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

    throw new Error('timed out waiting for random-choice result');
}

function sampledItems(result) {
    const resultItems = result?.items || result?.result;
    if (Array.isArray(resultItems)) return resultItems;

    const embed = result?.embed || [];
    const text = embed
        .filter(part => part?.type === 'text' && typeof part.text === 'string')
        .map(part => part.text)
        .join('\n');
    return text
        .split(/Selected \d+ item\(s\): /)
        .pop()
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
}

function embedText(result) {
    return (result?.embed || [])
        .filter(part => part?.type === 'text' && typeof part.text === 'string')
        .map(part => part.text)
        .join('\n');
}

test.describe('random-choice toolcard direct invoke', () => {
    test.setTimeout(60_000);

    test('samples N unique items from the input list without replacement', async ({ request }) => {
        const items = ['fool', 'magician', 'empress', 'tower', 'star'];
        const n = 3;
        const result = await invokeRandomChoice(request, {
            items,
            n,
            with_replacement: false,
        });

        const picks = sampledItems(result);
        expect(Array.isArray(result.embed), 'result.embed contains sampled items text')
            .toBe(true);
        const text = embedText(result);
        expect(text, 'embed text is present').toContain('Selected');
        expect(picks, 'sample count matches n').toHaveLength(n);

        for (const pick of picks) {
            expect(items, `sampled item ${pick} came from input list`).toContain(pick);
            expect(text, `embed text contains sampled item ${pick}`).toContain(pick);
        }

        expect(new Set(picks).size, 'with_replacement=false has no duplicates')
            .toBe(picks.length);
    });

    test('100-item list produces different subsets across repeated runs', async ({ request }) => {
        const items = Array.from({ length: 100 }, (_, i) => `item-${String(i + 1).padStart(3, '0')}`);
        const runs = [];

        for (let i = 0; i < 3; i++) {
            const result = await invokeRandomChoice(request, {
                items,
                n: 10,
                with_replacement: false,
            });
            const picks = sampledItems(result);
            expect(picks).toHaveLength(10);
            expect(new Set(picks).size, 'no duplicates in each run').toBe(10);
            for (const pick of picks) {
                expect(items, `sampled item ${pick} came from input list`).toContain(pick);
            }
            runs.push(picks.join('|'));
        }

        expect(new Set(runs).size, 'at least 2 of 3 sampled subsets differ')
            .toBeGreaterThanOrEqual(2);
    });
});
