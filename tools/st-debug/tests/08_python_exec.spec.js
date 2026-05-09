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

async function invokePythonExec(request, task) {
    const startResp = await request.post(
        'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/python-exec/run',
        {
            data: {
                args: { task },
                profile: directInvokeProfile(),
            },
        });
    expect(startResp.status(), 'start_invoke 200').toBe(200);

    const { session_id } = await startResp.json();
    const deadline = Date.now() + 60_000;
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

    throw new Error('timed out waiting for python-exec result');
}

function parseIntegers(stdout) {
    return (stdout || '')
        .match(/-?\d+/g)
        ?.map(s => Number.parseInt(s, 10)) || [];
}

test.describe('python-exec toolcard direct invoke', () => {
    test.setTimeout(75_000);

    test('computes a deterministic SHA-256 digest', async ({ request }) => {
        const result = await invokePythonExec(
            request,
            'compute the SHA-256 of the string "hello world" and print the hex digest',
        );

        expect(result.stdout, 'stdout contains SHA-256 digest')
            .toContain('b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9');
        expect(result.script, 'script is non-empty Python source')
            .toEqual(expect.any(String));
        expect(result.script.trim().length, 'script is non-empty Python source')
            .toBeGreaterThan(0);
        expect(result.returncode, `stderr: ${result.stderr || ''}`).toBe(0);
    });

    test('prints five unique sorted random integers in range', async ({ request }) => {
        const result = await invokePythonExec(
            request,
            'print 5 unique random integers between 1 and 100, sorted ascending',
        );

        expect(result.returncode, `stderr: ${result.stderr || ''}`).toBe(0);

        const values = parseIntegers(result.stdout);
        expect(values, `stdout: ${result.stdout}`).toHaveLength(5);
        expect(new Set(values).size, 'values are distinct').toBe(5);

        const sorted = [...values].sort((a, b) => a - b);
        expect(values, 'values are sorted ascending').toEqual(sorted);
        for (const value of values) {
            expect(value, `${value} is at least 1`).toBeGreaterThanOrEqual(1);
            expect(value, `${value} is at most 100`).toBeLessThanOrEqual(100);
        }
    });
});
