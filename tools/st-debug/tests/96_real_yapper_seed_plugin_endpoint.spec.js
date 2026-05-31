import { test, expect } from '@playwright/test';
import fs from 'node:fs';

const PLUGIN_BASE = '/api/plugins/user-personas';
const BASIC_AUTH_FALLBACK = { username: 'sussy', password: 'amongus' };

function loadManifest() {
    const manifestPath = process.env.ST_MATRIX_MANIFEST;
    if (!manifestPath || !fs.existsSync(manifestPath)) return null;
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
}

function authHeader(manifest) {
    const auth = manifest.basicAuth || BASIC_AUTH_FALLBACK;
    return `Basic ${Buffer.from(`${auth.username}:${auth.password}`).toString('base64')}`;
}

test.describe('real user-personas /yapper-seed endpoint', () => {
    test.setTimeout(60_000);

    test('sillytavern-fork plugin ranks real top-k without Playwright route stubs', async ({ request }) => {
        const manifest = loadManifest();
        test.skip(!manifest, 'ST_MATRIX_MANIFEST is required; run sillytavern-fork/tests/scripts/multi_st_matrix.mjs (run from the fork) start');
        const instance = manifest.instances[0];

        const runtime = await request.get(`${instance.url}${PLUGIN_BASE}/runtime-config`, {
            headers: { Authorization: authHeader(manifest) },
        });
        expect(runtime.ok()).toBeTruthy();
        const runtimeBody = await runtime.json();
        expect(runtimeBody.plugin_dir).toContain(instance.cloneDir);
        expect(runtimeBody.bridge_url).toBe(manifest.bridgeUrl);

        const response = await request.post(`${instance.url}${PLUGIN_BASE}/yapper-seed`, {
            headers: {
                Authorization: authHeader(manifest),
                'Content-Type': 'application/json',
            },
            data: {
                chat_context_summary:
                    'USER: I am a reckless Sagittarius rogue who wants to steal the sealed message tube from the night-jay, flirt with the innkeeper, and run toward danger. DICEMOTHER: the corridor opens into a torchlit hall and asks what you do next.',
                K_top: 1,
                K_side: 1,
            },
        });

        expect(response.ok(), await response.text()).toBeTruthy();
        const body = await response.json();
        expect(Array.isArray(body.top)).toBeTruthy();
        expect(Array.isArray(body.side)).toBeTruthy();
        expect(body.top.length).toBe(1);
        expect(body.side.length).toBe(1);
        expect(body._meta?.target_extraction_error ?? null).toBeNull();
        expect(body._meta?.target_completed_axes).toBeGreaterThan(0);
        expect(body._meta?.candidates_considered).toBeGreaterThanOrEqual(2);
        expect(body.top[0]?.persona?.id).toBeTruthy();
        expect(body.top[0]?.agent?.id).toBeTruthy();
    });
});
