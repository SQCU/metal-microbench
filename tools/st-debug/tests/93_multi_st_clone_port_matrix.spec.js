import { test, expect } from '@playwright/test';
import fs from 'node:fs';

const PLUGIN_PATH = '/api/plugins/user-personas';

test.use({ trace: 'off', video: 'off' });

function loadManifest() {
    const manifestPath = process.env.ST_MATRIX_MANIFEST;
    if (!manifestPath || !fs.existsSync(manifestPath)) return null;
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
}

function pairs(items) {
    const out = [];
    for (let i = 0; i < items.length; i++) {
        for (let j = i + 1; j < items.length; j++) out.push([items[i], items[j]]);
    }
    return out;
}

function authHeader(manifest) {
    const user = manifest.basicAuth?.username || 'sussy';
    const pass = manifest.basicAuth?.password || 'amongus';
    return `Basic ${Buffer.from(`${user}:${pass}`).toString('base64')}`;
}

function row(label, tier, i) {
    return {
        bio_id: `${label}-${tier}-bio-${i}`,
        agent_id: `${label}-${tier}-agent-${i}`,
        distance: tier === 'top' ? 0.1 + i / 10 : 2.2 + i / 10,
        why: `${label} ${tier} fixture row ${i}`,
        persona: { name: `${label.toUpperCase()} ${tier} bio ${i}`, provenance: { kind: 'matrix' } },
        agent: { name: `${label.toUpperCase()} ${tier} agent ${i}`, provenance: { kind: 'matrix' } },
    };
}

function seedResponse(label) {
    return {
        top: [row(label, 'top', 1), row(label, 'top', 2), row(label, 'top', 3)],
        side: [row(label, 'side', 1), row(label, 'side', 2), row(label, 'side', 3)],
        _meta: {
            K_top: 3,
            K_side: 3,
            target_signature: { tone: 3 },
            target_completed_axes: 1,
            candidates_considered: 6,
            bios_total: 6,
            agents_total: 6,
            pending_synthesis: [],
            pending_count: 0,
        },
    };
}

async function installParentChatHarness(page, label) {
    await page.addInitScript((instanceLabel) => {
        const eventHandlers = {};
        const eventSource = {
            on(type, cb) {
                eventHandlers[type] = eventHandlers[type] || [];
                eventHandlers[type].push(cb);
            },
            emit(type) {
                for (const cb of eventHandlers[type] || []) cb();
            },
        };
        window.SillyTavern = {
            getContext() {
                return {
                    characterId: `matrix-character-${instanceLabel}`,
                    chatId: `matrix-chat-${instanceLabel}`,
                    chat: [
                        { is_user: true, name: 'operator', mes: `matrix-${instanceLabel}: user message for ranking` },
                        { is_user: false, name: 'assistant', mes: `matrix-${instanceLabel}: assistant context for ranking` },
                    ],
                    characters: {
                        [`matrix-character-${instanceLabel}`]: { avatar: `matrix-character-${instanceLabel}.png` },
                    },
                    eventSource,
                    eventTypes: {
                        CHAT_CHANGED: 'CHAT_CHANGED',
                        CHAT_LOADED: 'CHAT_LOADED',
                        MESSAGE_SENT: 'MESSAGE_SENT',
                        MESSAGE_RECEIVED: 'MESSAGE_RECEIVED',
                    },
                };
            },
        };
    }, label);
}

async function installPluginFixtures(page, instance, allPorts) {
    const expectedOrigin = new URL(instance.url).origin;
    const violations = [];
    let seedTotal = 0;

    page.on('request', request => {
        const url = request.url();
        if (!url.includes(PLUGIN_PATH)) return;
        const origin = new URL(url).origin;
        if (origin !== expectedOrigin) {
            violations.push(`${url} escaped ${expectedOrigin}`);
        }
        for (const port of allPorts) {
            if (port !== instance.port && url.includes(`:${port}/`)) {
                violations.push(`${url} referenced peer port ${port}`);
            }
        }
    });

    await page.route('**/csrf-token', route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ token: 'matrix-disabled' }),
    }));
    await page.route(`**${PLUGIN_PATH}/bridge-status`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ reachable: true, status: 'matrix-fixture', model: 'fixture' }),
    }));
    await page.route(`**${PLUGIN_PATH}/agents`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ agents: [{ id: `agent-${instance.index}`, designed_for_bio_id: `bio-${instance.index}.png` }] }),
    }));
    await page.route(`**${PLUGIN_PATH}/personas`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ personas: [{ id: `bio-${instance.index}.png`, name: `Bio ${instance.index}` }] }),
    }));
    await page.route(`**${PLUGIN_PATH}/dispatch-missing-agent-synth`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, dispatched: 0, in_flight: 0 }),
    }));
    await page.route(`**${PLUGIN_PATH}/yapper-seed`, route => {
        seedTotal += 1;
        return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(seedResponse(`st${instance.index}`)),
        });
    });
    await page.route(`**${PLUGIN_PATH}/poll`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
            applied_overlay: {
                source: 'agent',
                agent_id: `agent-${instance.index}`,
                name: `matrix agent ${instance.index}`,
                depth: 1,
                text_chars: 40,
            },
            candidates: [{
                text: `Matrix fixture response for ${instance.name}. This proves the row can accumulate a suggestion from its own server origin.`,
                truncated: false,
            }],
        }),
    }));

    return {
        get seedTotal() { return seedTotal; },
        violations,
    };
}

async function assertRuntimeConfig(request, instance, manifest) {
    const response = await request.get(`${instance.url}${PLUGIN_PATH}/runtime-config`, {
        headers: { Authorization: authHeader(manifest) },
    });
    expect(response.ok(), `${instance.name} runtime-config HTTP status`).toBeTruthy();
    const body = await response.json();
    expect(body.st_url).toBe(instance.url);
    expect(body.plugin_url).toBe(`${instance.url}${PLUGIN_PATH}`);
    expect(body.bridge_url).toBe(manifest.bridgeUrl);
    expect(body.plugin_dir).toContain(instance.cloneDir);
    expect(body.disable_boot_autosynth).toBe(true);
}

async function exerciseSuggester(page, instance, allPorts) {
    await installParentChatHarness(page, instance.name);
    const fixtures = await installPluginFixtures(page, instance, allPorts);
    await page.goto(`${instance.url}${PLUGIN_PATH}/static/suggester.html`, { waitUntil: 'domcontentloaded' });
    await expect(page.locator('h1')).toBeVisible();
    await expect.poll(() => fixtures.seedTotal, { timeout: 10_000, intervals: [100, 250, 500] }).toBeGreaterThan(0);
    await expect(page.locator('.ranked-row').first()).toBeVisible();
    await expect(page.locator('body')).not.toContainText('yapper-seed failed');
    await expect(page.locator('body')).not.toContainText('Rank failed');
    expect(fixtures.violations, `${instance.name} plugin requests stayed on its own origin`).toEqual([]);
}

test.describe('multi ST clone port matrix', () => {
    const manifest = loadManifest();
    test.skip(!manifest, 'ST_MATRIX_MANIFEST is required; run sillytavern-fork/tests/scripts/multi_st_matrix.mjs (run from the fork) run');

    for (const [a, b] of pairs(manifest?.instances || [])) {
        test(`${a.name} and ${b.name} run user-personas UI without port bleed`, async ({ browser, request }) => {
            const allPorts = manifest.instances.map(instance => instance.port);
            await assertRuntimeConfig(request, a, manifest);
            await assertRuntimeConfig(request, b, manifest);

            const context = await browser.newContext({
                httpCredentials: manifest.basicAuth || { username: 'sussy', password: 'amongus' },
            });
            const pageA = await context.newPage();
            const pageB = await context.newPage();
            try {
                await Promise.all([
                    exerciseSuggester(pageA, a, allPorts),
                    exerciseSuggester(pageB, b, allPorts),
                ]);
            } finally {
                await context.close();
            }
        });
    }
});
