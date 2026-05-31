import { test, expect } from '@playwright/test';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const PLUGIN_BASE = '/api/plugins/user-personas';
const SUGGESTER_URL = `${PLUGIN_BASE}/static/suggester.html`;
const __dirname = dirname(fileURLToPath(import.meta.url));
const STATIC_DIR = resolve(__dirname, '../sillytavern-fork/plugins/user-personas/static');

test.use({ trace: 'off', video: 'off' });

function row(label, tier, i) {
    return {
        bio_id: `${label}-${tier}-${i}`,
        agent_id: `${label}-${tier}-agent-${i}`,
        distance: tier === 'top' ? 0.25 + i / 10 : 2.5 + i / 10,
        why: `${label} ${tier} ${i} matches the deterministic cache-tier fixture.`,
        persona: {
            name: `${label.toUpperCase()} ${tier} bio ${i}`,
            provenance: { kind: 'canonical' },
        },
        agent: {
            name: `${label.toUpperCase()} ${tier} agent ${i}`,
            provenance: { kind: 'canonical' },
        },
    };
}

function seedResponse(label, kTop = 3, kSide = 3, meta = {}) {
    return {
        top: Array.from({ length: kTop }, (_, i) => row(label, 'top', i + 1)),
        side: Array.from({ length: kSide }, (_, i) => row(label, 'side', i + 1)),
        _meta: {
            K_top: kTop,
            K_side: kSide,
            target_signature: { tone: label === 'a' ? 2 : 4 },
            target_completed_axes: 1,
            candidates_considered: kTop + kSide,
            bios_total: 6,
            agents_total: 6,
            pending_synthesis: [],
            pending_count: 0,
            ...meta,
        },
    };
}

async function installParentChatHarness(page, initialChat = 'A') {
    await page.addInitScript((initial) => {
        const states = {
            A: {
                characterId: 'cache-tier-character',
                chatId: 'chat-a',
                chat: [
                    { is_user: true, name: 'operator', mes: 'chat-a-marker: I need a grounded continuation from this chat.' },
                    { is_user: false, name: 'assistant', mes: 'chat-a-marker: The assistant answers with enough context to rank personas.' },
                ],
                characters: {
                    'cache-tier-character': { avatar: 'cache-tier-character.png' },
                },
            },
            B: {
                characterId: 'cache-tier-character',
                chatId: 'chat-b',
                chat: [
                    { is_user: true, name: 'operator', mes: 'chat-b-marker: Rank a different active chat.' },
                    { is_user: false, name: 'assistant', mes: 'chat-b-marker: This second chat must get its own cache key.' },
                ],
                characters: {
                    'cache-tier-character': { avatar: 'cache-tier-character.png' },
                },
            },
        };
        let active = initial;
        const handlers = {};
        const eventSource = {
            on(type, cb) {
                handlers[type] = handlers[type] || [];
                handlers[type].push(cb);
            },
            emit(type) {
                for (const cb of handlers[type] || []) cb();
            },
        };
        window.__suggesterHarness = {
            setChat(next) { active = next; },
            emit(type) { eventSource.emit(type); },
            getActive() { return active; },
        };
        window.SillyTavern = {
            getContext() {
                return {
                    ...states[active],
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
    }, initialChat);
}

async function installPluginRoutes(page, { slowPoll = false, targetExtractionError = false } = {}) {
    const counters = {
        seedTotal: 0,
        seedByChat: { a: 0, b: 0 },
        pollTotal: 0,
        pollByPersona: new Map(),
        releasePolls: null,
    };
    let releasePolls;
    const pollGate = new Promise(resolve => { releasePolls = resolve; });
    counters.releasePolls = releasePolls;

    await page.route(`**${PLUGIN_BASE}/static/suggester.html`, route => route.fulfill({
        status: 200,
        contentType: 'text/html',
        path: resolve(STATIC_DIR, 'suggester.html'),
    }));
    await page.route(`**${PLUGIN_BASE}/static/csrf-fetch.js`, route => route.fulfill({
        status: 200,
        contentType: 'application/javascript',
        path: resolve(STATIC_DIR, 'csrf-fetch.js'),
    }));
    await page.route('**/csrf-token', route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ token: 'disabled' }),
    }));
    await page.route(`**${PLUGIN_BASE}/bridge-status`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ reachable: true, url: 'stubbed' }),
    }));
    await page.route(`**${PLUGIN_BASE}/agents`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ agents: [{ id: 'agent-fixture' }] }),
    }));
    await page.route(`**${PLUGIN_BASE}/personas`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ personas: [{ id: 'persona-fixture' }] }),
    }));
    await page.route(`**${PLUGIN_BASE}/dispatch-missing-agent-synth`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ dispatched: 0, in_flight: 0 }),
    }));
    await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
        counters.seedTotal += 1;
        const body = route.request().postDataJSON();
        const summary = body.chat_context_summary || '';
        const label = summary.includes('chat-b-marker') ? 'b' : 'a';
        counters.seedByChat[label] += 1;
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify(seedResponse(label, body.K_top || 3, body.K_side || 3, targetExtractionError ? {
                target_signature: {},
                target_completed_axes: 0,
                target_extraction_error: 'fixture judge failure',
            } : {})),
        });
    });
    await page.route(`**${PLUGIN_BASE}/poll`, async route => {
        counters.pollTotal += 1;
        const body = route.request().postDataJSON();
        const personaId = body.persona_id;
        counters.pollByPersona.set(personaId, (counters.pollByPersona.get(personaId) || 0) + 1);
        if (slowPoll) await pollGate;
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: {
                    source: 'agent',
                    agent_id: body.agent_id,
                    name: `stub overlay ${body.agent_id}`,
                    depth: 1,
                    text_chars: 42,
                },
                candidates: [{
                    text: `Stubbed inline completion for ${personaId} using ${body.agent_id}. This deterministic response is long enough to prove row residue rendered.`,
                    truncated: false,
                }],
            }),
        });
    });
    return counters;
}

async function waitForPolls(counters, n) {
    await expect.poll(() => counters.pollTotal, {
        timeout: 5_000,
        intervals: [50, 100, 250],
    }).toBe(n);
}

async function switchChat(page, label) {
    await page.evaluate((next) => {
        window.__suggesterHarness.setChat(next);
        window.__suggesterHarness.emit('CHAT_CHANGED');
    }, label);
}

test.describe('suggester cache tiers and navigation contract', () => {
    test('finite top/side tiers, per-row cache, per-chat rank cache, and bounded navigation polling', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop', 'deterministic cache-contract spec runs once under desktop');

        await installParentChatHarness(page, 'A');
        const counters = await installPluginRoutes(page);

        await page.goto(SUGGESTER_URL);
        await expect(page.locator('h1')).toBeVisible();

        const topRows = page.locator('.ranked-row:not(.side):not([data-bio-id^="__skeleton_"])');
        const sideRows = page.locator('.ranked-row.side');

        await expect(topRows).toHaveCount(3);
        await expect(sideRows).toHaveCount(3);

        await waitForPolls(counters, 3);
        for (let i = 0; i < 3; i++) {
            await expect(topRows.nth(i).locator('.row-completion-text')).toContainText('Stubbed inline completion');
        }
        for (let i = 0; i < 3; i++) {
            await expect(sideRows.nth(i).locator('.row-completion')).not.toHaveClass(/visible/);
        }

        const sidePersona = await sideRows.first().getAttribute('data-bio-id');
        await sideRows.first().locator('.suggest-btn').click();
        await waitForPolls(counters, 4);
        expect(counters.pollByPersona.get(sidePersona)).toBe(1);
        await expect(sideRows.first().locator('.row-completion-text')).toContainText(sidePersona);

        await sideRows.first().locator('.suggest-btn').click();
        await page.waitForTimeout(100);
        expect(counters.pollByPersona.get(sidePersona)).toBe(1);
        await expect(sideRows.first().locator('.cache-badge')).toBeVisible();

        expect(counters.seedByChat.a).toBe(1);
        await switchChat(page, 'B');
        await expect(page.locator('.ranked-row[data-bio-id="b-top-1"]')).toBeVisible();
        await expect.poll(() => counters.seedByChat.b).toBe(1);

        await switchChat(page, 'A');
        await expect(page.locator('.ranked-row[data-bio-id="a-top-1"]')).toBeVisible();
        await page.waitForTimeout(100);
        expect(counters.seedByChat.a).toBe(1);
        expect(counters.seedTotal).toBe(2);
    });

    test('chat navigation during in-flight polls does not create a duplicate retry cascade', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop', 'deterministic cache-contract spec runs once under desktop');

        await installParentChatHarness(page, 'A');
        const counters = await installPluginRoutes(page, { slowPoll: true });

        await page.goto(SUGGESTER_URL);
        await expect(page.locator('.ranked-row[data-bio-id="a-top-1"]')).toBeVisible();
        await waitForPolls(counters, 3);

        await switchChat(page, 'B');
        await expect(page.locator('.ranked-row[data-bio-id="b-top-1"]')).toBeVisible();
        await waitForPolls(counters, 6);

        await switchChat(page, 'A');
        await expect(page.locator('.ranked-row[data-bio-id="a-top-1"]')).toBeVisible();
        await page.waitForTimeout(250);

        expect(counters.seedByChat.a).toBe(1);
        expect(counters.seedByChat.b).toBe(1);
        expect(counters.pollTotal).toBe(6);
        for (const [personaId, count] of counters.pollByPersona.entries()) {
            expect(count, `${personaId} should poll at most once during A -> B -> A navigation`).toBe(1);
        }

        counters.releasePolls();
        await page.waitForTimeout(250);
        await expect(page.locator('.ranked-row[data-bio-id="a-top-1"]')).toBeVisible();
        expect(counters.pollTotal).toBe(6);
    });

    test('target-signature extraction failure still renders finite-K selection rows', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop', 'deterministic failure-contract spec runs once under desktop');

        await installParentChatHarness(page, 'A');
        const counters = await installPluginRoutes(page, { targetExtractionError: true });

        await page.goto(SUGGESTER_URL);

        const status = page.locator('#ranked-status');
        const rows = page.locator('.ranked-row:not([data-bio-id^="__skeleton_"])');

        await expect(rows.first(), 'finite-K selection row renders despite target judge failure').toBeVisible();
        await expect(rows).toHaveCount(6);
        await expect(status, 'bare yapper-seed failure string is not the UI').not.toContainText(/Rank failed|yapper-seed failed|target signature extraction failed/i);
        await expect(page.locator('#meta-target-error')).toContainText(/neutral-3 baseline/);
        await expect(page.locator('.suggest-btn').first(), 'selection row still has Suggest affordance').toBeEnabled();
        await waitForPolls(counters, 3);
    });
});
