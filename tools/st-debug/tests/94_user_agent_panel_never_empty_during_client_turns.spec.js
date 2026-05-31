import { test, expect } from '@playwright/test';

const ST_URL = process.env.ST_URL || 'http://127.0.0.1:8002';
const PLUGIN_BASE = '/api/plugins/user-personas';
const BASIC_AUTH = { username: 'sussy', password: 'amongus' };
const FORBIDDEN_PANEL_TEXT = [
    /no chat content yet/i,
    /awaiting chat content/i,
    /once an assistant or user turn lands/i,
    /yapper-seed failed/i,
    /loadout empty/i,
    /click .*retry/i,
];

test.use({ httpCredentials: BASIC_AUTH, trace: 'off', video: 'off' });

async function requireStAndBridge(request) {
    const st = await request.get(ST_URL).catch(() => null);
    if (!st || ![200, 401].includes(st.status())) {
        test.skip(true, `st-debug not reachable at ${ST_URL}; run tools/st-debug/scripts/run.sh first`);
    }
    const bridge = await request.get('http://127.0.0.1:8001/health').catch(() => null);
    if (!bridge?.ok()) {
        test.skip(true, 'bridge not reachable at http://127.0.0.1:8001; run make serve first');
    }
}

async function requireStOnly(request) {
    const st = await request.get(ST_URL).catch(() => null);
    if (!st || ![200, 401].includes(st.status())) {
        test.skip(true, `st-debug not reachable at ${ST_URL}; run tools/st-debug/scripts/run.sh first`);
    }
}

async function loadAndConnectViaClient(page) {
    await page.goto(ST_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });

    await page.locator('#API-status-top').click();
    await expect(page.locator('#api_button_openai')).toBeVisible({ timeout: 15_000 });
    await page.locator('#api_button_openai').click();
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.onlineStatus === 'Valid';
    }, { timeout: 30_000 });
    await page.locator('#API-status-top').click().catch(() => {});
    await page.keyboard.press('Escape').catch(() => {});
}

async function loadClientOnly(page) {
    await page.goto(ST_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });
}

async function openUserAgentPanel(page) {
    const button = page.locator('#user_personas_btn');
    await expect(button).toBeVisible({ timeout: 30_000 });
    await button.click();
    const panel = page.locator('#user_personas_panel');
    await expect(panel).toBeVisible({ timeout: 15_000 });
    return panel;
}

async function assertAllowedPanelState(page, label) {
    const panel = page.locator('#user_personas_panel');
    await expect(panel, `${label}: user-agent panel visible`).toBeVisible({ timeout: 15_000 });
    const text = await panel.innerText();
    for (const forbidden of FORBIDDEN_PANEL_TEXT) {
        expect(text, `${label}: forbidden stalled/empty copy ${forbidden}`).not.toMatch(forbidden);
    }

    const cards = panel.locator('.user-personas-card');
    const pending = /pending user-agent top-k choices/i.test(text);
    const count = await cards.count();
    expect(pending || count > 0, `${label}: pending top-k or loaded user-agent cards`).toBeTruthy();

    for (let i = 0; i < count; i++) {
        const card = cards.nth(i);
        const mode = await card.locator('.user-personas-card-mode').inputValue().catch(() => '');
        const cardText = await card.innerText();
        const stopped = mode === 'off';
        const pendingSuggestion = /\(polling…\)|\(no candidate yet\)/i.test(cardText);
        const finishedSuggestion = await card.locator('.user-personas-card-preview:not(.is-loading):not(.is-error)').count() > 0;
        expect(stopped || pendingSuggestion || finishedSuggestion,
            `${label}: card ${i} is stopped, pending suggestion, or finished`).toBeTruthy();
    }
}

async function sendUserTurn(page, prompt) {
    await page.locator('#send_textarea').fill(prompt);
    await page.locator('#send_but').click();
    await expect(page.locator('#chat .mes[is_user="true"]:not(.smallSysMes)').last())
        .toContainText(prompt.slice(0, 24), { timeout: 10_000 });
}

async function waitForAssistantTurn(page) {
    await page.waitForFunction((floor) => {
        const messages = [...document.querySelectorAll('#chat .mes:not(.smallSysMes)')];
        const last = messages[messages.length - 1];
        if (!last) return false;
        if (last.getAttribute('is_user') !== 'false') return false;
        const text = last.querySelector('.mes_text')?.textContent || '';
        return text.trim().length > 0;
    }, null, { timeout: 120_000 });
    await page.waitForFunction(() => {
        const stop = document.querySelector('#mes_stop');
        return !(stop && stop.offsetParent !== null);
    }, { timeout: 120_000 });
    await expect(page.locator('#chat .mes[is_user="false"]:not(.smallSysMes)').last().locator('.mes_text'))
        .not.toBeEmpty({ timeout: 30_000 });
}

async function installDeterministicAgentEndpoints(page) {
    const topRows = ['alpha', 'beta'].map((id, i) => ({
        bio_id: `client-${id}.png`,
        agent_id: `client-${id}-agent`,
        distance: 0.1 + i / 10,
        why: `Client-turn fixture ${id}`,
        persona: {
            id: `client-${id}.png`,
            name: `Client ${id}`,
            bio: `Client ${id} bio`,
            provenance: { kind: 'matrix' },
        },
        agent: {
            id: `client-${id}-agent`,
            name: `Client ${id} agent`,
            designed_for_bio_id: `client-${id}.png`,
            provenance: { kind: 'matrix' },
        },
    }));
    const sideRows = ['gamma', 'delta'].map((id, i) => ({
        bio_id: `client-${id}.png`,
        agent_id: `client-${id}-agent`,
        distance: 2.1 + i / 10,
        why: `Client-turn side fixture ${id}`,
        persona: {
            id: `client-${id}.png`,
            name: `Client ${id}`,
            bio: `Client ${id} bio`,
            provenance: { kind: 'matrix' },
        },
        agent: {
            id: `client-${id}-agent`,
            name: `Client ${id} agent`,
            designed_for_bio_id: `client-${id}.png`,
            provenance: { kind: 'matrix' },
        },
    }));

    let releaseFirstRank;
    const firstRankGate = new Promise(resolve => { releaseFirstRank = resolve; });
    let rankCalls = 0;

    await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
        rankCalls += 1;
        if (rankCalls === 1) await firstRankGate;
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                top: topRows,
                side: sideRows,
                _meta: {
                    K_top: 2,
                    K_side: 2,
                    target_signature: { tone: 3 },
                    target_completed_axes: 1,
                    candidates_considered: 4,
                    bios_total: 4,
                    agents_total: 4,
                    pending_synthesis: [],
                    pending_count: 0,
                },
            }),
        });
    });
    await page.route(`**${PLUGIN_BASE}/poll`, async route => {
        const body = route.request().postDataJSON();
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: {
                    source: 'agent',
                    agent_id: body.agent_id || `${body.persona_id}-agent`,
                    name: `Overlay for ${body.persona_id}`,
                    depth: 1,
                    text_chars: 40,
                },
                candidates: [{
                    text: `Suggested user-agent continuation for ${body.persona_id} after the latest client turn.`,
                    truncated: false,
                }],
            }),
        });
    });
    await page.route(`**${PLUGIN_BASE}/dispatch-missing-agent-synth`, route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, dispatched: 0, in_flight: 0 }),
    }));

    return { releaseFirstRank };
}

function rankRows(label) {
    return ['alpha', 'beta'].map((id, i) => ({
        bio_id: `${label}-${id}.png`,
        agent_id: `${label}-${id}-agent`,
        distance: 0.1 + i / 10,
        why: `${label} fixture ${id}`,
        persona: {
            id: `${label}-${id}.png`,
            name: `${label} ${id}`,
            bio: `${label} ${id} bio`,
            provenance: { kind: 'matrix' },
        },
        agent: {
            id: `${label}-${id}-agent`,
            name: `${label} ${id} agent`,
            designed_for_bio_id: `${label}-${id}.png`,
            provenance: { kind: 'matrix' },
        },
    }));
}

async function installResuggestEndpoints(page) {
    let rankCalls = 0;
    await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
        rankCalls += 1;
        const label = `Call ${rankCalls}`;
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                top: rankRows(label),
                side: [],
                _meta: {
                    K_top: 2,
                    K_side: 0,
                    target_signature: { tone: 3 },
                    target_completed_axes: 1,
                    candidates_considered: 2,
                    bios_total: 2,
                    agents_total: 2,
                    pending_synthesis: [],
                    pending_count: 0,
                },
            }),
        });
    });
    await page.route(`**${PLUGIN_BASE}/poll`, async route => {
        const body = route.request().postDataJSON();
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: {
                    source: 'agent',
                    agent_id: `${body.persona_id}-agent`,
                    name: `Overlay for ${body.persona_id}`,
                    depth: 1,
                    text_chars: 40,
                },
                candidates: [{
                    text: `Candidate for ${body.persona_id}`,
                    truncated: false,
                }],
            }),
        });
    });
    return { get rankCalls() { return rankCalls; } };
}

test.describe('user-agent panel never empty during client chat turns', () => {
    test.setTimeout(240_000);

    test.beforeEach(async ({ request }) => {
        await requireStAndBridge(request);
    });

    test('panel is pending or populated before and after each user/Gemma turn', async ({ page }) => {
        const rankControl = await installDeterministicAgentEndpoints(page);
        await loadAndConnectViaClient(page);
        await openUserAgentPanel(page);

        await assertAllowedPanelState(page, 'first paint before rank resolves');
        rankControl.releaseFirstRank();
        await expect(page.locator('#user_personas_panel .user-personas-card').first())
            .toBeVisible({ timeout: 15_000 });
        await assertAllowedPanelState(page, 'first paint after top-k resolves');

        for (let i = 1; i <= 2; i++) {
            await sendUserTurn(page, `Panel invariant check turn ${i}. Reply in one short sentence.`);
            await assertAllowedPanelState(page, `after submitted user turn ${i}`);
            await waitForAssistantTurn(page);
            await assertAllowedPanelState(page, `after yielded Gemma turn ${i}`);
        }
    });
});

test.describe('user-agent panel retry controls', () => {
    test.setTimeout(90_000);

    test.beforeEach(async ({ request }) => {
        await requireStOnly(request);
    });

    test('Re-suggest click re-runs /yapper-seed and replaces rendered loadout', async ({ page }) => {
        const endpoints = await installResuggestEndpoints(page);
        await loadClientOnly(page);
        await openUserAgentPanel(page);

        const panel = page.locator('#user_personas_panel');
        await expect(panel.locator('.user-personas-card-name').first())
            .toContainText(/Call \d+ alpha/, { timeout: 15_000 });
        const beforeText = await panel.locator('.user-personas-card-name').first().innerText();
        const beforeCalls = endpoints.rankCalls;
        expect(beforeCalls, 'initial panel render called /yapper-seed').toBeGreaterThan(0);

        await panel.locator('#user_personas_resuggest_btn').click();
        await expect.poll(() => endpoints.rankCalls, {
            timeout: 10_000,
            intervals: [100, 250, 500],
        }).toBeGreaterThan(beforeCalls);
        await expect.poll(async () => (
            await panel.locator('.user-personas-card-name').first().innerText()
        ), { timeout: 10_000, intervals: [100, 250, 500] }).not.toBe(beforeText);
        await expect(panel.locator('.user-personas-card-name').first())
            .toContainText(`Call ${endpoints.rankCalls} alpha`);
    });
});
