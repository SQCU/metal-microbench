/**
 * Selection-Is-Design coverage from actual UI paths.
 *
 * Locks in P-SELECTION-IS-DESIGN / P-ONTOLOGICAL-CLOSURE / Designer + UX-T1:
 *  - agentless ranked rows visibly route to Designer instead of ending at raw bio use
 *  - designer.html?bio=<id> auto-fires synthesis with immediate status
 *  - usable bio x agent rows POST /poll with both persona_id and agent_id
 *  - Designer agent-creation context has no bare manual JSON/text fields
 *
 * Stubs endpoint responses so this spec does not wait for lock_in_iterative.
 */

import { test, expect } from '@playwright/test';

const ST_URL = 'http://127.0.0.1:8002';
const PLUGIN_BASE = '/api/plugins/user-personas';
const DESIGNER_SURFACE = '#user-personas-surface-designer';
const SUGGESTER_SURFACE = '#user-suggester-button';
const TOOLS_BUTTON = '#user-personas-tools-button .drawer-toggle';

const AGENTLESS_BIO_ID = 'selection-design-agentless.png';
const AGENTLESS_BIO_NAME = 'Selection Design Agentless';
const EXISTING_BIO_ID = 'selection-design-usable.png';
const EXISTING_BIO_NAME = 'Selection Design Usable';
const EXISTING_AGENT_ID = 'selection-design-usable-agent';
const EXISTING_AGENT_NAME = 'Usable Selection Overlay';

const BIOS = [
    {
        id: AGENTLESS_BIO_ID,
        name: AGENTLESS_BIO_NAME,
        description: 'An agentless persona with enough context to synthesize overlays from signature.',
        signature: { rpg_class: 2, star_sign: 4, money_orientation: 3 },
        provenance: { kind: 'canonical' },
    },
    {
        id: EXISTING_BIO_ID,
        name: EXISTING_BIO_NAME,
        description: 'A persona that already has a designed agent and can be used immediately.',
        signature: { rpg_class: 4, star_sign: 2, money_orientation: 5 },
        provenance: { kind: 'canonical' },
    },
];

const EXISTING_AGENT = {
    id: EXISTING_AGENT_ID,
    name: EXISTING_AGENT_NAME,
    designed_for_bio_id: EXISTING_BIO_ID,
    agent_text: 'Responds as a complete bio x agent composition with assertive, high-context residue.',
    injection_mode: 'authors_note',
    injection_depth: 1,
    signature: { rpg_class: 4, star_sign: 2, money_orientation: 5 },
    created_at: new Date().toISOString(),
    provenance: { kind: 'canonical' },
};

function synthesizedAgentsForAgentless() {
    const now = new Date().toISOString();
    return [1, 3, 5].map(v => ({
        id: `selection-design-agentless-money-${v}`,
        name: `${AGENTLESS_BIO_NAME} money ${v}`,
        designed_for_bio_id: AGENTLESS_BIO_ID,
        agent_text: `Candidate overlay ${v} synthesized from the agentless bio signature without manual JSON entry.`,
        injection_mode: 'authors_note',
        injection_depth: 1,
        signature: { money_orientation: v },
        created_at: now,
        provenance: { kind: 'experiment_output' },
    }));
}

async function requireStDebug(request) {
    const res = await request.get(ST_URL).catch(() => null);
    if (!res?.ok()) {
        test.skip(true, 'st-debug not reachable on :8002; run tools/st-debug/scripts/run.sh first');
    }
}

async function installFakeActiveChat(page) {
    await page.evaluate(() => {
        const fake = {
            chat: [
                { is_user: true, name: 'operator', mes: 'I need a high-signal participant for this scene.' },
                { is_user: false, name: 'assistant', mes: 'Pick a persona and overlay that can act immediately.' },
            ],
            characterId: 0,
            chatId: 'selection-is-design-e2e',
            characters: [{ avatar: 'selection-test.png', name: 'Selection Test' }],
        };
        const st = window.SillyTavern || {};
        const original = typeof st.getContext === 'function' ? st.getContext.bind(st) : null;
        st.getContext = () => ({ ...(original ? original() : {}), ...fake });
        window.SillyTavern = st;
    });
}

async function openToolsPopover(page) {
    const toolsBtn = page.locator(TOOLS_BUTTON);
    await expect(toolsBtn).toBeVisible({ timeout: 20_000 });
    await toolsBtn.click();
    await page.waitForSelector('#UserPersonasToolsMenu.openDrawer', { timeout: 5_000 });
}

async function openSurface(page, key, surfaceSelector) {
    await openToolsPopover(page);
    const item = page.locator(`.user-personas-tools-menuitem[data-surface-key="${key}"]`);
    await expect(item).toBeVisible({ timeout: 5_000 });
    await item.click();
    const frame = page.frameLocator(`${surfaceSelector} iframe`);
    await expect(frame.locator('body')).toBeVisible({ timeout: 20_000 });
    return frame;
}

async function openStWithSurfaces(page) {
    await page.goto(ST_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForSelector('body', { timeout: 10_000 });
    await installFakeActiveChat(page);

    const designer = await openSurface(page, 'designer', DESIGNER_SURFACE);
    await expect(page.locator(`${DESIGNER_SURFACE} iframe`)).toHaveAttribute('src', /designer\.html/);

    const suggester = await openSurface(page, 'suggester', SUGGESTER_SURFACE);
    await expect(page.locator(`${SUGGESTER_SURFACE} iframe`)).toHaveAttribute('src', /suggester\.html/);
    return { designer, suggester };
}

async function stubSharedCorpus(page, { getAgents = null } = {}) {
    await page.route(`**${PLUGIN_BASE}/personas`, async route => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ personas: BIOS, count: BIOS.length }),
        });
    });

    await page.route(`**${PLUGIN_BASE}/agents`, async route => {
        const agents = getAgents ? getAgents() : [EXISTING_AGENT];
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ agents, count: agents.length }),
        });
    });

    await page.route(`**${PLUGIN_BASE}/axes`, async route => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                axes: [
                    { id: 'rpg_class', name: 'RPG class', kind: 'bio', scale_min: 1, scale_max: 5, def: 'class posture' },
                    { id: 'star_sign', name: 'Star sign', kind: 'bio', scale_min: 1, scale_max: 5, def: 'astrological tone' },
                    { id: 'money_orientation', name: 'Money orientation', kind: 'either', scale_min: 1, scale_max: 5, def: 'material focus' },
                ],
                count: 3,
            }),
        });
    });

    await page.route(`**${PLUGIN_BASE}/bridge-status`, async route => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({ ok: true, bridge_up: true }),
        });
    });
}

async function stubRank(page, rows) {
    await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                top: rows,
                side: [],
                _meta: {
                    target_signature: { rpg_class: 3, star_sign: 3, money_orientation: 3 },
                    target_completed_axes: 3,
                    candidates_considered: rows.length,
                    bios_total: BIOS.length,
                    agents_total: 1,
                    K_top: 3,
                    K_side: 3,
                },
            }),
        });
    });
}

function agentlessRankRow() {
    return {
        bio_id: AGENTLESS_BIO_ID,
        agent_id: null,
        persona: BIOS[0],
        agent: null,
        distance: 0.42,
        why: 'Agentless bio is a pending design target and must route to Designer.',
    };
}

function existingCompositionRankRow() {
    return {
        bio_id: EXISTING_BIO_ID,
        agent_id: EXISTING_AGENT_ID,
        persona: BIOS[1],
        agent: EXISTING_AGENT,
        distance: 0.31,
        why: 'Existing bio x agent composition is immediately usable.',
    };
}

test.describe('Selection-Is-Design: Suggester to Designer', () => {
    test.setTimeout(90_000);

    test.beforeEach(async ({ request }) => {
        await requireStDebug(request);
    });

    test('agentless suggester row opens Designer ?bio= and auto-fires synthesis status', async ({ page }) => {
        let synthPost = null;
        await stubSharedCorpus(page);
        await stubRank(page, [agentlessRankRow()]);
        await page.route(`**${PLUGIN_BASE}/poll`, async route => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    candidates: [{ text: 'Agentless rows should prefer the Designer route over raw bio-only use.' }],
                    applied_overlay: null,
                }),
            });
        });
        await page.route(`**${PLUGIN_BASE}/synthesize-agents-for-persona/${encodeURIComponent(AGENTLESS_BIO_ID)}`, async route => {
            if (route.request().method() === 'POST') {
                synthPost = route.request();
                await route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({
                        ok: true,
                        run_id: 'selection-is-design-agentless',
                        experiment_id: 'selection-is-design-agentless',
                        persona_key: AGENTLESS_BIO_ID,
                        started_at: new Date().toISOString(),
                    }),
                });
                return;
            }
            await route.continue();
        });

        const { designer, suggester } = await openStWithSurfaces(page);

        const row = suggester.locator(`.ranked-row[data-bio-id="${AGENTLESS_BIO_ID}"]`);
        await expect(row).toBeVisible({ timeout: 20_000 });
        await expect(row.locator('.ranked-ids')).not.toContainText(EXISTING_AGENT_ID);

        const designCta = row.locator('.bio-without-agent-redirect');
        await expect(designCta).toBeVisible();
        await expect(designCta).toContainText(/Design agent for this bio/);
        await expect(designCta).toHaveAttribute('title', /Designer|auto-fire|synthesis/i);
        await expect(row.locator('a')).toHaveCount(0);

        await designCta.click();

        const designerIframe = page.locator(`${DESIGNER_SURFACE} iframe`);
        await expect.poll(async () => await designerIframe.getAttribute('src') || '', {
            timeout: 5_000,
            intervals: [250, 500, 1000],
        }).toContain(`bio=${encodeURIComponent(AGENTLESS_BIO_ID)}`);

        await expect(designer.locator('#agent-bio-name')).toContainText(AGENTLESS_BIO_NAME, { timeout: 10_000 });
        await expect(designer.locator('#bio-context-prose')).toContainText('agentless persona', { timeout: 10_000 });

        const status = designer.locator('#synth-status, #synth-status-top').filter({ hasText: /Synthesizing K=3|POST \/synthesize|Synth dispatched|Polling \/agents/ }).first();
        await expect(status).toBeVisible({ timeout: 10_000 });
        await expect.poll(() => Boolean(synthPost), { timeout: 5_000 }).toBe(true);
        expect(synthPost.url()).toContain(`/synthesize-agents-for-persona/${encodeURIComponent(AGENTLESS_BIO_ID)}`);

        const agentCreationContext = designer.locator('#synth-affordance-top, #pane-agent');
        await expect(agentCreationContext.locator('input[type="text"], textarea')).toHaveCount(0);
    });

    test('designer deep-link auto-fire renders synthesized candidates without manual fields', async ({ page }) => {
        let postCount = 0;
        await stubSharedCorpus(page, {
            getAgents: () => postCount > 0
                ? [EXISTING_AGENT, ...synthesizedAgentsForAgentless()]
                : [EXISTING_AGENT],
        });
        await page.route(`**${PLUGIN_BASE}/synthesize-agents-for-persona/${encodeURIComponent(AGENTLESS_BIO_ID)}`, async route => {
            if (route.request().method() === 'POST') {
                postCount += 1;
                await route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({
                        ok: true,
                        run_id: 'selection-is-design-deeplink',
                        experiment_id: 'selection-is-design-deeplink',
                        persona_key: AGENTLESS_BIO_ID,
                        started_at: new Date().toISOString(),
                    }),
                });
                return;
            }
            await route.continue();
        });

        await page.goto(`${ST_URL}${PLUGIN_BASE}/static/designer.html?bio=${encodeURIComponent(AGENTLESS_BIO_ID)}`, {
            waitUntil: 'domcontentloaded',
        });

        await expect(page.locator('#agent-bio-name')).toContainText(AGENTLESS_BIO_NAME, { timeout: 10_000 });
        await expect(page.locator('#synth-status, #synth-status-top').filter({ hasText: /Synthesizing K=3|POST \/synthesize|Synth dispatched|Polling \/agents/ }).first())
            .toBeVisible({ timeout: 10_000 });
        await expect.poll(() => postCount, { timeout: 5_000 }).toBe(1);

        const cards = page.locator('.candidate-agent');
        await expect(cards).toHaveCount(3, { timeout: 15_000 });
        for (let i = 0; i < 3; i++) {
            await expect(cards.nth(i).locator('.prose')).toContainText(/Candidate overlay/);
            await expect(cards.nth(i).locator('.persist-btn')).toContainText('Persist this one');
        }

        await expect(page.locator('#synth-affordance-top, #pane-agent').locator('input[type="text"], textarea')).toHaveCount(0);
        await expect(page.locator('#pane-bio input[type="range"]')).toHaveCount(3);
    });

    test('existing-agent composition posts persona_id and agent_id and leaves inline residue', async ({ page }) => {
        let pollPayload = null;
        await stubSharedCorpus(page);
        await stubRank(page, [existingCompositionRankRow()]);
        await page.route(`**${PLUGIN_BASE}/poll`, async route => {
            pollPayload = route.request().postDataJSON();
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    candidates: [{
                        text: 'Inline residue from the complete bio x agent composition, suitable for operator supervision.',
                        truncated: false,
                    }],
                    applied_overlay: {
                        source: 'agent',
                        agent_id: EXISTING_AGENT_ID,
                        name: EXISTING_AGENT_NAME,
                        depth: 1,
                        text_chars: EXISTING_AGENT.agent_text.length,
                    },
                }),
            });
        });

        const { suggester } = await openStWithSurfaces(page);

        const row = suggester.locator(`.ranked-row[data-bio-id="${EXISTING_BIO_ID}"][data-agent-id="${EXISTING_AGENT_ID}"]`);
        await expect(row).toBeVisible({ timeout: 20_000 });
        await expect(row.locator('.ranked-name')).toContainText(EXISTING_BIO_NAME);
        await expect(row.locator('.ranked-name')).toContainText(EXISTING_AGENT_NAME);
        await expect(row.locator('.bio-without-agent-redirect')).toHaveCount(0);

        await row.locator('.suggest-btn').click();

        await expect.poll(() => pollPayload, { timeout: 5_000 }).not.toBeNull();
        expect(pollPayload.persona_id).toBe(EXISTING_BIO_ID);
        expect(pollPayload.agent_id).toBe(EXISTING_AGENT_ID);
        expect(Array.isArray(pollPayload.chat)).toBe(true);
        expect(pollPayload.chat.length).toBeGreaterThan(0);

        await expect(row.locator('.row-completion')).toHaveClass(/visible/, { timeout: 10_000 });
        await expect(row.locator('.row-completion-head')).toContainText(EXISTING_AGENT_NAME);
        await expect(row.locator('.row-completion-text')).toContainText(/Inline residue/);
    });
});
