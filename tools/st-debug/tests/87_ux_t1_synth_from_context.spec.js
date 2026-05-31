/**
 * UX-T1 — Synthesize-from-context agent affordance
 *
 * Acceptance criteria (docs/ux_debt_followup_tickets_2026_05_21.md UX-T1
 * + docs/multi_user_agent_chat_interface_spec.md P-EMPTY-FORM):
 *
 *  A. Open Designer drawer — "Synthesize agent for this bio" button visible.
 *  B. Zero bare <input type="text"> elements in the agent-creation context
 *     (#pane-agent). ONLY <input type="range"> (axis sliders) allowed.
 *  C. Select a bio → guided affordance button becomes enabled + labels bio.
 *  D. Click → network panel shows POST /synthesize-agents-for-persona/<key>.
 *  E. K=3 candidate cards render with .candidate-agent, non-empty prose.
 *     (Poll-loop stub so the test doesn't wait for lock_in_iterative.)
 *  F. "Persist this one" button on each card; click → POST /agents/:id fires
 *     → button label changes to "Persisted ✓".
 *
 * This spec FAILS against the pre-UX-T1 state (no guided affordance, no
 * .candidate-agent class, Adopt-with-alert pattern). It locks in P-EMPTY-FORM
 * for the agent-creation surface permanently.
 *
 * Target: st-debug @ http://127.0.0.1:8002
 */

import { test, expect } from '@playwright/test';

const PLUGIN_BASE = '/api/plugins/user-personas';
// Use bios that exist in the st-debug fixture corpus (see tools/st-debug/_data).
// rpg-rogue-cancer.png has 2 agents on disk — suitable for the guided-affordance
// and persist tests (agents already exist, we add a mock 3rd to reach K=3).
const TEST_BIO_KEY   = 'rpg-rogue-cancer.png';
const TEST_BIO_NAME  = 'RPG Rogue Cancer';

// An agentless bio (Despotic Miscreant has 0 agents in the fixture corpus).
const AGENTLESS_BIO_KEY  = '1778631331275-DespoticMiscreant.png';

// ── helper ────────────────────────────────────────────────────────────────────
// Navigate directly to designer.html (outside ST shell for isolation).
// Stubs the /agents response so the bio list loads quickly and stub agents
// appear for the test bio immediately on the first poll.
async function openDesigner(page, { stubAgents } = {}) {
    // Route /agents BEFORE navigation so it's in place when boot() calls it.
    if (stubAgents) {
        await page.route(`${PLUGIN_BASE}/agents`, async (route) => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({ agents: stubAgents, count: stubAgents.length }),
                });
            } else {
                await route.continue();
            }
        });
    }

    const url = `http://127.0.0.1:8002${PLUGIN_BASE}/static/designer.html`;
    await page.goto(url, { waitUntil: 'networkidle' });
    // boot() populates #bio-list after /personas resolves.
    await page.waitForFunction(
        () => document.getElementById('bio-list')?.children.length > 0,
        { timeout: 15_000 }
    );
}

// Three mock agents for TEST_BIO_KEY — returned by the stub instead of the
// real harness output. Created_at is "now" so the synthesize poll loop
// picks them up (filter: created_at > t0 - 60_000).
function mockAgentsFor(bioKey) {
    const base = bioKey.replace(/\.png$/, '');
    const now = new Date().toISOString();
    return [
        {
            id: `${base}-money-1`,
            name: `${TEST_BIO_NAME} (acquisition-focus)`,
            designed_for_bio_id: bioKey,
            agent_text: 'This rogue acts with single-minded material focus, treating every interaction as a transaction to be won at any social cost.',
            injection_mode: 'authors_note',
            injection_depth: 1,
            signature: { money_orientation: 1 },
            created_at: now,
            agent_schema: 'agent-v1',
        },
        {
            id: `${base}-money-3`,
            name: `${TEST_BIO_NAME} (balanced)`,
            designed_for_bio_id: bioKey,
            agent_text: 'Balances pragmatic self-interest with genuine social engagement; opportunistic but capable of real warmth when it serves a purpose.',
            injection_mode: 'authors_note',
            injection_depth: 1,
            signature: { money_orientation: 3 },
            created_at: now,
            agent_schema: 'agent-v1',
        },
        {
            id: `${base}-money-5`,
            name: `${TEST_BIO_NAME} (charm-leveraged)`,
            designed_for_bio_id: bioKey,
            agent_text: 'Deploys charm and romantic energy as instrumental cover for material acquisition; seduction is a tool, not a genuine emotional state.',
            injection_mode: 'authors_note',
            injection_depth: 1,
            signature: { money_orientation: 5 },
            created_at: now,
            agent_schema: 'agent-v1',
        },
    ];
}

// ── tests ─────────────────────────────────────────────────────────────────────
test.describe('UX-T1: synthesize-from-context agent affordance (P-EMPTY-FORM)', () => {
    test.setTimeout(60_000);

    // Skip gracefully if st-debug is not running. Avoids confusing
    // ERR_CONNECTION_REFUSED errors when the server is between restarts.
    test.beforeEach(async ({ request }) => {
        let up = false;
        for (let i = 0; i < 3 && !up; i++) {
            const res = await request.get('http://127.0.0.1:8002/').catch(() => null);
            if (res?.ok()) up = true;
            else if (i < 2) await new Promise(r => setTimeout(r, 1500));
        }
        if (!up) test.skip(true, 'st-debug not reachable on port 8002 — run tools/st-debug/scripts/run.sh first');
    });

    // A. "Synthesize agent for this bio" button visible on first paint.
    // Fails against pre-UX-T1 state (affordance absent or labeled differently).
    test('A: "Synthesize agent for this bio" button visible on first paint', async ({ page }) => {
        await openDesigner(page);

        // The guided affordance panel must be visible immediately.
        const panel = page.locator('#synth-affordance-top');
        await expect(panel).toBeVisible();

        // The button must be present and carry the correct label.
        const btn = page.locator('#synth-affordance-btn');
        await expect(btn).toBeVisible();
        await expect(btn).toContainText('Synthesize agent for this bio');

        // On first paint (no bio selected) the button must be disabled.
        // This communicates the required flow without misleading the operator.
        await expect(btn).toBeDisabled();
    });

    // B. Zero bare <input type="text"> in the agent-creation context.
    // Fails against any state that introduces a bare text input in #pane-agent
    // or the top guided affordance panel — the canonical P-EMPTY-FORM violation.
    test('B: zero bare <input type="text"> in agent-creation context', async ({ page }) => {
        await openDesigner(page);

        // Check the Agent designer tab pane (the agent-creation surface).
        const agentPane = page.locator('#pane-agent');
        await expect(agentPane.locator('input[type="text"]')).toHaveCount(0);

        // Check the guided affordance panel at the top.
        const affordancePanel = page.locator('#synth-affordance-top');
        await expect(affordancePanel.locator('input[type="text"]')).toHaveCount(0);
    });

    // C. Selecting a bio enables the button and surfaces the bio name.
    // Fails if selectBio() doesn't enable the top affordance button or
    // doesn't update the label (regression against pre-UX-T1 state).
    test('C: selecting a bio enables button + shows bio name', async ({ page }) => {
        await openDesigner(page);

        const btn = page.locator('#synth-affordance-btn');
        await expect(btn).toBeDisabled();

        // Click the test bio in the list.
        const bioCard = page.locator(`#bio-list .bio-card[data-bio-key="${TEST_BIO_KEY}"]`);
        await expect(bioCard).toBeVisible({ timeout: 10_000 });
        await bioCard.click();

        // Button must now be enabled.
        await expect(btn).toBeEnabled({ timeout: 3_000 });

        // Bio label below the button must name the selected bio.
        const bioLabel = page.locator('#synth-bio-label');
        await expect(bioLabel).toContainText(TEST_BIO_NAME, { timeout: 3_000 });
    });

    // D. Click fires POST /synthesize-agents-for-persona/<key>.
    // Fails if the button is not wired to synthesize() or synthesize() calls
    // a different endpoint. The stub makes the POST return immediately.
    test('D: click fires POST /synthesize-agents-for-persona/<key>', async ({ page }) => {
        // Stub the synth endpoint to return immediately.
        const synthUrl = `${PLUGIN_BASE}/synthesize-agents-for-persona/${encodeURIComponent(TEST_BIO_KEY)}`;
        await page.route(synthUrl, async (route) => {
            if (route.request().method() === 'POST') {
                await route.fulfill({
                    status: 200,
                    contentType: 'application/json',
                    body: JSON.stringify({
                        ok: true,
                        run_id: 'test-ux-t1-d-00000',
                        experiment_id: 'synth-test-d-0000',
                        persona_key: TEST_BIO_KEY,
                        started_at: new Date().toISOString(),
                    }),
                });
            } else {
                await route.continue();
            }
        });

        await openDesigner(page);

        const bioCard = page.locator(`#bio-list .bio-card[data-bio-key="${TEST_BIO_KEY}"]`);
        await bioCard.click();
        await expect(page.locator('#synth-affordance-btn')).toBeEnabled();

        // Capture the POST request.
        const [request] = await Promise.all([
            page.waitForRequest(
                req => req.url().includes(encodeURIComponent(TEST_BIO_KEY)) &&
                       req.url().includes('synthesize-agents-for-persona') &&
                       req.method() === 'POST',
                { timeout: 5_000 }
            ),
            page.locator('#synth-affordance-btn').click(),
        ]);
        expect(request.method()).toBe('POST');
        expect(request.url()).toContain(`synthesize-agents-for-persona/${encodeURIComponent(TEST_BIO_KEY)}`);
    });

    // E. K=3 .candidate-agent cards render with non-empty prose.
    // Fails if:
    //   - candidate cards use a different class name (pre-UX-T1: candidate-card only)
    //   - the poll loop doesn't populate candidates
    //   - "Persist this one" button is absent (pre-UX-T1: "Adopt" label)
    test('E: K=3 .candidate-agent cards render with non-empty prose + "Persist this one" buttons', async ({ page }) => {
        const mocks = mockAgentsFor(TEST_BIO_KEY);

        // Stub synth → immediate success.
        await page.route(
            `${PLUGIN_BASE}/synthesize-agents-for-persona/${encodeURIComponent(TEST_BIO_KEY)}`,
            async (route) => {
                if (route.request().method() === 'POST') {
                    await route.fulfill({
                        status: 200, contentType: 'application/json',
                        body: JSON.stringify({ ok: true, run_id: 'test-ux-t1-e', experiment_id: 'synth-e', persona_key: TEST_BIO_KEY, started_at: new Date().toISOString() }),
                    });
                } else { await route.continue(); }
            }
        );

        // Stub /agents → return K=3 mocks immediately. This simulates harness
        // completion without waiting for lock_in_iterative (minutes).
        await page.route(`${PLUGIN_BASE}/agents`, async (route) => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200, contentType: 'application/json',
                    body: JSON.stringify({ agents: mocks, count: mocks.length }),
                });
            } else { await route.continue(); }
        });

        await openDesigner(page);

        const bioCard = page.locator(`#bio-list .bio-card[data-bio-key="${TEST_BIO_KEY}"]`);
        await bioCard.click();
        await page.locator('#synth-affordance-btn').click();

        // Poll loop fires every 4s. With stub it finds 3 on first tick.
        // Generous timeout: 15s to accommodate timer jitter.
        const candidateCards = page.locator('.candidate-agent');
        await expect(candidateCards).toHaveCount(3, { timeout: 15_000 });

        for (let i = 0; i < 3; i++) {
            // Each card must have .prose with content.
            const prose = candidateCards.nth(i).locator('.prose');
            const text = await prose.textContent();
            expect((text ?? '').trim().length).toBeGreaterThan(10);

            // Each card must have a "Persist this one" button (pre-UX-T1: absent or "Adopt").
            const persistBtn = candidateCards.nth(i).locator('.persist-btn');
            await expect(persistBtn).toBeVisible();
            await expect(persistBtn).toContainText('Persist this one');
        }
    });

    // F. "Persist this one" click → POST /agents/:id fires → button → "Persisted ✓".
    // Fails if persistCandidate() doesn't call POST /agents/:id, or if the
    // button label doesn't update (pre-UX-T1: alert() pattern).
    test('F: "Persist this one" fires POST /agents/:id + button shows "Persisted ✓"', async ({ page }) => {
        const mocks = mockAgentsFor(TEST_BIO_KEY);
        const persistId = mocks[1].id;  // balanced variant

        await page.route(
            `${PLUGIN_BASE}/synthesize-agents-for-persona/${encodeURIComponent(TEST_BIO_KEY)}`,
            async (route) => {
                if (route.request().method() === 'POST') {
                    await route.fulfill({
                        status: 200, contentType: 'application/json',
                        body: JSON.stringify({ ok: true, run_id: 'test-ux-t1-f', experiment_id: 'synth-f', persona_key: TEST_BIO_KEY, started_at: new Date().toISOString() }),
                    });
                } else { await route.continue(); }
            }
        );
        await page.route(`${PLUGIN_BASE}/agents`, async (route) => {
            if (route.request().method() === 'GET') {
                await route.fulfill({
                    status: 200, contentType: 'application/json',
                    body: JSON.stringify({ agents: mocks, count: mocks.length }),
                });
            } else { await route.continue(); }
        });

        // Capture and stub POST /agents/:id.
        let persistPostFired = false;
        await page.route(`${PLUGIN_BASE}/agents/${encodeURIComponent(persistId)}`, async (route) => {
            if (route.request().method() === 'POST') {
                persistPostFired = true;
                await route.fulfill({
                    status: 200, contentType: 'application/json',
                    body: JSON.stringify({ ok: true, agent: { id: persistId, ...mocks[1] } }),
                });
            } else { await route.continue(); }
        });

        await openDesigner(page);
        const bioCard = page.locator(`#bio-list .bio-card[data-bio-key="${TEST_BIO_KEY}"]`);
        await bioCard.click();
        await page.locator('#synth-affordance-btn').click();

        const candidateCards = page.locator('.candidate-agent');
        await expect(candidateCards).toHaveCount(3, { timeout: 15_000 });

        // Click "Persist this one" on the second card (balanced).
        const persistBtn = candidateCards.nth(1).locator('.persist-btn');
        await persistBtn.click();

        // POST must have fired.
        await expect.poll(() => persistPostFired, { timeout: 5_000 }).toBe(true);

        // Button label must change to "Persisted ✓".
        await expect(persistBtn).toContainText('Persisted ✓', { timeout: 5_000 });
    });

    // G. ?bio= deep-link: button enabled + status line visible immediately.
    // Fails if boot() doesn't call selectBio() for the ?bio= param, or if
    // the top affordance isn't updated on boot.
    test('G: ?bio= deep-link enables affordance button + fires auto-synth status', async ({ page }) => {
        // Route synth so auto-fire doesn't actually spawn the harness.
        await page.route(
            `${PLUGIN_BASE}/synthesize-agents-for-persona/${encodeURIComponent(AGENTLESS_BIO_KEY)}`,
            async (route) => {
                if (route.request().method() === 'POST') {
                    await route.fulfill({
                        status: 200, contentType: 'application/json',
                        body: JSON.stringify({ ok: true, run_id: 'test-ux-t1-g', experiment_id: 'synth-g', persona_key: AGENTLESS_BIO_KEY, started_at: new Date().toISOString() }),
                    });
                } else { await route.continue(); }
            }
        );

        const url = `http://127.0.0.1:8002${PLUGIN_BASE}/static/designer.html?bio=${encodeURIComponent(AGENTLESS_BIO_KEY)}`;
        await page.goto(url, { waitUntil: 'networkidle' });
        await page.waitForFunction(
            () => document.getElementById('bio-list')?.children.length > 0,
            { timeout: 15_000 }
        );

        // After boot() processes ?bio=, the top affordance button must be enabled.
        const btn = page.locator('#synth-affordance-btn');
        await expect(btn).toBeEnabled({ timeout: 5_000 });

        // The status line (either top or inline) must be visible —
        // auto-fire happened without operator click.
        const statusTop    = page.locator('#synth-status-top');
        const statusInline = page.locator('#synth-status');
        const topVisible    = await statusTop.isVisible();
        const inlineVisible = await statusInline.isVisible();
        expect(topVisible || inlineVisible).toBe(true);
    });
});
