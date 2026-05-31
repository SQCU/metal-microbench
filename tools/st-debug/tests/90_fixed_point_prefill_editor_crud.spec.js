// Fixed-point prefill + experiment editor CRUD/coherence coverage.
//
// Focuses the contract between the context-driven suggester and
// fixed_point.html:
//   - ?target_bio_signature opens a prefilled editor path
//   - editor controls are labelled/described, save/edit/delete cohere
//   - ?prefill_bio preserves visible context instead of showing a blank form

import { test, expect } from '@playwright/test';

const ST_URL = 'http://127.0.0.1:8002';
const PLUGIN_BASE = '/api/plugins/user-personas';
const FP_URL = `${ST_URL}${PLUGIN_BASE}/static/fixed_point.html`;
const TEST_EXP_ID = 'pw_fixed_point_prefill_crud';

test.describe('fixed_point.html prefill + editor CRUD', () => {
    test.setTimeout(90_000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'fixed-point editor modal coverage is desktop-scoped');
    });

    test.afterEach(async ({ request }) => {
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});
    });

    async function loadAxisFixture(request) {
        const resp = await request.get(`${PLUGIN_BASE}/axes`);
        expect(resp.ok(), `GET /axes must succeed, got ${resp.status()}`).toBe(true);
        const body = await resp.json();
        const axes = body.axes || [];
        const bioAxes = axes.filter(a => a.kind === 'bio' || a.kind === 'either').map(a => a.id);
        const agentAxes = axes.filter(a => a.kind === 'agent' || a.kind === 'either').map(a => a.id);
        expect(bioAxes.length, 'fixture needs at least one bio/either axis').toBeGreaterThan(0);
        expect(agentAxes.length, 'fixture needs at least one agent/either axis').toBeGreaterThan(0);
        return {
            bioAxis: bioAxes[0],
            secondBioAxis: bioAxes[1] || bioAxes[0],
            agentAxis: agentAxes[0],
        };
    }

    async function selectedValues(locator) {
        return await locator.evaluate(sel => [...sel.selectedOptions].map(o => o.value));
    }

    async function auditVisibleInputsHaveLabelsAndDescriptions(page) {
        return await page.locator('body').evaluate(() => {
            const failures = [];
            const inputs = [...document.querySelectorAll('#editor-overlay input, #editor-overlay textarea, #editor-overlay select')];

            function isVisible(el) {
                let cur = el;
                while (cur) {
                    if (cur.nodeType === 1) {
                        const cs = window.getComputedStyle(cur);
                        if (cur.hidden || cs.display === 'none' || cs.visibility === 'hidden') return false;
                    }
                    cur = cur.parentNode;
                }
                return true;
            }

            function cssPath(el) {
                if (el.id) return `#${el.id}`;
                const cls = typeof el.className === 'string' && el.className.trim()
                    ? `.${el.className.trim().split(/\s+/).slice(0, 2).join('.')}`
                    : '';
                return `${el.tagName.toLowerCase()}${cls}`;
            }

            function hasLabel(el) {
                if (el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)) return true;
                if ((el.getAttribute('aria-label') || '').trim()) return true;
                const previousLabels = [];
                let sib = el.previousElementSibling;
                while (sib && previousLabels.length < 2) {
                    if (sib.tagName?.toLowerCase() === 'label') previousLabels.push(sib);
                    sib = sib.previousElementSibling;
                }
                return previousLabels.some(l => l.textContent.trim().length > 0);
            }

            function hasDescription(el) {
                const container = el.closest('.form-row, .repeat-row, .stepper-cell, .loop-grid > div, .form-section');
                const descriptions = container ? [...container.querySelectorAll('.description')] : [];
                return descriptions.some(d => d.textContent.trim().length >= 20);
            }

            for (const input of inputs) {
                if (!isVisible(input)) continue;
                const labelled = hasLabel(input);
                const described = hasDescription(input);
                if (!labelled || !described) {
                    failures.push({
                        selector: cssPath(input),
                        problem: [
                            labelled ? null : 'missing label/aria-label',
                            described ? null : 'missing causal .description',
                        ].filter(Boolean).join(' and '),
                    });
                }
            }
            return failures;
        });
    }

    async function openTargetSignaturePrefill(page, signature) {
        const url = `${FP_URL}?target_bio_signature=${encodeURIComponent(JSON.stringify(signature))}`;
        await page.goto(url);
        await expect(page.locator('#editor-overlay')).toBeVisible({ timeout: 15_000 });
        await expect(page.locator('#editor-title')).toContainText(/new experiment/i);
    }

    test('?target_bio_signature pre-fills bio axes, target values, slug, and design brief', async ({ page, request }) => {
        const { bioAxis, secondBioAxis } = await loadAxisFixture(request);
        const signature = { [bioAxis]: 5, [secondBioAxis]: 2 };

        await openTargetSignaturePrefill(page, signature);

        const selectedBioAxes = await selectedValues(page.locator('#exp-bio-axes'));
        for (const axisId of Object.keys(signature)) {
            expect(selectedBioAxes, `bio axis ${axisId} should be selected from signature keys`).toContain(axisId);
            await expect(page.locator(`#bios-rows .bio-row .bio-target-wrap input[data-axis-id="${axisId}"]`))
                .toHaveValue(String(signature[axisId]));
        }

        await expect(page.locator('#bios-rows .bio-row .bio-slug')).toHaveValue(/^from-chat-context-/);
        await expect(page.locator('#bios-rows .bio-row .bio-design-brief'))
            .toHaveValue(/Synthesized from chat context/i);
        await expect(page.locator('#agent-targets-rows .agent-row'),
            'context prefill should not invent agent_targets').toHaveCount(0);
    });

    test('?prefill_bio preserves visible prefill context and does not expose an empty editor', async ({ page }) => {
        const bioId = 'user-personas-rpg-wizard-sagittarius.png';
        await page.goto(`${FP_URL}?prefill_bio=${encodeURIComponent(bioId)}`);
        await page.waitForLoadState('networkidle', { timeout: 15_000 }).catch(() => {});

        await expect.poll(async () => {
            const uiResidue = await page.locator([
                '#editor-overlay:visible',
                '#run-banner:visible',
                '#seed-warnings:visible',
                `[data-prefill-bio="${bioId}"]`,
            ].join(', ')).count();
            const textResidue = await page.locator('body').evaluate((body, id) => {
                const text = body.textContent || '';
                return text.includes(id) || /prefill|preload|from bio/i.test(text) ? 1 : 0;
            }, bioId);
            return uiResidue + textResidue;
        }, {
            message: '?prefill_bio must open a run/editor path or leave visible prefill residue; ignoring the param regresses the suggester handoff',
            timeout: 5_000,
        }).toBeGreaterThan(0);

        const emptyEditorVisible = await page.locator('#editor-overlay:visible').count() > 0
            && await page.locator('#exp-id').inputValue().then(v => v.trim() === '').catch(() => false)
            && await page.locator('#bios-rows .bio-row .bio-design-brief').inputValue().then(v => v.trim() === '').catch(() => false);
        expect(emptyEditorVisible, '?prefill_bio must not open an empty bare editor').toBe(false);
    });

    test('prefilled editor is labelled, saves via POST, refreshes list, edits immutables, and confirms delete', async ({ page, request }) => {
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});
        const { bioAxis, agentAxis } = await loadAxisFixture(request);
        await openTargetSignaturePrefill(page, { [bioAxis]: 4 });

        await page.locator('#exp-id').fill(TEST_EXP_ID);
        await page.locator('#exp-name').fill('Playwright fixed-point prefill CRUD');
        await page.locator('#exp-description').fill('Created by 90_fixed_point_prefill_editor_crud.spec.js');
        await page.locator('#bios-rows .bio-row .bio-name').fill('Prefill CRUD Bio');
        await page.locator('#exp-agent-axes').selectOption([agentAxis]);
        await page.locator('#add-agent-btn').click();
        await page.locator('#agent-targets-rows .agent-row .agent-slug').fill('prefill-crud-agent');
        await page.locator(`#agent-targets-rows .agent-row .agent-target-wrap input[data-axis-id="${agentAxis}"]`).fill('3');
        await page.locator('#agent-targets-rows .agent-row .agent-motive-hint').fill('A deterministic probe agent target for editor CRUD coverage.');
        await page.locator('#exp-counterparty').fill('the-rock.png');

        const labelFailures = await auditVisibleInputsHaveLabelsAndDescriptions(page);
        expect(labelFailures,
            `visible editor inputs must have labels and causal descriptions:\n${JSON.stringify(labelFailures, null, 2)}`
        ).toEqual([]);

        const saveResp = page.waitForResponse(resp =>
            resp.url().endsWith(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`)
            && resp.request().method() === 'POST'
        );
        await page.locator('#save-experiment-btn').click();
        expect((await saveResp).ok(), 'Save must POST /experiments/:id successfully').toBe(true);
        await expect(page.locator('#editor-overlay')).toBeHidden({ timeout: 10_000 });
        await expect(page.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"]`),
            'saved card must appear after list refresh').toBeVisible({ timeout: 10_000 });

        const savedResp = await request.get(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`);
        expect(savedResp.status()).toBe(200);
        const saved = await savedResp.json();
        expect(saved.id).toBe(TEST_EXP_ID);
        expect(saved.bio_axes).toContain(bioAxis);
        expect(saved.agent_axes).toContain(agentAxis);
        expect(saved.bios[0].target_bio[bioAxis]).toBe(4);
        expect(saved.bios[0].design_brief).toMatch(/Synthesized from chat context/i);

        await page.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"] .exp-edit-trigger`).click();
        await expect(page.locator('#editor-overlay')).toBeVisible();
        await expect(page.locator('#exp-id')).toBeDisabled();
        await expect(page.locator('#exp-id')).toHaveAttribute('title', /canonical key|rename/i);
        await expect(page.locator('#delete-experiment-btn')).toBeVisible();

        let sawDeleteConfirm = false;
        page.once('dialog', async dialog => {
            sawDeleteConfirm = /delete experiment/i.test(dialog.message());
            await dialog.dismiss();
        });
        await page.locator('#delete-experiment-btn').click();
        expect(sawDeleteConfirm, 'Delete must require a confirmation dialog').toBe(true);
        await expect(page.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"]`)).toBeVisible();

        page.once('dialog', dialog => dialog.accept());
        await page.locator('#delete-experiment-btn').click();
        await expect(page.locator('#editor-overlay')).toBeHidden({ timeout: 10_000 });
        await expect(page.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"]`)).toHaveCount(0, { timeout: 10_000 });

        const afterDelete = await request.get(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`);
        expect(afterDelete.status()).toBe(404);
    });
});
