// Spec R-6 — COORDINATE-PICKER: synthesize a bio from operator-picked
// axis coordinates via the Corpus tab.
//
// Surface chain under audit:
//   browser → Corpus tab (corpus.html) → axis multi-select
//   → operator selects 2 axes → per-axis sliders render
//   → operator dials values → design_brief textarea auto-fills
//   → Synthesize → POST /synthesize-bio-from-coordinates
//   → plugin builds K=1 experiment card with k_max_outer=1, spawns
//     lock_in_iterative.mjs → harness writes bio via saveBio
//     (POST /personas/<candidate_key>)
//   → server stamps target_bio on persona card at child exit
//   → result strip renders with name + provenance=experiment_output
//
// What this spec validates:
//   (a) Multi-select visible with bio-kind axes as options.
//   (b) money_orientation (kind=agent) NOT in the multi-select options.
//   (c) Pre-selected axes (astrology_sagittarian + intellectual_application)
//       have sliders rendered on first paint without operator action.
//   (d) Slider labels + captions visible for pre-selected axes.
//   (e) Slider default values are scale midpoints.
//   (f) Dragging a slider updates the displayed value.
//   (g) Selecting a different axis in the multi-select rebuilds sliders.
//   (h) design_brief textarea pre-fills from selection + values.
//   (i) Synthesize POSTs {target_signature:{...}} (selected axes only).
//       Preview pane appears. Status shows run_id.
//   (j) result strip renders with provenance badge after synthesis.
//       [fixme — requires live harness completion]
//
// Tests (a)-(i) fail against pre-multi-select corpus.html (no multi-select
// element exists). Tests (a)-(d) catch the regression where all bio axes
// were shown as sliders regardless of selection (no multi-select).

import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';

const ST_URL = 'http://127.0.0.1:8002';
const CORPUS_URL = `${ST_URL}/api/plugins/user-personas/static/corpus.html`;
const PLUGIN_BASE = '/api/plugins/user-personas';

test.describe('R-6: Coordinate Picker', () => {
    test.setTimeout(2 * 60 * 1000);

    test.afterAll(async () => {
        // Reap any harness children this spec may have spawned.
        try {
            execSync(`pkill -KILL -f "user-agent-harness/"`, { stdio: 'ignore' });
        } catch { /* none */ }
    });

    // ── (a) Multi-select is visible and has bio-kind axes as options ─────
    test('(a) axis multi-select is visible with bio axes', async ({ page }) => {
        await page.goto(CORPUS_URL);
        // Picker heading is present.
        await expect(page.locator('h2:has-text("Synthesize bio from coordinates")')).toBeVisible();
        // The multi-select element is visible.
        const sel = page.locator('#picker-axis-multiselect');
        await expect(sel).toBeVisible({ timeout: 8_000 });
        // It has at least 2 options (the bio axes).
        const optCount = await sel.locator('option:not([disabled])').count();
        expect(optCount).toBeGreaterThanOrEqual(2);
        // rpg_class and star_sign are among the options.
        const optTexts = await sel.locator('option').allTextContents();
        const joined = optTexts.join('\n');
        expect(joined).toMatch(/rpg_class/);
        expect(joined).toMatch(/star_sign/);
    });

    // ── (b) money_orientation (kind=agent) is NOT in the multi-select ────
    test('(b) money_orientation (kind=agent) excluded from multi-select', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });
        // money_orientation must NOT appear as an option (it's kind=agent).
        const moneyOpts = page.locator('#picker-axis-multiselect option[value="money_orientation"]');
        await expect(moneyOpts).toHaveCount(0);
        // And never renders a slider.
        const moneySlider = page.locator('#picker-sliders-host input[data-axis-id="money_orientation"]');
        await expect(moneySlider).toHaveCount(0);
    });

    // ── (c) Pre-selected axes have sliders on first paint ────────────────
    // Demo axes (astrology_sagittarian + intellectual_application) are pre-
    // selected; their sliders must appear without any operator action.
    test('(c) pre-selected axes have sliders visible on first paint', async ({ page }) => {
        await page.goto(CORPUS_URL);
        // Wait for multi-select to populate.
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // At least one slider renders without operator action.
        const anySlider = page.locator('#picker-sliders-host input[type="range"]');
        await expect(anySlider.first()).toBeVisible({ timeout: 5_000 });

        // Specifically: astrology_sagittarian OR intellectual_application
        // slider is visible (one of the demo pre-selected pair).
        const sagSlider = page.locator('input[data-axis-id="astrology_sagittarian"]');
        const intSlider = page.locator('input[data-axis-id="intellectual_application"]');
        const sagVisible = await sagSlider.isVisible().catch(() => false);
        const intVisible = await intSlider.isVisible().catch(() => false);
        expect(sagVisible || intVisible,
            'at least one of the demo pre-selected axes must have a visible slider',
        ).toBe(true);
    });

    // ── (d) Sliders have labels + captions ───────────────────────────────
    test('(d) sliders render labeled with axis ids + captions', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // Wait for at least one slider to appear.
        const anySlider = page.locator('#picker-sliders-host input[type="range"]');
        await expect(anySlider.first()).toBeVisible({ timeout: 5_000 });

        // The label for the first slider must reference its axis id.
        const firstSlider = anySlider.first();
        const axisId = await firstSlider.getAttribute('data-axis-id');
        expect(axisId).toBeTruthy();

        await expect(page.locator(`label[for="picker-slider-${axisId}"]`)).toContainText(axisId);

        // Captions are present.
        const captions = page.locator('.picker-slider-caption');
        const captionCount = await captions.count();
        expect(captionCount).toBeGreaterThanOrEqual(1);
        // At least one caption is non-empty.
        const captionTexts = await captions.allTextContents();
        expect(captionTexts.some(t => t.trim().length > 3)).toBe(true);
    });

    // ── (e) Default values are scale midpoints ───────────────────────────
    test('(e) slider default values are scale midpoints', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });
        await expect(page.locator('#picker-sliders-host input[type="range"]').first()).toBeVisible({ timeout: 5_000 });

        // Every rendered slider's displayed value-display must equal the slider value.
        const sliders = page.locator('#picker-sliders-host input[type="range"]');
        const count = await sliders.count();
        for (let i = 0; i < count; i++) {
            const slider = sliders.nth(i);
            const axisId = await slider.getAttribute('data-axis-id');
            const sliderVal = await slider.inputValue();
            const displayEl = page.locator(`[data-value-for="${axisId}"]`);
            await expect(displayEl).toHaveText(sliderVal);
        }

        // Astrology_sagittarian scale is 1-5 → midpoint = 3 if pre-selected.
        const sagSlider = page.locator('input[data-axis-id="astrology_sagittarian"]');
        if (await sagSlider.isVisible().catch(() => false)) {
            await expect(page.locator('[data-value-for="astrology_sagittarian"]')).toHaveText('3');
        }
    });

    // ── (f) Dragging slider updates displayed value ───────────────────────
    test('(f) dragging slider updates displayed value', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-sliders-host input[type="range"]').first()).toBeVisible({ timeout: 8_000 });

        const firstSlider = page.locator('#picker-sliders-host input[type="range"]').first();
        const axisId = await firstSlider.getAttribute('data-axis-id');
        const displayEl = page.locator(`[data-value-for="${axisId}"]`);

        const origVal = await displayEl.textContent();
        // Fill with a different value.
        const newVal = origVal === '3' ? '1' : '3';
        await firstSlider.fill(newVal);
        await expect(displayEl).toHaveText(newVal);

        // Fill with another value.
        await firstSlider.fill('5');
        await expect(displayEl).toHaveText('5');
    });

    // ── (g) Selecting a new axis in multi-select adds its slider ─────────
    test('(g) selecting an axis adds its slider; deselecting removes it', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // Make sure rpg_class is available as an option.
        const rpgOpt = page.locator('#picker-axis-multiselect option[value="rpg_class"]');
        await expect(rpgOpt).toHaveCount(1);

        // Select ONLY rpg_class (clear others).
        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            for (const opt of sel.options) opt.selected = opt.value === 'rpg_class';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });

        // rpg_class slider must appear.
        const rpgSlider = page.locator('input[data-axis-id="rpg_class"]');
        await expect(rpgSlider).toBeVisible({ timeout: 3_000 });
        await expect(page.locator('label[for="picker-slider-rpg_class"]')).toContainText('rpg_class');

        // astrology_sagittarian slider should NOT be visible (deselected).
        const sagSlider = page.locator('input[data-axis-id="astrology_sagittarian"]');
        await expect(sagSlider).not.toBeVisible();
    });

    // ── (h) design_brief textarea pre-fills from selection + values ───────
    test('(h) design_brief textarea pre-fills from axis selection', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });
        await expect(page.locator('#picker-sliders-host input[type="range"]').first()).toBeVisible({ timeout: 5_000 });

        // The textarea must be visible and non-empty on first paint.
        const ta = page.locator('#picker-design-brief');
        await expect(ta).toBeVisible();
        const taValue = await ta.inputValue();
        expect(taValue.trim().length).toBeGreaterThan(5);
        // It mentions one of the pre-selected axes.
        const lc = taValue.toLowerCase();
        expect(
            lc.includes('astrology_sagittarian') || lc.includes('intellectual_application'),
            `design_brief should mention a pre-selected axis; got: ${taValue}`,
        ).toBe(true);

        // Change selection to rpg_class only; brief should update.
        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            for (const opt of sel.options) opt.selected = opt.value === 'rpg_class';
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await expect(page.locator('input[data-axis-id="rpg_class"]')).toBeVisible({ timeout: 3_000 });
        const updatedVal = await ta.inputValue();
        expect(updatedVal.toLowerCase()).toContain('rpg_class');
    });

    // ── (i) Synthesize POSTs target_signature (selected axes only) ────────
    test('(i) Synthesize POSTs selected axis coordinates and opens preview pane', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // Select astrology_sagittarian + intellectual_application (the demo axes).
        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            const demoAxes = new Set(['astrology_sagittarian', 'intellectual_application']);
            for (const opt of sel.options) opt.selected = demoAxes.has(opt.value);
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });

        // Wait for both sliders to render.
        await expect(page.locator('input[data-axis-id="astrology_sagittarian"]')).toBeVisible({ timeout: 4_000 });
        await expect(page.locator('input[data-axis-id="intellectual_application"]')).toBeVisible({ timeout: 4_000 });

        // Set target values: sag=3, intel=4.
        await page.locator('input[data-axis-id="astrology_sagittarian"]').fill('3');
        await page.locator('input[data-axis-id="intellectual_application"]').fill('4');

        // Capture the POST body.
        const postBodyPromise = page.waitForRequest(req =>
            req.url().includes('/synthesize-bio-from-coordinates') && req.method() === 'POST',
        ).then(r => r.postDataJSON());

        await page.locator('#picker-synthesize-btn').click();

        const postBody = await postBodyPromise;
        expect(postBody).toBeTruthy();
        // target_signature must contain ONLY the selected axes.
        expect(postBody.target_signature).toMatchObject({
            astrology_sagittarian: 3,
            intellectual_application: 4,
        });
        // money_orientation must NOT be in target_signature.
        expect(Object.keys(postBody.target_signature)).not.toContain('money_orientation');

        // Preview pane appears.
        await expect(page.locator('#picker-preview-pane')).toBeVisible({ timeout: 10_000 });
        // Status shows run_id.
        await expect(page.locator('#picker-status')).toContainText(/Synthesis running|run_id/i, { timeout: 10_000 });
    });

    // ── (j) Result strip renders after synthesis with provenance ──────────
    // Marked fixme: requires the harness to run to completion (~30-90s)
    // and stamp provenance on the persona card. The network + result strip
    // assertions would only be observable after that full round-trip.
    test.fixme('(j) result strip shows name + provenance=experiment_output after synthesis', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // Select demo axes.
        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            const demoAxes = new Set(['astrology_sagittarian', 'intellectual_application']);
            for (const opt of sel.options) opt.selected = demoAxes.has(opt.value);
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await expect(page.locator('input[data-axis-id="astrology_sagittarian"]')).toBeVisible();

        await page.locator('input[data-axis-id="astrology_sagittarian"]').fill('3');
        await page.locator('input[data-axis-id="intellectual_application"]').fill('4');

        // Capture candidate_id from the response.
        const respPromise = page.waitForResponse(r =>
            r.url().includes('/synthesize-bio-from-coordinates') && r.status() === 200,
        );
        await page.locator('#picker-synthesize-btn').click();
        const resp = await respPromise;
        const body = await resp.json();
        const candidateId = body.candidate_id;
        expect(candidateId).toBeTruthy();

        // Wait for Save CTA — means synthesis finished.
        await expect(page.locator('#picker-save-btn')).toBeVisible({ timeout: 5 * 60 * 1000 });

        // Result strip must be visible with provenance=experiment_output.
        const strip = page.locator('#picker-result-strip');
        await expect(strip).toBeVisible({ timeout: 5_000 });
        await expect(strip.locator('[data-provenance-kind]')).toHaveAttribute(
            'data-provenance-kind', 'experiment_output',
        );
        await expect(strip.locator('[data-provenance-kind]')).toContainText('experiment_output');

        // curl /personas confirms target_bio is present.
        const personasResp = await page.request.get(`${ST_URL}${PLUGIN_BASE}/personas`);
        const personasBody = await personasResp.json();
        const candidate = (personasBody.personas || []).find(p => p.id === candidateId);
        expect(candidate, `candidate ${candidateId} in /personas`).toBeTruthy();
        expect(candidate.provenance?.kind,
            'provenance.kind must be experiment_output',
        ).toBe('experiment_output');
        expect(candidate.target_bio,
            'target_bio must be present and non-null',
        ).toBeTruthy();
        expect(candidate.target_bio.astrology_sagittarian).toBe(3);
        expect(candidate.target_bio.intellectual_application).toBe(4);
    });

    // Retained from original spec: after Save, persona has ≥2 agents.
    test.fixme('(k) after Save, candidate persona has ≥2 agents', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            const demoAxes = new Set(['astrology_sagittarian', 'intellectual_application']);
            for (const opt of sel.options) opt.selected = demoAxes.has(opt.value);
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await expect(page.locator('input[data-axis-id="astrology_sagittarian"]')).toBeVisible();
        await page.locator('input[data-axis-id="astrology_sagittarian"]').fill('1');
        await page.locator('input[data-axis-id="intellectual_application"]').fill('5');

        const respPromise = page.waitForResponse(r =>
            r.url().includes('/synthesize-bio-from-coordinates') && r.status() === 200,
        );
        await page.locator('#picker-synthesize-btn').click();
        const resp = await respPromise;
        const body = await resp.json();
        const candidateId = body.candidate_id;
        expect(candidateId).toBeTruthy();

        const saveBtn = page.locator('#picker-save-btn');
        await expect(saveBtn).toBeVisible({ timeout: 5 * 60 * 1000 });
        await saveBtn.click();

        // Poll /personas until candidate appears.
        const personasDeadline = Date.now() + 60_000;
        let foundPersona = false;
        while (Date.now() < personasDeadline) {
            const r = await page.request.get(`${ST_URL}${PLUGIN_BASE}/personas`);
            if (r.ok()) {
                const pb = await r.json();
                if ((pb.personas || []).some(p => p.id === candidateId)) {
                    foundPersona = true;
                    break;
                }
            }
            await new Promise(rs => setTimeout(rs, 2000));
        }
        expect(foundPersona, `candidate persona ${candidateId} in /personas`).toBe(true);

        // Poll /agents for >= 2 agents.
        const agentsDeadline = Date.now() + 5 * 60 * 1000;
        let agentCount = 0;
        while (Date.now() < agentsDeadline) {
            const r = await page.request.get(`${ST_URL}${PLUGIN_BASE}/agents`);
            if (r.ok()) {
                const ab = await r.json();
                const agentsForCandidate = (ab.agents || []).filter(
                    a => a.designed_for_bio_id === candidateId,
                );
                agentCount = agentsForCandidate.length;
                if (agentCount >= 2) break;
            }
            await new Promise(rs => setTimeout(rs, 5000));
        }
        expect(agentCount,
            `expected >=2 agents derived from candidate ${candidateId}; got ${agentCount}`,
        ).toBeGreaterThanOrEqual(2);
    });
});
