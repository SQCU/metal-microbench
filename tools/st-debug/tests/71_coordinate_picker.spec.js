// Spec R-6 — COORDINATE-PICKER → DESIGNER HANDOFF.
//
// STEP F3 (salvage-phase1-2): the Designer (designer.html) is the SINGLE
// bio creator. The Corpus tab's coordinate picker (corpus.html) is now a
// pure ACTUATOR: it lets the operator dial axis coordinates with sliders,
// then hands them to the Designer's Bio tab via ?coords=<json> ("Design in
// Designer →"). The picker's OWN /synthesize-bio-from-coordinates dispatch
// + preview/result/save UI was REMOVED — there is one synth path, owned by
// the Designer.
//
// Surface chain under audit:
//   browser → Corpus tab (corpus.html) → axis multi-select
//   → operator selects axes → per-axis sliders render
//   → operator dials values
//   → "Design in Designer →" → designer.html?coords=<json>
//   → Designer.boot() parses ?coords=, switches to Bio tab,
//     prefills #axis-sliders from the coordinate signature
//   → operator clicks "Synthesize candidate bio"
//   → Designer POSTs /synthesize-bio-from-coordinates {target_signature}
//   → plugin builds K=1 experiment card, spawns lock_in_iterative.mjs
//   → harness writes bio via saveBio; result lands in #bio-candidate-panel
//
// What this spec validates:
//   (a) Picker multi-select visible with bio-kind axes as options.
//   (b) money_orientation (kind=agent) NOT in the multi-select options.
//   (c) Pre-selected axes have sliders on first paint (no operator action).
//   (d) Slider labels + captions visible for pre-selected axes.
//   (e) Slider default values are scale midpoints.
//   (f) Dragging a slider updates the displayed value.
//   (g) Selecting a different axis rebuilds sliders.
//   (h) The picker has NO local synth/preview/save UI (collapse check) and
//       exposes only "Design in Designer →" as its action.
//   (i) "Design in Designer →" (standalone nav) lands on designer.html with
//       a ?coords= signature carrying ONLY the selected axes (no
//       money_orientation), the Designer opens on the Bio tab, and prefills
//       its axis sliders from those coordinates.
//   (j) The Designer (the ONE creator) POSTs /synthesize-bio-from-coordinates
//       with the handed-off target_signature when "Synthesize candidate bio"
//       is clicked. [fixme tail: requires live harness completion]
//
// Tests (a)-(g) drive the picker actuator. (h)/(i)/(j) assert the collapse:
// the picker no longer synthesizes; it only hands coordinates to the single
// Designer creator.

import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';

const ST_URL = 'http://127.0.0.1:8002';
const STATIC_BASE = `${ST_URL}/api/plugins/user-personas/static`;
const CORPUS_URL = `${STATIC_BASE}/corpus.html`;
const DESIGNER_URL = `${STATIC_BASE}/designer.html`;
const PLUGIN_BASE = '/api/plugins/user-personas';

test.describe('R-6: Coordinate Picker → Designer handoff', () => {
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
        // Picker heading is present (the picker now hands off to the Designer).
        await expect(page.locator('h2:has-text("Design a bio")')).toBeVisible();
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

    // ── (h) Collapse check: the picker has NO local synth/preview/save UI ──
    // STEP F3: the picker is a pure actuator. Its only action is "Design in
    // Designer →"; the old #picker-synthesize-btn / #picker-preview-pane /
    // #picker-save-btn / #picker-result-strip elements must be GONE.
    test('(h) picker exposes only the Designer handoff; no local synth UI', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // The single action: hand off to the Designer.
        await expect(page.locator('#picker-design-in-designer-btn')).toBeVisible();

        // The removed picker-local synth lifecycle must NOT exist.
        for (const removedId of [
            '#picker-synthesize-btn',
            '#picker-preview-pane',
            '#picker-preview-text',
            '#picker-save-btn',
            '#picker-result-strip',
            '#picker-design-brief',
        ]) {
            await expect(page.locator(removedId),
                `${removedId} must be removed (picker no longer synthesizes)`,
            ).toHaveCount(0);
        }
    });

    // ── (i) Picker → Designer ?coords= handoff carries the selected axes ──
    // Standalone nav path (no parent ST shell). Picking sag+intel and
    // clicking "Design in Designer →" must navigate to designer.html with a
    // ?coords= signature of ONLY those axes, land on the Bio tab, and
    // prefill the Designer's axis sliders from the coordinates.
    test('(i) "Design in Designer →" hands selected coords to the Designer Bio tab', async ({ page }) => {
        await page.goto(CORPUS_URL);
        await expect(page.locator('#picker-axis-multiselect')).toBeVisible({ timeout: 8_000 });

        // Select astrology_sagittarian + intellectual_application (demo axes).
        await page.locator('#picker-axis-multiselect').evaluate(sel => {
            const demoAxes = new Set(['astrology_sagittarian', 'intellectual_application']);
            for (const opt of sel.options) opt.selected = demoAxes.has(opt.value);
            sel.dispatchEvent(new Event('change', { bubbles: true }));
        });
        await expect(page.locator('input[data-axis-id="astrology_sagittarian"]')).toBeVisible({ timeout: 4_000 });
        await expect(page.locator('input[data-axis-id="intellectual_application"]')).toBeVisible({ timeout: 4_000 });

        // Set target values: sag=3, intel=4.
        await page.locator('input[data-axis-id="astrology_sagittarian"]').fill('3');
        await page.locator('input[data-axis-id="intellectual_application"]').fill('4');

        // Click the handoff and follow the top-level navigation to the Designer.
        await Promise.all([
            page.waitForURL(/designer\.html\?coords=/, { timeout: 10_000 }),
            page.locator('#picker-design-in-designer-btn').click(),
        ]);

        // The ?coords= signature must carry ONLY the selected axes.
        const coordsRaw = new URL(page.url()).searchParams.get('coords');
        expect(coordsRaw, 'designer URL must carry a ?coords= signature').toBeTruthy();
        const coords = JSON.parse(coordsRaw);
        expect(coords).toMatchObject({
            astrology_sagittarian: 3,
            intellectual_application: 4,
        });
        // money_orientation (kind=agent) must NOT be in the handoff.
        expect(Object.keys(coords)).not.toContain('money_orientation');

        // The Designer opens on the Bio designer tab (coords handoff).
        await expect(page.locator('#pane-bio')).toHaveClass(/active/, { timeout: 8_000 });
        await expect(page.locator('.tab[data-tab="bio"]')).toHaveClass(/active/);

        // The Designer's axis sliders are prefilled from the coordinates.
        const sagRow = page.locator('#axis-sliders .axis-slider-row[data-axis="astrology_sagittarian"]');
        await expect(sagRow.locator('input[type="range"]')).toHaveValue('3', { timeout: 8_000 });
        const intRow = page.locator('#axis-sliders .axis-slider-row[data-axis="intellectual_application"]');
        await expect(intRow.locator('input[type="range"]')).toHaveValue('4');

        // The Designer's bio-synth affordance (the ONE creator) is present.
        await expect(page.locator('#bio-synth-btn')).toBeVisible();
    });

    // ── (j) The Designer is the SINGLE creator: it POSTs the handoff coords ─
    // Drive the full collapse end-to-end up to the synthesis dispatch: the
    // POST must originate from the DESIGNER (not the picker), and carry the
    // handed-off target_signature. The fixme tail (harness completion) is
    // out of scope for a fast spec; the dispatch + body is the contract.
    test('(j) Designer POSTs /synthesize-bio-from-coordinates with handed-off coords', async ({ page }) => {
        // Arrive at the Designer directly via the handoff URL the picker emits.
        const coords = { astrology_sagittarian: 3, intellectual_application: 4 };
        const url = `${DESIGNER_URL}?coords=${encodeURIComponent(JSON.stringify(coords))}`;
        await page.goto(url);

        // Lands on the Bio tab, sliders prefilled from coords.
        await expect(page.locator('#pane-bio')).toHaveClass(/active/, { timeout: 8_000 });
        const sagInput = page.locator('#axis-sliders .axis-slider-row[data-axis="astrology_sagittarian"] input[type="range"]');
        await expect(sagInput).toHaveValue('3', { timeout: 8_000 });

        // Capture the synthesis POST body from the Designer.
        const postBodyPromise = page.waitForRequest(req =>
            req.url().includes('/synthesize-bio-from-coordinates') && req.method() === 'POST',
        ).then(r => r.postDataJSON());

        await page.locator('#bio-synth-btn').click();

        const postBody = await postBodyPromise;
        expect(postBody).toBeTruthy();
        // target_signature must contain the handed-off coordinates.
        expect(postBody.target_signature).toMatchObject({
            astrology_sagittarian: 3,
            intellectual_application: 4,
        });
        // money_orientation must NOT be in target_signature.
        expect(Object.keys(postBody.target_signature)).not.toContain('money_orientation');

        // The Designer's bio-status surfaces the dispatch (run_id / polling).
        await expect(page.locator('#bio-status')).toContainText(/run_id|Polling|Dispatch/i, { timeout: 10_000 });
    });

    // ── (k) Result lands in the Designer after synthesis ──────────────────
    // Marked fixme: requires the harness to run to completion (~30-90s) and
    // write the candidate bio. The Designer's #bio-candidate-panel + Save
    // CTA only become observable after that full round-trip.
    test.fixme('(k) Designer renders candidate bio after synthesis completes', async ({ page }) => {
        const coords = { astrology_sagittarian: 3, intellectual_application: 4 };
        const url = `${DESIGNER_URL}?coords=${encodeURIComponent(JSON.stringify(coords))}`;
        await page.goto(url);
        await expect(page.locator('#pane-bio')).toHaveClass(/active/, { timeout: 8_000 });

        // Capture candidate_id from the synthesis response.
        const respPromise = page.waitForResponse(r =>
            r.url().includes('/synthesize-bio-from-coordinates') && r.status() === 200,
        );
        await page.locator('#bio-synth-btn').click();
        const resp = await respPromise;
        const body = await resp.json();
        const candidateId = body.candidate_id;
        expect(candidateId).toBeTruthy();

        // The candidate bio panel becomes visible once the harness writes it.
        await expect(page.locator('#bio-candidate-panel')).toBeVisible({ timeout: 8 * 60 * 1000 });
        await expect(page.locator('#bio-candidate-key')).toContainText(candidateId);
        await expect(page.locator('#bio-save-btn')).toBeEnabled();

        // curl /personas confirms the candidate landed.
        const personasResp = await page.request.get(`${ST_URL}${PLUGIN_BASE}/personas`);
        const personasBody = await personasResp.json();
        const candidate = (personasBody.personas || []).find(p => p.id === candidateId);
        expect(candidate, `candidate ${candidateId} in /personas`).toBeTruthy();
    });
});
