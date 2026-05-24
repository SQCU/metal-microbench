// Experiment editor (fixed_point.html) — full e2e spec.
//
// Validates the New/Edit/Delete + query-param prefill + inline-validation
// flow against the live st-debug instance.
//
// Acceptance gate per docs/ui_spec_experiment_editor.md §"Acceptance: Playwright spec".

import { test, expect } from '@playwright/test';
import { loadAndConnect } from './_helpers/elicit_clean.mjs';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';

const PLUGIN_BASE = '/api/plugins/user-personas';
const TEST_EXP_ID = 'playwright_test_exp';

// Test scoped to desktop project — modal is a desktop affordance and we
// don't need to validate small-viewport reflow here (the existing FP
// smoke spec covers cross-viewport rendering of the surrounding tab).
test.describe('experiment editor — desktop only', () => {
    test.setTimeout(2 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'editor spec runs only on desktop project');
    });

    // Always clean up the test experiment, even if a test failed mid-way.
    test.afterEach(async ({ request }) => {
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});
    });

    // Helper: open the FP iframe and return the FrameLocator.
    async function openFixedPointTab(page) {
        await loadAndConnect(page);
        // Open via hamburger popover — .drawer-toggle is display:none after
        // sillytavern-fork e2973179d; direct click on wrapper is invalid.
        await openPersonaSurface(page, 'fixed-point');
        const iframe = page.frameLocator('iframe[src*="fixed_point.html"]');
        await expect(iframe.locator('h1, h2').first()).toBeVisible({ timeout: 15_000 });
        // Wait for the experiments list to settle (status text changes
        // from "Loading…" to "N experiments loaded" or similar).
        await expect(iframe.locator('#experiments-status')).toContainText(/loaded/i, { timeout: 10_000 });
        return iframe;
    }

    test('new button opens form, picking/unpicking axes (de)materializes steppers, save round-trips, edit pre-populates, delete works, validation error renders inline', async ({ page, request }) => {
        // Pre-clean: in case a prior failed run left the test card around.
        await request.delete(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`).catch(() => {});

        const iframe = await openFixedPointTab(page);

        // (1) lock_in_tetrad seed card is present.
        await expect(iframe.locator('text=/lock_in_tetrad|RPG Wizard\\/Rogue/').first()).toBeVisible();

        // (2) Open the New Experiment form.
        //
        // P-EMPTY-FORM (UX-T1, 2026-05-21, spec 78): the bare "+ New
        // Experiment" button was removed because clicking it produced a
        // blank modal with bare exp-id / exp-name / exp-description text
        // inputs — exactly the JSON-fields-as-strings anti-pattern. The
        // editor modal itself is still mounted (it's reached via the
        // Edit button on existing experiment cards, where every field
        // pre-fills, and via the ?target_bio_signature query-param
        // path which pre-fills from chat context). To keep this spec
        // exercising the underlying editor wiring (validation, save,
        // edit pre-populate, delete) without re-introducing the
        // forbidden surface, we invoke the editor opener programmatically
        // — the bare entry-point button stays deleted, the wiring stays
        // tested.
        await expect(iframe.locator('#new-experiment-btn'),
            'P-EMPTY-FORM (spec 78): bare "+ New Experiment" button must be absent'
        ).toHaveCount(0);
        await iframe.locator('body').evaluate(() => {
            // openEditorForNew is the module-local fn behind the deleted
            // button. Invoking it asserts the editor wiring is intact
            // even though the UI entry point that USED to call it is
            // gone — every other caller (query-param prefill, Edit on
            // an existing card) still depends on this code path.
            window.openEditorForNew && window.openEditorForNew();
        });
        // If openEditorForNew wasn't on window, fall back to calling
        // it directly via the (re-exported) globals the file declares.
        // Modern fixed_point.html declares it at top-level scope, so
        // we re-invoke if the first attempt didn't open the modal.
        const stillHidden = await iframe.locator('#editor-overlay').isHidden().catch(() => true);
        if (stillHidden) {
            await iframe.locator('body').evaluate(() => {
                // Trigger via the eval scope — the script is in a
                // classic <script> so identifiers are on window/global.
                if (typeof openEditorForNew === 'function') openEditorForNew();
            });
        }
        await expect(iframe.locator('#editor-overlay')).toBeVisible();
        await expect(iframe.locator('#exp-id')).toHaveValue('');
        await expect(iframe.locator('#exp-name')).toHaveValue('');
        await expect(iframe.locator('#exp-description')).toHaveValue('');
        await expect(iframe.locator('#exp-bio-axes')).toBeVisible();
        await expect(iframe.locator('#exp-agent-axes')).toBeVisible();
        // bios always start with one row; agent_targets is empty until +Add.
        await expect(iframe.locator('#bios-rows .bio-row')).toHaveCount(1);
        await expect(iframe.locator('#agent-targets-rows .agent-row')).toHaveCount(0);

        // (3) Picking a bio_axis materializes a stepper in every bio row.
        // Bio row starts empty (no axes selected) so its stepper grid
        // shows the "No axes selected" hint.
        await iframe.locator('#exp-bio-axes').selectOption(['astrology_sagittarian']);
        // The change event re-renders the grid in every bio row.
        const bioStepperInput = iframe.locator('#bios-rows .bio-row .bio-target-wrap input[data-axis-id="astrology_sagittarian"]');
        await expect(bioStepperInput).toHaveCount(1);
        await expect(bioStepperInput).toBeVisible();

        // (4) Un-picking the axis removes the stepper from the bio row.
        await iframe.locator('#exp-bio-axes').selectOption([]);
        await expect(bioStepperInput).toHaveCount(0);

        // Re-pick the axis (need it for the save round-trip below).
        await iframe.locator('#exp-bio-axes').selectOption(['astrology_sagittarian']);
        await expect(bioStepperInput).toHaveCount(1);

        // Pick one agent axis and add an agent_target row.
        await iframe.locator('#exp-agent-axes').selectOption(['theft_aggressiveness']);
        await iframe.locator('#add-agent-btn').click();
        await expect(iframe.locator('#agent-targets-rows .agent-row')).toHaveCount(1);
        const agentStepperInput = iframe.locator('#agent-targets-rows .agent-row .agent-target-wrap input[data-axis-id="theft_aggressiveness"]');
        await expect(agentStepperInput).toHaveCount(1);

        // (9) FIRST: try to save with empty agent_targets to assert the
        // inline validation error renders in red. We do this BEFORE
        // filling the agent_target so we can exercise the error path
        // without already-good state.
        //
        // Save the current agent-target slug/text and then remove the
        // agent_target row to exercise the empty-array validation.
        await iframe.locator('#exp-id').fill(TEST_EXP_ID);
        await iframe.locator('#exp-name').fill('Playwright test experiment');
        await iframe.locator('#exp-description').fill('Generated by 42_experiment_editor.spec.js');
        await iframe.locator('#bios-rows .bio-row .bio-slug').fill('pw-test-bio');
        await iframe.locator('#bios-rows .bio-row .bio-name').fill('Playwright Test Bio');
        await bioStepperInput.fill('5');
        await iframe.locator('#bios-rows .bio-row .bio-design-brief').fill('A test bio synthesized by 42_experiment_editor.spec.js');
        await iframe.locator('#exp-counterparty').fill('the-rock.png');

        // Remove the agent_target row to force the server to reject.
        await iframe.locator('#agent-targets-rows .agent-row .remove-agent-btn').first().click();
        await expect(iframe.locator('#agent-targets-rows .agent-row')).toHaveCount(0);

        await iframe.locator('#save-experiment-btn').click();
        const formError = iframe.locator('#form-error');
        await expect(formError).toBeVisible({ timeout: 10_000 });
        await expect(formError).toContainText(/agent_targets must be (a )?non-empty/i);
        // Form stays open after error.
        await expect(iframe.locator('#editor-overlay')).toBeVisible();
        // The error chip is rendered in the bad colour.
        const errorColor = await formError.evaluate(el => getComputedStyle(el).color);
        // bad rgb: var(--bad) = #d75f5f → rgb(215, 95, 95)
        expect(errorColor).toMatch(/rgb\(\s*215\s*,\s*95\s*,\s*95\s*\)/);

        // (5) Now fill the agent_target back and save successfully.
        await iframe.locator('#add-agent-btn').click();
        await expect(iframe.locator('#agent-targets-rows .agent-row')).toHaveCount(1);
        await iframe.locator('#agent-targets-rows .agent-row .agent-slug').fill('pw-test-agent');
        await iframe.locator('#agent-targets-rows .agent-row .agent-target-wrap input[data-axis-id="theft_aggressiveness"]').fill('5');
        await iframe.locator('#agent-targets-rows .agent-row .agent-motive-hint').fill('Plays the role of a test agent that picks pockets.');

        await iframe.locator('#save-experiment-btn').click();
        // Form closes on success.
        await expect(iframe.locator('#editor-overlay')).toBeHidden({ timeout: 10_000 });
        // List re-renders to include the new card.
        await expect(iframe.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"]`)).toBeVisible({ timeout: 10_000 });

        // Verify server-side via direct API.
        const getResp = await request.get(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`);
        expect(getResp.status()).toBe(200);
        const saved = await getResp.json();
        expect(saved.id).toBe(TEST_EXP_ID);
        expect(saved.name).toBe('Playwright test experiment');
        expect(saved.bio_axes).toEqual(['astrology_sagittarian']);
        expect(saved.agent_axes).toEqual(['theft_aggressiveness']);
        expect(saved.bios).toHaveLength(1);
        expect(saved.bios[0].slug).toBe('pw-test-bio');
        expect(saved.bios[0].canonical_key).toBe('user-personas-pw-test-bio.png');
        expect(saved.bios[0].target_bio).toEqual({ astrology_sagittarian: 5 });
        expect(saved.bios[0].design_brief).toContain('synthesized by');
        expect(saved.agent_targets).toHaveLength(1);
        expect(saved.agent_targets[0].slug).toBe('pw-test-agent');
        expect(saved.agent_targets[0].target_agent).toEqual({ theft_aggressiveness: 5 });
        expect(saved.counterparty_avatar).toBe('the-rock.png');

        // (6) Click the saved row to enter edit mode; assert fields pre-populated.
        await iframe.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"] .exp-edit-trigger`).click();
        await expect(iframe.locator('#editor-overlay')).toBeVisible();
        await expect(iframe.locator('#exp-id')).toBeDisabled();
        await expect(iframe.locator('#exp-id')).toHaveValue(TEST_EXP_ID);
        await expect(iframe.locator('#exp-name')).toHaveValue('Playwright test experiment');
        await expect(iframe.locator('#bios-rows .bio-row .bio-slug')).toHaveValue('pw-test-bio');
        await expect(iframe.locator('#bios-rows .bio-row .bio-design-brief')).toContainText('synthesized by');
        await expect(iframe.locator('#exp-counterparty')).toHaveValue('the-rock.png');
        // Delete button is visible in edit mode.
        await expect(iframe.locator('#delete-experiment-btn')).toBeVisible();

        // (7) Delete works. Accept the confirm() dialog, then assert row vanishes.
        page.once('dialog', d => d.accept());
        await iframe.locator('#delete-experiment-btn').click();
        await expect(iframe.locator('#editor-overlay')).toBeHidden({ timeout: 10_000 });
        await expect(iframe.locator(`.experiment-card[data-eid="${TEST_EXP_ID}"]`)).toHaveCount(0, { timeout: 10_000 });
        const after404 = await request.get(`${PLUGIN_BASE}/experiments/${TEST_EXP_ID}`);
        expect(after404.status()).toBe(404);
    });

    test('query-param prefill: ?target_bio_signature auto-opens the New form with bio_axes + stepper populated', async ({ page }) => {
        // Bring up the tab via the hamburger popover so the iframe URL routing works.
        await loadAndConnect(page);
        await openPersonaSurface(page, 'fixed-point');
        const iframeEl = page.locator('iframe[src*="fixed_point.html"]').first();
        await expect(iframeEl).toBeVisible({ timeout: 15_000 });

        // Now navigate the iframe to the prefill URL. We do this via
        // setAttribute so it goes through the iframe's own load cycle —
        // navigating page.goto() would unload the parent ST shell.
        const sig = { astrology_sagittarian: 5 };
        const prefillUrl = `/api/plugins/user-personas/static/fixed_point.html?target_bio_signature=${encodeURIComponent(JSON.stringify(sig))}`;
        await iframeEl.evaluate((el, url) => { el.src = url; }, prefillUrl);

        const iframe = page.frameLocator('iframe[src*="fixed_point.html"]');
        await expect(iframe.locator('h1, h2').first()).toBeVisible({ timeout: 15_000 });
        // Editor opens automatically.
        await expect(iframe.locator('#editor-overlay')).toBeVisible({ timeout: 10_000 });
        // bio_axes shows astrology_sagittarian selected.
        const bioAxes = iframe.locator('#exp-bio-axes');
        const selectedValues = await bioAxes.evaluate(sel =>
            [...sel.selectedOptions].map(o => o.value));
        expect(selectedValues).toContain('astrology_sagittarian');
        // The first bio row's stepper for that axis shows 5.
        const stepper = iframe.locator('#bios-rows .bio-row .bio-target-wrap input[data-axis-id="astrology_sagittarian"]').first();
        await expect(stepper).toHaveValue('5');
        // Bio slug pre-populated with the from-chat-context placeholder.
        const slugVal = await iframe.locator('#bios-rows .bio-row .bio-slug').inputValue();
        expect(slugVal).toMatch(/^from-chat-context-/);
        // Design brief pre-populated with the placeholder hint.
        await expect(iframe.locator('#bios-rows .bio-row .bio-design-brief')).toContainText(/Synthesized from chat context/i);
        // agent_targets left empty for operator to fill (spec §"Pre-population from query params" item 6).
        await expect(iframe.locator('#agent-targets-rows .agent-row')).toHaveCount(0);
    });
});
