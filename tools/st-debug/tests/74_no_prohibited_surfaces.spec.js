// Regression test: prohibited surfaces must not appear in the DOM.
//
// The operator has explicitly prohibited certain UI elements 77+ times
// each in the project transcript. This spec verifies the deletions
// landed AND captures a screenshot of the user-personas panel for
// visual proof. Re-runs catch regressions before the operator does.

import { test, expect } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const SCREENSHOT_DIR = '/tmp';

test.describe('prohibited surfaces — must not appear', () => {
    test.setTimeout(60_000);

    test('user-personas panel: no + Add user-agent button + no inline form + no design-one empty state', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'visual capture is desktop-only');

        await page.goto('/');
        await page.waitForFunction(() => document.getElementById('preloader') === null,
            { timeout: 60_000 });
        // Wait for the user-personas extension to install its panel
        // (the panel inserts via DOM mutation; give it time).
        await page.waitForTimeout(2500);

        // 1. The deprecated button must NOT exist.
        const addBtn = page.locator('#user_personas_add_btn');
        await expect(addBtn, 'NO "+ Add user-agent" button anywhere').toHaveCount(0);

        // 2. The deprecated inline form must NOT exist.
        const addForm = page.locator('#user_personas_add_form');
        await expect(addForm, 'NO inline "create new user-agent" form').toHaveCount(0);
        const nameInput = page.locator('#user_personas_add_name');
        await expect(nameInput, 'NO Name text input').toHaveCount(0);
        const voiceInput = page.locator('#user_personas_add_voice');
        await expect(voiceInput, 'NO Voice textarea').toHaveCount(0);

        // 3. The deprecated empty-state string must NOT appear anywhere.
        const docHtml = await page.content();
        expect(docHtml, 'no "no personas in inventory yet" text anywhere in DOM')
            .not.toContain('no personas in inventory yet');
        expect(docHtml, 'no "design one with + Add user-agent" text anywhere')
            .not.toContain('design one with + Add user-agent');

        // 4. The panel itself should exist + show its proper state.
        const panel = page.locator('#user_personas_panel');
        await expect(panel, 'user-personas panel installs').toBeAttached({ timeout: 10_000 });

        // Capture the panel for visual proof. Expand it first if it's
        // collapsed so the screenshot shows the actual UI.
        const isCollapsed = await panel.evaluate(el => el.classList.contains('is-collapsed'));
        if (isCollapsed) {
            await page.locator('.user-personas-panel-header').click();
            await page.waitForTimeout(400);
        }
        // Screenshot the panel only.
        const panelHandle = await panel.elementHandle();
        await panelHandle.screenshot({ path: `${SCREENSHOT_DIR}/no_prohibited_surfaces_panel.png` });
        console.log(`  panel screenshot saved: ${SCREENSHOT_DIR}/no_prohibited_surfaces_panel.png`);

        // Also dump the panel's text content for textual snapshot.
        const panelText = await panel.innerText();
        writeFileSync(`${SCREENSHOT_DIR}/no_prohibited_surfaces_panel.txt`, panelText);
        console.log(`  panel text content:\n${panelText.split('\n').map(s => '    ' + s).join('\n')}`);

        // 5. The cards container should either have cards OR the
        // proper P-NO-EMPTY-FIRST-PAINT-compliant empty state.
        const container = page.locator('#user_personas_cards_container');
        await expect(container, 'cards container exists').toBeAttached();
        const containerText = await container.innerText().catch(() => '');
        if (containerText) {
            // If non-empty, must NOT contain prohibited phrases.
            expect(containerText, 'no prohibited "add" affordance in container')
                .not.toMatch(/\+ ?Add user.?agent/i);
            expect(containerText, 'no prohibited "design one" affordance')
                .not.toMatch(/design one with/i);
        }

        // 6. CSS for the deleted surface must not be load-bearing on any
        // visible element. If a leftover .user-personas-add-btn class
        // got applied somewhere, it would have the green-tinted styling.
        const addBtnClassUsed = await page.locator('.user-personas-add-btn').count();
        expect(addBtnClassUsed, 'no element uses the deleted .user-personas-add-btn class').toBe(0);
        const addFormClassUsed = await page.locator('.user-personas-add-form').count();
        expect(addFormClassUsed, 'no element uses the deleted .user-personas-add-form class').toBe(0);
    });
});
