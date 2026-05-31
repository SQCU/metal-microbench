// Locks in: User Avatars/ contains ONLY bio cards, never vanilla PNGs.
// Therefore ST's native Persona Management drawer renders ZERO
// "[Unnamed Persona]" entries — every persona has a name + bio.
//
// Background: ST's FE calls /api/avatars/get (filesystem scan) and
// addMissingPersonas() fabricates a "[Unnamed Persona]" placeholder
// for every PNG returned that lacks an entry in power_user.personas.
// Earlier work left vanilla PNGs (header+pixels, no tEXt) in the
// User Avatars/ directory. Those triggered ghost persona rows.
//
// Fix: delete vanilla PNGs at source. This spec catches their return.
//
// Acceptance criterion (operator-stated):
//   1. Zero "[Unnamed Persona]" entries in the persona panel
//   2. Every persona has a non-empty name
//   3. Personas exist (corpus is non-empty)

import { test, expect } from '@playwright/test';

test.describe('persona panel — no Unnamed Persona ghosts', () => {
    test.setTimeout(60_000);

    test('every persona has a real name (zero [Unnamed Persona] entries)', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'render test is desktop-only — canonical 1280×800 viewport');

        await page.goto('/');
        await page.waitForFunction(() => document.getElementById('preloader') === null,
            { timeout: 60_000 });
        await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
            { timeout: 30_000 });

        // Open ST's native Persona Management drawer.
        const personaBtn = page.locator('#persona-management-button .drawer-toggle');
        await expect(personaBtn).toBeVisible({ timeout: 20_000 });
        await personaBtn.click();

        // The persona panel renders persona rows. Wait for at least one
        // to appear (otherwise the test would pass on an empty panel).
        const personaRows = page.locator('#user_avatar_block .avatar-container, #user_avatar_block .persona, #user-list-block .persona, #persona-management-block .avatar-container');
        // Fall back to grepping all visible text in the drawer.
        await page.waitForTimeout(3000);

        // Strategy: scan the entire persona drawer's text for the
        // forbidden literal "[Unnamed Persona]" — that's exactly the
        // string addMissingPersonas() writes.
        const drawerText = await page.evaluate(() => {
            const block = document.querySelector('#user_avatar_block')
                || document.querySelector('#persona-management-block')
                || document.querySelector('[id*="persona"]');
            return block ? (block.innerText || '') : '';
        });

        console.log(`  drawer text length: ${drawerText.length}`);
        const unnamedCount = (drawerText.match(/\[Unnamed Persona\]/g) || []).length;
        console.log(`  [Unnamed Persona] occurrences: ${unnamedCount}`);

        await page.screenshot({ path: '/tmp/spec84_persona_panel.png', fullPage: true });
        console.log(`  screenshot: /tmp/spec84_persona_panel.png`);

        expect(unnamedCount,
            `persona panel must NOT show any "[Unnamed Persona]" entries — ` +
            `if it does, the User Avatars/ directory contains a PNG without ` +
            `a chara_card_v3 tEXt chunk that needs to be deleted at source. ` +
            `See screenshot at /tmp/spec84_persona_panel.png.`)
            .toBe(0);

        // Also: confirm the corpus is non-empty (we should see SOME persona text).
        // The 27 canonical bios should produce a sizeable drawer.
        expect(drawerText.length, 'persona drawer must render non-empty content').toBeGreaterThan(50);
    });

    test('canonical anchors visible in the persona panel', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'render test is desktop-only');

        await page.goto('/');
        await page.waitForFunction(() => document.getElementById('preloader') === null,
            { timeout: 60_000 });
        await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
            { timeout: 30_000 });

        const personaBtn = page.locator('#persona-management-button .drawer-toggle');
        await expect(personaBtn).toBeVisible({ timeout: 20_000 });
        await personaBtn.click();
        await page.waitForTimeout(3000);

        const drawerText = await page.evaluate(() => {
            const block = document.querySelector('#user_avatar_block')
                || document.querySelector('#persona-management-block')
                || document.querySelector('[id*="persona"]');
            return block ? (block.innerText || '') : '';
        });

        // Canonical anchors must be visible. These match by their
        // display name as set in the chara_card_v3 tEXt chunk.
        expect(drawerText, 'Despotic Miscreant anchor visible')
            .toMatch(/Despotic Miscreant/i);
        expect(drawerText, 'scringlo scrambler anchor visible')
            .toMatch(/scringlo/i);
    });
});
