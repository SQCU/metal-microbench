// Spec 88 — top-bar at phone aspect ratio (375×812).
//
// REGRESSION OBJECT: Sub-16/19/20/22 added `buttonId` to each of their
// tabs in USER_PERSONAS_TOOLS_TABS, which the install code at
// public/scripts/extensions/user-personas/index.js:945 used to render
// the toggle visibly in the top bar. Net effect: 4 new always-visible
// top-row buttons that overflowed the header at phone aspect ratio and
// pushed existing ST dropdowns off-screen.
//
// OPERATOR FRAMING (verbatim, 2026-05-24):
//   "this is not a subjective conversation but an objectively pinned
//    measurement that is simple to conduct and doesn't require using
//    the backend tensor service to notice regressions, regressions like
//    doubling the number of buttons in the top/header div of menus and
//    interface elements for no reason."
//   "previous implementations were cautioned to use a single interface
//    button for all user agent and persona design features instead of
//    adding many buttons to the header interface."
//
// FIX (commit e2973179d in sillytavern-fork):
//   public/scripts/extensions/user-personas/index.js:945
//     -    const toggleStyle = t.buttonId ? '' : 'display: none;';
//     +    const toggleStyle = 'display: none;';
//
// THIS SPEC LOCKS IN THE FIX:
//   At phone aspect ratio (375×812), the user-personas plugin must
//   contribute EXACTLY ONE visible element to the ST top-bar — the
//   hamburger (#user-personas-tools-button). The per-surface wrappers
//   (#user-suggester-button / #user-corpus-button /
//   #user-fixed-point-button / #user-axes-button) must exist in the DOM
//   (test harnesses still need to programmatically open them), but their
//   .drawer-toggle children must be display:none so they take no top-row
//   real estate.

import { test, expect } from '@playwright/test';

const ST_URL = process.env.ST_URL || 'http://127.0.0.1:8002';
const PLUGIN_WRAPPER_IDS = [
    'user-suggester-button',
    'user-corpus-button',
    'user-fixed-point-button',
    // 'user-axes-button' retired 2026-06: the Axes tab was folded into
    // corpus.html (Corpus tab), so the standalone wrapper no longer exists.
];

test.use({ viewport: { width: 375, height: 812 } });

test.describe('top-bar at phone aspect ratio', () => {
    test('plugin contributes exactly one visible top-row entry', async ({ page }) => {
        await page.goto(ST_URL);
        // Wait for the extension to install.
        await expect(page.locator('#user-personas-tools-button')).toBeAttached({ timeout: 30_000 });

        // (1) The hamburger MUST be visible.
        await expect(page.locator('#user-personas-tools-button')).toBeVisible();

        // (2) Each per-surface wrapper MUST exist in DOM (for programmatic open).
        for (const id of PLUGIN_WRAPPER_IDS) {
            await expect(page.locator(`#${id}`)).toBeAttached();
        }

        // (3) Each per-surface wrapper's .drawer-toggle MUST be display:none.
        //     This is the regression check: if Sub-16/19/20/22's buttonId-driven
        //     visibility comes back, this asserts FAILS.
        for (const id of PLUGIN_WRAPPER_IDS) {
            const toggle = page.locator(`#${id} > .drawer-toggle`);
            const display = await toggle.evaluate(el => getComputedStyle(el).display);
            expect(display, `#${id} > .drawer-toggle must be display:none (regression: per-tab top-bar buttons)`).toBe('none');
        }

        // (4) Hamburger remains clickable + popover opens.
        await page.locator('#user-personas-tools-button').click();
        await expect(page.locator('#UserPersonasToolsMenu')).toBeVisible({ timeout: 5_000 });
        // The popover lists every surface as a menu item — single entry point.
        const items = await page.locator('.user-personas-tools-menuitem').count();
        expect(items, 'hamburger popover lists every surface').toBeGreaterThanOrEqual(4);
    });

    test('count of plugin-added visible top-bar elements is exactly 1', async ({ page }) => {
        // Stronger phrasing of the regression: count *visible* elements
        // that the user-personas plugin adds to the top bar.
        await page.goto(ST_URL);
        await expect(page.locator('#user-personas-tools-button')).toBeAttached({ timeout: 30_000 });

        const ids = ['user-personas-tools-button', ...PLUGIN_WRAPPER_IDS];
        let visibleCount = 0;
        for (const id of ids) {
            const visible = await page.locator(`#${id}`).isVisible();
            if (visible) visibleCount++;
        }
        expect(visibleCount, 'user-personas plugin must contribute exactly 1 visible top-bar element').toBe(1);
    });

    test('screenshot: top bar at phone aspect ratio (proof artifact)', async ({ page }) => {
        await page.goto(ST_URL);
        await expect(page.locator('#user-personas-tools-button')).toBeAttached({ timeout: 30_000 });
        // Wait for layout settle.
        await page.waitForTimeout(500);
        await page.screenshot({
            path: 'screenshots/88_topbar_phone_aspect.png',
            clip: { x: 0, y: 0, width: 375, height: 100 },
        });
    });
});
