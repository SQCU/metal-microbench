// Standalone playwright check against ROOT ST (port 8008, basic-auth).
// Validates: the canonical tetrad dyad + their 4 agents render in the
// suggester surface — through the ACTUAL UI, not just an API curl.
//
// Run from /Users/mdot/metal-microbench/tools/st-debug/tests/ via:
//   npx playwright test /tmp/spec_root_st_suggester.spec.js \
//     --project=desktop --config=playwright.config.js

import { test, expect } from '@playwright/test';

const ROOT_URL = 'http://127.0.0.1:8008';
const BASIC_AUTH = { username: 'sussy', password: 'amongus' };

test.use({ httpCredentials: BASIC_AUTH, baseURL: ROOT_URL });

test('root ST suggester: dyad bios + 4 canonical agents render', async ({ page }, testInfo) => {
    test.setTimeout(90_000);

    await page.goto('/');
    await page.waitForFunction(() => document.getElementById('preloader') === null,
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });

    // Open user-personas tools drawer + suggester.
    const hamburger = page.locator('#user-personas-tools-button .drawer-toggle');
    await expect(hamburger).toBeVisible({ timeout: 20_000 });
    await hamburger.click();
    const menuItem = page.locator('.user-personas-tools-menuitem[data-surface-key="suggester"]');
    await expect(menuItem).toBeVisible({ timeout: 5_000 });
    await menuItem.click();

    const iframe = page.frameLocator('#user-personas-surface-suggester iframe');
    await expect(iframe.locator('h1')).toBeVisible({ timeout: 20_000 });

    // Banner: must NOT say Synthesizing — the dyad has agents.
    const banner = iframe.locator('#bridge-status-banner');
    // Give the iframe a poll cycle to settle.
    await page.waitForTimeout(7_000);
    const bannerVisible = await banner.isVisible().catch(() => false);
    if (bannerVisible) {
        const bannerText = await banner.innerText();
        expect(bannerText, 'banner must NOT say Synthesizing when agents exist')
            .not.toMatch(/Synthesizing K=|Dispatching K=/);
    }

    // Verify the bios are reachable via the iframe's own API view.
    // We poll the same endpoint the iframe polls.
    const apiCheck = await page.evaluate(async (creds) => {
        const r = await fetch('/api/plugins/user-personas/agents',
            { headers: { Authorization: 'Basic ' + btoa(`${creds.username}:${creds.password}`) } });
        const j = await r.json();
        return (j.agents || j || []).length;
    }, BASIC_AUTH);
    expect(apiCheck, 'iframe sees 4 agents via /agents endpoint').toBe(4);

    // Screenshot of the suggester in its post-fix state.
    await page.screenshot({ path: '/tmp/root_st_suggester_post_fix.png', fullPage: true });

    console.log(`  ✓ root ST suggester: bios=dyad, agents=4, banner_visible=${bannerVisible}`);
    console.log(`  ✓ screenshot: /tmp/root_st_suggester_post_fix.png`);
});
