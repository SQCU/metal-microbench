// Visual proof: bridge-down banner renders as designed.
//
// This spec EXISTS for screenshot capture, not just assertion pass/fail.
// Two screenshots saved to /tmp/bridge_banner_{down,up}.png so the
// operator can eyeball the actual rendered pixels — runner pass/fail
// alone is insufficient empirical evidence per the project's spec
// (an e2e test must observe what a user would see, not just element
// presence).
//
// State 1: bridge actually unreachable (current real state — bridge
//   isn't running). Suggester opens, polls /bridge-status, banner
//   renders with "Bridge unreachable" copy. Screenshot saved.
//
// State 2: /bridge-status + /agents both mocked to healthy via
//   page.route(). Banner clears. Screenshot saved.

import { test, expect } from '@playwright/test';
import { writeFileSync } from 'node:fs';

const SCREENSHOT_DIR = '/tmp';

async function openSuggester(page) {
    await page.goto('/');
    await page.waitForFunction(() => document.getElementById('preloader') === null,
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });
    const hamburger = page.locator('#user-personas-tools-button .drawer-toggle');
    await expect(hamburger).toBeVisible({ timeout: 20_000 });
    await hamburger.click();
    const menuItem = page.locator('.user-personas-tools-menuitem[data-surface-key="suggester"]');
    await expect(menuItem).toBeVisible({ timeout: 5_000 });
    await menuItem.click();
    const iframe = page.frameLocator('#user-personas-surface-suggester iframe');
    await expect(iframe.locator('h1')).toBeVisible({ timeout: 20_000 });
    return iframe;
}

test.describe('bridge-down banner — visual capture', () => {
    test.setTimeout(60_000);

    test('state 1: bridge unreachable → banner visible (real state, no mock)', async ({ page }, testInfo) => {
        // Only run on desktop project — viewport size matters for visual
        // proof and we want the canonical 1280×800.
        test.skip(testInfo.project.name !== 'desktop',
            'screenshot capture is desktop-only — canonical 1280×800 viewport');
        // No page.route mocks — the live /bridge-status will report the
        // bridge's actual state. We expect it to be unreachable (bridge
        // isn't running in this test env).
        const iframe = await openSuggester(page);

        // Wait for the first poll to land + the banner to render.
        const banner = iframe.locator('#bridge-status-banner');
        await expect(banner, 'bridge-down banner visible').toBeVisible({ timeout: 15_000 });
        await expect(banner).toContainText(/bridge unreachable/i);

        // Settle wait — the hamburger menu popover takes a few hundred
        // ms to fully transition to display:none after the menu-item
        // click closes it. Without this, the screenshot catches the
        // popover still overlapping the banner. Test-timing artifact,
        // not a production bug. (Earlier version of this test added a
        // belt-and-suspenders body.click outside the popover, but that
        // ALSO closed the surface drawer — visible only as ST's home
        // page in the capture. Just wait.)
        await page.waitForTimeout(2000);

        // Capture FULL-PAGE screenshot — the banner is at the top of
        // the suggester iframe, but full-page captures the surrounding
        // context (drawer chrome, chat behind it, etc.) so the
        // operator can verify the banner placement is correct.
        const path = `${SCREENSHOT_DIR}/bridge_banner_down.png`;
        await page.screenshot({ path, fullPage: false });
        console.log(`  screenshot saved: ${path}`);

        // Also capture banner-only crop for clarity.
        const bannerHandle = await iframe.locator('#bridge-status-banner').elementHandle();
        const bannerBoxPath = `${SCREENSHOT_DIR}/bridge_banner_down_crop.png`;
        await bannerHandle.screenshot({ path: bannerBoxPath });
        console.log(`  banner crop saved: ${bannerBoxPath}`);

        // Dump the inner text so we have a textual snapshot too.
        const bannerText = await iframe.locator('#bridge-status-banner').innerText();
        writeFileSync(`${SCREENSHOT_DIR}/bridge_banner_down_text.txt`, bannerText);
        console.log(`  banner text:\n${bannerText.split('\n').map(s => '    ' + s).join('\n')}`);
    });

    test('state 2: bridge mocked healthy → banner hidden, suggester clear', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'screenshot capture is desktop-only');
        // Mock /bridge-status to report reachable.
        await page.route('**/api/plugins/user-personas/bridge-status', async (route) => {
            await route.fulfill({
                status: 200, contentType: 'application/json',
                body: JSON.stringify({
                    reachable: true, latency_ms: 8, status: 'ready',
                    model: 'gemma-4-a4b', active_streams: 0,
                }),
            });
        });
        // Mock /agents to have entries (so the reachable-but-empty
        // banner doesn't fire either).
        await page.route('**/api/plugins/user-personas/agents', async (route) => {
            await route.fulfill({
                status: 200, contentType: 'application/json',
                body: JSON.stringify({
                    agents: [
                        { id: 'mock-agent-1', name: 'Mock Agent 1', designed_for_bio_id: 'despotic-miscreant.png' },
                        { id: 'mock-agent-2', name: 'Mock Agent 2', designed_for_bio_id: 'brutish-miscreant.png' },
                    ],
                }),
            });
        });

        const iframe = await openSuggester(page);
        const banner = iframe.locator('#bridge-status-banner');

        // Banner should be hidden — bridge mocked-healthy + agents
        // mocked-present. Wait for at least one poll cycle to ensure
        // the boot-time poll has landed.
        await page.waitForTimeout(1500);
        await expect(banner, 'banner hidden when bridge mocked-up').toBeHidden({ timeout: 8_000 });

        // Capture screenshot of the cleared state.
        const path = `${SCREENSHOT_DIR}/bridge_banner_up.png`;
        await page.screenshot({ path, fullPage: false });
        console.log(`  screenshot saved: ${path}`);

        // Confirm via DOM probe: bridge-status-banner is display:none.
        const isVisible = await iframe.locator('#bridge-status-banner').isVisible();
        writeFileSync(`${SCREENSHOT_DIR}/bridge_banner_up_state.txt`,
            `banner_visible=${isVisible}\n` +
            `(expected: false — bridge mocked healthy + 2 agents)\n`);
        console.log(`  banner_visible=${isVisible} (expected: false)`);
    });
});
