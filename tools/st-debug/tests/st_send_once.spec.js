import { test, expect } from '@playwright/test';
test('drive one send (capture bridge request)', async ({ page }, testInfo) => {
    test.skip(testInfo.project.name !== 'desktop');
    await page.goto('/');
    await page.waitForFunction(() => document.getElementById('preloader') === null, { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function', { timeout: 30_000 });
    await page.locator('#API-status-top').click();
    await page.locator('#api_button_openai').click();
    await page.waitForFunction(() => window.SillyTavern?.getContext?.()?.onlineStatus === 'Valid', { timeout: 30_000 });
    await page.waitForTimeout(3000);
    await page.locator('#send_textarea').fill('uhhh hi');
    await page.locator('#send_but').click();
    // Don't assert anything — just give bridge time to receive the request.
    await page.waitForTimeout(8000);
});
