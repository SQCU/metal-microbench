// Happy-path: with the bridge UP, the user-personas panel must drive /poll
// successfully (200 + generated previews), NOT storm and NOT error-loop. This is
// the positive counterpart to 97_poll_storm_backoff (bridge DOWN -> backoff).
import { test, expect } from '@playwright/test';

const ST_URL = process.env.ST_URL || 'http://127.0.0.1:8002';
test.use({ httpCredentials: { username: 'sussy', password: 'amongus' }, trace: 'off', video: 'off' });

test('poll happy path: panel drives /poll to 200 (bridge up), no storm', async ({ page, request }) => {
    const st = await request.get(ST_URL).catch(() => null);
    test.skip(!st || ![200, 401].includes(st.status()), `st-debug not up at ${ST_URL}`);
    const bridge = await request.get('http://127.0.0.1:8001/health').catch(() => null);
    test.skip(!bridge?.ok(), 'bridge DOWN — this test needs it UP');

    const status = { 200: 0, 404: 0, other: 0, total: 0 };
    page.on('response', (r) => {
        if (!r.url().includes('/user-personas/poll')) return;
        status.total++;
        if (r.status() === 200) status['200']++;
        else if (r.status() === 404) status['404']++;
        else status.other++;
    });

    await page.goto(ST_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction('document.getElementById("preloader") === null', { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function', { timeout: 30_000 });
    // give the panel time to: yapper-seed -> pick top-k -> /poll each -> render previews
    await page.waitForTimeout(20_000);

    // eslint-disable-next-line no-console
    console.log(`[poll-happy] /poll responses: ${JSON.stringify(status)}`);

    // Must NOT storm (a few polls for the top-k picks, not thousands).
    expect(status.total, 'no storm with bridge up').toBeLessThan(60);
    // At least one poll must SUCCEED (the panel actually works end-to-end).
    expect(status['200'], 'at least one /poll must 200 (panel drives real generation)').toBeGreaterThan(0);
});
