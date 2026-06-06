// Regression: the user-personas FE must NOT hammer the backend when it is down.
// Pre-fix, a down/unreachable bridge made pollAndCachePreview cache 'error' while
// the refreshPanel render-cascade re-fired the poll every render -> ~10,344
// /poll requests in seconds (browser-observed). Fix: exponential backoff on the
// 'error' path (base 2s, cap 60s) for both /poll and /yapper-seed, with a single
// scheduled auto-recovery retry. This test loads the panel with the BRIDGE DOWN
// and asserts the request count over a window is backoff-bounded, not a storm.
import { test, expect } from '@playwright/test';

const ST_URL = process.env.ST_URL || 'http://127.0.0.1:8002';
test.use({ httpCredentials: { username: 'sussy', password: 'amongus' }, trace: 'off', video: 'off' });

test('poll storm: FE backs off (does not hammer) when the bridge is down', async ({ page, request }) => {
    const st = await request.get(ST_URL).catch(() => null);
    test.skip(!st || ![200, 401].includes(st.status()), `st-debug not up at ${ST_URL}`);
    // This test SPECIFICALLY requires the bridge DOWN (to exercise the error path).
    const bridge = await request.get('http://127.0.0.1:8001/health').catch(() => null);
    test.skip(!!bridge?.ok(), 'bridge is UP — restart this test with the bridge DOWN to exercise backoff');

    const counts = { poll: 0, yapper: 0 };
    const tStart = Date.now();
    page.on('request', (r) => {
        const u = r.url();
        if (u.includes('/user-personas/poll')) counts.poll++;
        else if (u.includes('/user-personas/yapper-seed')) counts.yapper++;
    });

    await page.goto(ST_URL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction('document.getElementById("preloader") === null', { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function', { timeout: 30_000 });

    // Let the panel auto-render + (with the fix) back off for a generous window.
    const WINDOW_MS = 10_000;
    await page.waitForTimeout(WINDOW_MS);
    const elapsedS = (Date.now() - tStart) / 1000;

    // eslint-disable-next-line no-console
    console.log(`[poll-storm] over ${elapsedS.toFixed(1)}s, bridge DOWN: poll=${counts.poll} yapper=${counts.yapper}`);

    // Backoff base = 2s, cap = 60s. Over ~10s a single key retries at most a few
    // times (t=0, ~2s, ~6s). With a couple of personas + yapper that is well
    // under a few dozen. Pre-fix this was ~10,344. Assert hard upper bounds.
    expect(counts.poll, `/poll must be backoff-bounded (was ~10,344 pre-fix); got ${counts.poll}`).toBeLessThan(40);
    expect(counts.yapper, `/yapper-seed must be backoff-bounded; got ${counts.yapper}`).toBeLessThan(25);
});
