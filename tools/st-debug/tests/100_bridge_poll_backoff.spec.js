import { test, expect } from '@playwright/test';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';

// Spec 100 — Bridge-poll backoff witness (log-n, not n-shaped).
//
// The iframe suggester (suggester.html) polls /bridge-status at boot and
// self-reschedules. This is the network surface the operator flagged as
// "spammy in a dangerous part of the client code": it used to fire a flat
//   setInterval(pollBridgeAndAgents, 5000)
// which hammered /bridge-status every 5s FOREVER even against a dead bridge
// (~13 reqs / 60s — an n-shaped, high-frequency linear retry).
//
// POST-FIX it is a single self-rescheduling timer with exponential backoff
// keyed off the consecutive-failure streak:
//   delay = min(5000 * 2**failStreak, 60000)   →  ~10s, 20s, 40s, cap 60s
// plus a "Retry now" banner button that bypasses the backoff (resets the
// streak + fires one immediate poll).
//
// This spec proves the n→log-n conversion at the wire by ABORTING
// /bridge-status (dead-bridge sim — /agents + /personas still succeed, as in
// a real bridge-down where only the engine probe fails) and asserting:
//   (a) the request count over a 62s window is backoff-bounded (≤6, was ~13),
//   (b) the inter-request gaps GROW (each ≳2× the previous), and
//   (c) clicking "Retry now" fires exactly one immediate poll, far inside
//       the (now ≥40s) backoff delay.

const BRIDGE_STATUS_GLOB = '**/api/plugins/user-personas/bridge-status';

// Load ST without connecting the chat API (mirrors spec 66): the bridge poll
// runs at suggester.html boot independently of chat content, so no character
// selection / chat turn is needed for this witness.
async function loadSTNoConnect(page) {
    await page.goto('/');
    await page.waitForFunction(() => document.getElementById('preloader') === null, { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function', { timeout: 30_000 });
}

test.describe('Bridge-poll backoff witness (spec 100)', () => {
    test.setTimeout(5 * 60 * 1000);

    test.beforeEach(async ({}, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop', 'desktop-only iframe-poll witness');
    });

    test('dead bridge: /bridge-status backs off log-n + Retry now fires one immediate poll', async ({ page }) => {
        // Dead-bridge sim: abort the plugin /bridge-status probe so every tick
        // takes the reachable:false path (failStreak++ → exponential backoff).
        await page.route(BRIDGE_STATUS_GLOB, route => route.abort());

        // Timestamp every /bridge-status request (relative-time analysis below).
        // page.on('request') fires for aborted requests too.
        const reqTimes = [];
        page.on('request', (r) => {
            if (r.url().includes('/user-personas/bridge-status')) reqTimes.push(Date.now());
        });

        await loadSTNoConnect(page);
        await openPersonaSurface(page, 'suggester');
        await expect(page.locator('#user-suggester-button iframe')).toBeAttached({ timeout: 30_000 });
        const iframe = page.frameLocator('#user-suggester-button iframe');
        await expect(iframe.locator('h1')).toBeVisible({ timeout: 60_000 });

        // Wait until we've observed ≥4 ticks (boot + 3 backed-off re-arms).
        // Post-fix the schedule is ~0, 10s, 30s, 70s → ~70s to reach 4.
        // Pre-fix (flat 5s) reaches 4 in ~15s; we cap the wait at 110s so a
        // regression still completes and the count/gap assertions catch it.
        const DEADLINE = Date.now() + 110_000;
        while (reqTimes.length < 4 && Date.now() < DEADLINE) {
            await page.waitForTimeout(1000);
        }
        expect(reqTimes.length, 'need ≥4 ticks to witness backoff growth').toBeGreaterThanOrEqual(4);

        const t0 = reqTimes[0];
        // eslint-disable-next-line no-console
        console.log(`[bridge-backoff] reqTimes(rel s)=${reqTimes.map(t => ((t - t0) / 1000).toFixed(1)).join(', ')}`);

        // (a) Backoff-bounded over 62s. Pre-fix flat 5s ⇒ ~13; post-fix ⇒ 3.
        const within62 = reqTimes.filter(t => (t - t0) <= 62_000).length;
        expect(within62,
            `/bridge-status must be backoff-bounded over 62s (flat-5s regression ⇒ ~13); got ${within62}`)
            .toBeLessThanOrEqual(6);

        // (b) Inter-request gaps GROW (exponential): ~10s → 20s → 40s.
        const gaps = [];
        for (let i = 1; i < reqTimes.length; i++) gaps.push(reqTimes[i] - reqTimes[i - 1]);
        // eslint-disable-next-line no-console
        console.log(`[bridge-backoff] gaps(s)=${gaps.map(g => (g / 1000).toFixed(1)).join(', ')}`);
        // Each successive gap ≈ 2× the previous; allow ×1.4 slack for scheduler
        // jitter + poll round-trip.
        expect(gaps[1], `gap2 (${gaps[1]}ms) must exceed gap1 (${gaps[0]}ms) — backoff grows`)
            .toBeGreaterThan(gaps[0] * 1.4);
        expect(gaps[2], `gap3 (${gaps[2]}ms) must exceed gap2 (${gaps[1]}ms) — backoff grows`)
            .toBeGreaterThan(gaps[1] * 1.4);

        // (c) "Retry now" bypass. At this quiescent point the next natural poll
        // is a (capped) backoff delay away (≥40s). Clicking Retry must fire ONE
        // immediate poll — a new /bridge-status request within a few seconds,
        // far inside that delay. The banner+button are rendered at the end of
        // each poll (renderBridgeBanner), so waiting for the button confirms
        // the prior poll fully settled (no in-flight-guard collision).
        const retryBtn = iframe.locator('#bridge-retry-now');
        await expect(retryBtn).toBeVisible({ timeout: 10_000 });
        const beforeClick = reqTimes.length;
        await retryBtn.click();
        await expect.poll(() => reqTimes.length, {
            message: 'Retry now must fire one immediate /bridge-status poll (bypassing the long backoff)',
            timeout: 6_000,
        }).toBeGreaterThan(beforeClick);
    });
});
