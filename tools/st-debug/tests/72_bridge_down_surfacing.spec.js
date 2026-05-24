// Spec — bridge-down surfacing (P-NO-EMPTY-FIRST-PAINT honesty).
//
// When the Metal bridge at :8001 is unreachable, the suggester must
// surface that fact via a visible banner. Without this banner, an
// empty /agents list looks identical to "synthesis in flight" — the
// operator can't tell whether anything's wrong.
//
// Two test paths:
//
//   1. INTERCEPTED — uses page.route() to mock /bridge-status as
//      unreachable. Fast, deterministic, no infrastructure required.
//      This is the canonical test that the FE wiring is correct.
//
//   2. DESTRUCTIVE (test.fixme) — actually kills the bridge process
//      via `pkill -f serve.py`, waits for the next 5s poll, asserts
//      banner appears. Marked fixme because it requires the bridge
//      to be running at test-start AND a manual restart via
//      `make serve` from /Users/mdot/metal-microbench/ after the
//      test (Metal engine load takes ~30-60s — too slow to do
//      mid-test).
//
// The intercepted variant is the load-bearing test (always-runnable).
// The destructive variant proves the same thing under real conditions
// the operator can opt into.

import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';

const PLUGIN_BASE = 'http://127.0.0.1:8002/api/plugins/user-personas';

// Open the suggester surface via the hamburger menu and return the
// iframe FrameLocator. Mirrors the pattern from 61_suggester_resynth.
async function openSuggester(page) {
    await page.goto('/');
    await page.waitForFunction(() => document.getElementById('preloader') === null,
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });
    // Open hamburger → Suggester
    const hamburger = page.locator('#user-personas-tools-button .drawer-toggle');
    await expect(hamburger, 'user-personas hamburger present').toBeVisible({ timeout: 20_000 });
    await hamburger.click();
    const menuItem = page.locator('.user-personas-tools-menuitem[data-surface-key="suggester"]');
    await expect(menuItem, 'suggester menu item present').toBeVisible({ timeout: 5_000 });
    await menuItem.click();
    const iframe = page.frameLocator('#user-personas-surface-suggester iframe');
    await expect(iframe.locator('h1'), 'suggester.html paints').toBeVisible({ timeout: 20_000 });
    return iframe;
}

test.describe('bridge-down surfacing (P-NO-EMPTY-FIRST-PAINT honesty)', () => {
    test.setTimeout(90_000);

    test('intercepted: page.route mocks /bridge-status as unreachable → banner appears', async ({ page }) => {
        // Intercept /bridge-status to force the unreachable state.
        // We mock at the page level (not via test.beforeAll) so the
        // intercept is per-test and self-contained.
        await page.route('**/api/plugins/user-personas/bridge-status', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({
                    reachable: false,
                    error: 'mock: bridge-unreachable',
                    latency_ms: 2001,
                }),
            });
        });
        // Also mock /agents as empty (worst-case state: bridge dead AND
        // no agents in the corpus).
        await page.route('**/api/plugins/user-personas/agents', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ agents: [] }),
            });
        });

        const iframe = await openSuggester(page);

        // The suggester polls /bridge-status on boot AND every 5s. The
        // initial poll fires before the iframe finishes layout, so we
        // wait for the banner to be visible — generous timeout because
        // first-paint races with the initial poll.
        const banner = iframe.locator('#bridge-status-banner');
        await expect(banner, 'bridge-down banner becomes visible').toBeVisible({ timeout: 10_000 });

        // The banner must NAME the failure clearly — operator should
        // see exactly what's wrong + how to fix it.
        await expect(banner, 'banner says "Bridge unreachable"').toContainText(/bridge unreachable/i);
        await expect(banner, 'banner cites the canonical bridge URL').toContainText(/127\.0\.0\.1:8001/);
        await expect(banner, 'banner tells operator how to restart').toContainText(/make serve/i);

        // Now flip the mock to reachable + non-empty agents, and assert
        // the banner disappears within one poll cycle (~5s).
        await page.unroute('**/api/plugins/user-personas/bridge-status');
        await page.route('**/api/plugins/user-personas/bridge-status', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ reachable: true, latency_ms: 12, status: 'ready' }),
            });
        });
        await page.unroute('**/api/plugins/user-personas/agents');
        await page.route('**/api/plugins/user-personas/agents', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ agents: [{ id: 'mock-agent-1' }, { id: 'mock-agent-2' }] }),
            });
        });

        // Wait for the next poll cycle.
        await expect(banner, 'banner clears once bridge becomes reachable').toBeHidden({ timeout: 8_000 });
    });

    test('intercepted: bridge reachable but /agents empty → reachable-empty banner appears', async ({ page }) => {
        await page.route('**/api/plugins/user-personas/bridge-status', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ reachable: true, latency_ms: 12, status: 'ready' }),
            });
        });
        await page.route('**/api/plugins/user-personas/agents', async (route) => {
            await route.fulfill({
                status: 200,
                contentType: 'application/json',
                body: JSON.stringify({ agents: [] }),
            });
        });

        const iframe = await openSuggester(page);
        const banner = iframe.locator('#bridge-status-banner');
        await expect(banner, 'reachable-but-empty banner shows').toBeVisible({ timeout: 10_000 });
        await expect(banner, 'banner says "Bridge reachable"').toContainText(/bridge reachable/i);
        await expect(banner, 'banner mentions restart ST').toContainText(/restart ST|run\.sh/i);
    });

    test.fixme('destructive: actually kill the bridge process + verify banner', async ({ page }) => {
        // PRECONDITION: bridge must be running at test start. If it
        // isn't, this test would be vacuous.
        const bridgeRunning = (() => {
            try {
                execSync('curl -fsS -m 2 http://127.0.0.1:8001/health > /dev/null', { stdio: 'ignore' });
                return true;
            } catch { return false; }
        })();
        test.skip(!bridgeRunning, 'bridge not running at test start — skipping destructive variant');

        const iframe = await openSuggester(page);
        const banner = iframe.locator('#bridge-status-banner');

        // Banner should NOT be visible initially (bridge alive).
        await expect(banner, 'banner hidden when bridge is alive').toBeHidden({ timeout: 8_000 });

        // KILL the bridge. This is the operator's "what if I close the
        // bridge mid-session?" probe. After this test ends, the bridge
        // remains dead — restart manually via `make serve` from
        // /Users/mdot/metal-microbench/.
        console.warn('  [DESTRUCTIVE] killing bridge via pkill -f "server/serve\\.py"');
        execSync('pkill -f "server/serve\\.py" || pkill -f "bridge\\.py" || true');

        // Wait for the next 5s poll + buffer.
        await page.waitForTimeout(7000);

        await expect(banner, 'banner appears after bridge killed').toBeVisible({ timeout: 10_000 });
        await expect(banner).toContainText(/bridge unreachable/i);

        // Bridge stays dead — operator restart required. Annotate.
        test.info().annotations.push({
            type: 'destructive-test-aftermath',
            description: 'Bridge was killed by this test. Restart with `make serve` from /Users/mdot/metal-microbench/.',
        });
    });
});
