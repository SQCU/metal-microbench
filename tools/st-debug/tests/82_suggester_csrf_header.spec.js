// CSRF wrapper coverage: iframe POSTs must include X-CSRF-Token.
//
// Background: the suggester iframe (and the other plugin surfaces)
// historically sent POSTs with no CSRF token, which 403'd under
// SillyTavern's default CSRF middleware. They worked in st-debug
// only because st-debug runs with --disableCsrf. Sub-13 added a
// shared static/csrf-fetch.js wrapper that:
//   1. lazy-fetches /csrf-token on first unsafe-method call
//   2. injects X-CSRF-Token on every unsafe-method request
//   3. on 403+EBADCSRFTOKEN, refetches and retries once
//
// This spec captures real iframe traffic via page.on('request') and
// asserts:
//   A. A GET to /csrf-token is issued by the suggester iframe at
//      least once before its first unsafe-method POST.
//   B. Every unsafe-method POST to the plugin from the iframe
//      carries a non-empty X-CSRF-Token header.
//
// We do NOT require st-debug to be running with CSRF enabled — the
// wrapper's correctness is about HEADER PRESENCE, which is observable
// regardless of server-side enforcement. (st-debug's --disableCsrf
// makes the server return `{token: 'disabled'}` from /csrf-token; the
// wrapper still sends that value as the header.)
//
// To trigger guaranteed iframe POSTs, we use the same empty-agents
// setup pattern as spec 81: move agents/ aside, restart st-debug, open
// suggester. The auto-dispatch fires within a poll cycle and produces
// the POST we want to inspect.

import { test, expect } from '@playwright/test';
import { execSync } from 'node:child_process';
import { existsSync, renameSync, rmSync } from 'node:fs';

const AGENTS_DIR = '/Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork/plugins/user-personas/agents';
const AGENTS_BAK = '/Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork/plugins/user-personas/agents.bak';

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

test.describe('suggester iframe — CSRF wrapper presence', () => {
    test.setTimeout(120_000);

    test.beforeAll(async () => {
        // Empty-agents pattern from spec 81 ensures the suggester's
        // auto-dispatch path fires within a poll cycle.
        if (existsSync(AGENTS_BAK)) {
            rmSync(AGENTS_BAK, { recursive: true, force: true });
        }
        if (existsSync(AGENTS_DIR)) {
            renameSync(AGENTS_DIR, AGENTS_BAK);
        }
        execSync(`pkill -f 'node server.js.*--port 8002' || true`, { stdio: 'ignore' });
        await new Promise(r => setTimeout(r, 1000));
        execSync(`cd /Users/mdot/metal-microbench/tools/st-debug && ./scripts/run.sh --bg`,
            { stdio: 'inherit' });
        for (let i = 0; i < 30; i++) {
            try {
                execSync(`curl -fsS --max-time 1 http://127.0.0.1:8002/ -o /dev/null`,
                    { stdio: 'ignore' });
                return;
            } catch { await new Promise(r => setTimeout(r, 1000)); }
        }
        throw new Error('st-debug did not come up within 30s');
    });

    test.afterAll(() => {
        if (existsSync(AGENTS_BAK)) {
            if (existsSync(AGENTS_DIR)) {
                rmSync(AGENTS_DIR, { recursive: true, force: true });
            }
            renameSync(AGENTS_BAK, AGENTS_DIR);
            try {
                execSync(`pkill -f 'node server.js.*--port 8002' || true`, { stdio: 'ignore' });
                execSync(`sleep 1 && cd /Users/mdot/metal-microbench/tools/st-debug && ./scripts/run.sh --bg`,
                    { stdio: 'inherit' });
            } catch (e) {
                console.warn(`afterAll: st-debug restart for restore failed: ${e.message}`);
            }
        }
    });

    test('iframe POSTs include X-CSRF-Token AND iframe issues a /csrf-token GET', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'CSRF wrapper test is desktop-only — canonical 1280×800 viewport');

        // Capture every request. We filter to iframe-origin POSTs to
        // the plugin (not the parent ST page's requests, which already
        // have CSRF via csrfRecoveringFetch in script.js).
        const iframePosts = [];   // {url, hasHeader, headerValue}
        const tokenFetches = [];  // {url, frameUrl}

        page.on('request', (req) => {
            const url = req.url();
            const method = req.method();
            const frameUrl = req.frame()?.url() || '';
            // GET /csrf-token from inside the suggester iframe.
            if (method === 'GET' && url.endsWith('/csrf-token')
                    && frameUrl.includes('/api/plugins/user-personas/static/suggester.html')) {
                tokenFetches.push({ url, frameUrl });
                return;
            }
            // Unsafe-method requests from the suggester iframe to the plugin.
            if (method !== 'GET' && method !== 'HEAD' && method !== 'OPTIONS'
                    && url.includes('/api/plugins/user-personas/')
                    && frameUrl.includes('/api/plugins/user-personas/static/suggester.html')) {
                const headers = req.headers();
                // Header names are lowercased by Playwright/Chromium.
                const headerValue = headers['x-csrf-token'] || null;
                iframePosts.push({
                    url,
                    method,
                    hasHeader: !!headerValue,
                    headerValue,
                });
            }
        });

        // Open suggester. Triggers boot fetches + the empty-agents
        // auto-dispatch within one poll cycle (5s).
        await openSuggester(page);

        // Wait for the bridge banner to appear with status framing —
        // that proves we've reached the auto-dispatch path.
        const iframe = page.frameLocator('#user-personas-surface-suggester iframe');
        const banner = iframe.locator('#bridge-status-banner');
        await expect(banner).toContainText(/Synthesizing K=2|Dispatching K=2/, { timeout: 30_000 });

        // Give the iframe a couple of poll cycles so we observe the
        // POST (the first one fires immediately, but giving it room
        // for the safety guard at suggester.html ~L1541 doesn't hurt).
        await page.waitForTimeout(7_000);

        console.log(`  /csrf-token fetches from iframe: ${tokenFetches.length}`);
        console.log(`  iframe unsafe-method POSTs observed: ${iframePosts.length}`);
        for (const p of iframePosts.slice(0, 5)) {
            console.log(`    ${p.method} ${p.url} → X-CSRF-Token=${p.hasHeader ? `"${p.headerValue.slice(0,12)}…"` : 'MISSING'}`);
        }

        // Assertion A: the iframe issued at least one GET /csrf-token.
        expect(tokenFetches.length,
            'iframe must fetch /csrf-token at least once before its first unsafe-method POST')
            .toBeGreaterThan(0);

        // Assertion B: we observed at least one iframe POST (otherwise
        // we never tested the wrapper). Auto-dispatch should produce
        // one within 5-7s.
        expect(iframePosts.length,
            'iframe must have issued at least one unsafe-method POST (auto-dispatch)')
            .toBeGreaterThan(0);

        // Assertion C: EVERY observed POST carries the header.
        const missingHeader = iframePosts.filter(p => !p.hasHeader);
        expect(missingHeader.length,
            `every iframe POST must carry X-CSRF-Token. Missing on: ` +
            JSON.stringify(missingHeader.map(p => `${p.method} ${p.url}`)))
            .toBe(0);

        // Assertion D: every header value is non-empty.
        const emptyValue = iframePosts.filter(p => p.hasHeader && !p.headerValue.trim());
        expect(emptyValue.length,
            `every X-CSRF-Token header value must be non-empty. Empty on: ` +
            JSON.stringify(emptyValue.map(p => `${p.method} ${p.url}`)))
            .toBe(0);
    });
});
