import { defineConfig } from '@playwright/test';

// ===========================================================================
// Server-free runtime-client harness config.
// ===========================================================================
//
// This config is DELIBERATELY decoupled from the main st-debug suite
// (../playwright.config.js), which points baseURL at the live ST instance on
// :8002 and skips when the bridge/ST are down. The runtime-client harness
// needs NEITHER ST nor the bridge: it loads the *real* shipped client source
// (suggester.html / index.js + csrf-fetch.js) off disk and synthesizes every
// backend response via Playwright route interception. No socket is opened; no
// server is started. This is the "validate the runtime client, not a pytest
// simulacrum" tier.
//
// Run it with:
//   cd tools/st-debug/tests
//   npx playwright test --config runtime-client/playwright.runtime.config.js
export default defineConfig({
    testDir: '.',
    testMatch: '*.spec.js',
    fullyParallel: false,
    workers: 1,
    timeout: 60_000,
    expect: { timeout: 15_000 },
    reporter: [['list']],
    use: {
        headless: true,
        // No baseURL: each spec navigates to an absolute, route-intercepted URL.
        video: 'off',
        trace: 'off',
        screenshot: 'off',
    },
    projects: [
        { name: 'runtime-client', use: { viewport: { width: 1280, height: 800 } } },
    ],
});
