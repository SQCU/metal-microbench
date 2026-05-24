import { defineConfig } from '@playwright/test';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

// globalTeardown runs AFTER all tests finish. We invoke the cleanup
// script with --apply to reap any chromium-headless-shell processes
// that didn't get torn down by playwright itself (rare, but happens
// when a test is killed via SIGKILL mid-render). Conservative defaults
// in the script (MIN_AGE_MIN=5) prevent it from touching anything
// that just spawned for the next test run; this is end-of-suite, not
// mid-suite.
const __dirname = dirname(fileURLToPath(import.meta.url));
const CLEANUP_SCRIPT = resolve(__dirname, '..', 'scripts', 'cleanup_playwright.sh');

// Talks to the st-debug instance launched by ../scripts/run.sh on :8002.
// (Our bridge sits at :8001; ST forwards chat-completions traffic there
// based on the bootstrap-patched settings.json.)
//
// Viewport projects:
//   - desktop (default): 1280×800, what a laptop / desktop user sees
//   - tablet:            768×1024, iPad-ish
//   - mobile:            390×844,  iPhone 12/13/14-ish
//
// Tests that don't care about viewport rendering should run under
// `desktop`. Tests validating UI affordances across form factors
// should run under all three (or filter via --project=mobile etc.).
export default defineConfig({
    testDir: '.',
    testMatch: '*.spec.js',
    globalTeardown: resolve(__dirname, 'global_teardown.js'),
    use: {
        baseURL: 'http://127.0.0.1:8002',
        // Useful artifacts when something fails during e2e:
        video: 'retain-on-failure',
        screenshot: 'only-on-failure',
        trace: 'retain-on-failure',
    },
    // Single worker by default — the bridge is a shared resource and
    // some integration tests assume specific KV-cache / page-pool state.
    // Bump for parallel-distribution tests later.
    workers: 1,
    fullyParallel: false,
    timeout: 120_000,
    expect: {
        timeout: 30_000,
    },
    projects: [
        {
            name: 'desktop',
            use: { viewport: { width: 1280, height: 800 } },
        },
        {
            name: 'tablet',
            use: { viewport: { width: 768, height: 1024 } },
        },
        {
            name: 'mobile',
            use: { viewport: { width: 390, height: 844 } },
        },
    ],
});
