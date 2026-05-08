import { defineConfig } from '@playwright/test';

// Talks to the st-debug instance launched by ../scripts/run.sh on :8002.
// (Our bridge sits at :8001; ST forwards chat-completions traffic there
// based on the bootstrap-patched settings.json.)
export default defineConfig({
    testDir: '.',
    testMatch: '*.spec.js',
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
});
