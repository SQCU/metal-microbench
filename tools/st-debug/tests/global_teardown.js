// End-of-suite cleanup: reap any chromium-headless-shell processes
// that didn't get torn down by Playwright itself. Healthy test runs
// leave nothing behind, but SIGKILL'd or crashed tests can leak.
//
// This is the test-side companion to the run.sh pre-launch sweep:
// both call the same cleanup script. Pre-launch handles old MCP
// residue; post-teardown handles current-run residue.

import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));
const CLEANUP_SCRIPT = resolve(__dirname, '..', 'scripts', 'cleanup_playwright.sh');

export default async function globalTeardown() {
    // Use a tight age threshold (1 minute) for end-of-suite since
    // anything older than 1m at THIS point is definitely from a prior
    // run that wasn't cleaned up. Pass --apply so it actually kills.
    return new Promise((resolve) => {
        const child = spawn('bash', [CLEANUP_SCRIPT, '--apply'], {
            env: { ...process.env, MIN_AGE_MIN: '1' },
            stdio: 'inherit',
        });
        child.on('exit', () => resolve());
        // 10s wallclock cap so a hung cleanup never blocks the test
        // reporter from finalizing.
        setTimeout(() => {
            try { child.kill('SIGTERM'); } catch {}
            resolve();
        }, 10_000);
    });
}
