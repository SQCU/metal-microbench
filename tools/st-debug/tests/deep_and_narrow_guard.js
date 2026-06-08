// deep_and_narrow_guard.js — globalSetup that BLOCKS a full-suite run.
//
// Validation here is DEEP-AND-NARROW (see metal-microbench memory
// validation_deep_and_narrow): pick the FEWEST specs that each exercise a real
// risk of the change under test, and iterate them to green — depth, not breadth.
// Running the ENTIRE suite ("npx playwright test" with no spec/grep) is breadth
// theater: it reads as "I covered everything" without a single spec having been
// justified. This guard fails such a run INSTANTLY, before any browser launches.
//
// Allowed:
//   - name spec file(s):  playwright test 66_foo.spec.js [94_bar.spec.js ...]
//   - a grep pattern:      playwright test --grep "poll storm"
//   - interactive/triage:  --ui / --list / --last-failed / --only-changed
// Deliberate full run (genuine release gate ONLY): RUN_FULL_SUITE=1
//
// It is a globalSetup (not a reporter) on purpose: a CLI --reporter=... replaces
// config reporters and would bypass a reporter-based guard; globalSetup always
// runs and cannot be silently skipped.

import { readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Flags that consume the FOLLOWING token as their value (space form). Their
// values must not be mistaken for spec-file filters.
const VALUE_FLAGS = new Set([
    '--browser', '--config', '-c', '--grep', '-g', '--grep-invert',
    '--global-timeout', '--output', '-o', '--project', '--repeat-each',
    '--reporter', '-r', '--retries', '--shard', '--timeout', '--trace',
    '--tsconfig', '--workers', '-j', '--max-failures',
]);
// Modes that are NOT "run all tests" — interactive selection / triage. Allowed.
const ALLOW_MODES = ['--ui', '--list', '--last-failed', '--only-changed'];

export default async function deepAndNarrowGuard() {
    if (process.env.RUN_FULL_SUITE === '1') {
        // eslint-disable-next-line no-console
        console.warn('[deep-and-narrow] RUN_FULL_SUITE=1 — full-suite run permitted (release gate). Iterative work should name specific specs.');
        return;
    }

    const argv = process.argv;
    const testIdx = argv.findIndex(a => a === 'test');
    const args = argv.slice(testIdx >= 0 ? testIdx + 1 : 2);

    let grep = false;
    let allowMode = false;
    const positionals = [];
    for (let i = 0; i < args.length; i++) {
        const a = args[i];
        if (a === '--grep' || a === '-g' || a.startsWith('--grep=') ||
            a === '--grep-invert' || a.startsWith('--grep-invert=')) {
            grep = true;
            if (!a.includes('=')) i++; // consume value token
            continue;
        }
        if (ALLOW_MODES.includes(a) || ALLOW_MODES.some(m => a.startsWith(m + '='))) { allowMode = true; continue; }
        if (a.startsWith('-')) {
            if (VALUE_FLAGS.has(a) && !a.includes('=')) i++; // consume value token
            continue;
        }
        positionals.push(a);
    }

    if (grep || allowMode) return;

    // A positional counts as a real file filter only if it is spec-like or
    // matches an actual spec file — so a leaked flag value can't masquerade as a
    // filter, and a typo'd filter that matches nothing is failed (not run as 0).
    let specFiles = [];
    try { specFiles = readdirSync(__dirname).filter(f => f.endsWith('.spec.js')); } catch { /* ignore */ }
    const isFileFilter = positionals.some(p =>
        p.includes('.spec') ||
        specFiles.some(f => f === p || f.includes(p) || p.includes(f)));

    if (isFileFilter) return;

    const total = specFiles.length;
    throw new Error(
        '\n' +
        '================================================================\n' +
        '  FULL-SUITE RUN BLOCKED — validation here is DEEP-AND-NARROW.\n' +
        '================================================================\n' +
        `You invoked the runner with no spec selection — that would run ALL ${total} ` +
        'spec files.\n' +
        '"Run everything" is breadth theater (fake coverage), not validation.\n\n' +
        'Pick the FEWEST specs that each exercise a real risk of your change and\n' +
        'iterate them to green (deep, not wide). For example:\n' +
        '  npx playwright test 66_auto_poll_k1.spec.js --project=desktop\n' +
        '  npx playwright test 66_auto_poll_k1.spec.js 94_user_agent_panel*.spec.js --project=desktop\n' +
        '  npx playwright test --grep "poll storm"\n\n' +
        'Each spec you run should be justified by the riskiest change it covers.\n' +
        'Genuine full-suite release gate ONLY:  RUN_FULL_SUITE=1 npx playwright test\n' +
        '================================================================\n'
    );
}
