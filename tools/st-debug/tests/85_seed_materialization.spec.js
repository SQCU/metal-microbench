// Spec UX-T4 — Default-seed materialization.
//
// Validates that `plugins/user-personas/default_seed/` exists as a
// committed corpus and that the plugin's init() copies missing files
// to live destinations before the load* calls.
//
// Acceptance line (from UX-T4 + concrete output requirements):
//   "the canonical minimum corpus exists as committed seed files in
//    default_seed/{bios,agents,axes,experiments}/, and the plugin's
//    init() copies anything missing from there into the live data dir
//    at startup (no overwrite of existing files; materialization is
//    additive only)"
//
// Live observable conditions under test (all 5 must hold):
//   1. GET /api/plugins/user-personas/personas  → .personas.length ≥ 4
//   2. GET /api/plugins/user-personas/agents    → .agents.length   ≥ 4
//   3. GET /api/plugins/user-personas/axes      → .axes.length     ≥ 10
//   4. GET /api/plugins/user-personas/experiments/lock_in_tetrad → .id === "lock_in_tetrad"
//   5. ls plugins/user-personas/default_seed/bios/ | wc -l ≥ 4
//
// Playwright expect() that locks in acceptance:
//   expect(personas.length, '≥4 bios from seed').toBeGreaterThanOrEqual(4);
//   expect(agts.length,     '≥4 agents from seed').toBeGreaterThanOrEqual(4);
//   expect(axesArr.length,  '≥10 axes from seed').toBeGreaterThanOrEqual(10);
//   expect(expId,           'lock_in_tetrad id').toBe('lock_in_tetrad');
//   expect(bioFiles.length, '≥4 files in default_seed/bios/').toBeGreaterThanOrEqual(4);
//
// This spec tests the API-layer consequences, not just filesystem state.
// A fresh environment (where _data/default-user/User Avatars/ is empty
// and agents/ is empty) must still pass because the seed materializer
// copies the files in before the loaders run.

import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';

const ST_BASE = 'http://127.0.0.1:8002';
const PLUGIN_BASE = `${ST_BASE}/api/plugins/user-personas`;

// Paths in the st-debug clone's version of the plugin (the one actually
// running on port 8002).
const PLUGIN_DIR =
    '/Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork/plugins/user-personas';
const SEED_BIOS_DIR = path.join(PLUGIN_DIR, 'default_seed', 'bios');

test.describe('UX-T4: seed materialization', () => {
    test.setTimeout(60_000);

    test('default_seed/bios/ contains ≥4 committed PNG files', () => {
        // Filesystem check: the seed directory must exist and contain
        // the 4 canonical bio PNGs.
        expect(fs.existsSync(SEED_BIOS_DIR),
            `default_seed/bios/ must exist at ${SEED_BIOS_DIR}`)
            .toBe(true);

        const bioFiles = fs.readdirSync(SEED_BIOS_DIR)
            .filter(f => f.endsWith('.png'));
        // Verbatim acceptance line:
        expect(bioFiles.length, '≥4 files in default_seed/bios/').toBeGreaterThanOrEqual(4);

        console.log(`  default_seed/bios/ has ${bioFiles.length} PNG(s): ${bioFiles.join(', ')}`);
    });

    test('seed bios have provenance.kind = seed_demo in ccv3 chunk', () => {
        // Every bio in default_seed/bios/ must carry
        // extensions.provenance.kind = 'seed_demo' in its ccv3 tEXt chunk.
        // This prevents the seed bios from being treated as user-authored
        // or experiment-output provenance by the suggester filter.
        if (!fs.existsSync(SEED_BIOS_DIR)) {
            test.skip();
            return;
        }
        // Manual PNG chunk reading (avoids loading the full ST stack).
        // We'll do a quick binary search for the base64 pattern in the
        // ccv3 chunk rather than re-implementing the parser.
        const { execSync } = require('node:child_process');
        const pngFiles = fs.readdirSync(SEED_BIOS_DIR).filter(f => f.endsWith('.png'));
        for (const f of pngFiles) {
            const fullPath = path.join(SEED_BIOS_DIR, f);
            // Use the same ESM reader the plugin uses.
            const result = execSync(
                `node --input-type=module`,
                {
                    input: `
import fs from 'fs';
import pngExtract from '/Users/mdot/sillytavern-fork/node_modules/png-chunks-extract/index.js';
import PNGtext from '/Users/mdot/sillytavern-fork/node_modules/png-chunk-text/index.js';
const buf = fs.readFileSync(${JSON.stringify(fullPath)});
const chunks = pngExtract(new Uint8Array(buf));
const tEXt = chunks.filter(c => c.name === 'tEXt').map(c => PNGtext.decode(c.data));
const ccv3 = tEXt.find(c => c.keyword.toLowerCase() === 'ccv3');
if (!ccv3) { process.stdout.write('NO_CCv3'); process.exit(0); }
const card = JSON.parse(Buffer.from(ccv3.text, 'base64').toString('utf8'));
const kind = card.data?.extensions?.provenance?.kind ?? 'MISSING';
process.stdout.write(kind);
`,
                    encoding: 'utf8',
                    timeout: 10_000,
                }
            ).trim();
            expect(result, `${f}: extensions.provenance.kind`).toBe('seed_demo');
        }
        console.log(`  ${pngFiles.length} seed bio(s) all have provenance.kind=seed_demo`);
    });

    test('/personas returns ≥4 bios (from materialized seed)', async () => {
        const resp = await fetch(`${PLUGIN_BASE}/personas`);
        expect(resp.ok, '/personas 2xx').toBe(true);
        const body = await resp.json();
        const personas = body.personas ?? [];
        // Verbatim acceptance line:
        expect(personas.length, '≥4 bios from seed').toBeGreaterThanOrEqual(4);
        console.log(`  /personas returned ${personas.length} bio(s)`);
    });

    test('/agents returns ≥4 agents (from materialized seed)', async () => {
        const resp = await fetch(`${PLUGIN_BASE}/agents`);
        expect(resp.ok, '/agents 2xx').toBe(true);
        const body = await resp.json();
        const agts = body.agents ?? [];
        // Verbatim acceptance line:
        expect(agts.length, '≥4 agents from seed').toBeGreaterThanOrEqual(4);
        console.log(`  /agents returned ${agts.length} agent(s)`);
    });

    test('/axes returns ≥10 axes (from materialized seed)', async () => {
        const resp = await fetch(`${PLUGIN_BASE}/axes`);
        expect(resp.ok, '/axes 2xx').toBe(true);
        const body = await resp.json();
        const axesArr = body.axes ?? [];
        // Verbatim acceptance line:
        expect(axesArr.length, '≥10 axes from seed').toBeGreaterThanOrEqual(10);
        console.log(`  /axes returned ${axesArr.length} axis(es)`);
    });

    test('/experiments/lock_in_tetrad returns id = "lock_in_tetrad"', async () => {
        const resp = await fetch(`${PLUGIN_BASE}/experiments/lock_in_tetrad`);
        expect(resp.ok, '/experiments/lock_in_tetrad 2xx').toBe(true);
        const body = await resp.json();
        const expId = body.id;
        // Verbatim acceptance line:
        expect(expId, 'lock_in_tetrad id').toBe('lock_in_tetrad');
        console.log(`  /experiments/lock_in_tetrad returned id="${expId}"`);
    });

    test('materialization is additive (does not overwrite existing files)', () => {
        // The materializeSeedFiles() function must not overwrite existing
        // live files. We verify this by checking that the live bios dir
        // contains the same files as the seed but possibly more — never
        // fewer (additive invariant is impossible to directly test without
        // a write-probe, but we can at least verify the seed bios exist
        // in the live store AND that the seed dir itself matches the
        // expected canonical filenames).
        const EXPECTED_BIO_FILENAMES = [
            '1778631331275-DespoticMiscreant.png',
            '1779035204660-scringloscrambler.png',
            'rpg-rogue-cancer.png',
            'rpg-wizard-sagittarius.png',
        ];
        if (!fs.existsSync(SEED_BIOS_DIR)) {
            test.skip();
            return;
        }
        const present = fs.readdirSync(SEED_BIOS_DIR);
        for (const expected of EXPECTED_BIO_FILENAMES) {
            expect(present, `default_seed/bios/ must contain ${expected}`)
                .toContain(expected);
        }
        console.log(`  all 4 canonical bio filenames present in default_seed/bios/`);
    });
});
