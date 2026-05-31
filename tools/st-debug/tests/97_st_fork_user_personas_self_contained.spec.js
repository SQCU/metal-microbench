import { test, expect } from '@playwright/test';
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

function loadManifest() {
    const manifestPath = process.env.ST_MATRIX_MANIFEST;
    if (!manifestPath || !fs.existsSync(manifestPath)) return null;
    return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
}

function resolveStForkRoot() {
    const manifest = loadManifest();
    if (manifest?.instances?.[0]?.cloneDir) return manifest.instances[0].cloneDir;
    return process.env.ST_FORK_ROOT || path.resolve(process.cwd(), 'tools/st-debug/sillytavern-fork');
}

test.describe('st-fork user-personas ownership boundary', () => {
    test('plugin runtime has no workstation absolute path defaults', async () => {
        const stRoot = resolveStForkRoot();
        const pluginRoot = path.join(stRoot, 'plugins', 'user-personas');
        const lintScript = path.join(pluginRoot, 'scripts', 'lint_no_host_paths.mjs');

        expect(fs.existsSync(lintScript), `${lintScript} must exist inside st-fork`).toBeTruthy();
        const output = execFileSync('node', [lintScript], {
            cwd: stRoot,
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'pipe'],
        });
        expect(output).toContain('[lint-no-host-paths] clean');
    });
});
