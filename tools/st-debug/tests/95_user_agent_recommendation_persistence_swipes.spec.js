import { test, expect } from '@playwright/test';
import fs from 'node:fs/promises';
import fsSync from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';

const PLUGIN_BASE = '/api/plugins/user-personas';
const TEST_CHAT_NAME = 'dicemother - 2026-05-26@12h34m56s789ms user-agent-persistence';
const TEST_CHAT_FILE = `${TEST_CHAT_NAME}.jsonl`;
const BASIC_AUTH_FALLBACK = { username: 'sussy', password: 'amongus' };
const HERE = path.dirname(fileURLToPath(import.meta.url));
const SCREENSHOT_DIR = path.join(HERE, 'screenshots');

function loadManifest() {
    const manifestPath = process.env.ST_MATRIX_MANIFEST;
    if (!manifestPath || !fsSync.existsSync(manifestPath)) return null;
    return {
        path: manifestPath,
        dir: path.dirname(manifestPath),
        body: JSON.parse(fsSync.readFileSync(manifestPath, 'utf8')),
    };
}

function authHeader(manifest) {
    const auth = manifest.basicAuth || BASIC_AUTH_FALLBACK;
    return `Basic ${Buffer.from(`${auth.username}:${auth.password}`).toString('base64')}`;
}

function rowSet(label) {
    return ['alpha', 'beta'].map((id, i) => ({
        bio_id: `${label}-${id}.png`,
        agent_id: `${label}-${id}-agent`,
        distance: 0.1 + i / 10,
        why: `${label} fixture ${id}`,
        persona: {
            id: `${label}-${id}.png`,
            name: `${label} ${id}`,
            bio: `${label} ${id} bio`,
            provenance: { kind: 'swipe-persistence-e2e' },
        },
        agent: {
            id: `${label}-${id}-agent`,
            name: `${label} ${id} agent`,
            designed_for_bio_id: `${label}-${id}.png`,
            provenance: { kind: 'swipe-persistence-e2e' },
        },
    }));
}

function chatJsonl() {
    const header = {
        chat_metadata: {
            integrity: 'user-agent-recommendation-persistence-v1',
            seed_id: 'dicemother_six_turn_swipe_recommendation_persistence',
        },
        user_name: 'scringlo scrambler',
        character_name: 'dicemother',
    };
    const swipeA = 'SWIPE_A_MARKER: the night-jay pecks once at the message tube, then goes still; the seal smells faintly of rain and iron.';
    const swipeB = 'SWIPE_B_MARKER: the night-jay hops onto your wrist and turns its banded leg toward the stable door as if pointing you away.';
    const lines = [
        header,
        {
            name: 'dicemother',
            is_user: false,
            is_system: false,
            send_date: '2026-05-26T12:30:00.000Z',
            mes: 'The inn yard is muddy and quiet. A sealed satchel rests on an overturned cask while rain taps the stable roof.',
            extra: {},
            swipes: ['The inn yard is muddy and quiet. A sealed satchel rests on an overturned cask while rain taps the stable roof.'],
            swipe_id: 0,
            swipe_info: [{ send_date: '2026-05-26T12:30:00.000Z', extra: {} }],
        },
        {
            name: 'scringlo scrambler',
            is_user: true,
            is_system: false,
            send_date: '2026-05-26T12:30:10.000Z',
            mes: 'I set the satchel down carefully and check whether anyone followed me into the yard.',
            extra: { isSmallSys: false },
            force_avatar: '/thumbnail?type=persona&file=1779035204660-scringloscrambler.png',
        },
        {
            name: 'dicemother',
            is_user: false,
            is_system: false,
            send_date: '2026-05-26T12:30:20.000Z',
            mes: 'No one follows. The stable latch clicks in the wind, and something inside the satchel shifts against the leather.',
            extra: {},
            swipes: ['No one follows. The stable latch clicks in the wind, and something inside the satchel shifts against the leather.'],
            swipe_id: 0,
            swipe_info: [{ send_date: '2026-05-26T12:30:20.000Z', extra: {} }],
        },
        {
            name: 'scringlo scrambler',
            is_user: true,
            is_system: false,
            send_date: '2026-05-26T12:30:35.000Z',
            mes: 'I open the clasp just enough to see what is breathing in there.',
            extra: { isSmallSys: false },
            force_avatar: '/thumbnail?type=persona&file=1779035204660-scringloscrambler.png',
        },
        {
            name: 'dicemother',
            is_user: false,
            is_system: false,
            send_date: '2026-05-26T12:30:45.000Z',
            mes: 'A hooded night-jay blinks up at you, one leg banded silver with a sealed message tube. It does not panic.',
            extra: {},
            swipes: ['A hooded night-jay blinks up at you, one leg banded silver with a sealed message tube. It does not panic.'],
            swipe_id: 0,
            swipe_info: [{ send_date: '2026-05-26T12:30:45.000Z', extra: {} }],
        },
        {
            name: 'dicemother',
            is_user: false,
            is_system: false,
            send_date: '2026-05-26T12:31:00.000Z',
            mes: swipeA,
            extra: {},
            swipes: [swipeA, swipeB],
            swipe_id: 0,
            swipe_info: [
                { send_date: '2026-05-26T12:31:00.000Z', extra: {} },
                { send_date: '2026-05-26T12:31:01.000Z', extra: {} },
            ],
        },
    ];
    return `${lines.map(line => JSON.stringify(line)).join('\n')}\n`;
}

async function installConversation(instance) {
    const chatDir = path.join(instance.dataRoot, 'default-user', 'chats', 'dicemother');
    await fs.mkdir(chatDir, { recursive: true });
    await fs.writeFile(path.join(chatDir, TEST_CHAT_FILE), chatJsonl(), 'utf8');
}

async function loadClient(page, instance) {
    await page.goto(instance.url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });
}

async function openSeededChat(page) {
    await page.evaluate(async (chatName) => {
        const st = await import('/script.js');
        if (!Array.isArray(st.characters) || st.characters.length === 0) {
            await st.getCharacters();
        }
        const dicemotherId = st.characters.findIndex(c => c?.avatar === 'dicemother.png' || c?.name === 'dicemother');
        if (dicemotherId >= 0) {
            await st.selectCharacterById(dicemotherId);
        }
        await st.openCharacterChat(chatName);
    }, TEST_CHAT_NAME);
    await page.waitForFunction((chatName) => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.chatId === chatName && ctx?.chat?.some(m => /SWIPE_[AB]_MARKER/.test(String(m.mes || '')));
    }, TEST_CHAT_NAME, { timeout: 30_000 });
}

async function openPanel(page) {
    const button = page.locator('#user_personas_btn');
    await expect(button).toBeVisible({ timeout: 30_000 });
    const panel = page.locator('#user_personas_panel');
    if (!await panel.isVisible().catch(() => false)) {
        await button.click();
    }
    await expect(panel).toBeVisible({ timeout: 15_000 });
    return panel;
}

async function savePanelScreenshot(testInfo, panel, fileName) {
    await fs.mkdir(SCREENSHOT_DIR, { recursive: true });
    const body = await panel.screenshot();
    await fs.writeFile(path.join(SCREENSHOT_DIR, fileName), body);
    await testInfo.attach(fileName, { body, contentType: 'image/png' });
}

async function installRecommendationRoutes(page) {
    const calls = [];
    let releaseFirstTarget;
    let releaseFirstSwipeB;
    const firstTargetGate = new Promise(resolve => { releaseFirstTarget = resolve; });
    const firstSwipeBGate = new Promise(resolve => { releaseFirstSwipeB = resolve; });

    await page.route(`**${PLUGIN_BASE}/yapper-seed`, async route => {
        const body = route.request().postDataJSON();
        const summary = String(body.chat_context_summary || '');
        const variant = summary.includes('SWIPE_B_MARKER')
            ? 'Swipe B'
            : summary.includes('SWIPE_A_MARKER')
                ? 'Swipe A'
                : 'Other';
        calls.push({ variant, summary });
        if (variant === 'Swipe A' && calls.filter(c => c.variant === 'Swipe A').length === 1) {
            await firstTargetGate;
        }
        if (variant === 'Swipe B' && calls.filter(c => c.variant === 'Swipe B').length === 1) {
            await firstSwipeBGate;
        }
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                top: rowSet(variant),
                side: [],
                _meta: {
                    K_top: 2,
                    K_side: 0,
                    target_signature: { variant },
                    target_completed_axes: 1,
                    candidates_considered: 2,
                    bios_total: 2,
                    agents_total: 2,
                    pending_synthesis: [],
                    pending_count: 0,
                },
            }),
        });
    });
    await page.route(`**${PLUGIN_BASE}/poll`, async route => {
        const body = route.request().postDataJSON();
        await route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
                applied_overlay: {
                    source: 'agent',
                    agent_id: `${body.persona_id}-agent`,
                    name: `Overlay for ${body.persona_id}`,
                    depth: 1,
                    text_chars: 40,
                },
                candidates: [{
                    text: `Buffered continuation for ${body.persona_id}`,
                    truncated: false,
                }],
            }),
        });
    });

    return {
        releaseFirstTarget,
        releaseFirstSwipeB,
        get calls() { return calls; },
        targetCount(variant) { return calls.filter(c => c.variant === variant).length; },
    };
}

async function swipeLastAssistantToSecondSwipe(page) {
    await page.evaluate(async () => {
        const st = await import('/script.js');
        const constants = await import('/scripts/constants.js');
        const ctx = window.SillyTavern.getContext();
        await st.swipe(null, constants.SWIPE_DIRECTION.RIGHT, {
            forceMesId: ctx.chat.length - 1,
            forceSwipeId: 1,
            forceDuration: 0,
        });
        await st.saveChatConditional();
    });
    await page.waitForFunction(() => {
        const ctx = window.SillyTavern?.getContext?.();
        return ctx?.chat?.at(-1)?.swipe_id === 1 &&
            String(ctx.chat.at(-1).mes || '').includes('SWIPE_B_MARKER');
    }, { timeout: 15_000 });
}

async function restartInstance(manifestInfo, instance, request) {
    const pidsPath = path.join(manifestInfo.dir, 'pids.json');
    const pids = JSON.parse(await fs.readFile(pidsPath, 'utf8'));
    const rec = pids.find(p => p.name === instance.name);
    if (!rec) throw new Error(`no pid record for ${instance.name}`);
    try { process.kill(-rec.pid, 'SIGTERM'); } catch { try { process.kill(rec.pid, 'SIGTERM'); } catch {} }
    await new Promise(resolve => setTimeout(resolve, 1500));
    try { process.kill(-rec.pid, 'SIGKILL'); } catch { try { process.kill(rec.pid, 'SIGKILL'); } catch {} }

    const logFd = fsSync.openSync(instance.logPath, 'a');
    const child = spawn('node', ['server.js', '--configPath', instance.configPath], {
        cwd: instance.cloneDir,
        detached: true,
        stdio: ['ignore', logFd, logFd],
        env: {
            ...process.env,
            ST_PORT: String(instance.port),
            SERVER_PORT: String(instance.port),
            ST_URL: instance.url,
            USER_PERSONAS_ST_URL: instance.url,
            BRIDGE_URL: manifestInfo.body.bridgeUrl,
            USER_PERSONAS_BRIDGE_URL: manifestInfo.body.bridgeUrl,
            USER_PERSONAS_DISABLE_BOOT_AUTOSYNTH: '1',
        },
    });
    child.unref();
    rec.pid = child.pid;
    await fs.writeFile(pidsPath, JSON.stringify(pids, null, 2));

    const deadline = Date.now() + 90_000;
    let last = '';
    while (Date.now() < deadline) {
        const response = await request.get(`${instance.url}${PLUGIN_BASE}/runtime-config`, {
            headers: { Authorization: authHeader(manifestInfo.body) },
        }).catch(e => {
            last = e.message;
            return null;
        });
        if (response?.ok()) return;
        if (response) last = `HTTP ${response.status()}`;
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    throw new Error(`${instance.name} did not restart cleanly: ${last}`);
}

test.describe('user-agent top-k recommendation persistence across chat swipes', () => {
    test.setTimeout(180_000);

    test('seeded dicemother chat has pending real top-k call, distinct swipe choices, and persisted reload/server-restart choices', async ({ browser, request }, testInfo) => {
        const manifestInfo = loadManifest();
        test.skip(!manifestInfo, 'ST_MATRIX_MANIFEST is required; run sillytavern-fork/tests/scripts/multi_st_matrix.mjs (run from the fork) start');
        const manifest = manifestInfo.body;
        const instance = manifest.instances[Math.floor(Math.random() * manifest.instances.length)];
        await installConversation(instance);

        const context = await browser.newContext({
            httpCredentials: manifest.basicAuth || BASIC_AUTH_FALLBACK,
        });
        const page = await context.newPage();
        const routes = await installRecommendationRoutes(page);

        await loadClient(page, instance);
        await openSeededChat(page);
        const panel = await openPanel(page);

        await expect.poll(() => routes.targetCount('Swipe A'), {
            timeout: 10_000,
            intervals: [100, 250, 500],
        }).toBe(1);
        await expect(panel).toContainText(/pending user-agent top-k choices/i, { timeout: 5_000 });
        await savePanelScreenshot(testInfo, panel, '95_pending_yapper_seed_first_paint.png');

        routes.releaseFirstTarget();
        await expect(panel.locator('.user-personas-card-name').first())
            .toContainText('Swipe A alpha', { timeout: 15_000 });
        await savePanelScreenshot(testInfo, panel, '95_resolved_yapper_seed_swipe_a.png');

        await swipeLastAssistantToSecondSwipe(page);
        await expect.poll(() => routes.targetCount('Swipe B'), {
            timeout: 10_000,
            intervals: [100, 250, 500],
        }).toBe(1);
        await expect(panel).toContainText(/pending user-agent top-k choices/i, { timeout: 10_000 });
        routes.releaseFirstSwipeB();
        await expect(panel.locator('.user-personas-card-name').first())
            .toContainText('Swipe B alpha', { timeout: 15_000 });
        await savePanelScreenshot(testInfo, panel, '95_resolved_yapper_seed_swipe_b.png');
        expect(routes.targetCount('Swipe B'), 'swiping selected last turn triggered a distinct top-k call').toBe(1);

        const aBeforeReload = routes.targetCount('Swipe A');
        const bBeforeReload = routes.targetCount('Swipe B');
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction('document.getElementById("preloader") === null', { timeout: 60_000 });
        await openSeededChat(page);
        const reloadedPanel = await openPanel(page);
        await expect(reloadedPanel.locator('.user-personas-card-name').first())
            .toContainText('Swipe B alpha', { timeout: 15_000 });
        await savePanelScreenshot(testInfo, reloadedPanel, '95_hydrated_after_browser_refresh.png');
        expect(routes.targetCount('Swipe A'), 'hard refresh did not re-rank the first swipe context').toBe(aBeforeReload);
        expect(routes.targetCount('Swipe B'), 'hard refresh hydrated persisted selected-swipe top-k without a new yapper-seed call').toBe(bBeforeReload);

        await restartInstance(manifestInfo, instance, request);
        await page.reload({ waitUntil: 'domcontentloaded' });
        await page.waitForFunction('document.getElementById("preloader") === null', { timeout: 60_000 });
        await openSeededChat(page);
        const restartedPanel = await openPanel(page);
        await expect(restartedPanel.locator('.user-personas-card-name').first())
            .toContainText('Swipe B alpha', { timeout: 15_000 });
        await savePanelScreenshot(testInfo, restartedPanel, '95_hydrated_after_st_server_restart.png');
        expect(routes.targetCount('Swipe A'), 'ST restart did not re-rank the first swipe context').toBe(aBeforeReload);
        expect(routes.targetCount('Swipe B'), 'ST server restart hydrated file-persisted selected-swipe top-k without a new yapper-seed call').toBe(bBeforeReload);

        await context.close();
    });
});
