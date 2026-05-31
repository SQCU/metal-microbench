// UX-T4 — P-NO-EMPTY-FIRST-PAINT: welcome chat must be loaded on first paint.
//
// Source spec: docs/ux_debt_followup_tickets_2026_05_21.md §UX-T4
// Principle: P-NO-EMPTY-FIRST-PAINT
//
// The ticket says:
//   "stock SillyTavern ships with an Assistant character + working chat by
//    default precisely so a new install has zero empty state. Our st-debug
//    + plugin install should match this."
//
// This spec verifies the bootstrap mechanism that makes this possible:
//   1. The welcome chat JSONL exists in the _data directory (seeded by
//      bootstrap.sh from default/content/chats/dicemother/).
//   2. settings.json has active_character = dicemother.png AND
//      power_user.auto_load_chat = true so ST restores the chat on startup.
//   3. On first paint (no operator interaction), #chat has ≥3 turns
//      visible (≥1 user turn + ≥1 assistant turn as a minimum proxy).
//   4. No "select a character" or "no chats" empty state is shown.
//   5. Opening the suggester surface shows K=2 ranked rows within 30s
//      (suggester has chat content to score against).
//
// Verbatim acceptance lines (must fail against pre-bootstrap state):
//   expect(welcomeChatExists,         'welcome chat file must exist in _data').toBe(true);
//   expect(autoLoadChat,              'power_user.auto_load_chat must be true').toBe(true);
//   expect(activeCharacter,           'active_character must be dicemother.png').toBe('dicemother.png');
//   expect(chatMesCount,              '≥3 turns in #chat on first paint').toBeGreaterThanOrEqual(3);
//   expect(userMesCount,              '≥1 user turn in #chat').toBeGreaterThanOrEqual(1);
//   expect(assistantMesCount,         '≥1 assistant turn in #chat').toBeGreaterThanOrEqual(1);
//   expect(emptyStateVisible,         'no "select a character" empty state').toBe(false);
//   expect(suggesterRowCount,         'suggester has K=2 rows within 30s').toBeGreaterThanOrEqual(2);

import { test, expect } from '@playwright/test';
import { existsSync, readdirSync } from 'node:fs';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import { openPersonaSurface } from './_helpers/open_persona_surface.js';

const ST_BASE = 'http://127.0.0.1:8002';
const PLUGIN_BASE = `${ST_BASE}/api/plugins/user-personas`;

// Paths for filesystem checks (uses the live _data and the st-debug clone).
const DATA_ROOT =
    '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user';
const SETTINGS_PATH = path.join(DATA_ROOT, 'settings.json');
const DICEMOTHER_CHATS_DIR = path.join(DATA_ROOT, 'chats', 'dicemother');
const WELCOME_CHAT_FILENAME = 'dicemother - 2026-05-24@10h00m00s000ms.jsonl';

// Helper: open ST, wait for preloader to clear, wait for SillyTavern context.
async function loadST(page) {
    await page.goto('/');
    await page.waitForFunction(() => document.getElementById('preloader') === null,
        { timeout: 60_000 });
    await page.waitForFunction(() => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });
}

// Helper: open the suggester surface via the hamburger menu.
async function openSuggester(page) {
    await openPersonaSurface(page, 'suggester');
    const iframe = page.frameLocator('#user-suggester-button iframe');
    await expect(iframe.locator('h1')).toBeVisible({ timeout: 20_000 });
    return iframe;
}

test.describe('UX-T4: welcome chat on first paint', () => {
    test.setTimeout(90_000);

    // ── Filesystem checks (no browser needed) ──────────────────────────────

    test('welcome chat file exists in _data/default-user/chats/dicemother/', () => {
        const welcomeChatExists = existsSync(
            path.join(DICEMOTHER_CHATS_DIR, WELCOME_CHAT_FILENAME)
        );
        // Verbatim acceptance line:
        expect(welcomeChatExists, 'welcome chat file must exist in _data').toBe(true);

        console.log(`  welcome chat: ${path.join(DICEMOTHER_CHATS_DIR, WELCOME_CHAT_FILENAME)}`);
    });

    test('welcome chat is the most recent file in chats/dicemother/', () => {
        if (!existsSync(DICEMOTHER_CHATS_DIR)) {
            test.skip();
            return;
        }
        const files = readdirSync(DICEMOTHER_CHATS_DIR)
            .filter(f => f.endsWith('.jsonl'))
            .sort();
        // The welcome chat timestamp (2026-05-24) sorts after all existing
        // session timestamps, so it should be last in the sorted list.
        const lastFile = files[files.length - 1];
        expect(lastFile, 'welcome chat must sort as most recent filename').toBe(WELCOME_CHAT_FILENAME);
        console.log(`  most recent chat: ${lastFile} (out of ${files.length} total)`);
    });

    test('welcome chat has ≥3 user + ≥3 assistant turns in JSONL content', () => {
        const chatPath = path.join(DICEMOTHER_CHATS_DIR, WELCOME_CHAT_FILENAME);
        if (!existsSync(chatPath)) {
            test.skip();
            return;
        }
        const lines = readFileSync(chatPath, 'utf8')
            .split('\n')
            .filter(l => l.trim())
            .map(l => JSON.parse(l));

        // Skip the header (index 0).
        const messages = lines.slice(1);
        const userTurns = messages.filter(m => m.is_user === true);
        const assistantTurns = messages.filter(m => m.is_user === false && !m.is_system);

        expect(userTurns.length, '≥3 user turns in welcome chat JSONL').toBeGreaterThanOrEqual(3);
        expect(assistantTurns.length, '≥3 assistant turns in welcome chat JSONL').toBeGreaterThanOrEqual(3);
        console.log(`  turns: ${userTurns.length} user, ${assistantTurns.length} assistant`);
    });

    test('settings.json has auto_load_chat=true and active_character=dicemother.png', () => {
        if (!existsSync(SETTINGS_PATH)) {
            test.skip();
            return;
        }
        const settings = JSON.parse(readFileSync(SETTINGS_PATH, 'utf8'));
        const autoLoadChat = settings.power_user?.auto_load_chat;
        const activeCharacter = settings.active_character;

        // Verbatim acceptance lines:
        expect(autoLoadChat, 'power_user.auto_load_chat must be true').toBe(true);
        expect(activeCharacter, 'active_character must be dicemother.png').toBe('dicemother.png');
        console.log(`  auto_load_chat=${autoLoadChat}, active_character=${activeCharacter}`);
    });

    // ── Browser / DOM checks ───────────────────────────────────────────────

    test('first paint: #chat has ≥3 turns with no select-character empty state', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'first-paint render test is desktop-only');

        await loadST(page);

        // Give auto_load_chat time to fire and render the welcome chat.
        // ST calls RA_autoloadchat() on init — allow up to 10s for
        // the character to be selected and the chat to render.
        const chatMessages = page.locator('#chat .mes:not(.smallSysMes)');
        await expect(chatMessages.first(), 'first chat message appears within 10s')
            .toBeVisible({ timeout: 10_000 });

        const chatMesCount = await chatMessages.count();
        const userMesCount = await page.locator('#chat .mes[is_user="true"]:not(.smallSysMes)').count();
        const assistantMesCount = await page.locator('#chat .mes[is_user="false"]:not(.smallSysMes)').count();

        console.log(`  #chat messages: ${chatMesCount} total, ${userMesCount} user, ${assistantMesCount} assistant`);

        // Verbatim acceptance lines:
        expect(chatMesCount, '≥3 turns in #chat on first paint').toBeGreaterThanOrEqual(3);
        expect(userMesCount, '≥1 user turn in #chat').toBeGreaterThanOrEqual(1);
        expect(assistantMesCount, '≥1 assistant turn in #chat').toBeGreaterThanOrEqual(1);

        // No select-character empty-state text.
        // ST shows "Select a character to start chatting" or similar when
        // no character is loaded.
        const pageText = await page.locator('body').innerText();
        const emptyStateVisible =
            /select a character/i.test(pageText) ||
            /no chats.*create one/i.test(pageText) ||
            /click here to begin/i.test(pageText);
        // Verbatim acceptance line:
        expect(emptyStateVisible, 'no "select a character" empty state').toBe(false);

        await page.screenshot({ path: '/tmp/spec86_first_paint.png', fullPage: true });
        console.log('  screenshot: /tmp/spec86_first_paint.png');
    });

    test('suggester has K=2 ranked rows within 30s when chat is loaded', async ({ page }, testInfo) => {
        test.skip(testInfo.project.name !== 'desktop',
            'suggester render test is desktop-only');

        await loadST(page);

        // Wait for the chat to be loaded (the suggester needs a non-empty
        // chat to score against).
        const chatMessages = page.locator('#chat .mes:not(.smallSysMes)');
        await expect(chatMessages.first(), 'chat must be loaded before opening suggester')
            .toBeVisible({ timeout: 10_000 });

        // Open the suggester.
        const iframe = await openSuggester(page);

        // Wait for K=2 ranked rows to appear. The suggester calls /yapper-seed
        // + ranks the corpus by L2 distance. With a non-empty chat present,
        // this should complete within 30s (cold extraction) or instantly
        // (if the cache was primed by a prior run).
        const rankedRows = iframe.locator('#ranked-list .ranked-row');
        await expect(rankedRows.first(), 'first ranked row appears within 30s')
            .toBeVisible({ timeout: 30_000 });

        const suggesterRowCount = await rankedRows.count();
        // Verbatim acceptance line:
        expect(suggesterRowCount, 'suggester has K=2 rows within 30s').toBeGreaterThanOrEqual(2);

        console.log(`  suggester rows: ${suggesterRowCount}`);
        await page.screenshot({ path: '/tmp/spec86_suggester_rows.png', fullPage: true });
        console.log('  screenshot: /tmp/spec86_suggester_rows.png');
    });
});
