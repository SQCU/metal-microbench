// Seed-chat fixture loader.
//
// The R&D fork commits canonical seed chats under
// sillytavern-fork/default/content/chats/<character>/<seed_id>.jsonl.
// Those are NOT picked up by ST's content-seeding flow (chat isn't a
// registered seedable content type), so for playwright runs we copy
// the desired fixture into the live data dir at test setup time. The
// filename gets a future timestamp so ST sorts it as the most-recent
// chat, which is what ST loads when the character is clicked.
//
// Usage:
//
//     import { loadSeedChat } from './_helpers/seed_chat.mjs';
//
//     await loadSeedChat(page, {
//         character: 'dicemother',
//         seed_id: 'accusation',
//     });
//
// After this returns the chat is loaded in ST's UI with the seed's
// 3 turns of history.
import fs from 'node:fs';
import path from 'node:path';
import { selectCharacterByClick } from './elicit_clean.mjs';

const FORK_ROOT = '/Users/mdot/sillytavern-fork';
const DATA_USER = path.join(FORK_ROOT, 'data/default-user');

/**
 * Copy a seed chat fixture from the canonical content path into the
 * live chats dir, then open the chat via ST's UI.
 */
export async function loadSeedChat(page, { character, seed_id }) {
    const srcPath = path.join(
        FORK_ROOT, 'default/content/chats', character, `seed_${seed_id}.jsonl`);
    if (!fs.existsSync(srcPath)) {
        throw new Error(`seed chat fixture missing: ${srcPath}`);
    }
    // Make the seed the ONLY chat for this character. Move any
    // pre-existing chats aside into a sibling _backup_pre_seed/ dir
    // so they don't compete with the seed as "most recent." This
    // makes ST's character-click → chat-load deterministic: there's
    // only one chat to load.
    const liveDir = path.join(DATA_USER, 'chats', character);
    const backupDir = path.join(DATA_USER, 'chats', `${character}__backup_pre_seed`);
    fs.mkdirSync(liveDir, { recursive: true });
    fs.mkdirSync(backupDir, { recursive: true });
    for (const f of fs.readdirSync(liveDir)) {
        if (!f.endsWith('.jsonl')) continue;
        fs.renameSync(path.join(liveDir, f), path.join(backupDir, f));
    }
    const dstName = `${character} - 2099-01-01@00h00m00s000ms_SEED_${seed_id}.jsonl`;
    const dstPath = path.join(liveDir, dstName);
    fs.copyFileSync(srcPath, dstPath);
    const now = Date.now() / 1000;
    fs.utimesSync(dstPath, now, now);

    // Wipe session storage so ST doesn't restore the prior chat
    // pointer from before the seed write, then reload. This guarantees
    // ST reads the chat list fresh from disk on bootup.
    await page.evaluate(() => {
        try {
            localStorage.removeItem('SillyTavern_ChatHistory');
            localStorage.removeItem('SillyTavern_LastChat');
        } catch {}
    });
    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForFunction(
        'document.getElementById("preloader") === null',
        { timeout: 60_000 });
    await page.waitForFunction(
        () => typeof window.SillyTavern?.getContext === 'function',
        { timeout: 30_000 });

    // Switch into the character — ST will load the most recent chat,
    // which is our just-written seed (only chat for this character).
    await selectCharacterByClick(page, character);

    // Verify the loaded chat contains the seed's last message.
    // Read the seed JSONL ourselves to know what to look for.
    const seedLines = fs.readFileSync(srcPath, 'utf8').trim().split('\n');
    const lastMsg = JSON.parse(seedLines[seedLines.length - 1]);
    const expectedFragment = (lastMsg.mes || '').slice(0, 40);

    await page.waitForFunction((fragment) => {
        try {
            const ctx = window.SillyTavern?.getContext?.();
            const chat = ctx?.chat || [];
            const lastMes = chat[chat.length - 1]?.mes || '';
            return lastMes.includes(fragment);
        } catch { return false; }
    }, expectedFragment, { timeout: 10_000 });

    return { srcPath, dstPath, expectedFragment };
}
