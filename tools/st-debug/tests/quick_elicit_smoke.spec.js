// One-trial smoke: does the clean harness produce a tool call on
// a single "draw me X" prompt, given the current chat-template fix
// + persona + tool descriptions? ~2-min wall.
//
// Not a reliability claim. Just "does it ever fire under this path."

import { test, expect } from '@playwright/test';
import { loadAndConnect, sendAndObserve, selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';

test.describe('elicit smoke', () => {
    test.setTimeout(600_000);

    test('single trial: ask for voronoi via real UI, see what happens', async ({ page }, testInfo) => {
        // Capture all outbound requests at the network layer (more
        // reliable than fetch-monkey-patch, which we wired earlier and
        // saw fire 0 times — likely ST uses XHR or an internal helper).
        const capturedReqs = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('chat/completions') ||
                url.includes('chat-completions') ||
                url.includes('generate') ||
                url.includes('/api/backends/openai/')) {
                let body = null;
                try { body = req.postDataJSON(); } catch (e) {
                    body = { _parse_error: String(e), raw: req.postData()?.slice(0, 500) };
                }
                capturedReqs.push({ url, method: req.method(), body });
            }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        // Inspect chat[] shape after character select so we can see what
        // entries exist and whether is_system / is_user flags are set.
        const postSelect = await page.evaluate(() => {
            const ctx = window.SillyTavern?.getContext?.();
            const chat = ctx?.chat || [];
            return chat.map((e, i) => ({
                i, is_user: !!e.is_user, is_system: !!e.is_system,
                name: e.name || null, mes_head: (e.mes || '').slice(0, 80),
            }));
        });
        console.log('chat[] after select:', JSON.stringify(postSelect, null, 2));

        const r = await sendAndObserve(
            page,
            'draw me a voronoi diagram with 12 seeds, distinct fill colors per cell',
            { timeoutMs: 300_000 },
        );
        console.log('=== smoke result ===');
        console.log(`finishState: ${r.finishState}`);
        console.log(`elapsedMs: ${Math.round(r.elapsedMs)}`);
        console.log(`toolInvocations: ${JSON.stringify(r.toolInvocations)}`);
        console.log(`toolProgress: ${JSON.stringify(r.toolProgress.map(p => ({l:p.label,s:p.status})))}`);
        const txt = (r.assistantText || '').slice(0, 400);
        console.log(`assistantText: ${txt}`);

        for (let i = 0; i < capturedReqs.length; i++) {
            const { url, method, body } = capturedReqs[i];
            console.log(`--- captured request ${i}: ${method} ${url} ---`);
            const b = body || {};
            const msgs = b.messages || [];
            console.log(`  msg count: ${msgs.length}`);
            console.log(`  msg roles: ${msgs.map(m => m.role).join(', ')}`);
            for (let mi = 0; mi < msgs.length; mi++) {
                const m = msgs[mi];
                const c = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
                console.log(`    msg[${mi}] role=${m.role} content_len=${(c || '').length}`);
                console.log(`      head: ${(c || '').slice(0, 200)}`);
                if ((c || '').length > 200) {
                    console.log(`      tail: ...${(c || '').slice(-200)}`);
                }
            }
            console.log(`  has tools field: ${'tools' in b}`);
            console.log(`  tool count: ${(b.tools || []).length}`);
            console.log(`  tool names: ${(b.tools || []).map(t => t.function?.name || t.name).filter(Boolean).join(', ')}`);
            console.log(`  tool_choice: ${JSON.stringify(b.tool_choice)}`);
            console.log(`  stream: ${b.stream}`);
            console.log(`  temperature: ${b.temperature}`);
        }
        console.log(`total captured requests: ${capturedReqs.length}`);

        // No-rate assertion — just liveness
        expect(['completed', 'tool_handled']).toContain(r.finishState);
    });
});
