// End-to-end streaming tool-call residue test.
//
// The user reported: "first few tokens of a tool call stream out into
// the SillyTavern client, then vanish out of existence once parsed,
// with no return EVER from the persona-effort-schema tool, and a
// client state indicating the streaming request finished."
//
// Tests 28 + 29 (now deleted) used ctx.ToolManager.invokeFunctionTool
// to drive the FE plugin directly, bypassing the entire streaming
// pipeline. They proved post-dispatch rendering contracts but never
// exercised the path where tool_calls are parsed from SSE chunks and
// then dispatched through ST's chat-completion handler. The user's bug
// lives in that gap.
//
// This test does the real thing: drives the actual chat UI, sends a
// message that should elicit a tool call, records continuous video,
// captures every SSE chunk arriving from the bridge, takes screenshots
// at fixed intervals during streaming, and asserts on what a real user
// would see at the end.
//
// THE PRINCIPLE: rendered pixels (and DOM as the user-visible surface)
// are the only ground truth. Internal state probes are deleted; only
// what the chat surface actually displays matters.
//
// FORENSIC EVIDENCE captured per run, regardless of pass/fail:
//   - video.webm (continuous recording of the entire session)
//   - 01_pre_send.png  ... 99_final.png (interval screenshots)
//   - sse_chunks.jsonl (every SSE delta from the bridge)
//   - request_bodies.jsonl (every /generate, /start_invoke, etc.)
//   - dom_snapshots.json (DOM state at each screenshot)
//   - final_visible_text.txt (everything visible in #chat at end)

import { test, expect } from '@playwright/test';
import { loadAndConnect,
    selectCharacterByClick, freshChatByClick } from './_helpers/elicit_clean.mjs';
import fs from 'node:fs';

test.use({ video: 'on' });

test.describe('streaming tool-call residue, end-to-end', () => {
    test.setTimeout(8 * 60 * 1000);

    test('scringlo invoking persona-effort-schema: tool call leaves persistent visible artifact OR persistent error', async ({ page }, testInfo) => {
        // ── HTTP traffic capture ──────────────────────────────────────
        const sseChunks = [];
        const requestBodies = [];
        page.on('request', (req) => {
            const url = req.url();
            if (url.includes('/generate') || url.includes('/start_invoke') ||
                url.includes('/poll/') || url.includes('/cancel') ||
                url.includes('/chat/completions') || url.includes('/sessions')) {
                let body = null;
                try { body = req.postDataJSON(); } catch { body = req.postData()?.slice(0, 2000) || null; }
                requestBodies.push({
                    t: Date.now(), method: req.method(), url, body,
                });
            }
        });
        page.on('response', async (resp) => {
            const ct = resp.headers()['content-type'] || '';
            if (!ct.includes('text/event-stream')) return;
            try {
                const text = await resp.text();
                // Split on SSE record boundaries (blank line).
                for (const record of text.split('\n\n')) {
                    if (!record.trim()) continue;
                    sseChunks.push({
                        t: Date.now(), url: resp.url(), raw: record,
                    });
                }
            } catch { /* response body unavailable */ }
        });

        await loadAndConnect(page);
        await selectCharacterByClick(page, 'scringlo');
        await freshChatByClick(page);
        await page.screenshot({ path: testInfo.outputPath('01_pre_send.png'), fullPage: true });

        // ── Interval-screenshot timer ─────────────────────────────────
        // Take a screenshot every 1.5s during the entire test. Each
        // frame is saved with a sequential index + timestamp so we can
        // reconstruct the visual timeline of what the user actually saw.
        const intervalShots = [];
        let shotCounter = 10;
        const shotTimer = setInterval(async () => {
            try {
                const i = shotCounter++;
                const fname = `${String(i).padStart(2, '0')}_interval.png`;
                const t = Date.now();
                await page.screenshot({
                    path: testInfo.outputPath(fname),
                    fullPage: true,
                });
                intervalShots.push({ t, file: fname });
            } catch { /* page may have closed */ }
        }, 1500);

        // ── Trigger a tool-eliciting prompt ───────────────────────────
        // NATURAL prompt — the simplest possible integration. No
        // spoonfeeding of tool name or args. The earlier version of
        // this test used a contrived prompt that named the tool and
        // dictated every arg verbatim; that masked the actual tool-
        // design problem (persona-effort-schema required
        // persona_system_prompt as a string arg, an input the
        // integration could not consistently deliver in a single
        // tool_call body — long string with nested quoting, atomic
        // strings, etc.). The fix was structural: redesign the tool's
        // input shape so the failure case is unreachable. The tool
        // now takes zero required args and reads the character's
        // system prompt from caller_messages server-side. The
        // diagnostic-residue infrastructure in the bridge stays as
        // a safety net for any tool whose input grammar admits an
        // unparseable shape — but the goal is for that residue to
        // never need to render.
        const userPrompt = 'hey scringlo, what should your reasoning effort levels be?';
        await page.locator('#send_textarea').fill(userPrompt);
        const sendT0 = Date.now();
        await page.locator('#send_but').click();

        // Wait for the chat surface to settle. Settle = generation done
        // (no #mes_stop visible) + last assistant message present + any
        // in-flight tool_progress entries reached terminal state.
        // Bounded by the test timeout.
        let settled = false;
        const settleTimeoutMs = 5 * 60 * 1000;
        const settleEnd = Date.now() + settleTimeoutMs;
        while (Date.now() < settleEnd) {
            const state = await page.evaluate(() => {
                const ctx = window.SillyTavern.getContext();
                const chat = ctx.chat || [];
                if (!chat.length) return { generation_done: false };
                const stopBtn = document.querySelector('#mes_stop');
                const generating = stopBtn && stopBtn.offsetParent !== null;
                const last = chat[chat.length - 1];
                const isAssistantTerminal = last && !last.is_user && !last.is_system;
                // Count any tool_progress entries on the last assistant
                // turn that aren't terminal.
                const nonTerminal = isAssistantTerminal
                    ? ((last.extra?.tool_progress || []).filter(e =>
                        !['done', 'failed', 'cancelled'].includes(e.status)).length)
                    : 0;
                return {
                    generation_done: !generating,
                    has_assistant_msg: isAssistantTerminal,
                    non_terminal_tool_progress: nonTerminal,
                };
            });
            if (state.generation_done && state.has_assistant_msg && state.non_terminal_tool_progress === 0) {
                settled = true;
                break;
            }
            await page.waitForTimeout(500);
        }
        clearInterval(shotTimer);
        const settleElapsed = Date.now() - sendT0;
        console.log(`[settle] settled=${settled} elapsed=${settleElapsed}ms`);

        // ── Capture final rendered state ──────────────────────────────
        // Open any collapsibles so their bodies appear in screenshots
        // and innerText reads.
        await page.evaluate(() => {
            for (const d of document.querySelectorAll('#chat details')) {
                d.setAttribute('open', '');
            }
        });
        await page.waitForTimeout(500);
        await page.screenshot({ path: testInfo.outputPath('99_final.png'), fullPage: true });

        // Rip the entire #chat subtree as a graph. Each node carries:
        //   id (path), tag, classes (array), text (own innerText),
        //   visible (offsetHeight>0 AND display!=none AND visibility!=hidden),
        //   parent_id (null for root), child_ids.
        // The test makes assertions via graph queries against this dump,
        // not via innerText substring matching.
        const finalState = await page.evaluate(() => {
            // ── DOM graph dump ────────────────────────────────────────
            const root = document.getElementById('chat');
            const nodes = [];
            const idForNode = new Map();  // Element → id
            let nextId = 0;

            function visit(el, parentId) {
                if (el.nodeType !== 1) return;
                const id = nextId++;
                idForNode.set(el, id);
                const style = getComputedStyle(el);
                const h = el.offsetHeight;
                const visible = h > 0 && style.display !== 'none' && style.visibility !== 'hidden';
                // own_text: direct text-node descendants only, not deep
                let ownText = '';
                for (const c of el.childNodes) {
                    if (c.nodeType === 3) ownText += c.textContent;
                }
                ownText = ownText.trim();
                const childIds = [];
                for (const c of el.children) {
                    childIds.push(nextId);  // will be assigned by recursive visit
                    visit(c, id);
                }
                // Re-collect actual ids assigned to children (their idForNode entries).
                const realChildIds = Array.from(el.children).map(c => idForNode.get(c)).filter(x => x !== undefined);
                nodes.push({
                    id,
                    parent_id: parentId,
                    tag: el.tagName.toLowerCase(),
                    classes: Array.from(el.classList),
                    attrs: {
                        mesid: el.getAttribute('mesid'),
                        ch_name: el.getAttribute('ch_name'),
                        is_user: el.getAttribute('is_user'),
                        is_system: el.getAttribute('is_system'),
                    },
                    own_text_first_240: ownText.slice(0, 240),
                    full_text_first_240: (el.innerText || el.textContent || '').slice(0, 240),
                    visible,
                    offset_height: h,
                    display: style.display,
                    child_ids: realChildIds,
                });
            }
            if (root) visit(root, null);

            // ── chat[] side-channel: which mesid does each tool entry live on?
            const ctx = window.SillyTavern.getContext();
            const chatEntries = (ctx.chat || []).map((m, i) => {
                const domMes = document.querySelector(`#chat .mes[mesid="${i}"]`);
                return {
                    idx: i, is_user: !!m.is_user, is_system: !!m.is_system, name: m.name || '',
                    mes_first_240: (m.mes || '').slice(0, 240),
                    tool_progress_labels: (m.extra?.tool_progress || []).map(e =>
                        `${e.label || ''} • ${e.status || '?'}`),
                    tool_invocations_names: (m.extra?.tool_invocations || []).map(i =>
                        i.displayName || i.name),
                    mes_text_innerHTML_full: domMes
                        ? (domMes.querySelector('.mes_text')?.innerHTML || '')
                        : null,
                };
            });
            return { dom_graph_nodes: nodes, chat_entries: chatEntries };
        });

        // Persist forensic evidence.
        fs.writeFileSync(testInfo.outputPath('sse_chunks.jsonl'),
            sseChunks.map(c => JSON.stringify(c)).join('\n'));
        fs.writeFileSync(testInfo.outputPath('request_bodies.jsonl'),
            requestBodies.map(c => JSON.stringify(c)).join('\n'));
        fs.writeFileSync(testInfo.outputPath('final_state.json'),
            JSON.stringify({ settled, settleElapsed, intervalShots, ...finalState }, null, 2));

        console.log(`[evidence] sse_chunks=${sseChunks.length} request_bodies=${requestBodies.length}`);
        console.log(`[evidence] interval screenshots: ${intervalShots.length}`);
        console.log(`[evidence] DOM graph nodes under #chat: ${finalState.dom_graph_nodes.length}`);
        console.log(`[evidence] chat[] entries: ${finalState.chat_entries.length}`);

        // ── Detect whether a tool call was actually emitted by the model ──
        // Scan the SSE chunks for any sign of a tool_call: OpenAI shape
        // (`"tool_calls"`) or the literal gemma tokenizer-grammar marker
        // ("<|tool_call>" or "tool_call|>"). If neither shows up, the
        // model didn't try — this is an elicitation rate study, not a
        // residue bug, and we skip the strong assertion.
        const sseJoined = sseChunks.map(c => c.raw).join('\n');
        const sawOaiToolCalls = sseJoined.includes('"tool_calls"');
        const sawGemmaMarker = sseJoined.includes('<|tool_call>') || sseJoined.includes('tool_call|>');
        const modelAttemptedToolCall = sawOaiToolCalls || sawGemmaMarker;
        console.log(`[attempt] sawOaiToolCalls=${sawOaiToolCalls} sawGemmaMarker=${sawGemmaMarker}`);

        // ── HARD INVARIANT, visual-only assertion ─────────────────────
        // If the SSE shows the model attempted a tool call (either as
        // OpenAI tool_calls or as raw gemma markers), then the final
        // rendered chat MUST contain a permanent visible artifact about
        // that tool call. That artifact is one of:
        //   (a) a custom-tool-progress-collapsible element (toolcards FE plugin)
        //   (b) a custom-tool-invocations-collapsible element (ST's standard tool record)
        //   (c) a visible error message naming the tool
        //
        // If NONE of those exist after streaming ends, it's the
        // user-reported bug: tokens emerged, vanished after parsing,
        // and nothing persistent was left for the user/agent to see.
        if (!modelAttemptedToolCall) {
            console.log('[skip] model did not attempt a tool call in this run. ' +
                'Elicitation rate is a separate study — this test asserts on residue ' +
                'CONDITIONAL ON an attempt being made. Saving artifacts and passing.');
            return;
        }

        // ── Graph-based residue query ─────────────────────────────────
        // Build adjacency maps from the graph dump.
        const G = finalState.dom_graph_nodes;
        const byId = new Map(G.map(n => [n.id, n]));

        // ancestorChainVisible: walk parents; every ancestor must be visible.
        function ancestorChainVisible(node) {
            let cur = node;
            while (cur) {
                if (!cur.visible) return false;
                if (cur.parent_id == null) return true;
                cur = byId.get(cur.parent_id);
            }
            return true;
        }

        function classMatches(node, classFragment) {
            return node.classes.some(c => c.includes(classFragment));
        }

        // Find every visible descendant of #chat that's a tool-residue
        // node. A node qualifies iff:
        //   (a) it has class containing 'tool-progress-collapsible' OR
        //                              'tool-invocations-collapsible'
        //   (b) it is itself visible (visible: true)
        //   (c) every ancestor in its parent chain is also visible
        const toolResidueNodes = G.filter(n =>
            (classMatches(n, 'tool-progress-collapsible') ||
             classMatches(n, 'tool-invocations-collapsible')) &&
            n.visible && ancestorChainVisible(n)
        );

        // Find orphaned chat[] entries: those that have tool data but
        // whose corresponding .mes DOM node either doesn't exist or is
        // not visible.
        const orphans = [];
        for (const ce of finalState.chat_entries) {
            const hasToolData = ce.tool_progress_labels.length > 0 ||
                                ce.tool_invocations_names.length > 0;
            if (!hasToolData) continue;
            const mesNode = G.find(n => n.classes.includes('mes') &&
                                        n.attrs.mesid === String(ce.idx));
            if (!mesNode || !mesNode.visible || !ancestorChainVisible(mesNode)) {
                orphans.push({
                    chat_idx: ce.idx, name: ce.name,
                    tool_progress_labels: ce.tool_progress_labels,
                    tool_invocations_names: ce.tool_invocations_names,
                    mes_node_present: !!mesNode,
                    mes_node_visible: mesNode ? mesNode.visible : false,
                    mes_node_height: mesNode ? mesNode.offset_height : null,
                    mes_node_display: mesNode ? mesNode.display : null,
                });
            }
        }

        const evidence = {
            modelAttemptedToolCall, sawOaiToolCalls, sawGemmaMarker,
            visible_tool_residue_nodes: toolResidueNodes.length,
            tool_residue_summary: toolResidueNodes.map(n => ({
                tag: n.tag, classes: n.classes,
                text_first_240: n.full_text_first_240,
            })),
            orphan_chat_entries: orphans,
            graph_node_count: G.length,
            chat_entry_count: finalState.chat_entries.length,
            interval_screenshots: intervalShots.map(s => s.file),
        };
        fs.writeFileSync(testInfo.outputPath('invariant_evidence.json'),
            JSON.stringify(evidence, null, 2));
        fs.writeFileSync(testInfo.outputPath('dom_graph.json'),
            JSON.stringify(finalState.dom_graph_nodes, null, 2));

        console.log(`[graph] tool residue nodes (visible+ancestor-visible): ${toolResidueNodes.length}`);
        console.log(`[graph] orphan chat entries (have tool data, no visible DOM): ${orphans.length}`);
        for (const o of orphans) {
            console.log(`  - chat[${o.chat_idx}] ${o.name} tool_data=${[...o.tool_progress_labels, ...o.tool_invocations_names]} mes_node=${o.mes_node_present} visible=${o.mes_node_visible} height=${o.mes_node_height} display=${o.mes_node_display}`);
        }

        // HARD INVARIANT (graph query):
        //   IF the SSE shows the model emitted tool-call markers
        //   THEN the rendered DOM under #chat must contain at least
        //        one visible tool-residue node whose entire ancestor
        //        chain is also visible.
        expect(toolResidueNodes.length,
            `HARD INVARIANT VIOLATED: model emitted tool-call markers in SSE ` +
            `(oai=${sawOaiToolCalls} gemma=${sawGemmaMarker}) but the rendered ` +
            `DOM graph contains ZERO visible tool-residue nodes (collapsibles ` +
            `with class containing 'tool-progress-collapsible' or ` +
            `'tool-invocations-collapsible'). ${orphans.length} chat[] entries ` +
            `have tool data but no visible .mes DOM node. ` +
            `See dom_graph.json + invariant_evidence.json + 99_final.png.`
        ).toBeGreaterThan(0);
    });
});
