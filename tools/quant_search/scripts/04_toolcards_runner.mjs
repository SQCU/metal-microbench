#!/usr/bin/env node
/*
 * Standalone toolcards runner — imports the SillyTavern toolcards plugin
 * verbatim and serves its HTTP API on a configurable port without
 * requiring SillyTavern to be running.
 *
 * IMPORTANT: this script depends on `express`, which lives in the
 * SillyTavern fork's node_modules. Either run from the fork directory:
 *   cd /Users/mdot/sillytavern-fork
 *   node /Users/mdot/metal-microbench/tools/quant_search/scripts/04_toolcards_runner.mjs
 * or set NODE_PATH:
 *   NODE_PATH=/Users/mdot/sillytavern-fork/node_modules \
 *     node tools/quant_search/scripts/04_toolcards_runner.mjs
 *
 * Why: the plugin's lifecycle work (server-side llm_call dispatch,
 * persistent sessions, /sessions reconcile, /cancel) is exactly the
 * runner architecture the quant search needs. Extracting it as a
 * standalone process avoids re-implementing the runner inline (the
 * "CLI shim" anti-pattern). Same code as the plugin → same behavior →
 * any improvements to the plugin propagate here for free.
 *
 * Run:
 *   node 04_toolcards_runner.mjs
 *
 * Env:
 *   TOOLCARDS_PORT       (default 8002)
 *   TOOLCARDS_DATA_ROOT  (default ~/sillytavern-fork/data)
 *   QUANT_BRIDGE_URL     (default http://127.0.0.1:8001 — what plugin's
 *                         server-side llm_call dispatcher targets when
 *                         a request comes in without auth headers)
 */

// Import express from the SillyTavern fork's node_modules. ESM module
// resolution walks up from the importer's directory, so we need a full
// path here (NODE_PATH is no help for ESM).
const expressMod = await import(
    '/Users/mdot/sillytavern-fork/node_modules/express/index.js'
);
const express = expressMod.default || expressMod;
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const PORT = parseInt(process.env.TOOLCARDS_PORT || '8002', 10);
const DATA_ROOT = process.env.TOOLCARDS_DATA_ROOT
    || '/Users/mdot/sillytavern-fork/data';

// Plugin reads globalThis.DATA_ROOT for cards/installed paths.
globalThis.DATA_ROOT = DATA_ROOT;

// Import the plugin from the ST fork. It's an ES module that exports
// init(router) + exit() + info.
const PLUGIN_PATH = '/Users/mdot/sillytavern-fork/plugins/toolcards/index.mjs';
const plugin = await import(PLUGIN_PATH);

const app = express();
app.use(express.json({ limit: '50mb' }));

// CORS (the search harness will be hitting this from Python over HTTP;
// no browser involved, but better to be explicit).
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    next();
});

// Health endpoint for the search harness to verify we're up.
app.get('/health', (_req, res) => {
    res.json({
        status: 'ready',
        plugin_id: plugin.info.id,
        plugin_name: plugin.info.name,
        data_root: DATA_ROOT,
        bridge_url: process.env.QUANT_BRIDGE_URL || 'http://127.0.0.1:8001',
    });
});

// SillyTavern-shape → OAI-shape proxy. The plugin's server-side llm_call
// dispatcher assumes it's running inside SillyTavern and posts to
// /api/backends/chat-completions/generate. In standalone mode we don't
// have that route, so we serve a thin compatibility shim that maps the
// plugin's request body to the bridge's /v1/chat/completions and
// translates the response back.
//
// The plugin's request body has fields like chat_completion_source,
// model, custom_url, reverse_proxy, proxy_password etc. — all routing
// hints the bridge ignores. We extract just messages/max_tokens/temp/
// stream and forward.
const BRIDGE_URL = process.env.QUANT_BRIDGE_URL || 'http://127.0.0.1:8001';
app.post('/api/backends/chat-completions/generate', async (req, res) => {
    try {
        const body = req.body || {};
        const oai_body = {
            messages: body.messages,
            max_tokens: body.max_tokens || 4096,
            temperature: body.temperature ?? 0.7,
            stream: false,  // we always block here since the plugin awaits text
        };
        const upstream = await fetch(`${BRIDGE_URL}/v1/chat/completions`, {
            method: 'POST',
            headers: { 'content-type': 'application/json' },
            body: JSON.stringify(oai_body),
        });
        if (!upstream.ok) {
            const errBody = await upstream.text().catch(() => '');
            res.status(upstream.status).send(errBody);
            return;
        }
        const data = await upstream.json();
        res.json(data);
    } catch (e) {
        console.error(`[toolcards-runner] proxy error:`, e);
        res.status(500).json({ error: String(e?.message || e) });
    }
});

// Mount the plugin's router under /api/plugins/toolcards (same path the
// SillyTavern host uses, so request shapes are identical).
const router = express.Router();
await plugin.init(router);
app.use('/api/plugins/toolcards', router);

// Graceful shutdown — call plugin's exit() to kill any spawned services.
async function shutdown(signal) {
    console.log(`[toolcards-runner] ${signal} received, shutting down...`);
    try { await plugin.exit?.(); } catch (e) { console.warn(e); }
    process.exit(0);
}
process.on('SIGINT', () => shutdown('SIGINT'));
process.on('SIGTERM', () => shutdown('SIGTERM'));

const server = app.listen(PORT, '127.0.0.1', () => {
    console.log(`[toolcards-runner] listening on http://127.0.0.1:${PORT}`);
    console.log(`[toolcards-runner] DATA_ROOT=${DATA_ROOT}`);
    console.log(`[toolcards-runner] plugin: ${plugin.info.name} (${plugin.info.id})`);
    console.log(`[toolcards-runner] API: http://127.0.0.1:${PORT}/api/plugins/toolcards/list`);
});
