import { test, expect } from '@playwright/test';

// Phase-2c: validate the SVG-rendering pipeline independently of
// "the model decides to call the tool". Hits the toolcards plugin
// directly via /api/plugins/toolcards/start_invoke, then drains
// /poll until the result event arrives. This is what the FE does
// internally when the model emits a tool_call — but we skip the
// model's decision step and prove the rest of the chain works.
//
// Why split this from 03: in 03, Gemma-4 at temp=0 was observed to
// hallucinate "OK I rendered it" prose without actually calling
// query-to-svg__generate. This is a model-behavior regression (the
// qualitative slump the user reported) and is orthogonal to the
// pipeline correctness. By invoking the plugin endpoint directly we
// can validate the descendant-agent + rendering chain in a way that
// stays green when the model's tool-call decisions get flaky.

test.describe('toolcards direct-invoke', () => {
    test.setTimeout(8 * 60 * 1000);

    test('start_invoke + poll → result with rasterized SVG', async ({ request }) => {
        // The plugin needs a chat-completion `profile` so the descendant
        // agent's llm_call events get dispatched to our bridge. Mirror
        // the shape ST's frontend would send: chat_completion_source=
        // custom + custom_url=:8001.
        const profile = {
            api: 'openai',
            mode: 'cc',
            preset: 'default',
            chat_completion_source: 'custom',
            custom_url: 'http://127.0.0.1:8001',
            custom_model: 'gemma-4-a4b',
            stream_openai: false,
            temperature_openai: 0.4,
            openai_max_tokens: 4096,
        };

        // Kick off invocation. Body shape per plugins/toolcards/index.mjs.
        const startResp = await request.post(
            'http://127.0.0.1:8002/api/plugins/toolcards/start_invoke/query-to-svg/generate',
            {
                data: {
                    args: {
                        query: 'a small red circle',
                        max_iters: 1,
                        width: 256,
                        height: 256,
                    },
                    profile: profile,
                },
            });
        expect(startResp.status(), 'start_invoke 200').toBe(200);
        const { session_id } = await startResp.json();
        console.log(`  session_id: ${session_id}`);

        // Poll until we see a `result` event. /poll returns one event
        // per call (or heartbeat); long-poll up to 90s per call.
        let resultEvent = null;
        let progressCount = 0;
        const deadline = Date.now() + 7 * 60 * 1000;
        while (Date.now() < deadline) {
            const pollResp = await request.get(
                `http://127.0.0.1:8002/api/plugins/toolcards/poll/${session_id}`);
            if (pollResp.status() === 404) {
                // session vanished — fail loudly
                throw new Error(`session ${session_id} not found at poll`);
            }
            if (pollResp.status() !== 200) {
                console.log(`  poll status ${pollResp.status()}, retrying`);
                await new Promise(r => setTimeout(r, 500));
                continue;
            }
            const event = await pollResp.json();
            if (event.type === 'progress') {
                progressCount++;
                console.log(`  progress: ${(event.text || '').slice(0, 100)}`);
            } else if (event.type === 'heartbeat') {
                continue;
            } else if (event.type === 'result' || event.type === 'error') {
                resultEvent = event;
                break;
            } else {
                console.log(`  unknown event type: ${event.type}`);
            }
        }

        expect(resultEvent, 'received a result/error event before deadline')
            .not.toBeNull();
        console.log(`  total progress events: ${progressCount}`);

        if (resultEvent.type === 'error') {
            throw new Error(`toolcard returned error: ${JSON.stringify(resultEvent).slice(0, 500)}`);
        }

        // Validate the result shape. query-to-svg returns:
        //   embed: array of OAI message-content parts (text + image_url)
        //   metadata: per-iter trajectory info
        const result = resultEvent.result || resultEvent;
        console.log(`  result keys: ${Object.keys(result).join(', ')}`);

        const embed = result.embed || [];
        expect(Array.isArray(embed), 'result.embed is an array').toBe(true);
        expect(embed.length, 'embed has at least one part').toBeGreaterThan(0);

        // The embed should contain at least one image_url part with a
        // data: URI (the rasterized SVG).
        const imagePart = embed.find(p =>
            p.type === 'image_url' &&
            typeof p?.image_url?.url === 'string' &&
            p.image_url.url.startsWith('data:image/'));
        expect(imagePart, 'embed contains a data-URI image_url part').toBeTruthy();

        const dataUri = imagePart.image_url.url;
        console.log(`  data-URI length: ${dataUri.length} chars (first 60: ${dataUri.slice(0, 60)})`);
        expect(dataUri.length, 'data URI is non-trivial').toBeGreaterThan(100);
    });
});
