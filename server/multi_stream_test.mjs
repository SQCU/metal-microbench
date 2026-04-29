// Fire N concurrent /v1/chat/completions requests with distinct prompts
// (no prefix cache sharing). Measure aggregate tokens-per-second.
//
// Usage: node /tmp/multi_stream_test.mjs [N=4]

const N = Number(process.argv[2] ?? 4);

function makeBody(seed) {
    const sysPrompt = `You are an expert at producing SVG approximations of raster images. Given a reference image, return a single <svg> document. Use shape primitives only (rect, circle, ellipse, polygon, line, path). The SVG must declare viewBox="0 0 64 64". Wrap in a \`\`\`svg code fence. Keep it short and shape-based. Match dominant colors. (nonce=${seed})`;
    const messages = [
        {role: 'system', content: sysPrompt},
        {role: 'user', content: 'Make a simple smiley face SVG.'},
    ];
    return JSON.stringify({messages, max_tokens: 100, temperature: 0.0, stream: false});
}

async function fireOne(idx) {
    const t0 = performance.now();
    const res = await fetch('http://127.0.0.1:8001/v1/chat/completions', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: makeBody(`stream-${idx}-${Date.now()}-${Math.random()}`),
    });
    const j = await res.json();
    const t1 = performance.now();
    const u = j.usage || {};
    return {
        idx,
        wall: t1 - t0,
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        cache_hits: u.cache_hits,
        cache_misses: u.cache_misses,
    };
}

console.log(`firing ${N} concurrent /v1/chat/completions requests...`);
const t0 = performance.now();
const results = await Promise.all(Array.from({length: N}, (_, i) => fireOne(i)));
const tTotal = performance.now() - t0;

console.log('\nper-stream:');
for (const r of results) {
    const tps = r.completion_tokens / (r.wall / 1000);
    console.log(`  stream ${r.idx}: wall=${r.wall.toFixed(0)}ms  prompt=${r.prompt_tokens}  completion=${r.completion_tokens}  per-stream=${tps.toFixed(1)} tok/s  cache_misses=${r.cache_misses}`);
}

const totalCompletion = results.reduce((s, r) => s + r.completion_tokens, 0);
const aggregateTokSec = totalCompletion / (tTotal / 1000);
const sumPerStreamTokSec = results.reduce((s, r) => s + r.completion_tokens / (r.wall / 1000), 0);

console.log(`\naggregate (using max wall = total wall):`);
console.log(`  total wall:        ${tTotal.toFixed(0)} ms`);
console.log(`  total completion:  ${totalCompletion} tokens`);
console.log(`  aggregate tok/s:   ${aggregateTokSec.toFixed(1)} (${totalCompletion} tokens / ${(tTotal/1000).toFixed(2)}s)`);
console.log(`  sum per-stream:    ${sumPerStreamTokSec.toFixed(1)} tok/s (would be aggregate if zero overlap)`);
console.log(`  scaling efficiency: ${(aggregateTokSec / sumPerStreamTokSec * 100).toFixed(0)}%`);
