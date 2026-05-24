#!/usr/bin/env node
// Output-format × retry-mode ablation.
//
// Q: Are output-parse failures attributable to format choice (closing-`}`
//    hard for the model, indentation tricky, etc.) or to task difficulty
//    (this prose is genuinely confusing) or to harness design (we don't
//    feed the failure back to the agent for a fixup)?
//
// Method: pick a fixed task — score N axes on a piece of prose. Run K
// trials per (format, mode) cell. Measure per-cell:
//   - one-shot success rate (parsed cleanly on attempt 1)
//   - terminal success rate (parsed cleanly within maxAttempts)
//   - mean attempts to first complete parse
//   - mean axes-extracted on attempt 1
//   - mean wall-clock per terminal completion
//   - mean tokens emitted (latency proxy)
//
// Formats:    prose-lines, json, yaml, toml
// Modes:      feedback (in-context fixup), fresh-seed (independent retry),
//             one-shot (no retry, baseline reliability)
// Cells:      4 × 3 = 12
// Trials/cell:  configurable via TRIALS env var, default 10
//
// Prose samples are read from /yapper-seed candidate prose sources so
// they look like the actual workload. Total trials: 12 × TRIALS.
// At ~5s wall per trial: TRIALS=10 → ~10min, TRIALS=20 → ~20min.

import * as L from './harness_lib.mjs';
import { FORMATS, judgeWithFeedback } from './schema_lib.mjs';

const TRIALS = Number(process.env.TRIALS) || 10;
const MAX_ATTEMPTS = Number(process.env.MAX_ATTEMPTS) || 3;
const MODES = ['one-shot', 'feedback', 'fresh-seed'];

const AXES_CARDS = await L.fetchAxes();
const AXES = AXES_CARDS.map(c => ({ name: c.id, def: c.def }));
console.log(`[ablation] ${AXES.length} axes loaded`);

// 5 prose samples picked to span "easy / on-corpus" → "hard / off-corpus".
// Each gets TRIALS reruns per (format, mode) cell, so cross-prose noise
// averages out within-cell. Same prose used across all cells so the
// task-difficulty variable is controlled.
const PROSE_SAMPLES = [
    "An RPG wizard whose communication style is textbook Sagittarius: fire-sign, philosophical, blunt, restless, big-idea-loving. References spells, planes, alignment, the weave.",
    "An RPG rogue whose communication style is textbook Cancer: water-sign, moody, sentimental, defensive, sensitive. References shadows, locks, oaths, family.",
    "rolls a die and grins, ready for the next move",
    "discusses early-modern Spanish viticulture in iambic pentameter, with detailed reference to Garcilaso de la Vega and grafting practices",
    "A nomad of the astral winds, this wizard views the Weave not as a tool for safety, but as a vast frontier for discovery and provocation, peppering replies with planar trivia.",
];

// Cell key for the result table.
function cellKey(format, mode) { return `${format}__${mode}`; }

// Per-trial record.
//   trial_idx, sample_idx, format, mode, attempts, axes_extracted (terminal),
//   complete (bool), wall_ms, raws_chars (sum of raw output bytes).
const records = [];

function summarize() {
    const groups = {};
    for (const r of records) {
        const k = cellKey(r.format, r.mode);
        if (!groups[k]) groups[k] = [];
        groups[k].push(r);
    }
    const rows = [];
    for (const format of FORMATS) {
        for (const mode of MODES) {
            const k = cellKey(format, mode);
            const rs = groups[k] || [];
            if (rs.length === 0) continue;
            const n = rs.length;
            const oneshot = rs.filter(r => r.attempts === 1 && r.complete).length / n;
            const terminal = rs.filter(r => r.complete).length / n;
            const meanAttempts = rs.reduce((s, r) => s + r.attempts, 0) / n;
            const meanAxes1 = rs.reduce((s, r) => s + (r.axes_extracted_attempt1 ?? 0), 0) / n;
            const meanWallComplete = (() => {
                const completed = rs.filter(r => r.complete);
                if (!completed.length) return null;
                return completed.reduce((s, r) => s + r.wall_ms, 0) / completed.length;
            })();
            const meanRawChars = rs.reduce((s, r) => s + (r.raws_chars || 0), 0) / n;
            rows.push({
                format, mode, n,
                oneshot_success: oneshot,
                terminal_success: terminal,
                mean_attempts: meanAttempts,
                mean_axes_attempt1: meanAxes1,
                mean_wall_ms_complete: meanWallComplete,
                mean_raw_chars: meanRawChars,
            });
        }
    }
    return rows;
}

function printTable(rows) {
    // pad columns for human reading
    const header = ['format', 'mode', 'n', 'oneshot%', 'terminal%', 'mean_attempts', 'axes@1', 'wall_ms_ok', 'raw_chars'];
    const w = header.map(h => h.length);
    const data = rows.map(r => [
        r.format,
        r.mode,
        String(r.n),
        (r.oneshot_success * 100).toFixed(0),
        (r.terminal_success * 100).toFixed(0),
        r.mean_attempts.toFixed(2),
        r.mean_axes_attempt1.toFixed(1),
        r.mean_wall_ms_complete != null ? Math.round(r.mean_wall_ms_complete).toString() : '—',
        Math.round(r.mean_raw_chars).toString(),
    ]);
    for (const row of data) for (let i = 0; i < row.length; i++) w[i] = Math.max(w[i], row[i].length);
    const pad = (s, i) => s.padEnd(w[i] + 2);
    console.log('\n' + header.map(pad).join(''));
    console.log(header.map((_, i) => '─'.repeat(w[i]) + '  ').join(''));
    for (const row of data) console.log(row.map(pad).join(''));
}

// Drive the matrix.
const t0 = Date.now();
let cellIdx = 0;
const totalCells = FORMATS.length * MODES.length;
for (const format of FORMATS) {
    for (const mode of MODES) {
        cellIdx++;
        console.log(`\n[ablation] cell ${cellIdx}/${totalCells}: format=${format} mode=${mode}`);
        for (let trial = 0; trial < TRIALS; trial++) {
            const sampleIdx = trial % PROSE_SAMPLES.length;
            const prose = PROSE_SAMPLES[sampleIdx];
            const trialT0 = Date.now();
            let attempts = 0, axesExtracted = 0, complete = false, rawsChars = 0, axesAttempt1 = 0;
            try {
                const r = await judgeWithFeedback({
                    axes: AXES, prose, format,
                    bridgeCall: async (msgs, opts = {}) => {
                        return await L.bridgeCall(msgs, opts);
                    },
                    maxAttempts: MAX_ATTEMPTS,
                    mode,
                    // Vary seed by (trial, format, mode) for distinct rolls,
                    // but keep cell-rerun seeds deterministic across runs.
                    seedBase: 70000 + trial * 100,
                });
                attempts = r.attempts;
                axesExtracted = Object.keys(r.scores).length;
                complete = axesExtracted === AXES.length;
                rawsChars = (r.raws || []).reduce((s, t) => s + (t?.length || 0), 0);
                // Parse just attempt-1 raw to track first-shot reliability,
                // independent of whether a retry rescued it.
                const firstRaw = (r.raws || [])[0] || '';
                const { parse } = await import('./schema_lib.mjs');
                const a1 = parse(format, AXES.map(a => a.name), firstRaw);
                axesAttempt1 = Object.keys(a1.scores).length;
            } catch (e) {
                console.warn(`  trial ${trial}: error ${e.message}`);
                attempts = MAX_ATTEMPTS;
                complete = false;
            }
            const wall = Date.now() - trialT0;
            records.push({
                trial_idx: trial, sample_idx: sampleIdx, format, mode,
                attempts, axes_extracted: axesExtracted,
                axes_extracted_attempt1: axesAttempt1,
                complete, wall_ms: wall, raws_chars: rawsChars,
            });
            const flag = complete ? '✓' : '✗';
            console.log(`  trial ${trial+1}/${TRIALS} sample=${sampleIdx} ${flag} attempts=${attempts} axes=${axesExtracted}/${AXES.length} wall=${wall}ms`);
        }
    }
}

const totalElapsed = ((Date.now() - t0) / 1000).toFixed(1);
console.log(`\n[ablation] DONE. ${records.length} trials in ${totalElapsed}s`);

const rows = summarize();
printTable(rows);

// Write the raw records to JSONL for downstream analysis.
import * as fs from 'fs';
const outDir = '/Users/mdot/metal-microbench/data/format_ablation';
fs.mkdirSync(outDir, { recursive: true });
const ts = new Date().toISOString().replace(/[:.]/g, '-');
const outPath = `${outDir}/run_${ts}.jsonl`;
fs.writeFileSync(outPath, records.map(r => JSON.stringify(r)).join('\n') + '\n');
console.log(`[ablation] raw records → ${outPath}`);
const sumPath = `${outDir}/summary_${ts}.json`;
fs.writeFileSync(sumPath, JSON.stringify({ rows, n_axes: AXES.length, n_trials_per_cell: TRIALS, prose_samples: PROSE_SAMPLES.length }, null, 2));
console.log(`[ablation] summary → ${sumPath}`);
