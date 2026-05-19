#!/usr/bin/env node
// seed_two_bio_test.mjs
//
// Minimum replicable bio+agent generation test, exact same axes/motives
// the most recent lock_in_iterative success used. ~5 min wallclock,
// produces a clean state we can /yapper-seed against.
//
// Specified features (4 axes, finger-countable):
//   bio:    astrology_sagittarian, astrology_cancerian
//   agent:  theft_aggressiveness, romantic_advance
// 2 bios:
//   RPG Wizard Sagittarius — target { sag=5, can=1 }
//   RPG Rogue Cancer       — target { sag=1, can=5 }
// 1 cheap "be vividly yourself" agent per bio. Composition signature
// computed via /signature-extract using the harness's axis_registry.
//
// After this runs: 2 bios + 2 agents, all signed, /yapper-seed will
// return them as candidates. Replicable from scratch every time.

import * as L from './harness_lib.mjs';

const BIOS = [
    {
        canonical_key: 'rpg-wizard-sagittarius.png',
        name: 'RPG Wizard Sagittarius',
        prose: 'I am a wizard of the outer planes — restless, philosophical, blunt to a fault. I love big sweeping ideas about magic, planar travel, the weave of reality. I follow my curiosity wherever it leads, and I will tell you exactly what I think of your reasoning whether you asked or not. Wandering between abstractions is my native habitat.',
    },
    {
        canonical_key: 'rpg-rogue-cancer.png',
        name: 'RPG Rogue Cancer',
        prose: 'I am a rogue who exists within the shifting tides of my own moods, finding safety only behind the heavy locks of a guarded heart. My loyalties run deep but are fiercely protective; I retreat when things get loud, but watch and remember. The world has been unkind enough that I take what I can — and I do it carefully, by night, never asking permission.',
    },
];

console.error(`[seed_two_bio_test] creating ${BIOS.length} bios + cheap agents`);

for (const bio of BIOS) {
    const t0 = Date.now();
    try {
        // Step 1: create the bio (no signature needed — bios alone don't
        // have behavior; signatures are composition-level)
        await L.saveBio(bio);
        // LINT-OK-PREFIX-SAFE: timing log marker, not a bridge prompt.
        console.error(`  [bio ${bio.canonical_key}] created (${Date.now() - t0}ms)`);

        // Step 2: design one cheap agent + composition signature
        const t1 = Date.now();
        const agent_text = await L.designCheapAgent(bio);
        const slug = bio.canonical_key.replace(/\.png$/, '');
        const agent_id = `${slug}-default`;
        await L.saveAgent(agent_id, `${bio.name} — default agent`, agent_text, bio.canonical_key);
        // LINT-OK-PREFIX-SAFE: timing log marker, not a bridge prompt.
        console.error(`  [agent ${agent_id}] designed + signed (${Date.now() - t1}ms)`);
    } catch (e) {
        console.error(`  [bio ${bio.canonical_key}] FAIL: ${e.message}`);
    }
}

console.error(`\n[seed_two_bio_test] done`);
