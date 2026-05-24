// One-time migration: move the 3 KEEP bio PNGs from the plugin's
// players/ store into ST's canonical User Avatars/ directory, and add
// their descriptions to settings.json → power_user.persona_descriptions.
//
// Not test fixture code. Run once via:
//   node tools/migrate_keep_bios_to_user_avatars.mjs
//
// After verification, the plugin source refactor will point at User
// Avatars/ as the canonical store and the players/ directory will be
// removed.

import fs from 'fs';
import path from 'path';
import { read as readCharaCardPng } from '/Users/mdot/sillytavern-fork/src/character-card-parser.js';

const KEEP = [
    '1778631331275-DespoticMiscreant.png',
    '1778634272476-BrutishMiscreant.png',
    '1779035204660-scringloscrambler.png',
];
// Source: the clone's plugin players/ dir holds the chara_card-enriched
// PNGs (tEXt chunks with embedded bio + extensions). The existing
// User Avatars/ PNGs are plain PNGs without metadata — we OVERWRITE
// them with the enriched versions so the plugin's extension data
// (signature, derived agents, provenance) rides along in the canonical
// store. Then we lift each card's `description` into settings.json so
// ST's persona-management UI surfaces the bio text.
const SOURCE_DIR = '/Users/mdot/metal-microbench/tools/st-debug/sillytavern-fork/plugins/user-personas/players';
const USER_AVATARS_DIR = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/User Avatars';
const SETTINGS_JSON = '/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/settings.json';

if (!fs.existsSync(USER_AVATARS_DIR)) {
    fs.mkdirSync(USER_AVATARS_DIR, { recursive: true });
}

// Load current settings.json.
const settingsRaw = fs.readFileSync(SETTINGS_JSON, 'utf8');
const settings = JSON.parse(settingsRaw);
if (!settings.power_user) settings.power_user = {};
if (!settings.power_user.persona_descriptions) settings.power_user.persona_descriptions = {};
if (!settings.power_user.personas) settings.power_user.personas = {};

let migrated = 0;
for (const key of KEEP) {
    const src = path.join(SOURCE_DIR, key);
    if (!fs.existsSync(src)) {
        console.warn(`  SKIP ${key}: not found in ${SOURCE_DIR}`);
        continue;
    }
    // Read the chara-card JSON from the PNG's tEXt chunk to extract the
    // bio (= description field of the chara card v2/v3 spec).
    let card;
    try {
        const cardJsonStr = readCharaCardPng(fs.readFileSync(src));
        card = JSON.parse(cardJsonStr);
    } catch (e) {
        console.warn(`  SKIP ${key}: failed to read embedded card JSON: ${e.message}`);
        continue;
    }
    // chara_card_v3: top-level {spec, spec_version, data: {name, description, ...}}
    // chara_card_v2: top-level {name, description, ...}
    const d = card.data || card;
    const description = d.description || '';
    const name = d.name || key.replace(/\.png$/, '').replace(/[-_]/g, ' ');

    // Overwrite the plain User Avatars/ PNG with the chara_card-enriched
    // one from players/, so the plugin's extensions (signature, derived
    // agents, provenance — embedded as tEXt chunks) live in the
    // canonical store.
    const dst = path.join(USER_AVATARS_DIR, key);
    fs.copyFileSync(src, dst);

    // Write description + name to settings.json. Schema matches what
    // ST's persona-management UI reads:
    //   power_user.persona_descriptions[key] = {description, title, depth, position, role, lorebook}
    //   power_user.personas[key]               = displayName
    settings.power_user.persona_descriptions[key] = {
        description,
        title: '',
        depth: 2,
        position: 0,
        role: 0,
        lorebook: '',
    };
    settings.power_user.personas[key] = name;

    console.log(`  MIGRATED ${key}: name="${name}" desc-len=${description.length}`);
    migrated++;
}

// Atomic write of settings.json.
const tmp = `${SETTINGS_JSON}.tmp-${process.pid}-${Date.now()}`;
fs.writeFileSync(tmp, JSON.stringify(settings, null, 4));
fs.renameSync(tmp, SETTINGS_JSON);

console.log(`\nDone. Migrated ${migrated} / ${KEEP.length} bios to User Avatars/ + settings.json.`);
console.log(`User Avatars/ now contains:`);
for (const f of fs.readdirSync(USER_AVATARS_DIR)) console.log(`  - ${f}`);
