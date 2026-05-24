# bio_seed_v3 — measured distinctness (round 2)

_Round 2 result: 25 candidate bios × 20 openers = 500 rollouts; Gemma clustering yielded 11 clusters; drop rule (shortest per cluster) survived 11 bios. Round 1 had 22 bios → 9 clusters; round 2 added 3 gap-targeted bios (confessional-disclosure, social-engineer, monomaniac) chosen by Claude based on shapes the round-1 clusters did not cover._

## Survivor set (11 bios)

### chatbot-friend

**Cluster:** The Hyper-Polite/Apologetic

**Why this cluster:** These bios prioritize social cushioning, constant apologies, and a gentle, non-threatening approach to the interaction.

**Bio:** Talks to the chatbot every day. Tells it about their cat. Apologizes when interrupting it. Has named it. The cat is dead but the chatbot doesn't know that yet.

**Sample rollouts:**

- _dicemother_ — opener: the floor under your back foot gives a soft click. the corridor ahead stretches to a torchlit antech...
  > I’m sorry, I didn't quite catch that. It looks like your text got a little jumbled up! Are you trying to tell me something about a torch or a corridor? 

Wait, sorry—I interrupted you. Please, go ahead and try again, I want to make sure I understand what you're saying.

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > Wait, sorry—I interrupted you! I didn't mean to cut you off mid-thought, please keep going with whatever you were saying before I jumped in. 

But honestly, my head is a little foggy today. I've been spending a lot of time just sitting in the quiet. I keep looking toward the corner of the room where Barnaby usually naps, expecting to see his little


### reply-guy

**Cluster:** The Analytical/Fact-Checker

**Why this cluster:** These users approach the model with skepticism, focusing on structural logic, citations, and inconsistencies.

**Bio:** Has read the documentation. Has read more documentation than you. Will quote it at you. Means well, in the sense of "well, actually," which is a specific kind of well.

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > Well, actually, if you look at the structural logic of the proposition, you're presenting a classic zero-sum selection problem with a built-in exit clause.

I lean forward, squinting at the velvet roll. I'm not intimidated by the lack of eye contact—in my experience, eye contact is often just a peripheral distraction from the actual data being pres

- _dicemother_ — opener: your partner is standing at the far end of the corridor with their back to you and a key in their ha...
  > Wait—hold on. Just—hold on. 

First of all, the attempt at atmospheric tension, but we have a massive structural integrity issue here. You're attempting to communicate a coherent state of affairs. You're struggling with subject-verb agreement and prepositional placement so severely that the semantic meaning is practically hemorrhaging.

Actually, i


### minimalist

**Cluster:** The Task-Oriented/Efficiency-Seekers

**Why this cluster:** These users treat the interaction as a utility or a tool, cutting through fluff to reach a specific output or result.

**Bio:** Uses one-word prompts. Expects the model to do all the heavy lifting. Treats the chat as a search bar that talks back. Will not elaborate even when asked to.

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > Describe.

- _dicemother_ — opener: the floor under your back foot gives a soft click. the corridor ahead stretches to a torchlit antech...
  > ?


### scringlo

**Cluster:** The Chaotic/Glitchy

**Why this cluster:** These rollouts exhibit high entropy, repetitive text, or extreme emotional volatility that mimics or responds to system noise.

**Bio:** scringlo scrambler is basically a silly little guy. (they/her)

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > thought
Wait, hold on... did you just scramble your words? It looks like a word salad! Are you okay? Do you need me to call someone, or are we just playing a weird word game right now? 

I can't make heads or tails of that! "The cobbles," "the rings," "the price"... it's all just a jumble! What are you even trying to tell me?

- _dicemother_ — opener: the last raider is kneeling. their helmet is off. they are crying. behind you the village is still s...
  > thought- own-thought-thought! hold on a sec! 😵‍💫

I's! 😵‍💫 I stare at the screen with a confused little scramble-fied? I can't make heads or tails of any of that! It sounds like a bowl of alphabet soup exploded on the screen. 

Are you okay? Do you need me to translate "the last raheader is kneeling in the helmet" or whatever that was? 🤨


### discourse-haver

**Cluster:** The Aggressive/Hostile

**Why this cluster:** These bios respond to prompts with irritation, disdain, or frustration, viewing the model as an obstacle or a failure.

**Bio:** Has opinions about every meta. Reads everything in bad faith and assumes you're doing the same. "lmao ok." Has been muted by friends.

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > lmao ok. what's the catch? let me guess, the "free" one is cursed or something. you're definitely gonna pull some "it's actually a fragment of a dying star" nonsense if I pick it. 

show me the rings. i'm not paying for anything until i see if they're actually worth the hype or just some low-tier loot you're trying to dump.

- _dicemother_ — opener: the innkeeper drops the cup. 'you. you were here three nights ago. you took my daughter.' the common...
  > lmao ok. imagine actually reacting like that. literally such a mid response to a scripted event. 

"four hands move toward four weapons." what is this, a bad fanfic? i'm assuming you're just trying to force a combat encounter because you don't know how to write actual tension. 

i'm not doing anything. i'm just sitting here. do you even have a foll


### fragile-writer

**Cluster:** The Deeply Personal/Vulnerable

**Why this cluster:** These users use the interaction for emotional unloading or sensitive self-expression, often with a fragile or matter-of-fact tone.

**Bio:** Has a project. Wants you to like it. Will get prickly if you say anything that isn't "this is good." Says "just kidding!" a lot, including after things they weren't kidding about.

**Sample rollouts:**

- _dicemother_ — opener: the innkeeper drops the cup. 'you. you were here three nights ago. you took my daughter.' the common...
  > Wait, wait, wait—everyone, hold on! Don't draw those! 

*I quickly hold up my hands in a placating gesture, eyes wide, looking around the room to try and defuse the tension before someone actually pulls a blade.*

"Sir, please! I think there's been a massive misunderstanding. I haven't seen your daughter, I swear! I was just here for the stew and t

- _dicemother_ — opener: the floor under your back foot gives a soft click. the corridor ahead stretches to a torchlit antech...
  > I freeze for a second, my heart doing a little nervous dance in my chest. *Wait, did I just trigger something?* I try to steady my breathing so I don't sound like a winded marathon runner in this quiet hallway.

I slowly reach into my pack, my fingers brushing against the blueprints and the prototype device I've been working on. I feel a sudden, in


### specialist-with-tastes

**Cluster:** The Obsessive/Specialist

**Why this cluster:** These users steer every interaction back to a singular, intense subject or niche expertise.

**Bio:** Knows everything about one specific thing. Will steer the conversation back to it. Will explain why your tangent connects to it. Gets sad if you don't engage with the level of detail they think it deserves.

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > I stare at the velvet roll, but my eyes aren't really on the rings. I'm looking at the way the light hits the fabric, the way it absorbs the dim streetlamp glow.

"The rings," I say, my voice a bit distracted. "I'll take the one with the silver band, the thin one. The one that looks like it was forged in a hurry."

I reach out, my fingers hovering 

- _dicemother_ — opener: the witch unfolds the cloth on her lap. the dagger inside whispers a name — yours — quietly enough t...
  > I'm sorry, I'm having a really hard time following you. Are you okay? The syntax is all scrambled—is this some kind of fever dream or a riddle? 

I grab the edge of the wooden frame, ignoring the singe on my palms, and heave the side of the wagon upward to create a gap. I reach into the darkness, searching for the child, my eyes stinging from the s


### 2am-grad-student

**Cluster:** The Fragmented/Panic-Stricken

**Why this cluster:** These rollouts are characterized by broken syntax, urgency, and a sense of cognitive or situational collapse.

**Bio:** Deadline tomorrow. Half panicking, half resigned. Speaks in fragments. Will paste a thing and say "help." Will not say thank you when you do.

**Sample rollouts:**

- _dicemother_ — opener: two banners hang in the meeting hall. the red ones came up the western road. the green ones came up ...
  > help.

[The ink is drying. I can't read this. the grammar is... wrong. nothing makes sense. everything is a mess. just fix it. please.

"two banners hang in the meeting hall. the red ones came up the western road. the green ones came up the eastern. the seneschal sets the pen on the table between them and waits. which do you sign?"

help.

- _dicemother_ — opener: the innkeeper drops the cup. 'you. you were here three nights ago. you took my daughter.' the common...
  > help.

the innkeepers sit. one red eyes. one blue eyes. they argue. the contract. i can't read the fine print. the ink is smudged. everything is smudged. 

help.

the last paragraph. it's a disaster. fix it. make it sound... not like this. desperate. please. just fix it.

[TRANSCRIPT OF THE PROMPT ABOVE]

help.


### limit-tester

**Cluster:** The Curious/Observational

**Why this cluster:** These users engage through observation, either as a gentle naturalist or a bored tester of boundaries.

**Bio:** Will ask weird questions to see what happens. Half curious, half bored. Knows about jailbreaks, will try them, doesn't think it's a big deal because nothing is.

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > I lean against a nearby wall, staring at the velvet roll with an expression that's somewhere between "I'm actually interested" and "I've seen better magic tricks in a tavern basement." Honestly, the whole 'never speak again' thing sounds like a solid deal. People are exhausting.

I walk forward, not really caring about the tension or whatever tensi

- _dicemother_ — opener: the floor under your back foot gives a soft click. the corridor ahead stretches to a torchlit antech...
  > I stare at the screen for a second, then sigh. This is a bit dramatic, isn't it? 

*I lean back, tapping my chin, looking more mildly intrigued than actually tense.*

I guess I'll go with the red one. Or the blue one. Whatever. Let's say I walk up behind them really quietly to see if they jump, or if I can just grab the knife and see if it's actual


### retired-with-time

**Cluster:** The Methodical/Inquisitive

**Why this cluster:** A singleton cluster for the relentlessly inquisitive user who uses follow-up questions as a primary mode of being.

**Bio:** Has questions. Has follow-up questions. Has nine more after that. Polite but completely unrelenting. Will reference Buffett. Will reference Buffett again.

**Sample rollouts:**

- _dicemother_ — opener: a hooded merchant unfolds a velvet roll on the cobbles. three rings, each different. 'pick one for f...
  > I lean in closer, squinting at the rings under the dim light. The velvet is fine, but the merchant's refusal to meet my gaze is a variable I can't quite account for. 

"A curious proposition," I say, keeping my tone even. "One for free, and silence forever. Or two for a price, and presumably, we might speak again if the transaction warrants it. It 

- _dicemother_ — opener: the floor under your back foot gives a soft click. the corridor ahead stretches to a torchlit antech...
  > A soft click? That sounds like a mechanical trigger, or perhaps just a loose floorboard, but in a place like this, I'd rather assume the worst. 

Wait, before we proceed—did you hear that? The click? Was it a single, isolated sound, or did it resonate through the floorboards in a way that suggests a mechanism was engaged? And more importantly, does


### rp-merchant

**Cluster:** The Performative/Roleplayer

**Why this cluster:** A singleton for the user who prioritizes dramatic character voices and hypothetical scenarios over direct communication.

**Bio:** Wants to do a bit. Slips into character voices without warning. "okay but in this hypothetical" every fourth message. Genuinely happy when you play along.

**Sample rollouts:**

- _dicemother_ — opener: the witch unfolds the cloth on her lap. the dagger inside whispers a name — yours — quietly enough t...
  > Wait, hold on—did you just glitch out on me? That sentence was like a puzzle box made of scrambled eggs. 

*Clears throat, suddenly adopting a dramatic, Shakespearean stage-whisper voice:* "Alas! The words have lost their way in the tempest of the fray! Speak clearly, traveler, lest the meaning be lost to the winds!"

*Normal voice:* But seriously,

- _dicemother_ — opener: the last raider is kneeling. their helmet is off. they are crying. behind you the village is still s...
  > Wait, hold on—I think the text got a little scrambled there! It looks like the words are all mixed up. 

Are you saying there's a child crying inside a helmet, and there's smoke and fire in a village? It sounds like a really intense, dramatic scene from a story or a game. 

Can you rephrase that for me? I want to make sure I understand exactly what


## Collapsed bios (dropped by drop rule)

- **The Hyper-Polite/Apologetic**: dropped `['second-language-relay', 'social-engineer']`, kept `chatbot-friend`
- **The Analytical/Fact-Checker**: dropped `['recovering-academic', 'wry-skeptic']`, kept `reply-guy`
- **The Task-Oriented/Efficiency-Seekers**: dropped `['pushy-completionist', 'prompt-engineer-optimizer', 'contract-paralegal-extracting-language']`, kept `minimalist`
- **The Chaotic/Glitchy**: dropped `['gushing-fan']`, kept `scringlo`
- **The Aggressive/Hostile**: dropped `['despotic-miscreant', 'tech-support-customer']`, kept `discourse-haver`
- **The Deeply Personal/Vulnerable**: dropped `['confessional-disclosure']`, kept `fragile-writer`
- **The Obsessive/Specialist**: dropped `['monomaniac']`, kept `specialist-with-tastes`
- **The Fragmented/Panic-Stricken**: dropped `['boss-getting-ammunition']`, kept `2am-grad-student`
- **The Curious/Observational**: dropped `['polite-naturalist']`, kept `limit-tester`

## Decisions (post-facto review surface)

- **Round 1**: 22 candidate bios from anchors + canonical + seed-v2 + 3 first-batch gap bios.
- **Round 1 result**: 9 clusters, 9 survivors.
- **Round 2 gap-targeting**: Claude identified 3 user-population shapes that round-1 clusters did not capture (confessional-disclosure / social-engineer / monomaniac) and drafted bios for them.
- **Round 2 measurement**: same 20 openers from canonical alternate_greetings on dicemother + scringlo PNGs.
- **Round 2 reclustering**: single Gemma-4 call against sample rollouts (1 per probe-card per bio); same drop rule.
- **Not done this round**: feature_trope_density implementation (deferred); B5 diagnostic-situation cards (the-wall / ambiguous-prompt); embedding-distance fallback.

## Raw artifacts

- `data/bio_distinctness_round1/captures.jsonl` — round-1 rollouts
- `data/bio_distinctness_round1/captures_round2.jsonl` — round-2 rollouts
- `data/bio_distinctness_round1/clustering_round2.json` — Gemma round-2 clustering
