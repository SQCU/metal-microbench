# User-descriptivism brief

## Work so far

The user-personas project builds user-side chat-agents that interact with assistants in SillyTavern. The pipeline produces `(bio × situation × overlay)` tuples whose multiturn rollouts can be scored along behavioral feature axes and used as inputs to discovery cartography. The user-agent pipeline with the orthogonal BIO + MOTIVATION + SCENARIO + RELATIONSHIP + COMM_STYLE + TURN + AUDIT schema (run through a discovery loop with PCA-driven targeting and per-axis drift reports) is working: variety in rollouts emerges from the orthogonal commitments the designer maintains simultaneously across the schema fields.

The two strongly-separating bios in the project's history are **scringlo scrambler** ("basically a silly little guy. (they/her)") and **Despotic Miscreant** ("a brutish and irresponsible cad who experiences the world as a series of problems whose solutions should be found with the least mental and social effort possible. Inconsiderate and uses force of will and force of force whenever that can get what they want from a situation. Think 'neutral evil'."). Both compress a recognizable type-of-person to its minimum legible signature using shared cultural shorthand; the signature itself constitutes a behavioral commitment about how the person speaks and engages. The canonical store also contains four weakly-separating bios (polite-naturalist, wry-skeptic, gushing-fan, pushy-completionist) which can be redesign candidates for the discovery harness once it has stronger seed examples to learn from.

## Research orientation

User-descriptivism: we want `(bio, situation, overlay)` tuples whose user-side behavior plausibly samples from the population of actual users engaging chatbots through APIs and chat UIs, as that population appears in webtext. The decomposition is:

- **Bio = user-as-state.** Who this person looks like to other people; what's dispositionally true about them. Texture, register, recognizable type. One to three sentences, up to about sixty words. The signature is itself a behavioral claim — not a Wikipedia infobox of facts, but a compressed organizing principle that explains how everything else about this person tends to go. Fictional-character introductions are the reference class: scringlo scrambler, Despotic Miscreant, Tony Soprano, Walter White — each introduced with the one thing that organizes everything else they do.

- **Situation = the shared frame the user and assistant share.** Multi-user-mono-assistant: the assistant has a known role/shape, multiple users with different bios react to the same situation differently. Two anchors so far — **dicemother** (TTRPG GM-led scene with player participants) and **scringlo** (chatroom-affordance-aware assistant character). Two more situations to be added so the roster covers enough territory that user-variety draws on chat-room dynamics rather than narrowing to private/secret-elicitation as its sole distinguishing axis.

- **Overlay = user-as-action.** Wants, conversational moves, goal-shape, what this user wants out of THIS interaction with THIS assistant right now. Lives at the runtime layer (author's-notes injected at depth-N), separate from bio.

## Goals for the current substep

Produce a seed corpus of bios (`docs/bio_seed_v2.md`) in the texture of scringlo and Despotic Miscreant — compressed user-as-state cards that constitute behavioral claims, span a representative slice of types-of-person plausibly present in API user populations as documented in webtext, and use shared cultural shorthand freely. These bios feed the discovery harness's inductive prior, and (once the harness is rewired) anchor the `bio × situation × overlay` cartesian product whose cells the discovery loop explores.

Concretely after this substep: the harness's `load_inventory_bios` reads from the canonical store (`tools/st-debug/_data/default-user/settings.json → power_user.persona_descriptions`) plus the new seed file. The Claude-written entries currently in the canonical store stay or go on per-bio merit measured on-policy in subsequent runs. The discovery harness's bio-mode designer prompt uses the seed corpus as inductive examples, and (per the broader plan) Gemma is looped in as a third reviewer on the design surface so the elicitation strategy gets cross-validation that a single-model loop cannot give itself.
