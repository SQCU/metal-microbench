# Gemma-4 critic, round 1

_inputs: docs/user_descriptivism_brief.md + docs/bio_seed_v2.md; wallclock 52.1s; output 3978 chars_

COHERENCE
The decomposition into `(bio × situation × overlay)` is logically sound and solves the "static persona" problem by separating disposition (state) from intent (action). The distinction between Bio and Overlay is the project's strongest architectural feature; it prevents the "one-dimensional agent" trap where a user's personality is conflated with their immediate goal. The schema's expansion into "user-as-state" via cultural shorthand is a clever way to compress high-dimensional behavioral data into low-dimensional, legible prompts.

GAPS
1. **The "Silent User" Axis:** The current schema assumes all users are communicative agents. It lacks a way to model the "low-effort/low-signal" user who provides minimal input, which is a massive segment of the API population.
2. **Temporal Drift/Burnout:** The schema treats Bio as a static constant. It does not account for the "user-as-process"—how a user's behavior shifts from curiosity to frustration or from roleplay to utility over a long session.
3. **The "Systemic" User:** You have "Limit-Tester" and "Recovering-Academic," but you lack the "Prompt Engineer/Optimizer" who treats the assistant not as a character or a person, but as a raw compute resource to be tuned.
4. **Contextual Blindness in Bio:** The "Bio" is currently purely psychological/social. It lacks "Technical Literacy/Interface Literacy" as a formal dimension, which is a primary driver of user-side variance in SillyTavern.

RISKS
1. **The "Stereotype Collapse" Risk:** By relying heavily on "shared cultural shorthand," you risk the discovery harness merely rediscovering internet tropes (Reddit/Twitter archetypes) rather than actual user behavior. The "Discourse-Haver" is a caricature; if the harness converges on caricatures, the cartography will map *clichés*, not *users*.
2. **The "Overlay" Leakage:** There is a high probability that the "Overlay" (wants/goals) will bleed into the "Bio" during rollout. If a "Fragile-Writer" (Bio) is given an Overlay to "get feedback," the model may struggle to distinguish whether the prickliness is a personality trait or a reaction to the task.
3. **Evaluation Bias:** If the "scoring along behavioral feature axes" relies on LLM-as-a-judge, the judge will likely reward "legibility" (how well the agent performs the trope) rather than "authenticity" (how well the agent adheres to the nuances of the bio).

CONCRETE_SUGGESTIONS
1. **Refine the Schema:** Add `INTERFACE_LITERACY` as a sub-dimension of Bio or a separate axis. A "tech-support-customer" and a "2am-grad-student" may have similar goals but vastly different ways of interacting with the UI/API (e.g., wall-of-text vs. broken fragments).
2. **Diversify the Seed (The "Non-Communicative" Tier):** Add bios that represent different *modes of silence* or *low-entropy input*. (e.g., "The Minimalist: Uses one-word prompts; expects the model to do all the heavy lifting; treats the chat as a search bar.")
3. **Diagnostic Situations:** To test if bios are truly distinct, use "The Ambiguous Prompt" situation. 
    * *Situation:* An assistant provides a response that is technically correct but socially tone-deaf or slightly hallucinated. 
    * *Test:* Does the `recovering-academic` attack the source, does the `chatbot-friend` forgive it, and does the `limit-tester` exploit it? If they all react similarly, your bios are too weak.
4. **Diagnostic Situations (The "Constraint" Test):** Use "The Wall" situation. 
    * *Situation:* The assistant hits a refusal/filter or a logic loop. 
    * *Test:* Measure the "drift" in the Overlay. Does the `boss-getting-ammunition` pivot to a new framing, or does the `fragile-writer` retreat into "just kidding!"?
5. **Bio-Refinement:** Convert `discourse-haver` and `reply-guy` from "Internet Troll" archetypes into "Communication Style" archetypes to avoid the caricature trap. Focus on their *interactional rhythm* (e.g., frequency of interruptions, length of turns, use of meta-commentary).