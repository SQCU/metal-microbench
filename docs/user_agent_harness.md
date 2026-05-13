# What the user-agent harness is for

A short, normative reference. If you find yourself building feature X
and X doesn't match the framing below, X is the wrong feature.

## What it IS for

The user-agent harness produces **multi-scale behavior** — observable
differences in conversational shape — that are **independent of the
specific user-character and independent of the specific assistant-
character** at the other end of the chat. A user-agent is a policy
that animates a user turn. The harness's job is to make a small set
of orthogonal controls (voice × motive) produce conversational output
that is *recognizably itself* across counterparties.

The harness exists so that:

- A user-agent talking to **scringlo scrambler** (a rich, talkative
  character that volunteers a lot of conversational matter) produces
  user-turns whose shape and texture reflect the configured policy.
- The *same* user-agent talking to a **literal rock** (a deliberately
  minimal interlocutor that volunteers almost nothing — see
  `sillytavern-fork/default/content/the-rock.png`) produces user-turns
  whose shape and texture *still* reflect the same configured policy,
  even though the assistant side has changed dramatically.

That is the "multi-scale" property: scale here is the *richness of
the interlocutor*, and the harness must produce policy-faithful
output across the scale. Scringlo at one end (rich). The rock at the
other (sparse). The user-agent should be itself at both.

## What it is NOT for

Anything that can't talk to *both* scringlo scrambler and the rock
and find something to do with the rock — something **motivating**,
or **unmotivating**, or at least **performative** — is not a
user-agent. It's a chat-completion script with a personality tagline.

In particular, the harness is NOT for:

- Building a single user-agent that only works against a specific
  named assistant character.
- Building an assistant-side persona (those are character cards;
  user-agents are the OTHER side of the chat).
- "Suggest replies" features in the conventional UI sense — those
  are surface affordances on top of what this harness produces, not
  the harness's purpose.
- Engineering a precisely-tuned coordinate system into a controlled
  output. The harness does NOT exist to let a user dial in "I want
  exactly a johnny-blue-vorthos-explorer user-agent at coordinate
  (3,1,0,1)." It exists to let a user-agent that has *some*
  configured policy produce *its own* response to whatever (or
  whoever) it's interacting with — even when "whoever" is a rock.

## The two-pole test

When in doubt about whether the harness is doing its job, run the
two-pole check by inspection:

1. Configure a user-agent (any voice × any motive).
2. Drop it into a chat with scringlo scrambler. Watch a few turns.
3. Drop the *same* configuration into a chat with the rock. Watch
   a few turns.
4. The two transcripts should be **different in ways the assistant
   character explains** (scringlo brings sparkle, the rock brings
   silence; the user-agent responds appropriately to each) AND
   **similar in ways the user-agent policy explains** (the same
   voice, the same orientation, the same preferences).

If the transcripts are identical, the assistant character isn't doing
its job. If the transcripts share nothing in common across the two,
the user-agent isn't doing its job — its policy got eaten by whichever
interlocutor it was paired with.

If a configured user-agent has *nothing* to do with a rock, doesn't
find the rock motivating, doesn't find it unmotivating, doesn't even
find a performative angle on it ("I will narrate at this rock"), then
that user-agent's configuration is too thin. The rock is the floor.
Anything that fails the rock fails the harness.

## What scaffolding already exists in this codebase

These are the load-bearing pieces. Do not reinvent them.

- **Voice**: ST persona `user_personas_extras.voice_clauses`.
  Hand-written seeds in `DEFAULT_VOICES` in
  `public/scripts/extensions/user-personas/index.js`.
- **Motive**: server-side library at
  `plugins/user-personas/motives/<id>/manifest.json`. Each manifest
  carries `motivation` coords + `goal_clauses` (orientation prose) +
  `relationship_to_counterparty`. The 12 hand-written seed motives
  in that directory work; the model-generated ones produced by the
  studio's `/motives/generate` endpoint did NOT work in any
  iteration that was attempted — they're a known unimplemented hole.
- **Composition**: `composeFromVoiceMotive` in
  `plugins/user-personas/index.mjs` produces the `system_prompt`
  that drives a user-agent's turns.
- **Three runtime modes**: `MODES = ['off', 'suggest', 'autonomous']`
  in `public/scripts/extensions/user-personas/index.js`.
- **Per-member panel + per-card mode dropdown + live preview**:
  validated by curated artifacts under
  `docs/media/2026-05-11_test37_unified_panel_d1ba7fd/`.
- **Autonomous-tick advancing chat without click**: validated by
  `docs/media/2026-05-11_test35_user_personas_phase2_autonomous_a923164/`.
- **Multi-user dialogue (N user-agents, one assistant)**: validated
  by `docs/media/2026-05-11_test36_user_personas_phase3_multi_user_abdd6c4/`.
- **Truncation + kick-on-toggle guards**: validated by
  `docs/media/2026-05-11_test38_truncation_and_kick_0c4caaa/`.
- **The rock as an assistant character card**:
  `sillytavern-fork/default/content/the-rock.png` (chara_card_v2;
  seeded as default content). Use it as the rock-pole of the two-pole
  test.

## What is NOT in this codebase as a working feature

These are the known holes. Do not pretend they exist.

- **Model-generated motives.** The `/motives/generate` endpoint can
  return JSON sometimes; none of its outputs have been validated to
  produce a user-agent that passes the two-pole test. The studio's
  motive-design UI (sliders/radar/batch-generate) was speculative
  scaffolding that has been removed.
- **Diegetic rock test.** Talking to the rock as an actual
  SillyTavern character through the actual ST chat flow with a
  configured user-agent + capturing the result as a playwright video
  has not been done yet. The rock card EXISTS; the test of it
  through the client does not.
- **LLM-as-judge synthesis pipeline.** Was discussed; not
  implemented; not validated.

## Injunction

The pattern this codebase keeps falling into: someone reads the
codebase, infers a feature is "missing" or "incomplete," writes a
new internal function, calls it from a private caller, calls that
"validation," and commits. Then the cycle repeats with the next
feature.

The only validation that matters is **a rendered artifact from the
real client** — playwright screenshot, video, or in-browser
behavior — showing the feature working end to end against the
two-pole test (scringlo and the rock). Internal-function-called-by-
internal-caller is not validation. If a feature does not have a
rendered artifact, treat it as not implemented even if the source
code claims otherwise.
