# User-Agent Recommendation Persistence Spec

Date: 2026-05-26

This document amends `docs/multi_user_agent_chat_interface_spec.md` for the
chat-side user-agent recommendation panel. Any older language that permits an
empty or dormant panel before a chat turn is superseded here.

## Thesis

User-agent recommendation transactions are part of the chat session state. Once
a top-k recommendation or per-agent draft has completed, it must be persisted
with the chat and restored like any other committed chat-session fact.

## Invariants

1. **USERS READ SLOWER THAN REST PACKETS SEND.** When a chat client opens,
   switches chats, or changes the active swipe, the user-agent panel must
   already be either showing a persisted result or showing a pending top-k
   state backed by a real `/api/plugins/user-personas/yapper-seed` request.

2. The panel must never display an empty, apologetic, or retry-only state in
   place of recommendation work. Forbidden states include “wait for a chat
   turn”, “yapper-seed failed, click retry”, and any blank loadout panel.

3. A recommendation cache key is not merely `(character_id, chat_id)`. It must
   include the active continuation state of the chat, including selected swipe
   ids/content for recent messages. Two selected swipes of the same last
   assistant turn are different recommendation contexts.

4. Completed top-k recommendation results must be written to chat-backed
   metadata. In-memory maps are allowed only as hot caches over that canonical
   persisted state.

5. Completed per-agent continuation drafts must also be persisted under the
   same chat-context key. A hard refresh must restore the same cards and drafts
   without needing a new backend call.

6. Restarting the ST client or ST server must not lose completed recommendation
   transactions. If the chat file survived, the recommendation state survived.

7. Explicit operator invalidation, such as `Re-suggest`, may delete the
   persisted recommendation for the active context and start a new pending
   transaction. Silent invalidation on render, reload, or navigation is not
   allowed.

## Acceptance Coverage

The required browser-level acceptance test seeds a six-message dicemother chat
with a last assistant message that has two swipes. It then:

1. Opens a randomly selected ST server from the four-server matrix.
2. Installs the seeded chat into that server's real chat files.
3. Opens the ST client and switches to that chat.
4. Observes a pending top-k UI while a real `/yapper-seed` request is in flight.
5. Resolves the first selected swipe to top-k cards.
6. Switches the last assistant turn to the second swipe and requires a distinct
   `/yapper-seed` result.
7. Hard-refreshes the browser and requires the same selected-swipe cards without
   a new `/yapper-seed` request.
8. Restarts the selected ST server and requires the same selected-swipe cards
   without a new `/yapper-seed` request.

Passing this test proves that the recommendation layer is keyed by active chat
state, not by a stale singleton; that completed recommendations are persisted;
and that the UI cannot hide missing work behind a plausible empty-state string.
