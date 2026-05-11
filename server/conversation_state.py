"""Bridge-side conversation state for prefix-cache-aware multi-turn dialog.

Why this exists — chat-template asymmetry:
  The canonical jinja chat template's `add_generation_prompt` epilogue
  emits `<|channel>thought\\n<channel|>` (the no-thinking hint, ~4
  tokens) at the trailing position of each rendered conversation. The
  per-message loop that re-renders HISTORICAL assistant turns omits
  that wrapper. So at any given conversation position, turn N's KV
  carries 4 wrapper tokens that turn N+1's canonical render at the
  same byte offsets does NOT have — page hashes diverge at that point
  and the offset cascades through the rest of the conversation.

  Plus a smaller secondary divergence: OAI is stateless, so the
  client returns assistant text in the next turn's `messages` list and
  the bridge re-tokenizes. BPE round-trip
  `tokenize(detokenize(emitted_ids)) != emitted_ids` is a known
  pitfall.

How this fixes both:
  On stream completion, store the EXACT segment list submitted to the
  engine plus a synthetic tokens-segment for the model-emitted tokens
  — keyed by `hash_messages(messages_so_far + [response_msg], tools)`.
  On the next turn, the bridge looks up by `hash_messages(messages[:-1],
  tools)`; on hit, submission is `stored.segments + delta_segments`
  where `delta_segments` covers just the new turn's content + open
  model turn. No re-tokenize of historical content. No canonical
  re-render. The engine's content-hash page cache then adopts every
  page of `stored.segments` because the byte sequence is bit-identical.

Segment-based storage (vs flat token list):
  Segments preserve image_bytes / softs payloads so warm-path replay
  works for multimodal turns too. Each Segment is the bridge-side
  shape (kind=0 tokens, kind=1 image_bytes, kind=2 softs_bytes).
  A typical multi-turn multimodal conversation's stored segment list
  alternates text-tokens / image-bytes / text-tokens / model-emitted-
  tokens / boundary-tokens / new-image-bytes / etc.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any


def hash_messages(messages: list[dict], tools: list | None) -> str:
    """Stable conversation identity hash. Canonicalizes via JSON
    sort_keys; non-serializable values become strings via default=str.
    """
    payload = json.dumps(
        {"messages": messages, "tools": tools or []},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class StoredSegment:
    """Mirror of bridge `Segment` with the same wire shape.
    Kept here so conversation_state.py is dylib-free and tests can
    import without spinning up the engine."""
    kind: int                          # 0=tokens, 1=image_bytes, 2=softs (future)
    tokens: list[int] = field(default_factory=list)
    image_bytes: bytes = b""


@dataclass
class ConversationState:
    conversation_id: str
    last_used: float
    # Every segment that landed in the engine's KV state after this
    # turn finished — including the model-emitted text appended as a
    # final tokens-segment.
    segments: list[StoredSegment] = field(default_factory=list)


class ConversationCache:
    """Bounded LRU. In-memory only — dies with the bridge process, which
    is the right invalidation policy: a new bridge means a new dylib +
    possibly a different GGUF, so prior token sequences are not
    necessarily replayable."""

    def __init__(self, max_entries: int = 128):
        self._d: OrderedDict[str, ConversationState] = OrderedDict()
        self._max = max_entries
        self._lock = threading.Lock()

    def lookup(self, key: str) -> ConversationState | None:
        with self._lock:
            state = self._d.get(key)
            if state is None:
                return None
            state.last_used = time.time()
            self._d.move_to_end(key)
            return state

    def record(self, key: str, segments: list[StoredSegment]) -> None:
        with self._lock:
            self._d[key] = ConversationState(
                conversation_id=key,
                last_used=time.time(),
                segments=list(segments),
            )
            self._d.move_to_end(key)
            while len(self._d) > self._max:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        return len(self._d)
