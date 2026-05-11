"""Single source of truth for chat-template rendering.

Renders OpenAI-style `messages` lists through the model's jinja chat
template (same file the HF processor consumes), then emits an ordered
list of submission chunks the bridge feeds into the tensor server:

  - TextChunk:  plain string segment. Bridge tokenizes with atomic-ID
                recognition for the chat-scaffolding special markers
                (<|turn>, <turn|>, <|channel>, <channel|>) and submits
                token IDs via g.submit().
  - ImageChunk: raw image bytes (PNG/JPEG/whatever CGImageSource decodes).
                Bridge hands the bytes to g.submit_image_bytes() which
                brackets with BOI/softs/EOI internally.

The jinja template itself has no knowledge of token IDs, bytes, or the
BOI/EOI scaffolding around soft tokens — it only emits a `<|image|>`
placeholder. Mapping placeholders back to per-message image bytes
happens here, so the template is the only place chat-formatting lives
and the FFI stays free of role/template semantics.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import jinja2


# ── Chunk types the bridge consumes ────────────────────────────────

@dataclass
class TextChunk:
    text: str


@dataclass
class ImageChunk:
    data: bytes


Chunk = Union[TextChunk, ImageChunk]


# ── Template loading ──────────────────────────────────────────────

_TEMPLATE_CACHE: dict[str, jinja2.Template] = {}


def _default_template_path() -> Path:
    """Resolve the chat_template.jinja file from (in priority order):

      1. GEMMA_CHAT_TEMPLATE env var (absolute path)
      2. Sibling of GEMMA_SAFETENSORS (the HF model dir that shipped it)
      3. Sibling of GEMMA_GGUF (if a tokenizer_config.json is next to it)
    """
    explicit = os.environ.get("GEMMA_CHAT_TEMPLATE")
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"GEMMA_CHAT_TEMPLATE not found: {p}")

    for env_var in ("GEMMA_SAFETENSORS", "GEMMA_GGUF"):
        v = os.environ.get(env_var)
        if not v:
            continue
        candidate = Path(v).resolve().parent / "chat_template.jinja"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not locate chat_template.jinja — set GEMMA_CHAT_TEMPLATE "
        "or ensure the file is in the same dir as GEMMA_SAFETENSORS/GEMMA_GGUF")


def _safe_from_json(value):
    """Best-effort JSON parse for use as a Jinja filter.

    OpenAI's `tool_calls[*].function.arguments` is canonically a
    JSON-encoded string (e.g. `'{"query":"foo"}'`). The chat template
    needs the parsed mapping so it can render `key:value,key:value`
    pairs without double-bracketing the JSON object's own braces. If
    the input isn't a parseable JSON string, returns it unchanged so
    the template's string-fallback branch can still render it.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _load_template(path: Path) -> jinja2.Template:
    cached = _TEMPLATE_CACHE.get(str(path))
    if cached is not None:
        return cached
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                              keep_trailing_newline=False)
    env.filters['from_json'] = _safe_from_json
    tmpl = env.from_string(path.read_text())
    _TEMPLATE_CACHE[str(path)] = tmpl
    return tmpl


# ── Message normalization ─────────────────────────────────────────

def _decode_image_url(url: str) -> bytes:
    """`data:image/...;base64,...` URIs only.

    Earlier versions of this bridge synchronously fetched http(s) image
    URLs via `urllib.request.urlopen`. That is a blocking-I/O stall
    vector on the chat-completion hot path: a slow or unreachable URL
    would block the request task for up to 10 seconds, indirectly
    starving the engine's submit pump. The bridge is a tensor service;
    it does not get to make outbound HTTP requests on a client's
    behalf.

    Production policy: clients hand the bridge bytes (via `data:` URLs)
    or they hand the bridge nothing. If a real chat client wants to
    paste an `https://` image URL, that's a job for a separate proxy
    or async tool server — not the bridge.

    A non-`data:` URL therefore fails fast with a clear error.
    """
    if url.startswith("data:"):
        _, payload = url.split(",", 1)
        return base64.b64decode(payload)
    raise ValueError(
        f"image_url must be a data: URI (got {url[:32]!r}). The bridge "
        f"does not fetch remote URLs; clients must hand image bytes "
        f"directly. Use a separate proxy or async tool server if a "
        f"real chat client needs http(s) URL fetching.")


def _normalize_messages(
    messages: list[dict],
) -> tuple[list[dict], list[ImageChunk]]:
    """Walk messages; rewrite OpenAI content parts into jinja-friendly
    shapes and pull out image payloads in order of appearance.

    After this, the jinja template only ever sees text segments +
    {type: image} placeholders; the image bytes are a sidecar list.
    """
    out_msgs: list[dict] = []
    payloads: list[ImageChunk] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_parts = []
            for part in content:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                ptype = part.get("type")
                if ptype == "image_url":
                    url = part.get("image_url", {})
                    if isinstance(url, dict):
                        url = url.get("url", "")
                    payloads.append(ImageChunk(_decode_image_url(url)))
                    new_parts.append({"type": "image"})
                else:
                    new_parts.append(part)
            out_msgs.append({**msg, "content": new_parts})
        else:
            out_msgs.append(msg)
    return out_msgs, payloads


# ── Public renderer ───────────────────────────────────────────────

_IMAGE_PLACEHOLDER = "<|image|>"


def render_chat(messages: list[dict], *,
                add_generation_prompt: bool = True,
                enable_thinking: bool = False,
                tools: list | None = None,
                bos_token: str = "",
                template_path: Path | None = None) -> list[Chunk]:
    """Render OpenAI-style messages through the model's chat template.

    Returns an ordered list of chunks — text segments interleaved with
    binary payloads (image bytes, soft tokens). The bridge iterates this
    list and submits each chunk via the appropriate FFI call.

    Parameters:
      - add_generation_prompt: append the final `<|turn>model\\n...`
        scaffolding so the model knows it's its turn to reply. Default
        True (OpenAI-style). Pass False for completions-style prompts
        that already include a model turn.
      - enable_thinking: enable Gemma-4's thinking-channel prelude.
      - tools: optional list of tool function declarations.
      - bos_token: emitted literally by the template at the very top.
        Leave as empty string when the tokenizer will add BOS itself;
        set to the BOS literal (e.g. "<bos>") if the template output is
        being used raw without a subsequent add_bos tokenize.
      - template_path: override for the jinja source. Defaults to the
        location resolved from env vars.
    """
    norm_msgs, payloads = _normalize_messages(messages)
    path = template_path or _default_template_path()
    rendered: str = _load_template(path).render(
        messages=norm_msgs,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        tools=tools,
        bos_token=bos_token,
    )
    parts = rendered.split(_IMAGE_PLACEHOLDER)
    if len(parts) != len(payloads) + 1:
        raise RuntimeError(
            f"image placeholder mismatch: template emitted "
            f"{len(parts)-1} <|image|> markers, caller provided "
            f"{len(payloads)} payloads")

    chunks: list[Chunk] = []
    for i, text in enumerate(parts):
        if text:
            chunks.append(TextChunk(text))
        if i < len(payloads):
            chunks.append(payloads[i])
    return chunks


# ── Tokenization helpers (for the bridge to consume chunks) ───────

# Atomic-ID table for the chat-scaffolding special markers Gemma-4
# assigns to single vocab slots. BPE'ing these markers splits them into
# character pieces and pushes the model off-manifold; we emit the atomic
# ID instead. Extended via _register_atomic_id() when more get wired.
# Tokens we deliberately DON'T atomic-ize when they appear in prose:
#   bos / eos / pad / unk / mask:
#       Added by the tokenizer's `add_bos` parameter when needed; emitting
#       them as atomic IDs mid-string would break user prompts that
#       happen to contain the literal substring "<bos>" (etc.) — like a
#       documentation request about the tokenizer itself.
#   <|image|>:
#       The image placeholder. Already split out BEFORE tokenize_with_specials
#       runs (see render_chat / _IMAGE_PLACEHOLDER), so atomic-izing it here
#       would never match. Left out for clarity.
_EXCLUDE_FROM_ATOMIC: frozenset[str] = frozenset({
    "<bos>", "<eos>", "<pad>", "<unk>", "<mask>",
    "<|image|>",
})

# Hand-coded fallback list, used only when tokenizer.json can't be
# located at module-import time. Kept minimal: the auto-loader from
# tokenizer.json is the canonical path and pulls everything the model
# actually ships. Any entry here is just to keep the bridge from
# crashing on a misconfigured deployment.
_FALLBACK_SPECIAL_TOKENS: dict[str, int] = {
    "<|turn>":          105,
    "<turn|>":          106,
    "<|channel>":       100,
    "<channel|>":       101,
    "<|tool>":          46,
    "<tool|>":          47,
    "<|tool_call>":     48,
    "<tool_call|>":     49,
    "<|tool_response>": 50,
    "<tool_response|>": 51,
    "<|\"|>":           52,
    "<|think|>":        98,
}


def _load_special_tokens_from_tokenizer_json() -> dict[str, int] | None:
    """Walk likely tokenizer.json locations; return the model's full
    set of special added_tokens (id, content) — atomic-IDs the chat
    template can emit and the model was trained to consume directly.
    Returns None if no tokenizer.json is reachable; caller falls back
    to the hand-coded mini-dict.

    Why auto-load: hand-coded SPECIAL_TOKENS tables historically grew
    one regression at a time. Whenever the chat template introduced a
    new atomic marker (`<|tool_call>`, `<|"|>`, `<|think|>`, ...) the
    Python tokenize_with_specials would silently BPE-split it,
    feeding the model off-manifold input bytes. Pulling the full
    set from tokenizer.json eliminates that class of bug entirely:
    every special-true atomic ID the model knows about becomes
    eligible for atomic-emission on the input path.
    """
    paths_to_try: list[Path] = []
    chat_tpl_env = os.environ.get("GEMMA_CHAT_TEMPLATE")
    if chat_tpl_env:
        paths_to_try.append(Path(chat_tpl_env).parent / "tokenizer.json")
    st_env = os.environ.get("GEMMA_SAFETENSORS")
    if st_env:
        paths_to_try.append(Path(st_env).parent / "tokenizer.json")
    # Last-ditch: known canonical bf16 mirror location.
    paths_to_try.append(
        Path("/Users/mdot/models/gemma-4-a4b-bf16/tokenizer.json"))

    for p in paths_to_try:
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            print(f"[chat_template] failed to parse {p}: {e}", file=sys.stderr)
            continue
        out: dict[str, int] = {}
        for entry in data.get("added_tokens", []):
            if not isinstance(entry, dict):
                continue
            content = entry.get("content")
            tid = entry.get("id")
            if not content or tid is None:
                continue
            if not entry.get("special", False):
                continue
            if content in _EXCLUDE_FROM_ATOMIC:
                continue
            out[content] = int(tid)
        if out:
            print(f"[chat_template] loaded {len(out)} special tokens from {p}",
                  file=sys.stderr)
            return out
    return None


_loaded = _load_special_tokens_from_tokenizer_json()
if _loaded is not None:
    SPECIAL_TOKENS: dict[str, int] = _loaded
else:
    print("[chat_template] tokenizer.json unreachable; using hand-coded "
          "fallback SPECIAL_TOKENS (may miss model-specific atomic IDs)",
          file=sys.stderr)
    SPECIAL_TOKENS = dict(_FALLBACK_SPECIAL_TOKENS)

# Sort by length DESC so longer markers are matched ahead of any prefix
# that overlaps with a shorter one. (The current Gemma-4 set has no
# such overlaps, but it's a cheap invariant to maintain — if a future
# model adds e.g. `<|tool>` and `<|tool_extended>` we want the longer
# one to win.) Previously implemented as
#   re.compile("|".join(re.escape(k) for k in <length-desc>))
# but the alphabet is a finite set of LITERAL needles — `re.escape` was
# load-bearing precisely because the pattern is meant to be literal —
# so a plain longest-prefix scan over the sorted-by-length list is
# the same finite-state matcher with no regex dependency.
_SPECIAL_TOKENS_BY_LENGTH = sorted(SPECIAL_TOKENS.keys(), key=len, reverse=True)


def _find_special_at(text: str, pos: int) -> str | None:
    """Return the longest special-token literal that occurs at `text[pos]`
    or None if none does. Linear in the number of registered specials.
    """
    for marker in _SPECIAL_TOKENS_BY_LENGTH:
        if text.startswith(marker, pos):
            return marker
    return None


def _iter_specials(text: str):
    """Yield (start, end, marker) for each occurrence of a registered
    special-token literal in `text`, leftmost-longest. Equivalent to
    the prior `_SPECIAL_RE.finditer(text)`.
    """
    i = 0
    n = len(text)
    while i < n:
        marker = _find_special_at(text, i)
        if marker is not None:
            yield i, i + len(marker), marker
            i += len(marker)
        else:
            i += 1


# Boundary text fragments the canonical jinja template emits at turn
# transitions. Used by render_turn_delta() to assemble the suffix that
# extends a stored prior-turn prefix with one new turn + an open model
# turn. Must match chat_template.jinja or the conversation-state cache
# desyncs silently.
#
# After the prior assistant turn, the model emitted `<turn|>` (token
# id 106) as its EOS; the jinja template would have rendered the
# trailing "\n" + next role-open. So a "normal" turn boundary delta
# starts with "\n<|turn>{role}\n".
_TURN_OPEN_USER = "\n<|turn>user\n"
_TURN_CLOSE = "<turn|>\n"
_MODEL_GEN_PROMPT_NO_THINKING = "<|turn>model\n<|channel>thought\n<channel|>"

# When the prior message in the engine's KV was a tool_call (which
# emits `<|tool_call>...<tool_call|>` and does NOT close with `<turn|>`),
# the chat template appends `<|tool_response>` directly with no inter-
# turn newline. Used to walk back into the conversation from a tool
# round.
_TOOL_RESPONSE_OPEN = "<|tool_response>"
_TOOL_RESPONSE_CLOSE = "<tool_response|>"


def render_turn_delta(message: dict, *,
                       prev_was_tool_call: bool = False) -> list[Chunk]:
    """Render the chunk-list delta to APPEND to a stored prior-turn
    prefix so the conversation extends by one new turn + open model
    turn (with the canonical no-thinking gen-prompt epilogue).

    Mirrors the boundary-tokens portion of chat_template.jinja's
    per-message loop + add_generation_prompt epilogue, but rendered
    locally because the canonical template can only render whole
    conversations from BOS.

    Supported roles for `message['role']`:
      * 'user' — emits `\\n<|turn>user\\n{content}<turn|>\\n` then the
        model gen-prompt epilogue. Multimodal `image_url` content parts
        (must be `data:` URIs) are kept as ImageChunk in the returned
        list, matching `render_chat()`'s output shape.
      * 'tool' / 'function' — emits `<|tool_response>response:{name}{
        ...}<tool_response|>`. If `prev_was_tool_call=True` (i.e. the
        prior emitted message was an assistant tool_call), no inter-
        turn newline; otherwise the canonical template would have
        rendered a turn close, so we open with one. The gen-prompt
        epilogue is suppressed (template line 341 condition).

    `prev_was_tool_call` is determined by the bridge from inspecting
    the next-to-last message in the canonical message list. The
    bridge is the only caller and tracks this as part of warm-path
    eligibility logic.
    """
    role = message.get("role")
    if role in ("tool", "function"):
        return _render_tool_response_delta(message,
                                            prev_was_tool_call=prev_was_tool_call)
    if role != "user":
        raise ValueError(f"render_turn_delta: unsupported role {role!r}")
    return _render_user_turn_delta(message)


def _render_user_turn_delta(message: dict) -> list[Chunk]:
    norm_msgs, payloads = _normalize_messages([message])
    msg = norm_msgs[0]
    chunks: list[Chunk] = [TextChunk(_TURN_OPEN_USER)]
    payload_idx = 0
    content = msg.get("content")
    if isinstance(content, list):
        text_acc: list[str] = []

        def flush_text() -> None:
            if text_acc:
                chunks.append(TextChunk("".join(text_acc).strip()))
                text_acc.clear()

        for item in content:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "text":
                text_acc.append((item.get("text") or "").strip())
            elif t == "image":
                flush_text()
                chunks.append(payloads[payload_idx])
                payload_idx += 1
        flush_text()
    else:
        chunks.append(TextChunk((content or "").strip()))
    chunks.append(TextChunk(_TURN_CLOSE + _MODEL_GEN_PROMPT_NO_THINKING))
    return chunks


def _render_tool_response_delta(message: dict, *,
                                  prev_was_tool_call: bool) -> list[Chunk]:
    """A `role: tool` message renders as a tool_response block. The
    canonical template's tool_response shape (chat_template.jinja:160-
    173) is `<|tool_response>response:{tool_name}{<key>:<value>,...}<
    tool_response|>` for mapping bodies; for plain string bodies, it's
    `<|tool_response>response:{tool_name}{value:<text>}<tool_response|>`.
    """
    name = message.get("name") or "unknown"
    body = message.get("content")
    parts: list[str] = []
    if not prev_was_tool_call:
        parts.append(_TURN_CLOSE)
    parts.append(_TOOL_RESPONSE_OPEN)
    parts.append(f"response:{name}{{")
    if isinstance(body, dict):
        kvs = []
        for k, v in sorted(body.items()):
            kvs.append(f"{k}:{_format_simple_value(v)}")
        parts.append(",".join(kvs))
    else:
        parts.append(f"value:{_format_simple_value(body)}")
    parts.append("}")
    parts.append(_TOOL_RESPONSE_CLOSE)
    # NOTE: gen-prompt epilogue is intentionally OMITTED — the canonical
    # template suppresses it when prev_message_type ∈ {tool_call,
    # tool_response}. The next chat-completions call will likely be the
    # assistant's textual response after the tool round; if instead the
    # caller wants a fresh model turn, they pass it in the message list
    # with role:assistant ahead of time.
    return [TextChunk("".join(parts))]


def _format_simple_value(v: Any) -> str:
    """Match chat_template.jinja's `format_argument` (line 118-147) for
    common scalar / sequence shapes. Only a subset is needed for tool
    response bodies — strings, numbers, booleans, lists.
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return f'<|"|>{v}<|"|>'
    if isinstance(v, list):
        return "[" + ",".join(_format_simple_value(x) for x in v) + "]"
    if isinstance(v, dict):
        # Tool bodies generally don't nest deeply, but support one level.
        return "{" + ",".join(
            f"{k}:{_format_simple_value(val)}"
            for k, val in sorted(v.items())
        ) + "}"
    return f'<|"|>{str(v)}<|"|>'


# Back-compat alias kept while bridge.py transitions to render_turn_delta.
def render_user_turn_delta(content: str | list[dict]) -> str:
    """DEPRECATED: text-only delta. Use render_turn_delta() instead."""
    if isinstance(content, list):
        parts = []
        for item in content:
            t = item.get("type") if isinstance(item, dict) else None
            if t == "text":
                parts.append((item.get("text") or "").strip())
            elif t == "image":
                parts.append(_IMAGE_PLACEHOLDER)
        content = "".join(parts)
    else:
        content = (content or "").strip()
    return (_TURN_OPEN_USER + content
             + _TURN_CLOSE + _MODEL_GEN_PROMPT_NO_THINKING)


def tokenize_with_specials(text: str, *, tokenize_fn, add_bos: bool) -> list[int]:
    """Tokenize `text` with atomic-ID emission for scaffolding markers.

    `tokenize_fn` is `gemma_ffi.tokenize` (injected so this module has no
    FFI dependency). Pure BPE for prose, atomic IDs for registered specials.
    """
    tokens: list[int] = []
    did_bos = not add_bos
    last_end = 0
    for start, end, marker in _iter_specials(text):
        prose = text[last_end:start]
        if prose:
            tokens.extend(tokenize_fn(prose, add_bos=(not did_bos)))
            did_bos = True
        if not did_bos:
            # No prose yet and caller wants BOS — rely on the tokenizer
            # to prepend it via a zero-width call.
            tokens.extend(tokenize_fn("", add_bos=True))
            did_bos = True
        tokens.append(SPECIAL_TOKENS[marker])
        last_end = end
    tail = text[last_end:]
    if tail:
        tokens.extend(tokenize_fn(tail, add_bos=(not did_bos)))
        did_bos = True
    elif not did_bos:
        tokens.extend(tokenize_fn("", add_bos=True))
    return tokens
