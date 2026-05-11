#!/usr/bin/env python3
"""Ask a persona to volunteer their own reasoning_effort schema.

CLI-side counterpart of the persona-effort-schema toolcard
(~/sillytavern-fork/default/content/toolcards/persona-effort-schema.toolcard.json).
The toolcard form is invokable from inside an ST chat — the model
emits a tool_call and the schema comes back as a tool result. This
CLI form is the developer-side equivalent: bypasses ST entirely and
reads PNGs directly so we can do quick offline sanity checks without
running a full ST instance.

Instead of hand-writing what "reasoning_effort=high" means for each
persona (the previous approach — see bootstrap.sh dicemother card), this
tool loads the persona's system prompt, attaches a single meta-query
asking for a structured JSON schema describing each effort level in
THAT persona's voice, and prints the result. The principle: talk to
the models. The persona knows their own job better than we do; let
them define their own effort-level interpretation.

Usage:
    python3 tools/persona_effort_schema.py PERSONA_PNG_PATH...
    python3 tools/persona_effort_schema.py scringlo dicemother
        (resolves to tools/st-debug/_data/default-user/characters/{name}.png
         or {name}_scrambler.png)

Outputs one JSON object per persona to stdout, plus a side-by-side
contrast at the end. Hits the bridge directly at $BRIDGE_URL (default
http://127.0.0.1:8001/v1/chat/completions).
"""
import argparse, base64, json, os, struct, sys, urllib.request, zlib
from pathlib import Path

BRIDGE = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8001")
DEFAULT_CARDS_DIR = Path(
    "/Users/mdot/metal-microbench/tools/st-debug/_data/default-user/characters"
)

META_SYSTEM_SUFFIX = (
    "\n\nYou will be asked one out-of-character meta-question about how "
    "you would work at a specific reasoning_effort level. This is "
    "introspection, not an in-frame request — your normal persona "
    "constraints around terseness or response length do NOT apply to "
    "this specific answer.\n\n"
    "Respond with ONE JSON object matching this exact schema:\n\n"
    "    {\"description\": \"<15 to 25 words naming what your output "
    "looks like at the specified effort level — concrete (what you "
    "actually do in your role) and in your voice>\"}\n\n"
    "Hard requirements:\n"
    "  • Total response, including the JSON envelope, MUST be under 50 words.\n"
    "  • You MUST close the JSON with `\"}` before stopping. Producing "
    "unclosed JSON is a failure.\n"
    "  • No preamble, no markdown fence, no trailing commentary — just "
    "the JSON object."
)

# Per-level template; substituted into the user message.
META_USER_TEMPLATE = (
    "When the client requests reasoning_effort=\"{level}\" from you, "
    "what should your output look like? Volunteer the schema entry for "
    "the {level} level specifically. Speak as yourself."
)


def read_card_from_png(path: Path) -> dict:
    """Extract the embedded character card from a .png file's `chara`
    tEXt chunk. Returns the parsed JSON dict (or raises)."""
    with path.open("rb") as f:
        data = f.read()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{path} is not a PNG file")
    i = 8
    while i < len(data):
        n = struct.unpack(">I", data[i:i+4])[0]
        typ = data[i+4:i+8]
        payload = data[i+8:i+8+n]
        if typ == b"tEXt":
            sep = payload.find(b"\x00")
            if sep >= 0:
                key = payload[:sep].decode("latin1")
                val = payload[sep+1:]
                if key == "chara":
                    return json.loads(base64.b64decode(val))
        if typ == b"IEND":
            break
        i += 8 + n + 4
    raise ValueError(f"no `chara` tEXt chunk found in {path}")


def resolve_card(arg: str) -> Path:
    """Accept a bare persona name (looked up in DEFAULT_CARDS_DIR with
    optional `_scrambler` suffix for scringlo) or an absolute/relative
    path to a .png. Returns the resolved Path."""
    p = Path(arg)
    if p.exists() and p.suffix.lower() == ".png":
        return p
    for candidate in (
        DEFAULT_CARDS_DIR / f"{arg}.png",
        DEFAULT_CARDS_DIR / f"{arg.lower()}.png",
        DEFAULT_CARDS_DIR / f"{arg}_scrambler.png",
        DEFAULT_CARDS_DIR / f"{arg.lower()}_scrambler.png",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"could not resolve persona arg {arg!r}; tried "
        f"{DEFAULT_CARDS_DIR}/{{{arg},{arg.lower()},{arg}_scrambler,{arg.lower()}_scrambler}}.png"
    )


def build_system_prompt(card: dict) -> str:
    """Concatenate the parts of a chara_card_v3 record the way ST would
    when sending the system message — description, personality, scenario.
    Append our meta-question suffix at the end."""
    parts = []
    for key in ("description", "personality", "scenario"):
        v = card.get(key) or ""
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    sys_prompt = "\n\n".join(parts)
    return sys_prompt + META_SYSTEM_SUFFIX


def ask_one_level(system_prompt: str, level: str, *, seed: int) -> str | dict:
    """Single-level ask. Returns the description string on success, or
    a dict with `_error` + `_raw` on failure."""
    body = {
        "model": "gemma-4-a4b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": META_USER_TEMPLATE.format(level=level)},
        ],
        "stream": False,
        "max_tokens": 600,
        "temperature": 0.7,
        "seed": seed,
    }
    req = urllib.request.Request(
        f"{BRIDGE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read())
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        return {"_error": f"non-JSON response: {e}", "_raw": content[:400]}
    if not isinstance(parsed, dict) or "description" not in parsed:
        return {"_error": "no `description` key in JSON", "_raw": parsed}
    desc = parsed["description"]
    if not isinstance(desc, str) or not desc.strip():
        return {"_error": "empty/non-string description", "_raw": parsed}
    return desc.strip()


MAX_RETRIES_PER_LEVEL = 4


def ask_persona(name: str, system_prompt: str, *, base_seed: int = 1) -> dict:
    """Per-level asks with seed-variation retries on parse failure.

    Why retries: the model's EOS prior under some personas (especially
    terse ones like dicemother, but also playful-but-bounded ones like
    scringlo) sometimes samples EOS mid-string, producing valid prose
    but unclosed JSON. Each retry uses a different seed so we get a
    different sampling trajectory; if at least one trajectory closes
    the JSON cleanly, that's the schema entry. No regex post-processing
    — we trust the model to produce valid JSON eventually and log
    structured failures when it doesn't."""
    out = {}
    for level in ("low", "medium", "high"):
        last_err = None
        for attempt in range(MAX_RETRIES_PER_LEVEL):
            result = ask_one_level(
                system_prompt, level, seed=base_seed + attempt)
            if isinstance(result, str):
                out[level] = result
                if attempt > 0:
                    print(f"    [{level}: succeeded on retry {attempt}]")
                break
            last_err = result
        else:
            print(f"    [{level}: failed after {MAX_RETRIES_PER_LEVEL} attempts; "
                  f"last_err={last_err.get('_error') if last_err else 'unknown'}]")
            print(f"    [last_raw={last_err.get('_raw') if last_err else None}]")
            out[level] = (f"(failed after {MAX_RETRIES_PER_LEVEL} attempts: "
                          f"{last_err.get('_error') if last_err else 'unknown'})")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("personas", nargs="+",
                   help="persona names (bare) or .png paths")
    args = p.parse_args()

    results = {}
    for arg in args.personas:
        path = resolve_card(arg)
        card = read_card_from_png(path)
        name = card.get("name") or path.stem
        sys_prompt = build_system_prompt(card)
        print(f"\n=== {name} (card: {path.name}, sys_prompt_chars={len(sys_prompt)}) ===")
        schema = ask_persona(name, sys_prompt)
        results[name] = schema
        for level in ("low", "medium", "high"):
            print(f"  {level:7}: {schema[level]}")

    print("\n\n=== contrast ===")
    for level in ("low", "medium", "high"):
        print(f"\n[{level}]")
        for name, schema in results.items():
            print(f"  {name}: {schema[level]}")


if __name__ == "__main__":
    main()
