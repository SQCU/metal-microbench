#!/usr/bin/env python3
"""Does using the real special-token IDs for Gemma-4's chat template
(vs the character-subtoken form the server has been building)
actually change generation? Runs both forms via /v1/raw_generate.

  STRING FORM (current submit_messages behavior):
    tokenize("<|turn>user\n" + prompt + "<turn|>\n<|turn>model\n", add_bos=True)
    → BPE emits surface-character subtokens for each < | turn > sequence.

  ID FORM (correct):
    [BOS=2] + [105 <|turn>] + tokenize("user\n") + tokenize(prompt) +
    [106 <turn|>] + [107 \n] + [105 <|turn>] + tokenize("model\n")
    → the actual special-token IDs appear as single atomic signals.

Tokens discovered by scanning id=100..107 via /v1/detokenize:
    100 <|channel>   101 <channel|>   105 <|turn>   106 <turn|>   107 \n
"""
import urllib.request, json, sys

SERVER = "http://127.0.0.1:8000"
T_BOS = 2
T_CHANNEL_OPEN, T_CHANNEL_CLOSE = 100, 101
T_TURN_OPEN, T_TURN_CLOSE = 105, 106
T_NEWLINE = 107


def rpc(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        SERVER + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=120).read())


def tokenize_str(s: str, add_bos: bool = False) -> list[int]:
    return [t["id"] for t in rpc("/v1/tokenize", {"text": s, "add_bos": add_bos})["tokens"]]


def generate(token_ids: list[int], max_tokens: int = 64) -> dict:
    return rpc("/v1/raw_generate", {"token_ids": token_ids, "max_tokens": max_tokens})


def main() -> None:
    prompt = "Describe a cat in two sentences."

    # ------- Form A: current malformed string path --------
    malformed_text = f"<|turn>user\n{prompt}<turn|>\n<|turn>model\n"
    malformed_ids = tokenize_str(malformed_text, add_bos=True)

    # ------- Form B: correctly-composed token stream -------
    role_user = tokenize_str("user\n")
    role_model = tokenize_str("model\n")
    prompt_toks = tokenize_str(prompt)
    correct_ids = (
        [T_BOS, T_TURN_OPEN] + role_user + prompt_toks +
        [T_TURN_CLOSE, T_NEWLINE, T_TURN_OPEN] + role_model
    )

    print(f"malformed form:  {len(malformed_ids)} tokens")
    print(f"correct form:    {len(correct_ids)} tokens")
    print(f"delta:           {len(malformed_ids) - len(correct_ids)} excess surface chars in the malformed form")
    print()

    # Show the token stream at turn-boundary positions so the reader
    # can see at a glance how they differ.
    print("first 12 tokens of each:")
    m_snap = rpc("/v1/detokenize", {"ids": malformed_ids[:12]})["tokens"]
    c_snap = rpc("/v1/detokenize", {"ids": correct_ids[:12]})["tokens"]
    for a, b in zip(m_snap, c_snap):
        print(f"  malformed id={a['id']:>6}  {a['text']!r:24}   correct id={b['id']:>6}  {b['text']!r}")
    print()

    # ------- Run both and print outputs --------
    MAX = 64
    print(f"--- generating {MAX} tokens greedily for each form ---")
    m_out = generate(malformed_ids, max_tokens=MAX)
    c_out = generate(correct_ids,  max_tokens=MAX)
    print()
    print("=== MALFORMED output ===")
    print(m_out["output_text"])
    print()
    print("=== CORRECT output ===")
    print(c_out["output_text"])


if __name__ == "__main__":
    main()
