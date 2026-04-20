"""Smoke test for the Swift FFI bridge (no FastAPI).

Opens two sessions with different prompts, pumps the engine until both hit
.done, prints the outputs. Proves:
  - dylib loads
  - init / open_session / submit / tick / poll / close_session work
  - two concurrent sessions batch through B=4 scheduler correctly
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gemma_ffi as g


def main() -> int:
    gguf = os.environ.get(
        "GGUF_PATH",
        "/Users/mdot/models/gemma-4-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    )
    print(f"loading {gguf} ...")
    t0 = time.time()
    g.init(gguf)
    print(f"  ready in {time.time() - t0:.2f}s (bos={g.bos_id()}, eos={g.eos_id()})")

    prompts = [
        "<|turn>user\nWhat is the capital of France?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
        "<|turn>user\nWhat is 7 times 8?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>",
    ]

    sessions = []
    for i, p in enumerate(prompts):
        sid = g.open_session(max_new_tokens=32)
        toks = g.tokenize(p, add_bos=True)
        g.submit(sid, toks)
        print(f"  session {sid} submitted {len(toks)} tokens")
        sessions.append((sid, p, []))

    # Pump until every session hits .done (or a safety cap).
    print("\npumping:")
    start = time.time()
    ticks = 0
    while g.has_work() and ticks < 2000:
        g.tick()
        ticks += 1
        for sid, _prompt, out in sessions:
            tokens = g.poll(sid)
            if tokens:
                out.extend(tokens)
    wall = time.time() - start

    print(f"\nfinished in {wall:.2f}s ({ticks} ticks)")
    for sid, prompt, out in sessions:
        text = g.detokenize(out)
        print(f"\n  [session {sid}] {len(out)} tok")
        print(f"  prompt:   {prompt[-60:]!r}")
        print(f"  response: {text!r}")
        g.close_session(sid)

    return 0


if __name__ == "__main__":
    sys.exit(main())
