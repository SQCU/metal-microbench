#!/usr/bin/env python
"""
Greedy AR trajectory oracle. Tokenizes a prompt, runs HF's bf16 Gemma-4-A4B
forward over the prefix, then greedy-decodes num_gen additional tokens (each
step feeds the previous step's argmax). Writes two npys the Swift KL harness
can teacher-force against:

    lm_<tag>_tokens.npy   int32[S + num_gen]       full token sequence (prompt + generations)
    lm_<tag>_logits.npy   float32[S + num_gen, V]  logits at each position
                                                    (post-softcap, post-lm_head)

Swift-side: replay the same token sequence through its AR forward, capture
per-position logits, compare.

Run:
    .venv/bin/python extract_lm_trajectory.py --prompt "The capital of France is" --num-gen 16 --tag capfr
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--num-gen", type=int, default=16,
                    help="Tokens to greedy-decode past the prompt.")
    ap.add_argument("--tag", default="capfr")
    ap.add_argument("--model", default="/Users/mdot/models/gemma-4-a4b-bf16")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chat", action="store_true",
                    help="Wrap prompt with Gemma-4-it chat template (user→assistant turn).")
    args = ap.parse_args()

    out_dir = Path(__file__).parent
    tokens_path = out_dir / f"lm_{args.tag}_tokens.npy"
    logits_path = out_dir / f"lm_{args.tag}_logits.npy"
    hiddens_path = out_dir / f"lm_{args.tag}_hiddens.npy"

    print(f"loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if args.chat:
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False, add_generation_prompt=True)
        ids = tok.encode(formatted, add_special_tokens=False, return_tensors="pt")
        print(f"chat-wrapped prompt: {formatted!r}")
    else:
        ids = tok.encode(args.prompt, add_special_tokens=True, return_tensors="pt")
    S = int(ids.shape[1])
    print(f"prompt: {args.prompt!r}")
    print(f"  tokenized to S={S}: {ids[0].tolist()}")
    for t in ids[0].tolist():
        print(f"    {t:>6d}  {tok.decode([t])!r}")

    print(f"loading model (bf16, {args.device})")
    t0 = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        attn_implementation="eager",     # match our Swift reconstruction path
    ).to(args.device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    ids = ids.to(args.device)
    total = S + args.num_gen
    V = m.config.text_config.vocab_size if hasattr(m.config, "text_config") else m.config.vocab_size
    all_tokens = np.zeros(total, dtype=np.int32)
    all_logits = np.zeros((total, V), dtype=np.float32)
    all_tokens[:S] = ids[0].cpu().numpy().astype(np.int32)

    # Walk down to the actual decoder stack (Gemma4TextModel.layers).
    text_model = m
    for attr in ("model", "language_model"):
        cand = getattr(text_model, attr, None)
        if cand is not None:
            text_model = cand
            if hasattr(text_model, "layers"):
                break
    assert hasattr(text_model, "layers"), \
        f"no .layers on {type(text_model).__name__}"
    num_layers = len(text_model.layers)
    hidden_size = text_model.config.hidden_size
    print(f"  layers={num_layers}, hidden={hidden_size}")
    all_hiddens = np.zeros((num_layers + 1, total, hidden_size), dtype=np.float32)
    # Layer-0 intra-layer probes across all positions (same 5 slots as
    # extract_lm_logits.py): post_attn+res, ffw_norm_1 out, ffw_norm_2 out,
    # pre_ffw_2 out, experts raw output. Shape [5, total, HIDDEN] fp32.
    all_l0_probes = np.zeros((5, total, hidden_size), dtype=np.float32)

    # Hooks: slot 0 = input to layer 0 (post-embed+scale), slot L+1 = output of
    # layer L (for L in 0..num_layers-1). A pre-hook on layer 0 captures the
    # post-embed input; per-layer forward hooks capture the post-block residual.
    step_hidden_buf: list[list[torch.Tensor | None]] = [[None] * (num_layers + 1)]
    # step_hidden_buf[0] holds the current step's captures; the outer list is a
    # trick to make it mutable from nested closures.
    step_l0_buf: list[list[torch.Tensor | None]] = [[None] * 5]

    def pre_hook_layer0(_mod, inp):
        step_hidden_buf[0][0] = inp[0].detach().float().cpu()

    def make_fwd_hook(idx: int):
        def hook(_mod, _inp, out):
            t = out[0] if isinstance(out, tuple) else out
            step_hidden_buf[0][idx + 1] = t.detach().float().cpu()
        return hook

    handles = [text_model.layers[0].register_forward_pre_hook(pre_hook_layer0)]
    for L in range(num_layers):
        handles.append(text_model.layers[L].register_forward_hook(make_fwd_hook(L)))

    # Layer-0 intra-layer probes (same 5 probes as extract_lm_logits.py).
    layer0 = text_model.layers[0]
    def l0_pre_hook_pre_ffw(_m, inp): step_l0_buf[0][0] = inp[0].detach().float().cpu()
    def l0_hook_post_ffw_1(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out; step_l0_buf[0][1] = t.detach().float().cpu()
    def l0_hook_post_ffw_2(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out; step_l0_buf[0][2] = t.detach().float().cpu()
    def l0_hook_pre_ffw_2(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out; step_l0_buf[0][3] = t.detach().float().cpu()
    def l0_hook_experts(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out; step_l0_buf[0][4] = t.detach().float().cpu()
    handles.append(layer0.pre_feedforward_layernorm.register_forward_pre_hook(l0_pre_hook_pre_ffw))
    handles.append(layer0.post_feedforward_layernorm_1.register_forward_hook(l0_hook_post_ffw_1))
    handles.append(layer0.post_feedforward_layernorm_2.register_forward_hook(l0_hook_post_ffw_2))
    handles.append(layer0.pre_feedforward_layernorm_2.register_forward_hook(l0_hook_pre_ffw_2))
    handles.append(layer0.experts.register_forward_hook(l0_hook_experts))

    def drain_hiddens_into(positions: range):
        """Copy the current step's captures into all_hiddens at the given positions
        (1 position for AR steps, S positions for the prefill pass)."""
        for slot in range(num_layers + 1):
            t = step_hidden_buf[0][slot]
            assert t is not None, f"slot {slot} was not captured"
            arr = t[0].numpy()
            assert arr.shape[0] == len(positions), \
                f"slot {slot} captured {arr.shape[0]} positions, expected {len(positions)}"
            for i, p in enumerate(positions):
                all_hiddens[slot, p] = arr[i]
        # L0 intra-layer probes.
        for slot in range(5):
            t = step_l0_buf[0][slot]
            if t is None: continue
            # probe 4 (experts) returns [B*S, HIDDEN] flat; others [B, S, HIDDEN].
            arr = t.view(1, -1, hidden_size)[0].numpy() if t.dim() == 2 else t[0].numpy()
            assert arr.shape[0] == len(positions), \
                f"l0 slot {slot} captured {arr.shape[0]}, expected {len(positions)}"
            for i, p in enumerate(positions):
                all_l0_probes[slot, p] = arr[i]
        # Reset for the next step.
        step_hidden_buf[0] = [None] * (num_layers + 1)
        step_l0_buf[0] = [None] * 5

    # Pass 1: prefill over the prompt with use_cache=True so subsequent steps
    # only feed one new token. HF's KV cache object carries the state.
    print(f"prefill S={S} tokens...")
    t0 = time.time()
    with torch.no_grad():
        out = m(input_ids=ids, use_cache=True)
    past = out.past_key_values
    prefill_logits = out.logits[0].float().cpu().numpy()     # [S, V]
    all_logits[:S] = prefill_logits
    drain_hiddens_into(range(0, S))
    print(f"  prefill in {time.time() - t0:.1f}s; last-pos top5: "
          f"{np.argsort(-prefill_logits[-1])[:5].tolist()}")

    next_tok = int(prefill_logits[-1].argmax())
    print(f"greedy-decoding {args.num_gen} tokens...")
    t0 = time.time()
    for i in range(args.num_gen):
        pos = S + i
        all_tokens[pos] = next_tok
        inp = torch.tensor([[next_tok]], dtype=torch.long, device=args.device)
        with torch.no_grad():
            out = m(input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values
        step_logits = out.logits[0, 0].float().cpu().numpy()  # [V]
        all_logits[pos] = step_logits
        drain_hiddens_into(range(pos, pos + 1))
        next_tok = int(step_logits.argmax())
        if i < 8 or i == args.num_gen - 1:
            print(f"  step {i:>3d}  pos={pos}  gen={all_tokens[pos]:>6d}  "
                  f"next_argmax={next_tok:>6d}  {tok.decode([int(all_tokens[pos])])!r} → "
                  f"{tok.decode([next_tok])!r}")
    print(f"  {args.num_gen} steps in {time.time() - t0:.1f}s")

    for h in handles:
        h.remove()

    # Preview: render prompt + generated tokens.
    gen_ids = all_tokens[S:].tolist()
    print(f"\ngenerated continuation: {tok.decode(gen_ids)!r}")

    np.save(tokens_path, all_tokens)
    np.save(logits_path, all_logits)
    np.save(hiddens_path, all_hiddens)
    l0_probes_path = out_dir / f"lm_{args.tag}_l0_probes.npy"
    np.save(l0_probes_path, all_l0_probes)
    print(f"wrote {tokens_path}")
    print(f"wrote {logits_path}")
    print(f"wrote {hiddens_path}  shape={list(all_hiddens.shape)}")
    print(f"wrote {l0_probes_path}  shape={list(all_l0_probes.shape)}")


if __name__ == "__main__":
    main()
