#!/usr/bin/env python
"""
Reference oracle for Gemma-4-A4B-It LM (text-decoder) forward logits.

Loads the full bf16 model from /Users/mdot/models/gemma-4-a4b-bf16/ (CPU or
MPS), tokenizes a fixed prompt, runs a single causal-LM forward over the
prefill, and dumps:

  test_data/reference/lm_<tag>_tokens.npy   int32[S]                   prompt token ids
  test_data/reference/lm_<tag>_logits.npy   float32[S, VOCAB]          post-softcap logits
  test_data/reference/lm_<tag>_hiddens.npy  float32[NUM_LAYERS+1, S, HIDDEN]
      residual stream captured at NUM_LAYERS+1 boundaries:
        idx 0             = post-embed_tokens (already *sqrt(hidden_size))
        idx 1..NUM_LAYERS = after decoder_layer[L-1] (the residual returned
                            by that block, i.e. the input to layer L or to
                            self.norm if L==NUM_LAYERS)

S = prompt length in tokens. VOCAB = 262144. HIDDEN = 2816. NUM_LAYERS = 30.

The Swift-side KL harness (LM_KL_REF=<dir>) reads the logits file.
The Swift-side layer-dump harness (LM_DUMP_LAYERS=<dir>) writes a matching
lm_swift_hiddens.npy for a compare_lm_hiddens.py script to diff.

Hidden-state capture is via forward-hooks on each decoder layer — Gemma4's
Gemma4TextModel.forward does not plumb output_hidden_states and discards
per-layer activations.

Run:
    cd /Users/mdot/metal-microbench/test_data/reference
    .venv/bin/python extract_lm_logits.py --prompt "Hello, my name is" --tag hello
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="Hello, my name is",
                        help="Prompt string; tokenized with add_special_tokens=True.")
    parser.add_argument("--tag", type=str, default="hello",
                        help="Filename tag; outputs lm_<tag>_{tokens,logits}.npy.")
    parser.add_argument("--model", type=str,
                        default="/Users/mdot/models/gemma-4-a4b-bf16",
                        help="HF model directory (bf16 safetensors).")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu | mps (cpu is slow but safe for bf16 matmul).")
    args = parser.parse_args()

    out_dir = Path(__file__).parent
    tokens_path = out_dir / f"lm_{args.tag}_tokens.npy"
    logits_path = out_dir / f"lm_{args.tag}_logits.npy"
    hiddens_path = out_dir / f"lm_{args.tag}_hiddens.npy"
    l0_probes_path = out_dir / f"lm_{args.tag}_l0_probes.npy"

    print(f"loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok.encode(args.prompt, add_special_tokens=True, return_tensors="pt")
    print(f"prompt: {args.prompt!r}")
    print(f"tokenized to {ids.shape[1]} tokens: {ids[0].tolist()}")
    for t in ids[0].tolist():
        print(f"  {t:>6d}  {tok.decode([t])!r}")

    print(f"loading model from {args.model} on {args.device} (bf16)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(args.device).eval()
    print(f"  loaded in {time.time() - t0:.1f} s")

    ids = ids.to(args.device)
    S = ids.shape[1]

    # Install per-layer forward-hooks to capture the residual stream.
    # Gemma4ForConditionalGeneration wraps Gemma4Model which wraps
    # Gemma4TextModel as .language_model. Walk down until we find the module
    # that actually owns `.layers`.
    text_model = model
    for attr in ("model", "language_model"):
        candidate = getattr(text_model, attr, None)
        if candidate is not None:
            text_model = candidate
            if hasattr(text_model, "layers"):
                break
    if not hasattr(text_model, "layers"):
        raise RuntimeError(f"could not find .layers on {type(text_model).__name__}; "
                           f"walked from {type(model).__name__}")
    print(f"  found decoder stack at {type(text_model).__name__}")
    num_layers = len(text_model.layers)
    # text_config is nested on the multimodal Gemma4Config; text_model.config
    # points at the inner Gemma4TextConfig.
    hidden_size = text_model.config.hidden_size
    print(f"  num_hidden_layers={num_layers}, hidden_size={hidden_size}")

    # captured[L] is the residual AFTER layer L's full block (the decoder_layer
    # output). We also need the INPUT to layer 0 (= post-embed+scale), which we
    # get from a pre-hook on layers[0].
    layer_outputs: list[torch.Tensor | None] = [None] * num_layers
    embed_input: list[torch.Tensor | None] = [None]

    def make_fwd_hook(idx: int):
        def hook(_mod, _inp, out):
            # Decoder layer forward returns a Tensor (not a tuple) in Gemma4.
            t = out[0] if isinstance(out, tuple) else out
            layer_outputs[idx] = t.detach().float().cpu()
        return hook

    def pre_hook_layer0(_mod, inp):
        # inp is a tuple; first positional is hidden_states (post-embed+scale).
        layer_outputs  # silence linter
        embed_input[0] = inp[0].detach().float().cpu()

    handles = []
    handles.append(text_model.layers[0].register_forward_pre_hook(pre_hook_layer0))
    for L in range(num_layers):
        handles.append(text_model.layers[L].register_forward_hook(make_fwd_hook(L)))

    # Layer-0 intra-layer probes to bisect the first divergent decoder block.
    # Probe 0: hidden AFTER (self_attn → post_attn_norm → + residual).
    #          Captured via pre-hook on pre_feedforward_layernorm.
    # Probe 1: hidden_states_1 = post_feedforward_layernorm_1(mlp.down_proj_out).
    # Probe 2: hidden_states_2 = post_feedforward_layernorm_2(experts_combined).
    # Probe 3: pre_feedforward_layernorm_2 output = input to the experts.
    # Probe 4: experts module output = raw scatter-sum pre-post_feedforward_layernorm_2.
    l0_probes: list[torch.Tensor | None] = [None, None, None, None, None]
    l0_router: dict[str, torch.Tensor | None] = {"top_k_index": None, "top_k_weights": None}
    layer0 = text_model.layers[0]

    def probe0_pre_hook(_mod, inp):
        l0_probes[0] = inp[0].detach().float().cpu()

    def probe1_fwd_hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        l0_probes[1] = t.detach().float().cpu()

    def probe2_fwd_hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        l0_probes[2] = t.detach().float().cpu()

    def probe3_fwd_hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        l0_probes[3] = t.detach().float().cpu()

    def probe4_fwd_hook(_mod, _inp, out):
        # experts.forward returns final_hidden_states (the scatter-summed output,
        # weighted by top_k_weights). Shape [B*S, HIDDEN] — flattened per layer 1388.
        t = out[0] if isinstance(out, tuple) else out
        l0_probes[4] = t.detach().float().cpu()

    def router_fwd_hook(_mod, _inp, out):
        # Gemma4TextRouter.forward returns (router_probabilities, top_k_weights, top_k_index).
        _, w, idx = out
        l0_router["top_k_weights"] = w.detach().float().cpu()
        l0_router["top_k_index"] = idx.detach().to(torch.int32).cpu()

    assert hasattr(layer0, "pre_feedforward_layernorm"), "layer 0 missing pre_feedforward_layernorm"
    assert hasattr(layer0, "post_feedforward_layernorm_1"), "layer 0 missing post_feedforward_layernorm_1"
    assert hasattr(layer0, "post_feedforward_layernorm_2"), "layer 0 missing post_feedforward_layernorm_2"
    assert hasattr(layer0, "pre_feedforward_layernorm_2"), "layer 0 missing pre_feedforward_layernorm_2"
    assert hasattr(layer0, "experts"), "layer 0 missing experts"
    assert hasattr(layer0, "router"), "layer 0 missing router"
    handles.append(layer0.pre_feedforward_layernorm.register_forward_pre_hook(probe0_pre_hook))
    handles.append(layer0.post_feedforward_layernorm_1.register_forward_hook(probe1_fwd_hook))
    handles.append(layer0.post_feedforward_layernorm_2.register_forward_hook(probe2_fwd_hook))
    handles.append(layer0.pre_feedforward_layernorm_2.register_forward_hook(probe3_fwd_hook))
    handles.append(layer0.experts.register_forward_hook(probe4_fwd_hook))
    handles.append(layer0.router.register_forward_hook(router_fwd_hook))

    print(f"running forward (prefill, S={S}) — this may take a moment on CPU")
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    print(f"  forward in {time.time() - t0:.1f} s")

    for h in handles:
        h.remove()
    # Gemma-4 applies final_logit_softcapping internally (cap=30 by default).
    # out.logits is post-cap and post-lm_head.
    logits = out.logits[0].float().cpu().numpy()   # [S, VOCAB]
    tokens = ids[0].cpu().numpy().astype(np.int32)

    print(f"logits shape: {logits.shape}  dtype: {logits.dtype}")
    print(f"logits[last-pos] min={logits[-1].min():.3f}  max={logits[-1].max():.3f}")
    top5 = np.argsort(-logits[-1])[:5]
    print(f"top-5 at last position (greedy next token):")
    for t in top5:
        print(f"  {t:>6d}  {tok.decode([int(t)])!r}  logit={logits[-1, t]:.3f}")

    np.save(tokens_path, tokens)
    np.save(logits_path, logits)
    print(f"wrote {tokens_path}")
    print(f"wrote {logits_path}")

    # Stack the residual stream snapshots: idx 0 = post-embed (pre-layer-0),
    # idx L+1 = output of layer L. Shape [NUM_LAYERS+1, S, HIDDEN] fp32.
    assert embed_input[0] is not None, "pre_hook_layer0 did not fire"
    snapshots = [embed_input[0][0]]   # drop batch dim → [S, HIDDEN]
    for L in range(num_layers):
        assert layer_outputs[L] is not None, f"layer {L} forward hook did not fire"
        snapshots.append(layer_outputs[L][0])
    hiddens = torch.stack(snapshots, dim=0).numpy()  # [NUM_LAYERS+1, S, HIDDEN]
    assert hiddens.shape == (num_layers + 1, S, hidden_size), \
        f"unexpected hiddens shape {hiddens.shape}"
    print(f"hiddens shape: {hiddens.shape}  dtype: {hiddens.dtype}")
    print(f"  post-embed   (idx 0)   min={hiddens[0].min():.3f}  max={hiddens[0].max():.3f}")
    print(f"  after last L (idx {num_layers})  min={hiddens[-1].min():.3f}  max={hiddens[-1].max():.3f}")
    np.save(hiddens_path, hiddens)
    print(f"wrote {hiddens_path}")

    # Pack the 5 layer-0 intra-layer probes into [5, S, HIDDEN] fp32.
    for i, t in enumerate(l0_probes):
        assert t is not None, f"layer-0 probe {i} was not captured"
    # Probe 4 (experts output) is flattened [B*S, HIDDEN] whereas others are [B, S, HIDDEN];
    # reshape/pick slot 0 uniformly.
    def slot0(t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 3:
            return t[0]                         # [S, HIDDEN]
        elif t.dim() == 2:
            return t.view(1, -1, t.shape[-1])[0]  # B=1 → [S, HIDDEN]
        raise ValueError(f"unexpected probe shape {t.shape}")
    l0 = torch.stack([slot0(p) for p in l0_probes], dim=0).numpy()
    assert l0.shape == (5, S, hidden_size), f"unexpected l0 shape {l0.shape}"
    names = ["post_attn+res", "ffw_norm_1 out", "ffw_norm_2 out",
             "pre_ffw_2 out", "moe_sum raw   "]
    print(f"l0 probes shape: {l0.shape}")
    for i, name in enumerate(names):
        print(f"  probe {i} ({name:<17}) min={l0[i].min():.3f}  max={l0[i].max():.3f}")
    np.save(l0_probes_path, l0)
    print(f"wrote {l0_probes_path}")

    # Router captures. top_k_index [B, S, K]; top_k_weights [B, S, K]. Drop batch
    # and save as [S, K]. Note HF shapes here might be [B*S, K] — normalize.
    idx = l0_router["top_k_index"]
    w = l0_router["top_k_weights"]
    assert idx is not None and w is not None
    if idx.dim() == 3: idx = idx[0]         # [B=1, S, K] → [S, K]
    if w.dim() == 3: w = w[0]
    idx_np = idx.numpy().astype(np.int32)
    w_np = w.numpy().astype(np.float32)
    print(f"  top_k_index shape: {idx_np.shape}  top_k_weights shape: {w_np.shape}")
    print(f"  pos 0 expert_ids: {idx_np[0].tolist()}")
    print(f"  pos 0 gate_w    : {[round(float(v), 4) for v in w_np[0]]}")
    np.save(out_dir / f"lm_{args.tag}_l0_expert_ids.npy", idx_np)
    np.save(out_dir / f"lm_{args.tag}_l0_gate_w.npy", w_np)
    print(f"wrote lm_{args.tag}_l0_expert_ids.npy and lm_{args.tag}_l0_gate_w.npy")


if __name__ == "__main__":
    main()
