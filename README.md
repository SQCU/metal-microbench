![tetraplex — after the multi-slot soft-prefill fix (8.7s)](docs/media/tetraplex-after-multislot.gif)

# gemma-metal

A from-scratch Metal inference engine for **Gemma-4-A4B** on Apple Silicon, with three peer web demo clients consuming one OpenAI-compatible REST surface over paged attention, a content-hash prefix cache, and batched multi-session AR decode. See [**docs/QUICKSTART.md**](docs/QUICKSTART.md) to run it on your M-series Mac. The two animations on this page are the same four-way multimodal workload before and after the scheduler learned to fire soft-prefill on ≥ 2 ready slots instead of all-or-nothing — a 17.3 s → 8.7 s speedup on identical hardware, weights, and kernels ([before](docs/media/tetraplex-before-multislot.mp4) / [after](docs/media/tetraplex-after-multislot.mp4) MP4s if you want higher fidelity).

![tetraplex — before, serialized soft-prefill (17.3s)](docs/media/tetraplex-before-multislot.gif)
