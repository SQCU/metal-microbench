<video src="https://raw.githubusercontent.com/SQCU/metal-microbench/main/recordings/tetraplex-20260420-122258.mp4" controls muted playsinline width="720"></video>

# gemma-metal

A from-scratch Metal inference engine for **Gemma-4-A4B** on Apple Silicon, with three peer web demo clients consuming one OpenAI-compatible REST surface over paged attention, a content-hash prefix cache, and batched multi-session AR decode. See [**docs/QUICKSTART.md**](docs/QUICKSTART.md) to run it on your M-series Mac. The two recordings on this page show the same four-way multimodal workload before and after the scheduler learned to fire soft-prefill on ≥ 2 ready slots instead of all-or-nothing — a 17.3 s → 8.7 s speedup on identical hardware, weights, and kernels.

<video src="https://raw.githubusercontent.com/SQCU/metal-microbench/main/recordings/tetraplex-20260420-115156.mp4" controls muted playsinline width="720"></video>
