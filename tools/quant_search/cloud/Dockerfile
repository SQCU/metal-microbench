# Inference worker for the quant-search cloud fleet.
#
# Each worker runs a llama-server bound to one GGUF file on one GPU. The
# search driver POSTs prompts to /v1/chat/completions on each worker; KL
# eval, SVG MSE, multi-turn coherence etc. all get a normal OAI API.
#
# Build:
#   docker build -t metal-microbench/quant-worker:latest \
#     -f tools/quant_search/cloud/Dockerfile .
#
# Run (single-GPU, one config):
#   docker run --rm --gpus '"device=0"' \
#     -p 8080:8080 \
#     -v /path/to/ggufs:/models:ro \
#     -e GGUF=/models/gemma-4-26B-A4B-it-fp16.gguf \
#     metal-microbench/quant-worker:latest
#
#   (Replace fp16.gguf with whichever heterogeneous-quant probe you're
#    testing — uniform-quant configs aren't valid probes for this
#    project; see README.md for the methodology.)
#
# Run on a 7-GPU box, one worker per GPU (different ports, same image):
#   for i in $(seq 0 6); do
#     docker run -d --gpus "\"device=$i\"" \
#       -p $((8080+i)):8080 \
#       -v /path/to/ggufs:/models:ro \
#       -e GGUF=/models/config_$i.gguf \
#       --name quant-worker-$i \
#       metal-microbench/quant-worker:latest
#   done

# CUDA 12.8+ is required for Blackwell SM_100 (B100/B200) and SM_120 (RTX
# 50-series) support. Earlier CUDA versions silently fall back to PTX JIT
# on Blackwell, which skips the new tensor-core paths and leaves most of
# the perf on the table — defeating the entire point of running on the
# new silicon. Use 12.8 minimum; bump to whatever's current when this
# image is rebuilt.
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# llama.cpp build deps. Keep this lean — we only need the inference path.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential cmake ccache curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pin to a llama.cpp commit. Update via --build-arg LLAMA_CPP_REF=<sha>
# when you want a newer baseline; pinning ensures bit-exact GGUF reads
# across cloud workers and the local dev box.
ARG LLAMA_CPP_REF=master
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
        https://github.com/ggerganov/llama.cpp.git /opt/llama.cpp

WORKDIR /opt/llama.cpp
# CUDA build, fp16, all GGML quants enabled. CMAKE_CUDA_ARCHITECTURES
# covers the modern accelerator fleet:
#   80   — A100 (Ampere)
#   86   — RTX 30-series (Ampere consumer)
#   89   — RTX 40-series + Ada server (Ada Lovelace)
#   90   — H100 / H200 (Hopper)
#   100  — B100 / B200 / GB200 (Blackwell server)
#   120  — RTX 50-series (Blackwell consumer)
# One image runs on any of them. Image size grows with the arch list;
# trim if a specific deploy target is known.
RUN cmake -B build \
        -DGGML_CUDA=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90;100;120" \
    && cmake --build build --config Release --target llama-server -j

# Copy the binary out so we can run it on a leaner runtime image. We keep
# everything in one stage here for simplicity; multi-stage shrink can come
# later once we know what libraries the runtime needs.

EXPOSE 8080

# Default flags chosen for our use case:
#   -m: GGUF (must be set via env)
#   --port: bind on 8080 inside container
#   --host: 0.0.0.0 so docker port-forwarding works
#   -ngl 99: offload all layers to GPU
#   -c 8192: context (sufficient for SVG + multi-turn coherence harness)
#   -np 8: parallel slots (so the search driver can submit batched evals)
#   -t 8: 8 CPU threads for token io / sampling
#   --metrics: expose /metrics for prometheus-style scraping
#   --jinja: use the model's chat template if embedded in the GGUF
ENV PORT=8080 \
    CONTEXT=8192 \
    PARALLEL_SLOTS=8 \
    GPU_LAYERS=99

CMD ["/bin/bash", "-c", "exec /opt/llama.cpp/build/bin/llama-server \
        -m \"${GGUF:?env var GGUF (path to .gguf) must be set}\" \
        --host 0.0.0.0 \
        --port ${PORT} \
        -ngl ${GPU_LAYERS} \
        -c ${CONTEXT} \
        -np ${PARALLEL_SLOTS} \
        --metrics \
        --jinja"]
