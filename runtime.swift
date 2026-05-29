// Env-var-driven demo dispatchers and one-time bootstrap.
//
// Extracted 2026-05-14 from bootstrap.swift. Two responsibilities:
//   - runEnvDrivenDemos(): inspects env vars (VISION_ST, LM_PROMPT,
//     PROFILE_PREFILL, etc.) and fires the matching harness from
//     harness.swift / profile_*.swift. Called by main.swift in the
//     executable target; the dylib target never invokes it.
//   - bootstrapGlobalState(): the one-time print + active_exp index
//     fill that used to be top-level statements. Called by both
//     main.swift (executable) and ffi.swift's gemma_init (dylib).
//     Idempotent.

import Metal
import Foundation

// ---------- Env-var-driven test harnesses ----------
// All harness bodies live in harness.swift. main.swift is just a
// dispatcher that invokes the right harness based on which env vars
// the user set. Multiple harnesses can fire in one binary invocation.
//
// Wrapped in runEnvDrivenDemos() so the library target (libgemma_metal
// .dylib, built for the Python FFI bridge) can compile all of main.swift
// without executing any of this on dylib load. The executable target
// (forward_graph) calls the function at the bottom of this file.
func runEnvDrivenDemos() {
if let stPath = ProcessInfo.processInfo.environment["VISION_ST"],
   ProcessInfo.processInfo.environment["VISION_FORWARD"] == nil,
   ProcessInfo.processInfo.environment["VISION_ASPECT_DIR"] == nil,
   ProcessInfo.processInfo.environment["VISION_SWEEP_DIR"] == nil {
    // VISION_ST alone → patch-embed smoke test. If the user passed
    // VISION_FORWARD/VISION_ASPECT_DIR/VISION_SWEEP_DIR, let those drive.
    runVisionPatchEmbedSmokeTest(stPath: stPath)
}
if let framePath = ProcessInfo.processInfo.environment["VISION_FORWARD"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    runVisionEndToEndForward(framePath: framePath, stPath: stPath)
}
if let batchDir = ProcessInfo.processInfo.environment["VISION_BATCH_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    runVisionBatchForward(batchDir: batchDir, stPath: stPath)
}
if let batchDir = ProcessInfo.processInfo.environment["VISION_CONCURRENT_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    let n = Int(ProcessInfo.processInfo.environment["VISION_CONCURRENT_N"] ?? "2") ?? 2
    runVisionConcurrentQueues(batchDir: batchDir, stPath: stPath, nQueues: n)
}
if let aspectDir = ProcessInfo.processInfo.environment["VISION_ASPECT_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"] {
    runVisionAspectSweep(aspectDir: aspectDir, stPath: stPath)
}
if let sweepDir = ProcessInfo.processInfo.environment["VISION_SWEEP_DIR"],
   let stPath = ProcessInfo.processInfo.environment["VISION_ST"],
   let refPath = ProcessInfo.processInfo.environment["VISION_REF"],
   let orderPath = ProcessInfo.processInfo.environment["VISION_SWEEP_ORDER"] {
    runVisionMultiFrameSweep(sweepDir: sweepDir, stPath: stPath,
                              refPath: refPath, orderPath: orderPath)
}
if let stPath = ProcessInfo.processInfo.environment["VISION_LOAD"] {
    runVisionWeightLoadSmokeTest(stPath: stPath)
}
if let png = ProcessInfo.processInfo.environment["VISION_PREPROCESS"] {
    runVisionPreprocessSmokeTest(png: png)
}
let isDumpRun = ProcessInfo.processInfo.environment["LM_DUMP_LAYERS"] != nil
    || ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"] != nil
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_KL_REF"] == nil,
   ProcessInfo.processInfo.environment["LM_PREFILL_VALIDATE"] == nil,
   ProcessInfo.processInfo.environment["LM_GENERATE"] == nil,
   ProcessInfo.processInfo.environment["LM_MULTISESSION"] == nil,
   ProcessInfo.processInfo.environment["LM_TEST_CVEC_CACHE"] == nil,
   ProcessInfo.processInfo.environment["LM_TEST_CACHE_DIVERGENCE"] == nil,
   ProcessInfo.processInfo.environment["LM_PROFILE_PREFILL"] == nil,
   ProcessInfo.processInfo.environment["LM_PROFILE_PREFILL_SWEEP"] == nil,
   ProcessInfo.processInfo.environment["LM_PROFILE_AR"] == nil,
   !isDumpRun {
    // GGUF_PATH alone → LM forward benchmark. If LM_KL_REF, LM_PREFILL_VALIDATE,
    // LM_GENERATE, LM_MULTISESSION, LM_TEST_CVEC_CACHE or any dump flag is also
    // set, let those harnesses drive (all reuse loadLmWeights).
    runGgufPathHarness(ggufPath: ggufPath)
}
// LM_GENERATE was a reach-inside demo that took a raw prompt string
// and ran generation directly. Equivalent behavior is available through
// /v1/chat/completions via the HTTP API.
// LM_MULTISESSION / LM_MULTITURN_DEMO / LM_COMPOSITE_DEMO / LM_MULTIMODAL
// demos were inline-chat-template harnesses that bypassed the FFI/HTTP API
// and hand-crafted `<|turn>user\n...` strings. Removed — equivalent
// behavior is available through /v1/chat/completions with the same
// messages payloads, using the model's real jinja template.
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let refDir = ProcessInfo.processInfo.environment["LM_KL_REF"],
   !isDumpRun {
    let tag = ProcessInfo.processInfo.environment["LM_KL_TAG"] ?? "hello"
    runLmKLHarness(ggufPath: ggufPath, refDir: refDir, tag: tag)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let tag = ProcessInfo.processInfo.environment["LM_PREFILL_VALIDATE"] {
    let refDir = ProcessInfo.processInfo.environment["LM_KL_REF"]
        ?? "/Users/mdot/metal-microbench/test_data/reference"
    runLmPrefillValidate(ggufPath: ggufPath, refDir: refDir, tag: tag)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_PROFILE_PREFILL"] != nil {
    runLmPrefillProfile(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_PROFILE_PREFILL_SWEEP"] != nil {
    runLmPrefillBandwidthSweep(ggufPath: ggufPath)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   ProcessInfo.processInfo.environment["LM_PROFILE_AR"] != nil {
    runLmARProfile(ggufPath: ggufPath)
}
if let outDir = ProcessInfo.processInfo.environment["FLEX_ATTN_TEST"] {
    runFlexAttnSlideV1Test(outDir: outDir)
    runFlexAttnFullPrefillTest(outDir: outDir)
}
if ProcessInfo.processInfo.environment["ATTN_BENCH"] != nil {
    runAttnBench()
}
if ProcessInfo.processInfo.environment["KV_VIZ"] != nil {
    runKvVisualizer()
}
if ProcessInfo.processInfo.environment["LM_TEST_RADIX_TRIE"] != nil {
    runRadixTrieTests()
}
// LM_SHARED_PREFIX_DEMO / LM_BRANCH_DEMO / LM_PAUSE_DEMO were inline-
// template harnesses replicating what /v1/chat/completions does. Removed.
// The scheduler features these demos exercised (prefix caching, branch,
// pause/resume) are exercised through the HTTP API by normal clients.
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"],
   let outDir = ProcessInfo.processInfo.environment["LM_DUMP_EXPERT_W"] {
    let id = Int(ProcessInfo.processInfo.environment["LM_DUMP_EXPERT_ID"] ?? "52") ?? 52
    runDumpExpertWeights(ggufPath: ggufPath, expertId: id, outDir: outDir)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_PATH"], isDumpRun,
   ProcessInfo.processInfo.environment["LM_DUMP_EXPERT_W"] == nil {
    // LM_DUMP_LAYERS=<dir> and/or LM_DUMP_L0_INTERNALS=<dir> activate dumps.
    // Tokens come from LM_KL_REF's lm_<tag>_tokens.npy (reuses oracle files).
    let refDir = ProcessInfo.processInfo.environment["LM_KL_REF"]
        ?? "/Users/mdot/metal-microbench/test_data/reference"
    let tag = ProcessInfo.processInfo.environment["LM_KL_TAG"] ?? "hello"
    let outDir = ProcessInfo.processInfo.environment["LM_DUMP_LAYERS"]
        ?? ProcessInfo.processInfo.environment["LM_DUMP_L0_INTERNALS"]!
    runLmLayerDump(ggufPath: ggufPath, refDir: refDir, tag: tag, outDir: outDir)
}
if let ggufPath = ProcessInfo.processInfo.environment["GGUF_VALIDATE"] {
    runGgufValidateHarness(ggufPath: ggufPath)
}
// 2026-05-07: CVEC + prefix-cache test harnesses deleted. Their Swift
// test files (test_cvec_cache.swift, test_prefix_cache.swift) reached
// past the FFI to drive the engine's openSession/submit/closeSession
// internal lifecycle, freezing the bad two-phase API. Both were
// deleted under the user mandate "test files which freeze bad api
// contracts are bad tests."
// 2026-05-28: the below-the-bridge Python FFI tests that replaced them
// (server/test_batch_ffi*.py et al.) were ALSO deleted — driving the
// engine directly bypasses bridge.py's text processing (turn-marker
// stripping, chat templating) and yields misleading raw output. All
// engine coverage now runs THROUGH the bridge; cache reuse is verified
// end-to-end via cache_hits/cache_misses surfaced on the bridge's
// streaming response.
} // end runEnvDrivenDemos

// Run all the one-time top-level side effects (banners + active_exp fill)
// that used to be top-level statements. The executable (main.swift) calls
// this before runEnvDrivenDemos(); the library (ffi.swift/gemma_init) calls
// it right after weights load. Idempotent: safe to call multiple times.
private var _bootstrapped = false
func bootstrapGlobalState() {
    if _bootstrapped { return }
    _bootstrapped = true
    print("device: \(device.name)")
    print("")
    print("config: B=\(B), 30 layers, hidden=\(HIDDEN), experts=\(E_EXP) top_k=\(TOPK)")
    print("")
    let ap = active_exp.contents().bindMemory(to: UInt32.self, capacity: E_EXP)
    for e in 0..<E_EXP { ap[e] = UInt32(e) }
}
