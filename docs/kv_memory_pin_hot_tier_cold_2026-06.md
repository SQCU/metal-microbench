# KV memory: pin the hot set, tier the cold set — and why the model size is a red herring

**Status:** design for review (2026-06). Supersedes the reactive "free-RAM ceiling" experiment (stashed) and the "pin a fixed 64–72GB" sketch (strictly worse than SOTA).

## 0. The questions this answers (operator, 2026-06-04)
> if UD saves k GB but we implement double paging (KV → page table → NVMe) do we need to change the config? what's the largest working set for batch:8 on the largest plausible task? how much do we need that 9GB? what's the compromise?

Short answers, derived below: **(1)** No — double-paging makes the model size nearly irrelevant to *capacity*. **(2)** The max batch:8 hot working set = the whole pool ≈ **54 GB**. **(3)** We don't need the 9GB: Q8 model + 54GB hot pool + OS = ~91GB on a 128GB box (37GB headroom). **(4)** The compromise is quality-for-unneeded-headroom — so **keep Q8 for the deploy**; the kernel-parity work stands on its own and keeps UD available via `GEMMA_GGUF`.

## 1. The geometry (measured, not assumed)
- `PAGE=16`, `NUM_LAYERS=30` (25 slide: `KV_H=8,HD=256`; 5 full: `KV_H=2,HD=512`), fp16, K+V.
- **`perPageBytes = 3,604,480 B = 3.44 MiB`** (Σ all 30 layers for a 16-token page). Per token = **220 KiB**.
- `MAX_PAGES_PER_SLOT = 8192` → a single slot can reach **128K tokens = 8192 pages = 28 GB**.
- `B = 8`. Pool geometry cap `SCRATCH_PAGE_BASE = 16000 pages ≈ 54 GB` (runtime `poolCap` = `KV_MEM_BUDGET_FRAC·physMem − model`, ≤ this).
- **Decisive fact (page_manager.swift:58-59):** one physical page covers `[16P..16P+15]` in *every* layer's K/V buffer; full and slide layers both index `block_table[slot][pos/16]`. So the sliding window (`SLIDING_WINDOW=1024`) only *masks* old K positions in the 25 slide layers — it **does not free pages**, because the 5 full-layer slices in that same page stay live. Per-slot live pages therefore grow **linearly with context** and are not reclaimed mid-session.

## 2. The hot working set for batch:8 (the number that matters)
"Hot" = pages read on every forward pass = the union of all *active* slots' live pages. These **cannot be tiered to SSD** — they're touched every token — so they set the RAM floor.

| Scenario | Live pages | Hot KV |
|---|---|---|
| 1 slot @ 8K (typical chat) | 512 | 1.8 GB |
| batch:8 @ 8K each | 4,096 | 14 GB |
| batch:8 @ 32K each (≈ pool cap) | 16,000 | **54 GB** |
| 1 slot @ 128K (max single) | 8,192 | 28 GB |
| batch:2 @ 128K / batch:4 @ 64K | ~16,000 | **54 GB** |

**The max hot working set = the pool ≈ 54 GB**, regardless of how it's split across slots (the pool is the shared cap). Our actual workload (roleplay + multi-turn elicitation, 8 concurrent sessions, mostly ≤32K) sits far below this — typically 2–15 GB.

## 3. Do we need the 9GB (UD vs Q8)? — No.
RAM budget on the 128GB box, **with the hot pool fully pinned-resident**:

| | model | hot pool (pinned) | OS+activations | total | headroom |
|---|---|---|---|---|---|
| **Q8** | 25 | 54 | ~12 | **91 GB** | 37 GB |
| **UD-Q4_K_M** | 16 | 54 | ~12 | 82 GB | 46 GB |

The max batch:8 hot set fits with Q8 and 37GB to spare. The 9GB buys only ~16% more pinned pool (batch:8 @ ~37K instead of 32K) or more idle headroom — **not a capability we lack**. Meanwhile UD costs some expert precision (Q4_K/Q5_1 vs Q8). **Verdict: keep Q8 for quality; the 9GB is not the binding constraint. The original thrash was never a model-size problem — it was a page-lifecycle problem** (see §4). UD stays one `GEMMA_GGUF=` away for genuinely RAM-tight hosts (a box also running other heavy services, or batch:16 / 128K experiments).

## 4. The actual bug, and the fix
The 10× slowdown was: the KV pool is **lazy-committed** (`zeroPhysPageKV` first-touch) and **not wired** — `installWeightResidencySet` (weights.swift:197) pins the 719 weight buffers but **explicitly excludes K_chunks/V_chunks** (weights.swift:165); they only get `setPurgeableState(.nonVolatile)` (ffi.swift:125), which prevents *discard* but not *page-out*. So under memory pressure the OS compresses/pages the **hot** KV → faults on the next forward read → 4s/tok. The pool was then shrunk (33000→16000 pages) to dodge it. That's treating the symptom.

**Fix = SOTA tiering, sized by §2:**

### Tier 0 — pin the hot pool (the core fix; MUST)
- Add `K_chunks`/`V_chunks` to the persistent `MTLResidencySet` + `requestResidency` (extend `installWeightResidencySet`). The hot pool becomes **wired-resident → the OS cannot page it → faults structurally impossible** on active KV.
- This is **not** "pin everything arbitrarily" — it pins exactly the pages that must be resident (the §2 hot set, ≤54GB). It's what vLLM does (pre-reserve the KV region).
- **Pinning *removes the fault risk that forced the pool down to 16000.*** With Q8 we can confidently size the pinned pool up to `RAM − model − OS − margin` (~75GB / 20800 pages) if we want more concurrent capacity. Sizing is now a deliberate knob, not a fault-avoidance guess.
- Delete the reactive ceiling (stash@{0}); keep the min-heap reuse-by-recency allocator (also in the stash) for *which* page to reuse within the fixed pinned pool.

### Tier 1 — demote cold cache to SSD (capacity extension; OPTIONAL/next)
- "Cold" = refcount-0 cached prefixes from **finished** sessions that may be re-adopted (content-addressed via the radix trie). These are NOT in any active working set.
- Under cache pressure, demote the LRU-coldest cold pages to an NVMe-backed file (`block_table`/page gets a tier bit: `RAM | SSD@offset`) **instead of dropping them**. On re-adoption, reload into a free pinned page.
- **Bandwidth justification:** reload = 220 KiB/token ÷ ~6 GB/s = **37.5 µs/tok ≈ 26,700 tok/s**, vs re-prefill ~1–1.7 ms/tok (600–1000 tok/s) → **27–45× faster** (robust to ≥10× even at a pessimistic 3 GB/s SSD). Confirms the SOTA literature: the win is not re-running prefill, not NVMe speed. This is the *only* place "double paging" applies — and it's cold-only, off the hot path.

### Over-capacity (genuine)
- More concurrent *active* demand than the pinned pool holds → **exponential admission backoff on the newest request** (retry-after). Never evict/corrupt a refcount>0 page; never page out the hot set.

## 5. What changes vs. today
- weights.swift: KV chunks → residency set (+`requestResidency`); pinned from boot (or pin-on-grow).
- page_manager.swift: fixed pinned pool + min-heap recency reuse (un-stash); drop reactive ceiling/committedHighWater ratchet; add per-page tier bit (Tier 1).
- lm_engine.swift: pool sized as a deliberate pinned budget; exponential admission backoff (un-stash); SSD demote/reload hooks (Tier 1).
- bootstrap.swift: pool-size knob (pinned budget), SSD path/size config (Tier 1).
- **Deploy model: stays Q8.** Kernel-parity (v12 deleted, dense Q4_K btile added) committed independently; UD available via `GEMMA_GGUF`.

## 6. Open questions for review
1. **Tier 0 only, or Tier 0 + Tier 1 now?** Tier 0 alone fixes the thrash. Tier 1 adds cold-cache capacity (re-adoption beyond RAM) — worth it only if cross-session prefix re-adoption is common at scale.
2. **Pinned pool size:** keep 16000 (54GB, safe) or grow now that pinning removes the fault risk (e.g., 20000 = 72GB)?
3. **Pin-all-up-front vs pin-on-grow:** commit+wire the whole pool at boot (simple, 54GB resident even idle — fine on a dedicated box) vs wire pages as slots first touch them (dynamic, more bookkeeping)?
