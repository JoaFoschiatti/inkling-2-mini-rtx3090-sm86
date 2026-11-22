# Technical Architecture & Optimization Deep-Dive

This document details the CUDA kernel engineering, inference runtime changes, mathematical proofs, and multi-turn KV-cache systems work implemented across **Waves 1–10** for **Inkling-2-Mini** on **NVIDIA Ampere SM 8.6 (RTX 3090, 24 GB)**.

---

## 1. Target Model Bottlenecks (`Inkling-2-Mini`)

Inkling-2-Mini is a **looped-transformer MoE** with hybrid attention:
- **Depth**: 24 unique transformer blocks mapped to **48 effective layers** (looped core `[6, 18) × 3` passes), hidden size `D = 2304`, 24 query heads / 8 KV heads (`GQA = 3`, `head_dim = 128`), vocabulary `200,058` (padded `201,024`).
- **Hybrid Attention**: 40 Sliding-Window Attention (SWA, window `1,024`) layers + 8 Full-Context Global Attention layers with learned relative-position biases (`RelPos`) and zero-sinks.
- **Sparse MoE**: 128 experts, Top-8 routing, `FFN = 1152`; 20 blocks quantized in **NVFP4** (4.5-bit with E4M3 micro-scales) and 4 blocks + attention/LM head in **Q8_0**.
- **Roofline**: Each decode token reads ~3.77 GB of weights (~1.76 GiB active experts, ~1.31 GiB attention projections, 469 MiB LM head, 54 MiB routers). On an RTX 3090 delivering ~875 GB/s effective DRAM bandwidth, the physical memory-bandwidth floor is **~4.3 ms/tok (~230 t/s)**.

Unmodified `llama.cpp` (`pr390`) ran at only **55–59 t/s** because:
1. Software `<cuda_fp8.hpp>` `UE4M3 → FP32` conversion stalled NVFP4 dequantization on SM 8.6.
2. Looped-transformer primitives (`control()`, `mix()`, `shortconv4`, `SiTU`, `l2_clip`, `RelPos` masks) decomposed into **~11,508 ggml graph nodes / ~5,000+ micro-kernel launches** per token.
3. Speculative verification (`T = 2..8`) fell back to unfused paths and hit register spills across 128 experts.
4. `RelPos` materialized ~15 GiB of dense `F32`/`F16` bias & mask tensors at `-c 131072 -ub 512`, causing OOM past 32K.
5. `llama-server` dropped to CPU sampling whenever `tools` (lazy GBNF), `json_schema`, or `reasoning_budget` were attached — affecting 100 % of agentic harness traffic.

---

## 2. CUDA Kernel Engineering (`ggml/src/ggml-cuda/inkling-fused.cuh`)

All custom kernels live in [kernels/inkling-fused.cuh](../kernels/inkling-fused.cuh) (8,012 lines) and graph matchers in `ggml/src/ggml-cuda/ggml-cuda.cu`, with runtime A/B debug toggles (`W6_NO_*`, `W7_NO_*`, `W8_NO_*`, `W9_NO_*`, `W10_NO_*`) for isolated verification.

### Waves 1–6: Single-Token Decode Compression (`~5,000+ → 816 ops/tok`)
- **Branchless NVFP4 dequant (`common.cuh`, `vecdotq.cuh`, `mmvq.cu`)**: 5-instruction bit-manipulation `UE4M3 → FP32` for SM < 89 + vectorized `__ldg` `uint16_t` paired-scale loads + 8-warp `mul_mat_vec_q_nvfp4_nosync_1` without `__syncthreads()` or shared memory (`688–743 GB/s`).
- **Core Looped-Transformer Fusions**:
  - `inkling_fused_control_down` / `up` & triple `control()` Split-K=2 (`36 ops → 2 launches`).
  - `inkling_fused_shortconv4` (8-op, 9-op + in-place residual, 40-op dynamic, and K+V combined `SET_ROWS` + state store).
  - `inkling_fused_mix` using the algebraic identity $\text{dot}(\text{RMS}(x), q) = \text{inv\_rms} \cdot \text{dot}(x, q)$ in one block with zero transpose.
  - `inkling_fused_horiz_mmv` (horizontal Q+Gate+R+K+V+Sink multi-GEMV), `sigmoid·mul + wo`, `cp.async` NVFP4 `gate_up + SiTU` and `down + weighted_reduce`, fused Top-8 router (`topk-moe`), and Q/R clip pairs.
- **GPU Sampling Fix (`src/llama-sampler.cpp`)**: Scaled GPU `dist` uniform draw by final cumulative sum to match CPU `sum_cum` rounding — fixing tail-token (`<|content_invoke_tool_text|>`) mis-picks and achieving **4,000 / 4,000** identical draws vs CPU across seeds.

### Wave 7: Producer Quantization & Custom 1-Token Flash Attention (`816 → 524 ops/tok`)
- **Wave 7A**: Inline `Q8_1` activation quantization inside producer kernels (`mix_norm`, `rms_norm_mul_3`, `rms_norm_mul_1_q8`, `48/48` layers) + 3-kernel MoE pipeline packing router + `Q8_1` quantization onto idle SMs (73 blocks).
- **Wave 7B**: Custom 1-token RelPos Flash Attention (`3 warps/block`, `GQA = 3`, double-buffered `cp.async.cg`, inline zero-sink + `sigmoid` + `Q8_1` output quantization, empty-split-skipping `float4` reduction) directly feeding 32-bit aligned `wo` Q8_0 GEMV (`-192` launches).
- **Wave 7C**: 100 % bit-exact folding of `GET_ROWS[19968x1]` into the 4 `shortconv4` kernels via a persistent 4-byte GPU index slot (`-48` launches).
- **Front A1/A2**: `llama_set_sampler_need_probs()` skips the 800 KB/tok `t_sampled_probs` D2H copy and 200K CPU scan when unused; CPU `dist` skips `expf(-INFINITY)` for grammar-masked vocabularies.

### Wave 8: Unlocking Speculative Decoding (`T = 2..8` & Low-Rank DSpark)
- **Stages 8.1–8.2**: Expert-grouped `T = 2..8` MoE kernels (`inkling_moe_group_experts` via `__ballot_sync` + `template <int CNT>` NVFP4 `cp.async.cg` and Q8_0 kernels with **0 register spills**), batched `horiz_mmv`, deterministic `48/48` `sigmoid_mul + wo`, batched `control`/`control_3`, and `-147` redundant `CONT` copies removed in `inkling2.cpp`. Cut `T = 5` step latency from `26.20 ms → 16.47 ms` (**-37.1 %**).
- **Stage 8.3**: Unified `shortconv4` / `kv_shortconv4` fusion supporting 7-node `SET_ROWS` rollback (`n_rs_seq > 0`, `192/192` convs matched under DSpark).
- **Stage 8.4 (DSpark 4× / 260× breakthrough)**:
  - **Precomputed Low-Rank Head**: `M = W_out @ H` (`[256, 201024]` FP16 = 103 MiB via one-shot FP32-accum `cublasGemmEx`) with coalesced `uint4` kernel: `595 µs → 125 µs/step`.
  - **Multi-Block Deterministic First-Maximum `argmax.cu`**: 64-block reduction (`40 µs → 5 µs`).
  - **O(1) Rejection Plumbing**: Replaced $O(V)$ linear `prob_of()` (`4,100 ms → 0.15 ms`), added zero-copy GPU `sampled_probs` verification and recycled `q_pool` buffers (zero `mmap`/`munmap` page faults).

### Wave 9: 128K Context & Tensor Core Split-K Attention
- **Wave 9A (On-the-fly RelPos)**: Extended `ggml_flash_attn_ext_add_rel_pos` (`src[6..9]`) to compute relative position bias dynamically inside Flash Attention, eliminating **~12 GiB** of F32 `kq_b` + **~3 GiB** of F16 masks + `256 MiB` `self_rel_pos`.
- **Wave 9B (Selective KV + SM 8.6 Tensor Cores)**:
  - Selective `Q8_0` KV cache **only on the 8 global layers** (`kv_base`), leaving all 40 SWA-1K layers in lossless `F16` (`llama-kv-cache-iswa.cpp`).
  - Disabled redundant Hadamard KV rotation for Inkling-2 `Q8_0` (restoring `48/48` KV & `FA→wo` fusions and raising `inkcheck` to `100.00 %` top-1 at 9K and 32K).
  - `T = 1` kernel (`inkling_rel_pos_fattn_t1_kpos`): `dp4a + PRMT` zero-`I2F`, 0 SMEM, 0 `__syncthreads()`, 64 regs, 0 spills (`70.2 t/s` at 128K).
  - `T = 2..8` kernel (`inkling_rel_pos_fattn_batched_mma_kpos`): **Ampere SM 8.6 Tensor Core (`mma.sync.m16n8k16`) Split-K** without `__syncthreads()` in the KV loop (`+38 %` on `pp5 @ 128K = 180.0 t/s`, `+40 %` on `pp8 @ 128K = 197.1 t/s`).
- **Wave 9C (Hybrid GPU Grammar Sampling)**: Keeps GPU sampling active even with `tools`/`json_schema`/`reasoning_budget`; CPU validates the GPU-picked token in $O(1)$ and triggers lazy `fetch_deferred_logits()` only on the ~1–2 % grammar rejections; adds 256-bit first-byte GBNF pre-filter and Priority-0 strict GBNF forced-prefix drafting.
- **Wave 9D (Single-Shape `T = 5` Stacked Speculation)**: Chains `GBNF + ngram-mod(4..4, match=16) + DSpark(n=4, Q8_0 KV)` at a uniform `T = 5` verification shape (zero CUDA graph recaptures across drafters), adaptive `n_pad`, `--cache-ram 16384`, and CPU vision (`--no-mmproj-offload`, `0 MiB` VRAM).

---

## 3. Wave 10: Exact GPU Probabilistic Speculation & Multi-Turn Stability

### 3.1 Single-Graph GPU Probabilistic DSpark (`LLM_GRAPH_TYPE_DRAFT_PROBABILISTIC`)
Previously, probabilistic DSpark executed 4 separate `DRAFT_HEAD` graph launches with CPU/PCIe round-trips per step. Wave 10 fuses all `n_draft = 4` probabilistic draws into **one reused CUDA graph** (`src/models/dspark.cpp`, `common/speculative.cpp`):
- Passes `[inv_temp, u_1, ..., u_n]` in unused row 1 of `f_rnn`.
- Executes `ggml_soft_max` → `ggml_cumsum` → `ggml_step` → `ggml_sum` categorical sampling on GPU inside the single graph.
- Copies the resulting `q` probability distributions into owned flat `std::vector<std::vector<float>>` buffers with retained capacities (`~0.10 ms` memcpy, avoiding `ctx_dft->logits` aliasing across `batch_inject`).
- Isolated A/B gain vs 4× `DRAFT_HEAD`: **`+11.8 %` without tools (`107.67 → 120.35 t/s`)** and **`+20.4 %` with tools (`99.03 → 119.25 t/s`)**.

### 3.2 Exact Leviathan et al. Rejection Sampling Under Active GBNF Grammars
In `common/sampling.cpp`, let $p(x)$ be the target GPU post-filter/softmax distribution over vocabulary $V$, $q(x)$ be the DSpark draft distribution, $G \subseteq V$ be the grammar-valid token set, and $Z = \sum_{k \in G} p(k) \in (0, 1]$. The exact grammar-constrained target distribution is:

$$p'(x) = \begin{cases} p(x) / Z & x \in G \\ 0 & x \notin G \end{cases}$$

Because $0 < Z \le 1$, for every grammar-valid token $x \in G$:

$$p'(x) = \frac{p(x)}{Z} \ge p(x) \implies \frac{p(x)}{q(x)} \le \frac{p'(x)}{q(x)}$$

**Exact $O(1)$ Fast-Path Theorem**:
1. Draw $u \sim \text{Uniform}(0, 1)$.
2. If $x \in G$ (checked in $O(1)$ via the grammar automaton) **and** $u < p(x) / q(x)$, then $u < p'(x) / q(x)$ is **mathematically guaranteed** — accept $x$ immediately with **zero vocabulary scan**!
3. Only when $x \notin G$ or $u \ge p(x) / q(x)$ do we scan non-zero entries $p(k) > 0$ through `grmr` to compute exact $Z$, re-test $u < p'(x) / q(x)$ using the **same** $u$, or sample the exact residual distribution $\text{norm}(\max(0, p' - q))$.
4. For unconstrained tokens (lazy `tools` during reasoning/text), pure zero-copy GPU $p$ vs flat $q$ runs with zero grammar overhead. On `REASONING_BUDGET_FORCING`, `llama_get_raw_logits_ith` + `force_raw` fall back cleanly to CPU without double-applying temperature.
5. Per-step dynamic `llama_set_sampler_need_probs` / `llama_set_sampler_need_logits` gates D2H transfers to **0 bytes** on `ngram-mod`, `GBNF`, and non-`logprobs` steps.

### 3.3 Wave 10B: Periodic Generation Checkpointing for Hybrid/Recurrent Models
In `llama_memory_hybrid_iswa`, recurrent/SWA states have `n_swa_reuse == 0` (`pos_min == pos_max`). Upstream `llama-server` only created `PARTIAL_ONLY` context checkpoints during **prefill**, never during **generation**. Consequently, a single 1-token BPE retokenization drift inside a 6K–19K generated turn invalidated the entire generated response (6K–19K re-prefill = 4–22 s stall).

Wave 10B (`tools/server/server-context.cpp`) saves a `PARTIAL_ONLY` checkpoint (~168 MiB RAM, ~67 ms) every `clamp(checkpoint_min_step, 2048, 4096) = 4,096` generated tokens when `n_swa_reuse == 0 && get_text_tokens_direct() != nullptr`. On a controlled 4,400-token generation with tail drift, KV cache rescue jumped from **`87 → 4,183` tokens (92.7 % saved)**, cutting re-prefill by **13.4×** (`4,422 → 329` tok, `3.63 s → 1.15 s`).

### 3.4 DeepSeek Harness 105K Freeze Postmortem & `inkling-128k` Preset
Log analysis of a real 105K coding session revealed a **254 s (4.2 min) stall across 3 turns** caused by two harness policies designed for cloud APIs rather than local KV caches:
1. At 80 % context (`104.8K`), `tool-result-pruner` retroactively truncated one old tool output at token `23,203` — saving `2,153` tokens while invalidating **80,988 cached KV tokens** (`77.1 s` re-prefill!).
2. Two turns later, `compaction-basic` ran anyway with `maxTokens: 8192` (`44.8 s` prefill + `82.3 s` generating a 7,197-token summary + `50.3 s` post-compaction prefill).

Our local preset ([configs/deepseek-harness/agent-presets/inkling-128k/](../configs/deepseek-harness/agent-presets/inkling-128k/agent.cordis.yml)) fixes both cleanly:
- `tool-result-pruner: disabled: true` (zero retroactive prefix mutations).
- `spill-policy.maxInlineBytes: 24000` (bounds oversized tool outputs **at creation time** — append-only, `0 ms` KV invalidation).
- `compaction-basic`: `thresholdRatio: 0.85` (~111K), `retainTokens: 16384`, `maxTokens: 3072` (~3× faster compaction).
