# Technical Architecture & Optimization Deep-Dive

This document details the CUDA kernel engineering, inference runtime changes, mathematical proofs, and multi-turn KV-cache systems work implemented across **Waves 1–14** for **Inkling-2-Mini** on **NVIDIA Ampere SM 8.6 (RTX 3090, 24 GB)**.

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

All custom kernels live in [kernels/inkling-fused.cuh](../kernels/inkling-fused.cuh) (9,332 lines) and graph matchers in `ggml/src/ggml-cuda/ggml-cuda.cu`, with runtime A/B debug toggles (`W6_NO_*` … `W14_NO_*`) for isolated verification.

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

### 3.3 Wave 10B + 10D: Generation & Post-Image Checkpointing (SWA-1K & Hybrid/Recurrent)
Upstream `llama-server` creates `PARTIAL_ONLY` context checkpoints during **prefill** for SWA/hybrid/recurrent models (`ctx_tgt_seq_rm_type == FULL || RS || n_swa > 0`), never during **generation** and skipping batches containing `mtmd` image chunks (because M-RoPE pins `pos_max` inside image chunks). Consequently, a 1-token BPE drift inside a 6K–19K generated turn or right after a screenshot invalidated the entire tail.

- **Wave 10B** (`tools/server/server-context.cpp`, `W10_NO_GEN_CKPT=1`): saves a `PARTIAL_ONLY` checkpoint every `clamp(checkpoint_min_step, 2048, 4096) = 4,096` generated tokens for SWA/hybrid/recurrent models (`n_swa > 0` included). On a controlled 4,400-token generation with tail drift, KV cache rescue jumped from **`87 → 4,183` tokens (92.7 % saved)**, cutting re-prefill by **13.4×** (`4,422 → 329` tok, `3.63 s → 1.15 s`).
- **Wave 10D** (`W10_NO_MTMD_CKPT=1`): enables both periodic generation checkpoints in sessions with earlier images and post-image end-of-prefill checkpoints in `post_decode()` (`SLOT_STATE_DONE_PROMPT`) once trailing text tokens finish decoding and satisfy the strict boundary invariant (`tokens[n-1] != LLAMA_TOKEN_NULL && tokens[n-2] != LLAMA_TOKEN_NULL && pos_max + 1 == pos_next()`), ensuring `ctx_tgt`, `ctx_dft`, and DSpark's `context_pos` ring buffer are aligned.

### 3.4 Wave 10C: Calibrated Adaptive DSpark EMA Gate (`T = 5 ↔ T = 1`)
During extended creative reasoning stretches (`n_gen >= 256`) where DSpark acceptance drops persistently below ~28–31 %, speculative verification (`T=5`) can fall below memory-bound `T=1` (`78–85 t/s` vs `~105–110 t/s`). Wave 10C (`W10_NO_ADAPT_SPEC=1`, tunable via `W10_SPEC_PAUSE_THR=0.28`, `W10_SPEC_PAUSE_LEN=128`) tracks a smooth per-slot EMA (`α = 0.10`):
- Triggers only on sustained low-acceptance stretches (`n_gen >= 256`, `20` initial steps / `10` probe steps, `EMA < 0.28`), pausing into `T=1` (`need_probs=false`, `ctx_dft` synced via `common_speculative_process`) with progressive backoff (`128 → 256 → 384` tokens) to amortize CUDA graph switches.
- Wakes up **immediately** on structural transitions (`llama_vocab_is_control`, ```` ``` ````, `</thought>`, `<|call|>`, or GBNF forced prefixes), preserving **0 % regression** on code and tool calls (`133.6–180.2 t/s`) while lifting low-acceptance reasoning **`91.6–97.3 → 109.8 t/s` (+13–20 % cool, up to +30 % warm)**.

### 3.5 GPU Vision Offload (`VISION_OFFLOAD=1`) & DeepSeek Harness `fast-mcp.mjs`
- **GPU Vision at 128K**: `mmproj-F16.gguf` is only `610 MiB` (`22,823 MiB` idle / `23,331 MiB` peak during `1440×810` PNG encode out of `24,576 MiB`). Enabling `VISION_OFFLOAD=1` cuts image prefill **6.3–7.9×** (`1440×810` PNG `5,835 → 785 ms`; `960×540` JPEG `2,229 → 356 ms`, **16.4×** faster than `1440×810` PNG on CPU).
- **DeepSeek Harness 105K Freeze Postmortem & `inkling-128k` Preset**:
  - `tool-result-pruner: disabled: true` eliminates retroactive prefix truncation (which previously invalidated 80,988 KV tokens = `77.1 s` stall) + per-tool append-only `spill-policy` (`bash: 12 KB`, `grep: 12 KB`, `glob: 8 KB`, `read: 28 KB`) + `compaction-basic` (`thresholdRatio: 0.72`, `retainTokens: 16384`, `maxTokens: 24576` so `REASONING_BUDGET=12288` leaves 12K+ for summaries without fail-closed truncation).
  - Lean catalog (`26 → 16` tools including 3 `fast-mcp` tools, `-35 %` cold prompt vs default) + persona warning against `dsh`'s `@yao-pkg/pkg` `PKG_EXECPATH` bug that intercepts `bash` commands starting with bare `node ...`.
- **Persistent Zero-Dependency Local MCP Server ([configs/deepseek-harness/tools/fast-mcp.mjs](../configs/deepseek-harness/tools/fast-mcp.mjs))**:
  - Running `agent-browser` inside `dsh`'s ephemeral `bwrap --tmpfs /tmp --die-with-parent` sandbox killed Chromium after every call (`10–16 s` per screenshot + second `read_image` turn + `10.1 s` CPU vision encode).
  - `fast-mcp.mjs` runs outside `bwrap` over stdio and exposes **3 lean tools (~775 tokens total vs 33 tools / ~6,490 tokens in `agent-browser mcp`)**:
    1. `mcp__fast__browser`: warm persistent headless Chromium over raw CDP (`0 MiB` VRAM), compound actions (`click`/`drag`/`type`/`key`/`scroll`/`eval`/`wait_for`/`snapshot`), automatic console/exception/HTTP `≥400` capture, and **inline `960×540` JPEG screenshot** returned in the same tool result (`0.21–0.48 s` per check, zero extra `read_image` turn, native `clip` zoom).
    2. `mcp__fast__view_image`: fast Pillow Lanczos downscale (`max_dim: 960`) or pixel-crisp `clip` crop (`~82 ms`, 3–5× fewer vision tokens).
    3. `mcp__fast__multi_edit`: atomic multi-replacement file patcher with built-in syntax validation (`node --check` / `py_compile` / `JSON.parse`) in `~18 ms`, eliminating inline `python3 - <<EOF` bash patches.

---

## 4. Waves 11–13: Zero-D2H Speculation, Lossless Token/Media Caching & Production Agentic Stack

### 4.1 Wave 11: Zero-D2H GPU Leviathan Rejection & Tensor-Core Split-K Prefill
- **Wave 11 Spec 2A (`W11_NO_GPU_REJECT=1`)**: Keeps DSpark's probabilistic `q_probs` tensor (`[n_vocab, n_rows]`) **resident on GPU** inside a persistent device buffer (`~5 µs` D2D copy, **`0 B` PCIe D2H**), and executes exact Leviathan et al. rejection sampling on GPU (`inkling-fused.cuh`, `src/llama-context.cpp`, `common/sampling.cpp`), transferring only `16 B` back to host.
- **Wave 11 Spec 2B (`W11_NO_FUSED_DSPARK_SAMPLE=1`, `W11_DRAFT_MIN_P_SCALE`)**: Fuses DSpark's `ggml_soft_max` + optional lossless `draft_min_p` pruning + exact CDF categorical draw into a single CUDA kernel (`ggml-cuda.cu`), paired with a dedicated lightweight scheduler (`W11_NO_DFT_DUAL_SCHED=1`) for tiny `n_outputs == 0` DSpark inject graphs.
- **Wave 11 Prefill (`W11_NO_PREFILL_DYN=1`, `W11_NO_PREFILL_SPLITK=1`)**: Adds runtime-`T_s` zero-block-sync SM 8.6 `m16n8k16` Split-K Flash Attention (`Path 2B` for short tool-result prefills `T_s = 9..24`, `Path 3` Split-K for medium/large prefills) plus `W11_NO_CKPT_OPT` tail checkpoint batching (`n - 4` only, preventing every short tool turn from splitting into 3 separate decode batches + 3 synchronous 168 MiB checkpoints).

### 4.2 Wave 12: Measured-Rate Spec Gate & Host CPU Loop Hygiene
- **Measured-Rate Adaptive Spec Gate (`W12_NO_RATE_GATE=1`)**: Replaces the static acceptance threshold in Wave 10C with a live break-even comparison (`spec_ema_tok / spec_ema_ms` vs measured plain `T=1` rate), adapting automatically as context grows from `4K` to `128K` and backing off cleanly if a pause underperforms speculation.
- **CPU Hot-Path Memoization**: Memoizes GBNF grammar state transitions (`W12_NO_GRMR_MEMO=1`) and vocab token lookups (`W12_NO_TOK_MEMO=1`), skips redundant sampler candidate buffer reserves (`W12_NO_SAMPLER_RESERVE_SKIP=1`), and runs the decode loop with **`THREADS=1`** (`--threads 1 --threads-batch 1 --threads-draft 1`) so 7 idle host threads do not spin-wait per token while the vision encoder retains multithreading.

### 4.3 Wave 13: 100 % Lossless `preCPU` & BPE Rollback Elimination (`tools/server/`)
Driven by a 50.7-minute forensic production audit (`450` requests / `22` screenshots), Wave 13 eliminates every server-side CPU bottleneck on long multimodal agentic sessions with **0 % precision drift**:
- **Wave 13.1 — Fast LUT `base64_decode` + Zero-Copy `data:` URI**: Replaces $O(64)$ `std::string::find()` per character + triple payload copy with a 256-byte `constexpr` lookup table, pre-reserved buffer, and `std::string_view` slicing in `handle_media()`.
- **Wave 13.2 — Unified Incremental Prefix + Generated-Token + 48-Entry Image-Chunk LRU Cache (`W13_NO_TOK_CACHE=1`, `W13_NO_IMG_CACHE=1`)**:
  - Caches up to 8 conversation prefixes with boundaries placed **strictly after prefix-free control tokens** (verified once per `vocab`, excluding `LSTRIP`/`RSTRIP`/`SINGLE_WORD` and checking `!llama_vocab_get_add_eos(vocab)`), guaranteeing 100 % identical tokenization to full re-tokenization.
  - Clones matched prefixes in $O(1)$ with zero-byte media placeholders (`clone_prefix_placeholders`), pinning each chunk's `std::shared_ptr<mtmd_input_chunk>` on the task so that if a KV rollback ever occurs (`idx + n_tok > n_past`), `w13_hydrate_media_chunks()` materializes only the needed real chunks on demand in `SLOT_STATE_STARTED`.
  - Records exact slot-generated token IDs (`slot.w13_gen_tokens`) in `send_final_response()`, preserving non-canonical BPE merges across `write`/`edit` turns and eliminating BPE drift KV rollbacks (`102 / 102` `f_keep = 1.000` in live 107-step testing).
- **Wave 13.3 — `wave10d` Post-Image Checkpoint Spacing (`W13_NO_MEDIA_GAP=1`)**: Enforces a `media_min_gap` of `clamp(checkpoint_min_step / 2, 1024, 2048) = 2048` tokens before taking post-image checkpoints, preventing rapid screenshot bursts from spawning 30+ redundant 168 MiB checkpoints and exhausting host RAM/swap.
- **Wave 13.4 — `send_final_response` Guard**: Skips detokenizing the full 90K-token prompt when `res_type != TASK_RESPONSE_TYPE_NONE` (OpenAI / Anthropic / Responses endpoints never serialize `res->prompt`).

### 4.4 Production Stack: $O(1)$ Jinja, `inkling-proxy.mjs`, and `fast-mcp.mjs` v2
- **$O(1)$ Chat Template ([templates/chat-template-inkling-o1.jinja](../templates/chat-template-inkling-o1.jinja))**: Replaces the GGUF template's $O(M^2)$ `for prev in messages` scan for `tool_call_id` resolution with a single $O(M)$ `dict(tc_pairs)` lookup (verified 100 % byte-exact to the GGUF template).
- **Byte-Preserving Adaptive Proxy ([configs/deepseek-harness/tools/inkling-proxy.mjs](../configs/deepseek-harness/tools/inkling-proxy.mjs))**:
  - Splices/replaces only top-level JSON fields (depth 1) without touching a single byte inside `messages`, `tools`, or `system`.
  - Detects `dsh` compactions by header or prompt marker, **preserves the session's active `reasoning_effort`** even when `dsh` explicitly sends `"none"` (raising compaction KV cache hit from **`0 % → 94.9–97.6 %`**), and sets `reasoning_budget_tokens = 0`.
  - Keeps full `HIGH_BUDGET = 12288` after `bash` in `high`/`xhigh`/`max` (`bash_high_effort`), auto-escalates after any low-budget cap hit (`low_cap_escalate`), and triggers `loop_rescue` (`presence_penalty = 0.4`, `HIGH_BUDGET`) on repeated identical tool calls.
  - **Propagates client cancellation**: when the client disconnects before the response is complete (cancelled turn, timeout), the proxy drops the upstream request so `llama-server` stops generating and frees its single slot. Covers streaming, the wait before headers and non-streaming requests, and logs the turn with `cancelled: true`. Measured with the real server after abandoning a 1,500-token reply: the next request went from **`5.5–7.6 s → 78–379 ms`** (same as talking to the server directly). Regression test without GPU: `node tools/proxy-cancel-test.mjs`.
- **`fast-mcp.mjs` v2 ([configs/deepseek-harness/tools/fast-mcp.mjs](../configs/deepseek-harness/tools/fast-mcp.mjs))**: Adds `mcp__fast__read_batch` (multi-file/range batched reads), defaults to lossless PNG screenshots, disables HTTP cache (`Network.setCacheDisabled(true)`) + honors `reload: true` on same URL, installs a reversible `requestAnimationFrame` pause hook via `Page.addScriptToEvaluateOnNewDocument` + 10-min idle auto-close (dropping idle headless Chrome from `468–1,071 % CPU` to **`0.1–0.2 % CPU`**), reports SwiftShader software WebGL honestly on `<canvas>`, supports `duration_ms` alias, and sets `390×844` on `mobile: true`.

---

## 5. Wave 14: Correctness, Determinism & Speculation Economics

Wave 14 started as the validation of an external review (~45 claims: suspected bugs and proposed optimizations) against the code and, wherever a claim depended on hardware behaviour, against the GPU itself. It does not change the arithmetic of the 1-token path (logits bit-identical to Wave 13 over 1,024 tokens). It fixes one real non-determinism bug in batched decode and about twenty latent bugs, and it re-measures what speculation costs. Tables: [BENCHMARKS.md §4D](BENCHMARKS.md#4d-wave-14--determinism-draft-length--per-step-costs-265-w).

### 5.1 The Batched-Attention Shared-Memory Race
Batched decode (`T = 2..24`) was not reproducible from run to run: `inkcheck --decode-batch 5` deviated in 17 of 93 runs. The cause was not the L1 cache (the review's hypothesis) but a shared-memory overlap in the two `mma.sync` attention kernels that run without a block-level barrier, `inkling_rel_pos_fattn_batched_mma_kpos` (`T = 2..8`) and `inkling_rel_pos_fattn_batched_mma_dyn_kpos` (`T = 9..24`). Their per-warp scratch is a union of `s_K[4][16][136]` (4,352 B per warp) and `s_VT[4][128][18]` (4,608 B per warp), so the tail of `s_VT[w]` overlapped the head of `s_K[w+1]`, and the KV loop only synchronizes with `__syncwarp()`. The fix is one padding row, `s_K[4][17][136]`; the 136-element stride that avoids bank conflicts is unchanged.

After the fix `inkcheck --decode-batch 5` gives 0 deviating runs in 112 and `--decode-batch 12` 0 in 36, and the server at `temp 0` with speculation always on gives one output per prompt (it gave 7 and 5 distinct outputs in 8 runs). `bit-exact` between two runs is therefore a valid acceptance test for batches of 2–24 tokens too, not only for `T = 1`.

The server with its default configuration is still not reproducible at `temp 0`, for a different reason: the Wave 12 rate gate decides when to speculate from measured wall-clock rates, and the `T = 1` and batched paths differ within the int8-activation noise floor. `W10_NO_ADAPT_SPEC=1` (speculation always on) gives reproducible outputs.

### 5.2 Latent Bugs Fixed
- **DSpark low-rank head in true FP32**: `M = W_out @ H` was computed through a cuBLAS handle left in TF32 mode (relative error `2.97e-4`); it now uses `CUBLAS_COMPUTE_32F_PEDANTIC` (`6.0e-7`, [tools/microbench/tf32.cu](../tools/microbench/tf32.cu)). The cache also handed out a pointer into a `std::vector` that could be reallocated. Only the drafter is affected: verification never trusted it.
- **Per-context kernel state**: the Split-K counters and the persistent row-index slot were static per `(device, stream)` and shared by two contexts of the same device; they now belong to the CUDA backend context and are freed in its destructor.
- **`W6_NO_KVFUSE` A/B path**: with K/V fusion disabled, the `GET_ROWS[19968x1]` node folded away by Wave 7C was needed again but never materialized (KLD `0.29`, top-1 `87.5 %`). It is now materialized from the persistent index right before the unfused consumer, and the path is bit-identical to `W7_NO_SKIP_GET_ROWS=1`.
- **Graph matchers**: the full MoE matcher rejects a `down` destination that aliases `ids`/`weights` at `T = 1`; `flash_attn_ext_supported` checks tensor pointers and types.
- **Which scheduler owns the outputs** (`llama-context.cpp`): the graph callback pins tensors on the scheduler that runs the graph (it always used the main one, also for inject graphs); deferred logits and probabilities stay on the device only for the last ubatch of a batch, and `draft_step()`, `encode()` and `graph_reserve()` bring them to the host before the graph is invalidated.
- **Sampler chain signature**: it includes the node count of the sampling graph; `need_logits` no longer depends on `need_probs`; a failed `set_sampler` clears the flags.
- **Graph reuse**: `allow_reuse()` compares the output pattern of ubatches with partial outputs; the `DRAFT_HEAD` graph is rebuilt when its softmax changes.
- **GBNF**: the whitespace mode of the forced-prefix drafter is read before the loop and is explicit (`W14_GBNF_WS_MODE`); the grammar memo is created with the grammar (clones always share it) and evicts its least-used entry instead of clearing itself at 64 entries.
- **CPU sampling fast path**: the greedy shortcut requires a chain that really is an argmax (no logit bias, no model-suppressed tokens, no `dynatemp`, `infill` or `adaptive_p`); `common_sampler_get_candidates` no longer reads `data[-1]`; the host rejection path stops at EOG and bounds its copy to the drafter vocabulary.
- **DSpark drafter**: the draft seed is reset per request; missing anchor features skip the draft of that step instead of throwing (HTTP 500).
- **Server**: generation checkpoints (Wave 10B) are also taken without speculation; `__verbose.prompt` works on the OpenAI endpoints; `destroy()` purges the Wave 13 caches; the tokenization cache disables itself when the tokenizer does not split cleanly at control tokens (self-test once per vocabulary).
- **Library loading**: `$ORIGIN` reached the linker empty, which left the current directory first in the search path and the build tree ahead of `LD_LIBRARY_PATH`, so `bin-sm86/` was really running the libraries of the build tree. The RPATH now goes through CMake (`CMAKE_BUILD_WITH_INSTALL_RPATH=ON`, `CMAKE_INSTALL_RPATH=$ORIGIN`): each binary folder loads its own libraries, and deploying means copying.

### 5.3 Review Claims That Did Not Hold
Measured rather than argued, with probes in [tools/microbench/](../tools/microbench/):
- *Stale L1 reads in the Split-K kernels.* The L1 of the RTX 3090 is indeed incoherent across SMs (93 % stale reads in a probe without a fence), but `__threadfence()` flushes it at SM level (0 %, even when another warp executes it) and the last block always executes it before reading: 0 stale reads in 400,000 launches seeded with stale lines (`l1_probe.cu`, `l1_stale.cu`). `atomicAdd` alone does not flush it.
- *`return` before `__syncthreads()` hangs the GPU.* 14,000 launches with whole warps leaving before the barrier: no hang (`sync_exit.cu`).
- *Out-of-bounds `params_ptr[8..9]` in the fused DSpark sample*, *stacked-drafter crash at `block_size = 5`*, *fusions dropped at `T = 5`*: the matcher validates 16 floats, the GGUF `block_size` is 8 with GBNF capped at 7, and the `T = 5` profile shows MoE, horizontal GEMV, `wo` and K/V fused in 48 of 48 layers.

### 5.4 A Scheduler for the Plain Step (`sched_t1`)
A `ggml_backend_sched` holds one allocated graph. The target context alternates between the plain 1-token step (rate gate paused) and verification batches, and every switch rebuilt the graph and re-captured its CUDA graph: `7.0 + 3.6 ms` entering a pause and `6.9 + 2.1 ms` leaving it, ~20 ms per pause, ~3 pauses per 400-token request and 19 in a 2,400-token generation.

Wave 14 gives the plain step of a speculation target its own scheduler and graph arena (`W14_NO_T1_SCHED=1` disables it), next to the Wave 11 inject scheduler, and every reader of output tensors asks the scheduler that ran the last ubatch (`sched_last()`). One rebuild per request remains (~12 ms), because the graph is bound to the sampler object of that request. It costs 14 MiB of VRAM and outputs are identical with and without it. Gain: `-1.1 %` per plain step, `-0.7 %` per cycle, `-1.7 %` on chained short turns.

### 5.5 Draft Length: 3 Tokens by Default
With device-resident rejection sampling a verification row is cheap but not free. Measured with `W12_SPEC_STATS=1`: a cycle with 4 drafts costs `24.75 ms` and yields `2.80` tokens, a cycle with 3 drafts `22.10 ms` and `2.54` tokens, and a plain step `9.8 ms`. The 4th position buys `0.26` tokens for `2.65 ms`, which is `98 t/s` at the margin against `102 t/s` for a plain step: at `temp 1.0` it is a net loss when the drafter has to predict new code, reasoning or tool calls, and it only pays when the drafter is copying text.

Since 2026-11-23 the default is therefore 3 drafts per cycle (`T = 4` verification): `DSPARK_N=3`, `NGRAM_MIN=3`, `NGRAM_MAX=3`. Against 4 drafts that measures `+3.3 %` on new code, `+3.0 %` on reasoning, `+4.7 %` on tool calls, `-4.1 %` of wall time on chained short turns and `-4.0 %` on verbatim rewriting. Setting the three back to 4 restores the previous shape for workloads that mostly copy text. The three drafters must share one length: mixed lengths alternate batch shapes, and every change of shape rebuilds the verification graph (~8 ms), which measured far worse than either fixed length.

### 5.6 Measured and Not Shipped
- **Adaptive draft length** (3 or 4 from a moving average of per-position acceptance, two versions): `+2 %` on code and `-5` to `-7 %` on rewriting. The controller spends part of the gain on the ~60 tokens it needs to decide and on two graph rebuilds per request.
- **Lazy grammar through the GPU rejection path** (saves a 1.6 MB device-to-host copy per rejection): the cycle costs `28.07 ms` with it and `27.89 ms` without.
- **Strict GBNF whitespace prefix**: `96.5 ± 0.8 t/s` against `100.9 ± 2.0 t/s` for the compact mode on `json_schema` output; the compact mode stays the default, now explicitly.
- **Draft injection during prefill**: `11 + 1.8 ms` per 512-token batch against `280–295 ms` of the target model (4 %): no margin worth the change.

### 5.7 Follow-up: Layers Outside the GPU
A second external review found that the compact relative-position attention of Wave 9A only existed in CUDA while every layer used it. With all 49 layers on the GPU (the production setup) nothing changes: logits are bit-identical to the previous binaries at batch sizes 1, 5 and 12.
- **Wrong attention on CPU layers**: the CPU kernel of `FLASH_ATTN_EXT` ignored the relative-position inputs (`src[6..9]`) and still reported the op as supported, so a layer kept on the CPU lost its relative bias in the 1-token decode (batches were moved to the GPU by the scheduler and came out right). Each layer now picks its path by the device it runs on: the compact kernels on CUDA, the upstream per-head-mask path elsewhere. Against all layers on the GPU, 1-token decode with 44 of 49 layers offloaded: top-1 `78.1 % → 98.4 %`, KLD `0.51 → 5.9e-3`; with no layer offloaded: top-1 `45.8 % → 97.9 %`, KLD `2.6 → 7.7e-3`. Without a visible GPU the logits are bit-identical to the unmodified base commit.
- **Backends say what they implement**: the CPU backend rejects the op when it carries relative-position inputs, and the CUDA selector requires a mask shared by all heads (the kernels read `mask[tok*n_kv + j]`; a per-head mask was accepted and the first head's mask applied to all 24). Nothing builds such a mask today.
- **Null pointer in the dense path**: the input tensors were created by one condition (Flash Attention on) and the per-layer path chosen by a stricter one (24 Q heads, 8 KV heads, head size 128, 16 relative components), so a layer with another geometry dereferenced a tensor that was never created. Both tensor sets are now declared; the one no layer uses gets no buffer and is skipped.
- **Known and inherited, not fixed**: with layers on the CPU *and* a visible GPU, the scheduler offloads large batches of those layers to CUDA, and prefill logits drift from the all-GPU ones (KLD `3.4e-2`, top-1 `96.1 %` with no layer offloaded; `2.2e-3` with 44 of 49). The unmodified base commit drifts more (`0.19`, `86.3 %`), and neither does without a visible GPU, so the cause is in the op offload of the base, not in the paths above.
