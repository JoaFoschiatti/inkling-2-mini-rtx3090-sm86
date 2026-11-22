# Inkling-2-Mini — Custom SM 8.6 CUDA Engine & 128K Agentic Stack (RTX 3090)

[English](README.md) | [Español](README.es.md)

[![GPU](https://img.shields.io/badge/GPU-RTX%203090%20(SM%208.6)-76b900?logo=nvidia&logoColor=white)](docs/BENCHMARKS.md)
[![Context](https://img.shields.io/badge/Context-128K%20(131%2C072)-blue)](docs/ARCHITECTURE.md)
[![Decode](https://img.shields.io/badge/Agentic%20Decode-131--165%20t%2Fs-brightgreen)](docs/BENCHMARKS.md)
[![Precision](https://img.shields.io/badge/Precision-0%25%20Accuracy%20Loss%20(inkcheck)-success)](docs/BENCHMARKS.md#accuracy--statistical-verification)
[![Power](https://img.shields.io/badge/Thermal%20Cap-280W%20%E2%89%A482%C2%B0C-orange)](deploy/systemd/nvidia-power-cap.service)

> **From `55.5 t/s @ 32K` to `131–165 t/s` real-world multi-turn agentic coding and `114–125 t/s` at 118K context on a single consumer RTX 3090 (24 GB, 280 W) — with zero quantization degradation and exact Leviathan et al. probabilistic speculative decoding.**

This repository contains **~13,700 lines of custom Ampere SM 8.6 CUDA kernels and C++ inference-engine engineering** built on top of [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) (`pr390` @ `8aa2f23`), alongside a full production 128K agentic coding stack for **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (`UD-Q4_K_XL`, looped-transformer + 128-expert NVFP4 MoE + hybrid SWA/global attention + DSpark speculative drafter + vision projector).

---

## Test Hardware & Environment Disclosure

All benchmarks in this repository (both the **stock baseline** and **Waves 1–10**) were measured on the same machine:
- **GPU**: Gigabyte NVIDIA GeForce RTX 3090 (24 GB GDDR6X, Ampere SM 8.6), power-capped to **280 W** from Wave 8 onward (`nvidia-power-cap.service`).
- **Driver / OS**: Linux `x86_64`, NVIDIA Open Kernel Module `595.99.02`.
- **Tensor Core Unlock**: **[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0** active at boot (`gpu_unlock.targets=0000:09:00.0`, `SS0=0x88888888`, `SS1=0x8` — full-rate `FP16/BF16/TF32` Tensor Cores with `FP32` accumulation).
  - **What changes if you benchmark *without* `gpu-unlock`**: `T=1` decode and `NVFP4`/`Q8_0` `dp4a` MoE kernels are memory-bandwidth-bound and remain virtually identical. However, **long-context prompt prefill (`pp512`, 32K–128K)** and our **Wave 9B `mma.sync.m16n8k16` (`FP16→FP32`) Split-K Flash Attention kernel** (`T=2..8` at 64K–128K) directly use `FP16→FP32` Tensor Cores and will measure lower absolute throughput without `gpu-unlock`. Relative speedups over the stock baseline remain comparable since both baseline and optimized builds were tested under the same configuration.

---

## Executive Results (RTX 3090 24 GB)

| Metric | Stock Baseline (`bin/` / `pr390`) | **Final (Wave 10, 280 W Cap)** | Gain |
| :--- | ---: | ---: | ---: |
| Raw 1-token decode (`llama-bench tg256`, `d=0`) | `55.5 – 59.1 t/s` *(370 W)* | **`122.0 – 125.6 t/s`** *(280 W; `134.3` @ 370 W)* | **2.2× – 2.4×** |
| Prompt prefill (`llama-bench pp512`, `d=0`) | `1,264 – 1,477 t/s` *(370 W)* | **`2,345 t/s`** *(280 W; `2,656` @ 370 W)* | **1.8× – 2.1×** |
| CUDA graph kernel launches per decode token | `~5,000+` *(11,508 ggml nodes)* | **`524 ops/tok`** | **~10× fewer launches** |
| Max context in 24 GB VRAM (+ DSpark + Vision) | `32K` (`~21.3 GiB`) | **`128K (131,072)`** (`22.2 GiB`, ~2.3 GiB free) | **4× context** |
| Speculative verification `T = 5` (`llama-bench -d 512`) | `26.20 ms/step` *(Wave 7)* | **`16.47 ms/step`** *(303.7 t/s)* | **-37.1 % (1.59×)** |
| Long-ctx `T = 5` attention @ `d = 128K` (Tensor Core MMA) | `~130 t/s` *(pre-MMA)* | **`180.0 t/s`** *(+38 %, `pp8 = 197.1 t/s`)* | **+38 – 40 %** |
| DSpark `n=4` greedy (`temp=0.0` refactor, 100 % exact) | `~74 – 82 t/s` | **`194.0 t/s`** *(32K)* | **2.37×** |
| DSpark `n=4` probabilistic (`temp=1.0`, exact Leviathan) | `0.50 t/s` *(broken)* | **`135–143 t/s`** *(32–64K w/ tools)* · **`125 t/s`** *(118K)* | **>250×** |
| Live multi-turn `deepseek exec` (`tools` + `reasoning`) | `~55 – 58 t/s` *(no tools)* | **`131.6 – 164.8 t/s`** *(`f_keep = 1.000` 4/4 turns)* | **~2.5× – 2.9×** |
| Long-session KV resilience (105K pruner / 1-tok BPE drift) | `77.1 s` freeze / full re-prefill | **`0 ms` pruner** + **`92.7 %` KV rescued** *(13.4× less re-prefill)* | **Eliminated stalls** |
| Accuracy vs unmodified `pr390` F16 ref (`inkcheck`) | Reference noise floor: `96.5 %` top-1 | **`96.0–100.0 %` top-1**, KLD `1.25e-3 – 5.41e-3`, `|z| ≤ 1.47` | **0 % loss** |

---

## System Architecture

```mermaid
flowchart LR
    H["DeepSeek Harness (dsh)<br/>Preset: inkling-128k<br/>24 KB append-only spill<br/>No retroactive pruner"] -->|"OpenAI API (128K, tools, reasoning)"| S["llama-server (Wave 10)<br/>Cache-RAM 16 GiB<br/>Wave 10B Gen Checkpoints (4K)<br/>Stacked T=5 Speculation"]
    S --> D1["Priority 0: GBNF Forced Prefix"]
    S --> D2["Priority 1: ngram-mod (4..4, m=16)"]
    S --> D3["Priority 2: Single-Graph GPU DSpark<br/>Low-Rank M=W_out*H FP16 (125 us)<br/>On-GPU Softmax + Categorical Draw"]
    D1 & D2 & D3 -->|"Fixed T=5 (0 graph reallocs)"| V["Target Verification (SM 8.6 CUDA)<br/>524-op Fused Looped MoE Graph<br/>On-the-fly RelPos + MMA m16n8k16<br/>Selective KV: 8 Global Q8_0 + 40 SWA F16"]
    V --> R["Zero-Copy GPU Rejection Sampler<br/>Exact Leviathan et al. + Active GBNF O(1) Fast-Path<br/>Dynamic 0-byte D2H Gating"]
```

---

## 10-Wave Engineering Evolution

| Wave | Core Engineering Breakthrough | Before → After (Verified on RTX 3090) |
| :--- | :--- | :--- |
| **Baseline** | Stock Unsloth `b12588` & native `pr390` (`tools/ref`) | `55.5 – 59.1 t/s` `tg128`, `1,264 – 1,477 t/s` `pp512`, `32K` max ctx |
| **Waves 1–5** | Branchless 5-inst `UE4M3→FP32` NVFP4, nosync 8-warp GEMV, fused `control()` (W3), `shortconv4`, `mix()`, `SiTU`, `l2_clip+SET_ROWS`, router Top-8 | `55.5–59.1 → 98.0–99.4 t/s` (**+79 %**); `pp512 → 2,580.5 t/s` (**+104 %**) |
| **Wave 6 (6A–6D)** | Horizontal Q+Gate+R+K+V+Sink GEMV, `cp.async` NVFP4 MoE + SiTU, `down+reduce`, `sigmoid·mul+wo`, triple `control()`, Q/R clip pair + **fixed GPU `--backend-sampling`** (`4,000/4,000` GPU==CPU) | `99.4 → 129.8 t/s` `tg256` (370 W); `816 ops/tok`; `chat.sh` `107.3 → 119.2 t/s` |
| **Wave 7 (A1/A2, 7A–7C)** | Inline producer `Q8_1` (`48/48`), full 3-kernel MoE pipeline, custom 1-token GQA=3 `cp.async.cg` RelPos Flash Attention, `GET_ROWS` fold, `need_probs` D2H skip | `816 → 524 ops/tok` (**-292 launches**); `tg256 → 134.3 t/s` (**2.27×**); reasoning `126–129 t/s` |
| **Wave 8 (8.1–8.4)** | Zero-spill expert-grouped `T=2..8` MoE & attention fusions, rollback `shortconv4`, precomputed DSpark low-rank head `M=W_out·H` (`595→125 µs`), multi-block `argmax` (`40→5 µs`), `O(1)` `prob_of()`, **280 W cap** | `T=5` `26.20 → 16.47 ms` (**-37.1 %**); DSpark greedy `82 → 194.0 t/s` (**2.37×**); prob `0.50 → 129.9 t/s` (**260×**) |
| **Wave 9 (9A–9D)** | On-the-fly RelPos in Flash Attention (**-15 GiB** tables → **128K**), selective KV `Q8_0` (8 global) + `F16` (40 SWA), **SM 8.6 Tensor Core `mma.sync.m16n8k16` Split-K**, hybrid GPU grammar sampling, stacked `T=5` spec + CPU vision | `32K → 128K` in `22.2 GiB`; `pp5@128K` **+38 % (`180 t/s`)**; JSON/Tools **+41 % / +32 %**; 120K real coding **`91.5–104.2 t/s`** |
| **Wave 10 (Opt A + 10B + dsh)** | **Single-graph GPU probabilistic DSpark**, exact Leviathan GPU rejection sampling **with `tools`/`json_schema`/`reasoning_budget`**, **Wave 10B 4K generation checkpoints**, DeepSeek Harness `inkling-128k` preset | Single-graph prob **+20.4 %**; `32K/64K` w/ tools **`135–143 t/s` (+37–54 %)**; `118K` **`114–125 t/s`**; 1-tok drift re-prefill **13.4× less**; 77.1 s freeze **eliminated** |

Full technical breakdown → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Complete benchmark tables & accuracy proofs → [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

---

## Repository Contents

```text
├── patches/
│   ├── inkling2-sm86-wave10.patch   # Complete git-apply patch (+13,700 LOC over unslothai/llama.cpp@8aa2f23)
│   └── apply-and-build.sh           # Reproducible SM 8.6 CUDA build script
├── kernels/
│   └── inkling-fused.cuh            # Standalone browsable copy of the 8,012-line custom SM 8.6 CUDA kernels
├── docs/
│   ├── ARCHITECTURE.md              # Kernel design, math proofs, and KV-cache systems postmortem
│   └── BENCHMARKS.md                # Full Wave 1–10 measurements, accuracy checks, and thermal profiles
├── scripts/                         # Production launcher (128K Wave 10), CLI chat, GPU thermal monitor & power cap
├── deploy/                          # systemd units (inkling-server + 280W power cap) and production env file
├── configs/deepseek-harness/        # 128K local agent preset (inkling-128k), spill policy, and provider config
└── tools/                           # inkcheck.cpp (KLD/NLL verifier), A/B harness, smoke tests, CUDA microbenchmarks
```

---

## Quickstart (Reproduce on RTX 3090 / Ampere SM 8.6)

### 1. Build the custom SM 8.6 binaries
```bash
./patches/apply-and-build.sh
```
This clones [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) at base commit `8aa2f23`, applies [patches/inkling2-sm86-wave10.patch](patches/inkling2-sm86-wave10.patch), and compiles `llama-server`, `llama-cli`, `llama-bench`, `llama-perplexity`, and `llama-completion` for `SM 8.6` into `bin-sm86/`.

### 2. Download weights into `models/`
- `Inkling-2-Mini-UD-Q4_K_XL.gguf` (16.41 GiB)
- `dspark-Inkling-2-Mini-Q8_0.gguf` (384 MiB)
- `mmproj-F16.gguf` (610 MiB, runs on CPU with `0 MiB` VRAM via `--no-mmproj-offload`)

### 3. Apply thermal cap & launch the 128K Wave 10 server
```bash
sudo ./scripts/gpu-power-cap.sh 280
./scripts/start-server.sh
```

### 4. Verify endpoints & precision
```bash
python3 tools/server-smoke.py 8080          # Chat, Vision, JSON Schema, Tool Call, Logprobs (5/5 OK)
./tools/build-tools.sh && ./tools/validate.sh  # Teacher-forcing KLD / top-1 / paired-NLL vs stock reference
```

> **Reproducing exact prefill & long-context MMA numbers**: All benchmarks here were run with **[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0** active (`SS0=0x88888888`, `SS1=0x8`). Without it, `T=1` decode is nearly unchanged, while long-context prefill and Wave 9B `mma.sync.m16n8k16` (`FP16→FP32`) attention at 64K–128K will run at GeForce's stock half-rate Tensor Core speed.

## License

MIT License — matching upstream [`llama.cpp`](https://github.com/ggml-org/llama.cpp). See [LICENSE](LICENSE).
