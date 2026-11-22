# Verified Benchmarks & Accuracy Validation

All measurements (both stock baselines and Waves 1–10) were collected on the same **Gigabyte NVIDIA GeForce RTX 3090 (24 GB GDDR6X, Ampere SM 8.6)** system running Linux (`x86_64`, NVIDIA Open Kernel Module `595.99.02`), serving `Inkling-2-Mini-UD-Q4_K_XL.gguf` (16.41 GiB) + `dspark-Inkling-2-Mini-Q8_0.gguf` (384 MiB) + `mmproj-F16.gguf` (610 MiB, CPU offload `0 MiB` VRAM).

> **Environment disclosures (important for reproduction)**:
> 1. **Tensor Core Unlock — [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0 active at boot** (`SS0=0x88888888`, `SS1=0x8`, unlocking full-rate `FP16/BF16/TF32 → FP32` Tensor Core throughput on GeForce RTX 3090). Both the baseline reference and Waves 1–10 ran with `gpu-unlock` enabled. On a stock GeForce RTX 3090 **without** `gpu-unlock`, memory-bound `T=1` decode and `NVFP4`/`Q8_0` `dp4a` kernels perform nearly identically, whereas **prompt prefill (`pp512`, 32K–128K)** and **Wave 9B `mma.sync.m16n8k16` (`FP16→FP32`) Split-K Flash Attention (`T=2..8` at 64K–128K)** will exhibit lower absolute throughput due to NVIDIA's stock half-rate FP32-accumulation cap.
> 2. **Thermal methodology**: Stock 370 W runs reach 83–87 °C in ~20 s and hit continuous thermal throttling (~108 t/s sustained). Starting in **Wave 8**, all benchmarks and production runs use a persistent **280 W power cap** (`nvidia-power-cap.service`), holding **≤ 80–82 °C** with **0 % HW thermal throttling (`0x40`)**.

---

## 1. Wave 10 — Head-to-Head A/B (`temp=1.0`, WITH `tools` + `reasoning_budget=12288`, 280 W)

Comparing **Wave 9** (`DSpark n=4 greedy`) vs **Wave 10** (`Single-Graph GPU Probabilistic DSpark + Exact Leviathan GPU Rejection Sampling` with lazy/active GBNF grammars):

| Workload (`temp=1.0`, `tools` attached, `reasoning_budget=12288`) | Wave 9 (`DSpark greedy`) | **Wave 10 (`DSpark prob GPU`)** | Speedup |
| :--- | ---: | ---: | ---: |
| Isolated Single-Graph `DRAFT_PROBABILISTIC` vs 4× `DRAFT_HEAD` (no tools) | `107.67 t/s` | **`120.35 t/s`** | **+11.8 %** |
| Isolated Single-Graph `DRAFT_PROBABILISTIC` vs 4× `DRAFT_HEAD` (w/ tools) | `99.03 t/s` | **`119.25 t/s`** | **+20.4 %** |
| High-Entropy Reasoning + GLSL Shader (avg 2 runs) | `81.97 t/s` *(25.8 % acc)* | **`94.63 t/s`** *(29.6 % acc)* | **+15.4 %** |
| Tool Call `write_file` (avg 2 runs) | `111.18 t/s` | **`119.37 t/s`** | **+7.4 %** |
| **32K Turn 1** (C++ analysis + code) | `98.62 t/s` *(47.8 % acc)* | **`135.27 t/s`** *(71.8 % acc)* | **+37.2 %** |
| **64K Turn 1** (C++ analysis + code) | `92.98 t/s` *(51.3 % acc)* | **`134.92 – 142.96 t/s`** *(74–79 % acc)* | **+45 – 54 %** |
| **118K Turn 1** (`119,186` tok, analysis + code) | `121.79 t/s` | **`125.26 t/s`** *(78.7 % acc, `pp = 976.3 t/s`)* | **+2.8 %** |
| **118K Turn 2** (`119,697` tok, **100 % KV hit**, `pp = 28 tok`) | — | **`114.25 – 119.41 t/s`** *(75–77 % acc)* | — |
| `ReasoningBudget(64)` + `tools` (budget forcing transition) | `94.9 t/s` | **`139.1 – 155.6 t/s`** *(59–71 % acc)* | **+47 – 64 %** |
| `PureJSON` Schema (`temp=0.6`, `top_p=0.9`, 100 % valid JSON) | `66.7 t/s` | **`93.6 – 97.1 t/s`** | **+45.6 %** |

**VRAM footprint at 128K (`CTX_SIZE=131072`) + DSpark + Vision (CPU)**: **`22,133 – 22,244 / 24,576 MiB`** (~2.3–2.4 GiB free), **0 same-size CUDA graph reallocations**.

---

## 2. Wave 10B — Generation Checkpointing Resilience

Deliberate 1-token BPE drift injected near token ~4,300 of a 4,400-token generated turn (`CHECKPOINT_MIN_STEP=4096`):

| Metric | Without Wave 10B (`W10_NO_GEN_CKPT=1`) | **With Wave 10B (`ckpt` every 4,096 gen tok)** | Improvement |
| :--- | ---: | ---: | ---: |
| KV Cache tokens rescued | `87` | **`4,183` (92.7 % saved)** | **48×** |
| Tokens re-prefilled | `4,422` | **`329`** | **13.4× fewer** |
| Wall-clock recovery time | `3.63 s` | **`1.15 s`** | **3.16× faster** |
| Checkpoint overhead (RAM / time) | — | `168.0 MiB` / `~67 ms` per 4,096 gen tok | Negligible |

---

## 3. Live Production E2E — Multi-Turn DeepSeek Harness (`deepseek exec`)

Real multi-turn agentic coding task (`Create fib.py + unittest suite, execute via bash, verify pass`) against live `inkling-server.service` (`http://127.0.0.1:8080`, 280 W, `inkling-128k` preset):

| Turn | KV Reuse (`f_keep`) | Prompt Tokens Evaluated | Prefill Speed | Generated Tokens | **Decode Speed** | Graph Reallocs (same size) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Turn 1 (System + Tool Catalog) | Cold | `9,078` | `1,599.8 t/s` | `111` | **`147.59 t/s`** | `0` |
| Turn 2 (Tool result → `write_file`) | **`1.000`** | `74` | `250.8 t/s` | `319` | **`163.37 t/s`** | `0` |
| Turn 3 (Long reasoning + `bash`) | **`1.000`** | `45` | `162.5 t/s` | `947` | **`131.57 t/s`** | `0` |
| Turn 4 (Test output verification) | **`1.000`** | `46` | `152.5 t/s` | `51` | **`164.80 t/s`** | `0` |
| Turn 5 (Final concise answer) | **`1.000`** | `34` | `112.9 t/s` | `49` | **`157.86 t/s`** | `0` |

`tools/server-smoke.py 8080` on live Wave 10 production server: **5/5 PASS** (`chat 146.3 t/s`, `vision CPU OK`, `json_schema 70.5 t/s`, `tool_call 138.5 t/s`, `logprobs OK`).

---

## 4. Micro-Benchmarks (`llama-bench`, Context Scaling & Batch Verification)

### 4.1 Long-Context Scaling (`-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -ub 512`, 280 W, Wave 9B+ MMA)

| Context Depth (`d`) | Base `T = 1` (`tg`) | `T = 4` (`pp4`) | `T = 5` (`pp5`, DSpark `n=4`) | `T = 8` (`pp8`) | Prefill `pp512` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| `d = 0 / 512` | **`122.00 t/s`** *(8.20 ms)* | `230.7 t/s` | `250.4 t/s` | `310.2 t/s` | **`2,345 t/s`** |
| `d = 32K` | **`100.17 t/s`** *(9.98 ms)* | — | — | — | — |
| `d = 64K` | **`87.10 t/s`** *(11.48 ms)* | `151.0 t/s` | **`192.7 t/s`** *(26.0 ms)* | **`236.8 t/s`** | **`1,198 t/s`** |
| **`d = 128K` (`130,048`)** | **`70.17 t/s`** *(14.25 ms)* | **`135.8 t/s`** *(+30 %)* | **`180.0 t/s`** *(+38 %, 27.8 ms)* | **`197.1 t/s`** *(+40 %, 40.6 ms)* | **`917.7 t/s`** |

### 4.2 Speculative Batch Latency (`-d 512 -r 5`, 280 W: Wave 7 vs Wave 8+)

| Batch (`T`) | Wave 7 (`ms/step`) | **Wave 8+ (`ms/step`)** | **Throughput** | Latency Reduction |
| :--- | ---: | ---: | ---: | ---: |
| `T = 2` | `15.07 ms` | **`11.68 ms`** | **`171.2 t/s`** | **-22.5 %** |
| `T = 4` | `20.69 ms` | **`14.59 ms`** | **`274.2 t/s`** | **-29.5 %** |
| `T = 5` *(DSpark `n=4`)* | `26.20 ms` | **`16.47 ms`** | **`303.7 t/s`** | **-37.1 % (1.59×)** |
| `T = 6` | `27.46 ms` | **`18.46 ms`** | **`325.0 t/s`** | **-32.8 %** |
| `T = 8` *(DSpark `n=7`)* | `30.55 ms` | **`20.86 ms`** | **`383.5 t/s`** | **-31.7 % (1.46×)** |
| `T = 12` | `31.07 ms` | **`27.62 ms`** | **`434.5 t/s`** | **-11.1 %** |

### 4.3 Raw 1-Token Decode Evolution (`llama-bench`, cold 370 W vs 280 W)

| Milestone | `tg256` (`d=0`) | `tg128` (`d=0 / 2K / 8K / 24.5K`) | Prefill `pp512` (`-ub 512`) |
| :--- | ---: | ---: | ---: |
| Stock Unsloth (`b12588`, 370 W) | `55.5 t/s` | — | `1,476.7 t/s` |
| Unmodified `pr390` SM 8.6 (`tools/ref`, 370 W) | `59.1 t/s` | `59.1 / 56.5 / 53.3 / 49.0 t/s` | `1,279.8 t/s` |
| Waves 1–5 SM 8.6 Fused (370 W) | `98.0 – 99.4 t/s` | — | `2,580.5 t/s` |
| Wave 6D SM 8.6 Fused (370 W) | `129.8 t/s` | `128.7 / 121.7 / 115.7 / 104.0 t/s` | `2,656.3 t/s` |
| Wave 7 SM 8.6 Fused (370 W, `524 ops`) | **`133.8 – 134.3 t/s` (2.27×)** | **`132.4 / 122.9 / 116.0 / 104.9 t/s`** | **`2,652.6 t/s` (2.07×)** |
| Wave 8–10 SM 8.6 Fused (**280 W sustained**) | **`122.0 – 125.6 t/s`** | — | **`2,288 – 2,345 t/s`** |

### 4.4 Wave 9C Isolated Hybrid GPU Grammar Sampling Impact (`DSpark greedy`, 280 W)

| Structured Task | Wave 9C OFF (CPU grammar) | **Wave 9C ON (Hybrid GPU)** | Speedup |
| :--- | ---: | ---: | ---: |
| Pure JSON Schema (`temp=0.0` / `temp=1.0`) | `146.29` / `117.33 t/s` | **`172.06` / `165.74 t/s`** | **+17.6 % / +41.3 %** |
| Agentic Tool Call (`temp=0.0` / `temp=1.0`) | `167.09` / `146.22 t/s` | **`195.98` / `193.05 t/s`** | **+17.3 % / +32.0 %** |
| Chat + Reasoning + JSON (`temp=0.0` / `1.0`) | `121.93` / `96.89 t/s` | **`130.48` / `113.13 t/s`** | **+7.0 % / +16.8 %** |

---

## 5. Accuracy & Statistical Verification

Because Inkling-2-Mini quantizes activations to `int8` (`Q8_1`) before every GEMV, even the unmodified `pr390` reference differs from itself between batch size 1 and batch size 512 (`KLD = 4.36e-3`, `top-1 = 97.17 %`, `z = -1.96`). All builds were validated with [`tools/inkcheck.cpp`](../tools/inkcheck.cpp) against this exact noise floor:

| Teacher-Forcing Comparison (`inkcheck`) | Mean KLD | Top-1 Agreement | Paired Δ Log-Prob (`z`-score) |
| :--- | ---: | ---: | ---: |
| Stock `pr390` ref vs itself (`1,024` tok, batch 1 vs 512 — *noise floor*) | `4.36e-3` | `97.17 %` | `-0.0062 ± 0.0032` (`z = -1.96`) |
| `bin-sm86` Wave 7 vs stock ref (`1,024` tok, `T=1`) | `5.45e-3` | `96.19 %` | `z = -0.49` |
| `bin-sm86` Wave 8 vs stock ref (`1,024` tok, `T=1`) | `5.41e-3` | `96.00 %` | `-0.0050 ± 0.0039` (`z = -1.28`) |
| `bin-sm86` Wave 8 vs stock ref (`9K` ctx, 256 decode) | `3.45e-3` | `97.66 %` | `+0.0047 ± 0.0066` (`z = +0.71`) |
| `bin-sm86` Wave 8 vs stock ref (`T=5` & `T=8`, 256 tok) | `5.28e-3` | `96.88 %` | `-0.0112 ± 0.0077` (`z = -1.47`) |
| **Wave 9B `Q8_0` MMA vs F16 ref (`9K`, `T=5`, 64 decode)** | **`1.25e-3`** | **`100.00 %` (64/64)** | **`z = -1.28`** |
| **Wave 9B `Q8_0` MMA vs F16 ref (`9K`, `T=8`, 64 decode)** | **`1.47e-3`** | **`100.00 %` (64/64)** | **`z = -1.16`** |
| **Wave 9B `Q8_0` MMA vs F16 ref (`32K`, `T=5`, 64 decode)** | **`1.33e-3`** | **`100.00 %` (64/64)** | **`z = +1.03`** *(PPL `1.9259` vs `1.9444`)* |

- **GPU vs CPU Sampler Equivalence (`tools/sampler-compare.py`)**: `4,000 / 4,000` (Wave 7) and `3,999 / 4,000` (`99.975 %`, Wave 8+) exact same token across seeds, passing chi-squared against exact CPU probabilities on all 4 test contexts.
- **Deterministic Speculative Exactness (`temp=0.0`)**: `DSpark n=3/4 greedy`, `DSpark probabilistic` (auto-greedy at `temp=0`), and `ngram-mod n=7` produce **100 % token-for-token identical outputs** to baseline `T = 1`.
- **Wave 7C `GET_ROWS` Fold**: `bit-exact: YES` (`1024/1024`, max diff `0.000e+00`).
