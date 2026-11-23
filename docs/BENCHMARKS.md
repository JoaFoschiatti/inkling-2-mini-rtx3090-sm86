# Verified Benchmarks & Accuracy Validation

All measurements (both stock baselines and Waves 1–14) were collected on the same **Gigabyte NVIDIA GeForce RTX 3090 (24 GB GDDR6X, Ampere SM 8.6)** system running Linux (`x86_64`, NVIDIA Open Kernel Module `595.99.02`), serving `Inkling-2-Mini-UD-Q4_K_XL.gguf` (16.41 GiB) + `dspark-Inkling-2-Mini-Q8_0.gguf` (384 MiB) + `mmproj-F16.gguf` (610 MiB, GPU offload from Wave 10D onward).

> **Environment disclosures (important for reproduction)**:
> 1. **Tensor Core Unlock — [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0 active at boot** (`SS0=0x88888888`, `SS1=0x8`, unlocking full-rate `FP16/BF16/TF32 → FP32` Tensor Core throughput on GeForce RTX 3090). Both the baseline reference and Waves 1–10 ran with `gpu-unlock` enabled. On a stock GeForce RTX 3090 **without** `gpu-unlock`, memory-bound `T=1` decode and `NVFP4`/`Q8_0` `dp4a` kernels perform nearly identically, whereas **prompt prefill (`pp512`, 32K–128K)** and **Wave 9B `mma.sync.m16n8k16` (`FP16→FP32`) Split-K Flash Attention (`T=2..8` at 64K–128K)** will exhibit lower absolute throughput due to NVIDIA's stock half-rate FP32-accumulation cap.
> 2. **Thermal methodology**: Stock 370 W runs reach 83–87 °C in ~20 s and hit continuous thermal throttling (~108 t/s sustained). Starting in **Wave 8**, all benchmarks and production runs use a persistent **280 W power cap** (`nvidia-power-cap.service`), holding **≤ 80–82 °C** with **0 % HW thermal throttling (`0x40`)**. From **Wave 13** the cap is **265 W** (`-lgc 210,1800`, `75–77 °C`); figures measured at 280 W and at 265 W are not comparable with each other.

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

## 4B. Wave 10C/10D + GPU Vision (`VISION_OFFLOAD=1`) + `fast-mcp.mjs` A/B (128K, 280 W)

| Benchmark (RTX 3090 @ 280 W, `CTX=131072`) | Baseline (`W10C/D OFF`, CPU Vision, `agent-browser`) | **Wave 10C/10D + GPU Vision + `fast-mcp`** | Improvement |
| :--- | ---: | ---: | ---: |
| Vision encode: `1440×810` PNG (`788` tok) | `5,835 ms` *(CPU)* | **`785 ms`** *(GPU, `23.33 GiB` peak)* | **7.4× faster** |
| Vision encode: `960×540` JPEG (`355` tok) | `2,229 ms` *(CPU)* | **`356 ms`** *(GPU)* | **6.3× (16.4× vs PNG CPU)** |
| High-entropy creative reasoning (`900` tok, `acc≈0.31`) | `91.6 – 97.3 t/s` | **`109.8 t/s`** *(4 EMA pauses)* | **+13 % – +20 %** |
| Structured code + `tools` (`900` tok, `acc≈0.54`) | `133.2 t/s` | **`133.6 t/s`** | **0 % regression** |
| Browser verification (cold boot / warm action + inline JPEG) | `10,000 – 16,210 ms` + `read_image` turn | **`308 – 476 ms`** (cold) / **`208 ms`** (warm) | **~25–60× faster** |
| Multi-replacement file edit + `node --check` | 1–4 turns / `python3` heredoc | **`18.4 ms`** (`mcp__fast__multi_edit`) | **Atomic, 1 call** |
| Local PNG downscale (`1440×810` → `960×540` JPEG) | `1.25 MB` raw (`~680` tok) | **`82.3 ms`** (`~40–70 KB`, `~290` tok) | **2.3× fewer tokens** |
| **Live E2E `deepseek exec`** (edit HTML + Chrome + GPU vision) | `~48–60 s` per visual check | **`10.99 s` total** *(5 turns incl. `6.4K` cold pp, `114–180 t/s`, image turn `608 ms` pp)* | **Verified E2E** |
| `tools/server-smoke.py 8080` | 5/5 OK | **5/5 TODO OK** (`tool_call: 174.1 t/s`, `chat: 122.9 t/s`) | **100 % healthy** |

---

## 4C. Wave 13 Lossless A/B Verification & Live Multi-Hour Production Audit (Run 1 → Run 3)

### Isolated Wave 13 C++ A/B (`build-w12`, `temp=0.0`, `seed=42`, 15× `960×540` PNGs)
| Test | Baseline (`W13_NO_TOK_CACHE=1, W13_NO_MEDIA_GAP=1`) | **Wave 13 Enabled** | Result |
| :--- | ---: | ---: | :--- |
| Greedy multi-turn exactness (text + tools + 2 images) | `167 / 275 / 339` tok | `167 / 275 / 339` tok | **100 % bit-exact output** |
| KV eviction rollback + placeholder media hydration | `"Red"` (`396` tok) | `"Red"` (`396` tok, hydrated) | **PASS (lossless)** |
| Follow-up `preCPU` median with 15× `960×540` PNGs | `235.1 ms` (`241.7 ms` on img #15) | **`86.8 ms`** (`93.8 ms` on img #15) | **2.71× faster** (~9× lower server overhead) |
| BPE rollback after `write_note` w/ `----` & `""""` | — | **`0` tok rollback** (`hit=1`, `678/692`) | **PASS** |
| Redundant `wave10d` checkpoints across 15 PNGs | `16` ckpts (`~2.7 GiB`) | **`0` redundant** (covered by `2048`-tok ckpt) | **PASS** |

### Live Real-World Production Sessions (`deepseek` + `xhigh` + `fast-mcp` + `inkling-proxy`)
Three complete real-world multi-turn frontend/SVG/WebGL coding sessions on RTX 3090 (24 GB):
- **Run 1** (pre-Wave 13, 280 W): `50.7 min`, `443` steps, `237,604` generated tokens, `22` PNGs.
- **Run 2** (Wave 13 + Bloque 1 + 265 W): `25.8 min`, `77` steps, `146,729` generated tokens, `19` PNGs.
- **Run 3** (Run 2 + proxy compaction `effort:"none"` override & `bash_high_effort`, 265 W): `37.0 min`, `107` steps, `202,421` generated tokens, **`29` PNGs**, `58 KB` standalone deliverable (`daylight.html`).

| Metric | Run 1 (Before) | Run 2 (Wave 13 + Bloque 1) | **Run 3 (+ Proxy v2, 265 W)** |
| :--- | ---: | ---: | ---: |
| Compaction KV cache hit & prefill time | `0 %` (`736.3 s` / 4 cmp) | `0 %` (`98.7 s` / 2 cmp) | **`94.9 – 97.6 %`** (**`10.9 s` total** / 4 cmp) |
| Overall session KV cache hit | — | `94.64 %` | **`96.67 %`** (`7.14M / 7.39M` prompt tok) |
| Non-compaction turns with `f_keep = 1.000` | 8 BPE rollbacks (`26.5 s`) | `73 / 77` | **`102 / 102`** (**0 BPE rollbacks**, incl. `25,347`-tok `write`) |
| Repetitive tool loops | 235 stuck turns (`280.5 s`) | **`0`** | **`0`** (`max_consec = 0`) |
| Total `preCPU` time across entire session | `164.2 s` (~500 ms/turn) | **`13.8 s`** (med `109 ms`) | **`22.9 s`** / 112 reqs (med `158 ms` w/ up to 29 PNGs) |
| Idle headless Chrome CPU (`fast-mcp.mjs`) | `468 – 1,071 % CPU` | **`0.1 % CPU`** | **`0.2 % CPU`** after **48 browser calls** |
| Redundant `wave10d` post-image checkpoints | `31` (`5.2 GiB`) | **`0`** | **`0`** (29 simultaneous PNGs) |
| Host Swap growth (`CACHE_RAM=8192`) | `+3.0 GiB` swap | **`0` growth** | **`0` growth** (`15 GiB` RAM free) |
| Sustained GPU temp & mean decode speed | `80–82 °C` @ `280 W` (`121.8 t/s`) | **`75–77 °C`** @ `265 W` (`112.6 t/s`, peak `161`) | **`75–77 °C`** @ `265 W` (`102.4 t/s` @ 60–94K ctx; **`141.5 t/s`** on 58 KB `write`) |

`tools/server-smoke.py 8080` on live Wave 13 production server: **5/5 TODO OK** (`chat 115.3 t/s`, `vision OK`, `json_schema 96.6 t/s`, `tool_call 126.9 t/s`, `logprobs OK`).

---

## 4D. Wave 14 — Determinism, Draft Length & Per-Step Costs (265 W)

Protocol for every throughput table in this section: [`tools/gen-bench.py`](../tools/gen-bench.py) against `llama-server` with the production configuration, ~14K tokens of context, `temp = 1.0`, 400 generated tokens per request, four regimes (new code, reasoning, tool call, verbatim rewrite), the same seeds for every variant and the variants interleaved in blocks. `±` is the standard error of the mean.

> Under the 265 W cap the absolute level of one binary moves ~10 % with the thermal state of the card (the same binary gave `92.2` and `97.8 t/s` on code in two halves of one session). Only interleaved comparisons are meaningful, and a gain below ~3 % cannot be shown in t/s with 16 seeds: it has to be shown by mechanism (ms per step, graph rebuilds) and by equality of outputs.

### Determinism of batched decode (shared-memory race in the batched `mma.sync` attention kernels)
| Test | Wave 13 | **Wave 14** |
| :--- | ---: | ---: |
| `inkcheck --decode-batch 5`, 12 batches per run | `17 / 93` runs deviate | **`0 / 112`** (and the output is the majority one from before) |
| `inkcheck --decode-batch 12` (`dyn` kernel, `T = 9..24`) | `3 / 15` | **`0 / 36`** |
| Server, `temp 0`, 400 tokens, speculation always on (`W10_NO_ADAPT_SPEC=1`), 8 runs per prompt | `7` and `5` distinct outputs | **`1` and `1`** |
| 1-token decode over 1,024 tokens, Wave 14 vs Wave 13 binary | — | **`bit-exact: YES`** |

### Throughput before and after the fixes (same seeds, binaries interleaved)
| Case | Wave 13 binary | **Wave 14** |
| :--- | ---: | ---: |
| New code (8 seeds) | `97.8 t/s` | `104.0 t/s` |
| Reasoning (8) | `95.9 t/s` | `96.5 t/s` |
| Tool call (8) | `104.9 t/s` | `108.4 t/s` |
| Verbatim rewrite (24) | `115.8 ± 2.4 t/s` | `116.9 ± 2.5 t/s` |
| `json_schema` output (24) | `106.5 t/s` | `107.1 t/s` |
| Plain step / cycle with 4 drafts | `10.19 / 25.95 ms` | `10.26 / 25.66 ms` |

No measurable change, as expected from fixes that do not touch the cost per step (`±2–4 t/s` with 8 seeds).

### Scheduler for the plain step (`sched_t1`; `W14_NO_T1_SCHED=1` disables it)
| Metric | Without | **With** |
| :--- | ---: | ---: |
| Rebuilds of the 1-token graph in 4 requests | `14` | **`5`** (one per request remains) |
| Plain step / cycle with 4 drafts (mean of 16 paired comparisons) | — | **`-0.11 ms` (`-1.1 %`) / `-0.19 ms` (`-0.7 %`)** |
| 14 chained short turns ([`tools/turn-bench.py`](../tools/turn-bench.py)) | `4,457 ms` | **`4,382 ms` (`-1.7 %`)** |
| New code / reasoning / tool call / rewrite, 32 seeds | `100.3 ± 1.7` / `94.3 ± 0.7` / `103.2 ± 1.1` / `131.8 ± 2.9 t/s` | `102.4 ± 1.7` / `95.7 ± 0.9` / `103.9 ± 1.5` / `130.7 ± 2.6 t/s` |

Outputs are identical with and without it (same request at `temp 0` with forced pauses, 4 runs per prompt; same output hash in `turn-bench.py`). Cost of each batch-shape change it removes, from `[PROF_CTX]` over 4 requests:

| Change of shape | Count | Build + allocate | First compute, extra | Total |
| :--- | ---: | ---: | ---: | ---: |
| To 1 token after a batch of 5 (entering a pause) | `12` | `7.0 ms` | `3.6 ms` | `10.6 ms` |
| To 5 tokens after 1-token steps (leaving a pause) | `9` | `6.9 ms` | `2.1 ms` | `9.0 ms` |
| To 5 tokens after a batch of 4 | `3` | `6.3 ms` | `1.8 ms` | `8.1 ms` |

### Draft length (`DSPARK_N` = `NGRAM_MIN` = `NGRAM_MAX`)
Final binary, 16 seeds per case (`701–716`; 32 for the rewrite, `701–732`) in interleaved blocks:

| Variant | New code | Reasoning | Tool call | Verbatim rewrite |
| :--- | ---: | ---: | ---: | ---: |
| 4 drafts (`DSPARK_N=4`, n-gram `4..4`), the previous default | `105.2 ± 1.2` | `98.4 ± 0.9` | `107.3 ± 1.3` | `131.7 ± 2.8` |
| **3 drafts (`DSPARK_N=3`, n-gram `3..3`), default since 2026-11-23** | **`108.6 ± 1.9`** | **`101.4 ± 0.8`** | **`112.4 ± 1.5`** | **`126.4 ± 2.3`** |
| Difference | `+3.3 ± 2.2 %` | `+3.0 ± 1.3 %` | `+4.7 ± 1.9 %` | `-4.0 ± 2.7 %` |

First session (before `sched_t1`, seeds `601–616`), including the variants that were discarded:

| Variant | New code | Reasoning | Tool call | Verbatim rewrite |
| :--- | ---: | ---: | ---: | ---: |
| 4 drafts (32 samples) | `98.2 ± 1.0` | `94.2 ± 0.5` | `101.9 ± 0.9` | `137.1 ± 3.0` |
| 3 drafts (16) | `105.1 ± 1.4` | `97.9 ± 1.0` | `107.3 ± 1.5` | `126.4 ± 3.1` |
| 2 drafts (8) | `102.4 ± 1.6` | `96.8 ± 1.2` | `108.8 ± 1.5` | `120.9 ± 2.6` |
| DSpark 2 with n-gram at 4: mixed batch shapes (8) | `97.6 ± 1.2` | `94.9 ± 0.9` | `95.3 ± 1.5` | `98.6 ± 2.4` |
| Adaptive 3/4, state kept across requests (8) | `99.9 ± 1.4` | `98.2 ± 1.4` | `103.5 ± 1.6` | `130.3 ± 5.1` |
| Adaptive 3/4, every request starts at 4 (8) | `100.1 ± 3.9` | `94.7 ± 1.4` | `105.6 ± 2.5` | `126.9 ± 5.8` |

Chained short turns ([`tools/turn-bench.py`](../tools/turn-bench.py): 14 turns of 32 tokens on a cached ~14K context, `temp 0`, two interleaved passes per variant): `4,968` and `4,977 ms` with 4 drafts against **`4,769` and `4,764 ms` with 3 (`-4.1 %`)**, median generation per turn `283.6 → 261.6 ms`, and the same output hash in the four passes.

Mechanism, from `W12_SPEC_STATS=1` over the 68 requests of each variant: a cycle with 4 drafts costs `24.75 ms` and yields `2.80` tokens (`113.1 t/s` while speculating), a cycle with 3 drafts `22.10 ms` and `2.54` tokens (`114.8 t/s`), and a plain step `9.8–9.9 ms` (`102 t/s`). The 4th position buys `0.26` tokens for `2.65 ms`, `98 t/s` at the margin: less than a plain step delivers, unless the drafter is copying text. With 3 drafts the rate gate also pauses less (31 % of the tokens are generated in pause, against 42 %). Across both sessions 3 drafts gain 3–7 % on new code, reasoning and tool calls and lose 4–8 % on verbatim rewriting.

### Where the time goes (17.6K context, 265 W, ~76 °C)
`INKLING_PROF_PREFILL=1 W12_PROF_ALL=1`; each figure includes the profiling sync.

| Ubatch | ms |
| :--- | ---: |
| Plain step, 1 token (target model) | `10.03` |
| 1-token inject into the drafter context | `0.39` |
| 5-token verification | `22.44` |
| DSpark draft (8 tokens, 4 outputs) | `2.77` |
| 5-token inject | `0.55` |
| 512-token prefill (target / inject) | `280–295` / `11 + 1.8` |

A cycle with 4 drafts costs `27.0–28.0 ms` and a plain step `10.7–11.3 ms` (break-even ~2.5 tokens per cycle); with a cooler GPU, `25.0–25.9` and `9.3–10.3 ms`. In the `T = 5` graph the batched MoE is 51 % (`199 µs` per layer, already at the bandwidth limit for ~40 distinct experts), the horizontal GEMV 12 % and attention 7 %.

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
| Stock `pr390` ref vs itself, re-measured for Wave 14 (`1,024` tok, batch 1 vs 512 — *noise floor of that run*) | `4.03e-3` | `98.24 %` | `-0.0052 ± 0.0037` (`z = -1.38`) |
| **`bin-sm86` Wave 14 vs stock ref (`tools/validate.sh`, `1,024` tok, `T=1`)** | **`5.29e-3`** | **`98.44 %`** | **`-0.0021 ± 0.0040` (`z = -0.52`)** |

- **GPU vs CPU Sampler Equivalence (`tools/sampler-compare.py`)**: `4,000 / 4,000` (Wave 7) and `3,999 / 4,000` (`99.975 %`, Wave 8+) exact same token across seeds, passing chi-squared against exact CPU probabilities on all 4 test contexts.
- **Deterministic Speculative Exactness (`temp=0.0`)**: `DSpark n=3/4 greedy`, `DSpark probabilistic` (auto-greedy at `temp=0`), and `ngram-mod n=7` produce **100 % token-for-token identical outputs** to baseline `T = 1`.
- **Wave 7C `GET_ROWS` Fold**: `bit-exact: YES` (`1024/1024`, max diff `0.000e+00`).
- **Wave 14**: 1-token decode `bit-exact: YES` against the Wave 13 binary over 1,024 tokens; GPU vs CPU sampler `4,000 / 4,000`; `tools/server-smoke.py` 5/5 on `bin-sm86`. The two Wave 14 rows above use this repository's `README.md` as the test text (what `tools/validate.sh` does on a clone); with the previous internal test text the same binaries gave KLD `6.57e-3`, top-1 `97.27 %`, `z = -1.43` against a floor of `5.38e-3`, `97.46 %`, `z = +0.09`.
- **Scope of the `temp=0.0` exactness above (Wave 14 finding)**: until Wave 14 batched decode was not reproducible from run to run (§4D), and the `T = 1` and batched paths agree within the noise floor of the first rows of this table, not bit for bit, so a greedy speculative run can leave the `T = 1` output at a near-tie. From Wave 14 a batched run is reproducible; a reproducible server output at `temp 0` needs `W10_NO_ADAPT_SPEC=1`.
