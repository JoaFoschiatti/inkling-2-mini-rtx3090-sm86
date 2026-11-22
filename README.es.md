# Inkling-2-Mini — Motor CUDA SM 8.6 a Medida y Stack Agéntico 128K (RTX 3090)

[English](README.md) | [Español](README.es.md)

[![GPU](https://img.shields.io/badge/GPU-RTX%203090%20(SM%208.6)-76b900?logo=nvidia&logoColor=white)](docs/BENCHMARKS.md)
[![Contexto](https://img.shields.io/badge/Contexto-128K%20(131%2C072)-blue)](docs/ARCHITECTURE.md)
[![Decode](https://img.shields.io/badge/Decode%20Ag%C3%A9ntico-131--165%20t%2Fs-brightgreen)](docs/BENCHMARKS.md)
[![Precisión](https://img.shields.io/badge/Precisi%C3%B3n-0%25%20P%C3%A9rdida%20(inkcheck)-success)](docs/BENCHMARKS.md#accuracy--statistical-verification)
[![Potencia](https://img.shields.io/badge/Tope%20T%C3%A9rmico-280W%20%E2%89%A482%C2%B0C-orange)](deploy/systemd/nvidia-power-cap.service)

> **De `55.5 t/s @ 32K` a `131–165 t/s` en programación agéntica multi-turno real y `114–125 t/s` a 118K de contexto en una única RTX 3090 de consumo (24 GB, 280 W) — con cero pérdida de precisión y decodificación especulativa probabilística exacta de Leviathan et al.**

Este repositorio reúne **~13,700 líneas de kernels CUDA Ampere SM 8.6 e ingeniería de motor de inferencia C++** desarrolladas sobre [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) (`pr390` @ `8aa2f23`), junto con el stack de producción completo a **128K** para **[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (`UD-Q4_K_XL`, arquitectura *looped-transformer* + MoE NVFP4 de 128 expertos + atención híbrida SWA/global + drafter DSpark + proyector de visión).

---

## Hardware y Entorno de Pruebas (Importante para Reproducir)

Todos los benchmarks de este repositorio (tanto el **baseline stock** como las **Waves 1–10**) se midieron en la misma máquina:
- **GPU**: Gigabyte NVIDIA GeForce RTX 3090 (24 GB GDDR6X, Ampere SM 8.6), con tope térmico de **280 W** desde Wave 8 (`nvidia-power-cap.service`).
- **Driver / SO**: Linux `x86_64`, NVIDIA Open Kernel Module `595.99.02`.
- **Desbloqueo de Tensor Cores**: **[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0** activo desde el arranque (`SS0=0x88888888`, `SS1=0x8` — tasa completa `1:1` en Tensor Cores `FP16/BF16/TF32` con acumulador `FP32`).
  - **Qué cambia si alguien prueba *sin* `gpu-unlock`**: el decode `T=1` y los kernels MoE `NVFP4`/`Q8_0` (`dp4a`) están limitados por ancho de banda GDDR6X y rinden prácticamente igual. En cambio, el **prefill (`pp512`, 32K–128K)** y el kernel **Wave 9B Flash Attention `mma.sync.m16n8k16` (`FP16→FP32`) Split-K** (`T=2..8` a 64K–128K) sí dependen directamente de la tasa de Tensor Cores `FP16→FP32`, por lo que darán valores absolutos menores sin `gpu-unlock`. La mejora relativa frente al baseline se mantiene porque tanto el baseline como las Waves 1–10 se midieron bajo el mismo entorno.

---

## Resultados Clave (RTX 3090 24 GB)

| Métrica | Baseline Stock (`bin/` / `pr390`) | **Final (Wave 10, tope 280 W)** | Mejora |
| :--- | ---: | ---: | ---: |
| Decode puro 1 token (`llama-bench tg256`, `d=0`) | `55.5 – 59.1 t/s` *(370 W)* | **`122.0 – 125.6 t/s`** *(280 W; `134.3` @ 370 W)* | **2.2× – 2.4×** |
| Prefill (`llama-bench pp512`, `d=0`) | `1,264 – 1,477 t/s` *(370 W)* | **`2,345 t/s`** *(280 W; `2,656` @ 370 W)* | **1.8× – 2.1×** |
| Lanzamientos CUDA por token en decode | `~5,000+` *(11,508 nodos ggml)* | **`524 ops/tok`** | **~10× menos lanzamientos** |
| Contexto máximo en 24 GB (+ DSpark + Visión GPU) | `32K` (`~21.3 GiB`) | **`128K (131,072)`** (`22.8` reposo / `23.3 GiB` pico, ~1.25–1.75 GiB libres) | **4× contexto** |
| Encode de visión (`1440×810` PNG / `960×540` JPEG) | `5,835 ms` / `2,229 ms` *(CPU)* | **`785 ms`** / **`356 ms`** *(GPU `VISION_OFFLOAD=1`)* | **7.4× / 6.3× (16.4× vs PNG CPU)** |
| Verificación especulativa `T = 5` (`-d 512`) | `26.20 ms/paso` *(Wave 7)* | **`16.47 ms/paso`** *(303.7 t/s)* | **-37.1 % (1.59×)** |
| Atención `T = 5` a `d = 128K` (Tensor Core MMA) | `~130 t/s` *(pre-MMA)* | **`180.0 t/s`** *(+38 %, `pp8 = 197.1 t/s`)* | **+38 – 40 %** |
| DSpark `n=4` greedy (`temp=0.0` refactor, 100 % exacto) | `~74 – 82 t/s` | **`194.0 t/s`** *(32K)* | **2.37×** |
| DSpark `n=4` probabilístico (`temp=1.0`, exacto) | `0.50 t/s` *(inusable)* | **`135–143 t/s`** *(32–64K con tools)* · **`125 t/s`** *(118K)* | **>250×** |
| Razonamiento alta entropía (`temp=1.0`, Wave 10C) | `91.6 – 97.3 t/s` *(frío)* · `78–85` *(caliente)* | **`109.8 t/s`** *(gate EMA `T=5↔T=1`, 0 % regresión código)* | **+13 % – +30 %** |
| E2E real multi-turno `deepseek exec` (`fast-mcp` + visión GPU) | `~48–60 s` por chequeo visual (`agent-browser`) | **`10.99 s` totales** *(5 turnos edit+Chrome+visión GPU, `114–180 t/s`)* | **~30–50× más rápido** |
| Resiliencia KV sesiones largas (pruner 105K / deriva BPE) | Freeze `77.1 s` / re-prefill total | **`0 ms` pruner** + **`92.7 %` KV rescatado** *(SWA-1K + post-imagen)* | **Sin congelamientos** |
| Precisión frente a referencia `pr390` F16 (`inkcheck`) | Piso ruido ref: `96.5 %` top-1 | **`96.0–100.0 %` top-1**, KLD `1.25e-3 – 5.41e-3`, `|z| ≤ 1.47` | **0 % pérdida** |

---

## Evolución Wave por Wave (1 → 10)

| Wave | Hito de Ingeniería | Antes → Después (medido en RTX 3090) |
| :--- | :--- | :--- |
| **Baseline** | Unsloth `b12588` y `pr390` nativo sin fusiones | `55.5 – 59.1 t/s` `tg128`, `1,264 – 1,477 t/s` `pp512`, máx. `32K` |
| **Waves 1–5** | `UE4M3→FP32` branchless 5 insts, GEMV NVFP4 sin sync, fusiones `control()` (W3), `shortconv4`, `mix()`, `SiTU`, `l2_clip+SET_ROWS`, router Top-8 | `55.5–59.1 → 98.0–99.4 t/s` (**+79 %**); `pp512 → 2,580.5 t/s` (**+104 %**) |
| **Wave 6 (6A–6D)** | GEMV horizontal Q+Gate+R+K+V+Sink, MoE NVFP4 `cp.async`, `down+reduce`, `sigmoid·mul+wo`, `control()` triple + **muestreo GPU `--backend-sampling`** (`4,000/4,000` GPU==CPU) | `99.4 → 129.8 t/s` `tg256` (370 W); `816 ops/tok`; `chat.sh` `107.3 → 119.2 t/s` |
| **Wave 7 (A1/A2, 7A–7C)** | Cuantización `Q8_1` inline en productor (`48/48`), pipeline MoE 3-kernels, Flash Attention 1-token GQA=3 `cp.async.cg`, plegado `GET_ROWS`, omisión D2H `need_probs` | `816 → 524 ops/tok` (**-292 lanzamientos**); `tg256 → 134.3 t/s` (**2.27×**); reasoning `126–129 t/s` |
| **Wave 8 (8.1–8.4)** | Fusiones `T=2..8` zero-spill agrupadas por experto, `shortconv4` con rollback, cabeza DSpark low-rank `M=W_out·H` (`595→125 µs`), `argmax` multi-bloque (`40→5 µs`), `prob_of()` $O(1)$, **tope 280 W** | `T=5` `26.20 → 16.47 ms` (**-37.1 %**); DSpark greedy `82 → 194.0 t/s` (**2.37×**); prob `0.50 → 129.9 t/s` (**260×**) |
| **Wave 9 (9A–9D)** | `RelPos` al vuelo en Flash Attention (**-15 GiB** → **128K**), KV `Q8_0` selectivo (8 globales) + `F16` (40 SWA), **Tensor Cores SM 8.6 `mma.sync.m16n8k16` Split-K**, muestreo híbrido GPU con gramáticas, especulación apilada `T=5` + visión CPU | `32K → 128K` en `22.2 GiB`; `pp5@128K` **+38 % (`180 t/s`)**; JSON/Tools **+41 % / +32 %**; 120K real **`91.5–104.2 t/s`** |
| **Wave 10 (10A–10D + Visión GPU + `fast-mcp`)** | **DSpark probabilístico en 1 grafo GPU**, *rejection sampling* exacto con `tools`/`json_schema`/`reasoning_budget`, **Wave 10B/10D** checkpoints SWA-1K 4K + post-imagen, **Wave 10C** gate adaptativo EMA `T=5↔T=1`, **Visión GPU (`VISION_OFFLOAD=1`)**, MCP persistente **`fast-mcp.mjs`** (`browser`/`view_image`/`multi_edit`) | `32K/64K` con tools **`135–143 t/s` (+37–54 %)**; alta entropía **`97.3 → 109.8 t/s` (+13–30 %)**; visión **6.3–7.9× (`356–785 ms`)**; chequeo web **`10–16 s → 0.21–0.48 s`**; E2E 5 turnos **`10.99 s`** (`114–180 t/s`) |

Documentación técnica detallada → [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · Tablas completas de benchmarks y precisión → [docs/BENCHMARKS.md](docs/BENCHMARKS.md)

---

## Reproducción rápida (RTX 3090 / Ampere SM 8.6)

```bash
# 1. Clonar unslothai/llama.cpp@8aa2f23, aplicar el parche Wave 10 y compilar SM 8.6
./patches/apply-and-build.sh

# 2. Colocar los pesos GGUF en models/ y fijar el tope térmico de 280 W
sudo ./scripts/gpu-power-cap.sh 280

# 3. Iniciar el servidor 128K Wave 10 y verificar 5/5 endpoints
./scripts/start-server.sh
python3 tools/server-smoke.py 8080
```

> **Nota para reproducir los números de prefill y MMA en 64K–128K**: todas las mediciones de este repo se ejecutaron con **[Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock) v2.0.0** activo (`SS0=0x88888888`, `SS1=0x8`). Sin ese módulo, el decode `T=1` casi no cambia, pero el prefill y el kernel `mma.sync.m16n8k16` (`FP16→FP32`) de atención a contexto largo correrán a la ½ tasa de fábrica de GeForce.

## Licencia

Licencia MIT — alineada con [`llama.cpp`](https://github.com/ggml-org/llama.cpp). Ver [LICENSE](LICENSE).
