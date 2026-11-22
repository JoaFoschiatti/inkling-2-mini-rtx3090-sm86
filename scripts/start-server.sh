#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$([[ -d "$SCRIPT_DIR/bin-sm86" || -d "$SCRIPT_DIR/models" ]] && echo "$SCRIPT_DIR" || (cd "$SCRIPT_DIR/.." && pwd))"
BIN_DIR="${BIN_DIR:-$DIR/bin-sm86}"
MODELS_DIR="$DIR/models"

MODEL="${MODEL:-$MODELS_DIR/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-inkling-2-mini}"
MMPROJ="${MMPROJ:-$MODELS_DIR/mmproj-F16.gguf}"
DSPARK_MODEL="${DSPARK_MODEL:-$MODELS_DIR/dspark-Inkling-2-Mini-Q8_0.gguf}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
CTX_SIZE="${CTX_SIZE:-131072}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
GPU_LAYERS="${GPU_LAYERS:-99}"
# Wave 9B: Q8_0 selectivo en las 8 capas globales (las 40 capas SWA 1K se mantienen en F16 sin pérdida).
KV_TYPE="${KV_TYPE:-q8_0}"
DRAFT_KV_TYPE="${DRAFT_KV_TYPE:-q8_0}"
# Caché de prefijos en RAM del sistema (MiB) + checkpoints cada 4K tok (prefill y generación Wave 10B).
# 8192 MiB evita swap en hosts de 30-32 GiB RAM manteniendo varios checkpoints de 128K en memoria.
CACHE_RAM="${CACHE_RAM:-8192}"
CHECKPOINT_MIN_STEP="${CHECKPOINT_MIN_STEP:-4096}"
VISION="${VISION:-1}"
# mmproj-F16.gguf pesa solo 610 MiB (~23.3/24.5 GiB pico en 128K): VISION_OFFLOAD=1 acelera visión 6.3-7.9x en GPU.
VISION_OFFLOAD="${VISION_OFFLOAD:-1}"
# Wave 10-11: especulación apilada T=5 con DSpark probabilístico exacto en 1 grafo GPU + rejection sampling zero-D2H.
DSPARK="${DSPARK:-1}"
DSPARK_N="${DSPARK_N:-4}"
DSPARK_SAMPLING="${DSPARK_SAMPLING:-probabilistic}"
NGRAM="${NGRAM:-1}"
NGRAM_MIN="${NGRAM_MIN:-4}"
NGRAM_MAX="${NGRAM_MAX:-4}"
NGRAM_MATCH="${NGRAM_MATCH:-16}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"
# Techo de seguridad de pensamiento (tokens): -1 = sin límite; 12288 deja siempre >=20K para la respuesta final (maxTokens=32768).
REASONING_BUDGET="${REASONING_BUDGET:-12288}"
REASONING_BUDGET_MESSAGE="${REASONING_BUDGET_MESSAGE:-$'\nI have enough context; let me give the final answer or next tool call now.'}"
# Wave 10-11: muestreo híbrido en GPU + rejection sampling exacto en GPU compatible con gramáticas, tools y reasoning budget.
BACKEND_SAMPLING="${BACKEND_SAMPLING:-1}"
MIN_P="${MIN_P:-0.0}"
# Plantilla Jinja O(1) byte-exacta al GGUF (elimina escaneo O(M^2) de tool_call_id en sesiones largas).
DEFAULT_TEMPLATE="$DIR/templates/chat-template-inkling-o1.jinja"
if [[ -z "${CHAT_TEMPLATE_FILE:-}" && -f "$DEFAULT_TEMPLATE" ]]; then
  CHAT_TEMPLATE_FILE="$DEFAULT_TEMPLATE"
else
  CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE:-}"
fi
# Wave 12: un solo hilo de CPU en el bucle de decode. Por paso solo se copia una fila de embeddings y con 8 hilos
# los otros 7 giran en vacío (~0.5 núcleos). El codificador de imágenes conserva sus hilos. THREADS=0 deja el valor de llama.cpp.
THREADS="${THREADS:-1}"

export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

ARGS=(
  --model "$MODEL"
  --alias "$MODEL_ALIAS"
  --jinja
  --ctx-size "$CTX_SIZE"
  --ubatch-size "$UBATCH_SIZE"
  --n-gpu-layers "$GPU_LAYERS"
  --flash-attn on
  --cache-type-k "$KV_TYPE"
  --cache-type-v "$KV_TYPE"
  --cache-ram "$CACHE_RAM"
  --checkpoint-min-step "$CHECKPOINT_MIN_STEP"
  --parallel 1
  --temp 1.0
  --top-p 1.0
  --top-k 0
  --min-p "$MIN_P"
  --chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}"
  --host "$HOST"
  --port "$PORT"
)

if [[ -n "$CHAT_TEMPLATE_FILE" && -f "$CHAT_TEMPLATE_FILE" ]]; then
  ARGS+=(--chat-template-file "$CHAT_TEMPLATE_FILE")
fi

if [[ "$VISION" == "1" && -f "$MMPROJ" ]]; then
  ARGS+=(--mmproj "$MMPROJ")
  if [[ "$VISION_OFFLOAD" == "0" ]]; then
    ARGS+=(--no-mmproj-offload)
  fi
fi

USE_DSPARK=0
if [[ "$DSPARK" == "1" && -f "$DSPARK_MODEL" ]]; then
  USE_DSPARK=1
fi

if [[ "$NGRAM" == "1" && "$USE_DSPARK" == "1" ]]; then
  ARGS+=(
    --model-draft "$DSPARK_MODEL"
    --cache-type-k-draft "$DRAFT_KV_TYPE"
    --cache-type-v-draft "$DRAFT_KV_TYPE"
    --spec-type ngram-mod,draft-dspark
    --spec-ngram-mod-n-min "$NGRAM_MIN"
    --spec-ngram-mod-n-max "$NGRAM_MAX"
    --spec-ngram-mod-n-match "$NGRAM_MATCH"
    --spec-draft-n-max "$DSPARK_N"
    --spec-draft-p-min 0
    --spec-draft-sampling "$DSPARK_SAMPLING"
  )
elif [[ "$USE_DSPARK" == "1" ]]; then
  ARGS+=(
    --model-draft "$DSPARK_MODEL"
    --cache-type-k-draft "$DRAFT_KV_TYPE"
    --cache-type-v-draft "$DRAFT_KV_TYPE"
    --spec-type draft-dspark
    --spec-draft-n-max "$DSPARK_N"
    --spec-draft-p-min 0
    --spec-draft-sampling "$DSPARK_SAMPLING"
  )
elif [[ "$NGRAM" == "1" ]]; then
  ARGS+=(
    --spec-type ngram-mod
    --spec-ngram-mod-n-min "$NGRAM_MIN"
    --spec-ngram-mod-n-max "$NGRAM_MAX"
    --spec-ngram-mod-n-match "$NGRAM_MATCH"
  )
fi

if [[ "$BACKEND_SAMPLING" == "1" ]]; then
  ARGS+=(--backend-sampling)
fi

if [[ "$THREADS" != "0" ]]; then
  ARGS+=(--threads "$THREADS" --threads-batch "$THREADS")
  if [[ "$USE_DSPARK" == "1" ]]; then
    ARGS+=(--threads-draft "$THREADS" --threads-batch-draft "$THREADS")
  fi
fi

if [[ "$REASONING_BUDGET" != "-1" ]]; then
  ARGS+=(--reasoning-budget "$REASONING_BUDGET")
  if [[ -n "$REASONING_BUDGET_MESSAGE" ]]; then
    ARGS+=(--reasoning-budget-message "$REASONING_BUDGET_MESSAGE")
  fi
fi

GPU_STATE="$(nvidia-smi --query-gpu=temperature.gpu,power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"

echo "Iniciando Inkling 2 Mini (SM 8.6 Custom Fused CUDA - Wave 13 · 128K) en http://${HOST}:${PORT}"
echo "  Binarios: $BIN_DIR"
echo "  Modelo:   $MODEL"
if [[ -n "$CHAT_TEMPLATE_FILE" && -f "$CHAT_TEMPLATE_FILE" ]]; then
  echo "  Template: $CHAT_TEMPLATE_FILE (O(1) dict)"
fi
if [[ "$VISION" == "1" && -f "$MMPROJ" ]]; then
  echo "  Vision:   habilitada ($MMPROJ, $([[ "$VISION_OFFLOAD" == "0" ]] && echo "CPU 0 MiB VRAM" || echo "GPU"))"
else
  echo "  Vision:   deshabilitada"
fi
if [[ "$NGRAM" == "1" && "$USE_DSPARK" == "1" ]]; then
  echo "  Spec:     GBNF + ngram-mod (n=$NGRAM_MIN..$NGRAM_MAX, match=$NGRAM_MATCH) + DSpark (n=$DSPARK_N, $DSPARK_SAMPLING, KV=$DRAFT_KV_TYPE)"
elif [[ "$USE_DSPARK" == "1" ]]; then
  echo "  Spec:     GBNF + DSpark (n=$DSPARK_N, $DSPARK_SAMPLING, KV=$DRAFT_KV_TYPE)"
elif [[ "$NGRAM" == "1" ]]; then
  echo "  Spec:     GBNF + ngram-mod (n=$NGRAM_MIN..$NGRAM_MAX, match=$NGRAM_MATCH)"
else
  echo "  Spec:     deshabilitado (DSPARK=1 o NGRAM=1 para activar)"
fi
echo "  KV Cache: $KV_TYPE selectivo (8 globales $KV_TYPE + 40 SWA F16) | Cache RAM: ${CACHE_RAM} MiB (ckpt cada ${CHECKPOINT_MIN_STEP} tok)"
echo "  Muestreo: $([[ "$BACKEND_SAMPLING" == "1" ]] && echo "híbrido GPU exacto (probabilístico + gramáticas/tools + reasoning budget)" || echo "en CPU")"
echo "  Contexto: $CTX_SIZE (ubatch=$UBATCH_SIZE) | Reasoning: $REASONING_EFFORT (budget=$REASONING_BUDGET) | Hilos CPU: $([[ "$THREADS" == "0" ]] && echo "por defecto" || echo "$THREADS")"
if [[ -n "$GPU_STATE" ]]; then
  echo "  GPU:      ${GPU_STATE%%,*} C, tope de potencia${GPU_STATE##*,} W (ver ./gpu-watch.sh y ./gpu-power-cap.sh)"
fi
echo ""

exec "$BIN_DIR/llama-server" "${ARGS[@]}" "$@"
