#!/usr/bin/env bash
set -euo pipefail

# El script vive en scripts/; un enlace simbólico en la raíz (./chat.sh) también sirve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$([[ -d "$SCRIPT_DIR/bin-sm86" || -d "$SCRIPT_DIR/models" ]] && echo "$SCRIPT_DIR" || (cd "$SCRIPT_DIR/.." && pwd))"
BIN_DIR="${BIN_DIR:-$DIR/bin-sm86}"
MODELS_DIR="$DIR/models"

MODEL="${MODEL:-$MODELS_DIR/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
CTX_SIZE="${CTX_SIZE:-32768}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
GPU_LAYERS="${GPU_LAYERS:-99}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
# Caché KV: f16 por defecto (a 32K cabe sobrada). KV_TYPE=q8_0 cuantiza solo las 8 capas globales, como el servidor.
KV_TYPE="${KV_TYPE:-f16}"
# Muestreo en la GPU (misma distribución, ~1 ms menos de CPU por token). BACKEND_SAMPLING=0 lo desactiva.
BACKEND_SAMPLING="${BACKEND_SAMPLING:-1}"
# Un hilo de CPU en el bucle de decode, como el servidor (los demás girarían en vacío). THREADS=0 deja el valor de llama.cpp.
THREADS="${THREADS:-1}"
# Misma plantilla que el servidor. CHAT_TEMPLATE_FILE= (vacío) usa la del GGUF.
CHAT_TEMPLATE_FILE="${CHAT_TEMPLATE_FILE-$DIR/templates/chat-template-inkling-o1.jinja}"

# Los binarios llevan RPATH $ORIGIN: esto solo cubre binarios antiguos.
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

EXTRA=()
if [[ "$BACKEND_SAMPLING" == "1" ]]; then
  EXTRA+=(--backend-sampling)
fi
if [[ "$THREADS" != "0" ]]; then
  EXTRA+=(--threads "$THREADS" --threads-batch "$THREADS")
fi
if [[ -n "$CHAT_TEMPLATE_FILE" && -f "$CHAT_TEMPLATE_FILE" ]]; then
  EXTRA+=(--chat-template-file "$CHAT_TEMPLATE_FILE")
fi

exec "$BIN_DIR/llama-cli" \
  --model "$MODEL" \
  --jinja \
  --ctx-size "$CTX_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --n-gpu-layers "$GPU_LAYERS" \
  --flash-attn on \
  --cache-type-k "$KV_TYPE" \
  --cache-type-v "$KV_TYPE" \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --min-p 0.0 \
  --chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
