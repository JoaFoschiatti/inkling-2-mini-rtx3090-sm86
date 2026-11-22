#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR="$([[ -d "$SCRIPT_DIR/bin-sm86" || -d "$SCRIPT_DIR/models" ]] && echo "$SCRIPT_DIR" || (cd "$SCRIPT_DIR/.." && pwd))"
BIN_DIR="${BIN_DIR:-$DIR/bin-sm86}"
MODELS_DIR="$DIR/models"

MODEL="${MODEL:-$MODELS_DIR/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
CTX_SIZE="${CTX_SIZE:-32768}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
GPU_LAYERS="${GPU_LAYERS:-99}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
# Muestreo en la GPU (misma distribución, ~1 ms menos de CPU por token). BACKEND_SAMPLING=0 lo desactiva.
BACKEND_SAMPLING="${BACKEND_SAMPLING:-1}"

export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"

EXTRA=()
if [[ "$BACKEND_SAMPLING" == "1" ]]; then
  EXTRA+=(--backend-sampling)
fi

exec "$BIN_DIR/llama-cli" \
  --model "$MODEL" \
  --jinja \
  --ctx-size "$CTX_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --n-gpu-layers "$GPU_LAYERS" \
  --flash-attn on \
  --temp 1.0 \
  --top-p 1.0 \
  --top-k 0 \
  --min-p 0.0 \
  --chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}" \
  ${EXTRA[@]+"${EXTRA[@]}"} \
  "$@"
