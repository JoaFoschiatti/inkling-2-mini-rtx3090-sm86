#!/usr/bin/env bash
# Compara la precisión de un build contra la referencia sin modificar (teacher forcing, 1 token por paso).
#   ./validate.sh [bindir_a_probar] [n_tokens]
# Requiere ./build-tools.sh ref. Los volcados ocupan ~0,8 MB por token en TMPDIR (por defecto /tmp).
set -euo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TOOLS")"
BIN="${1:-$ROOT/bin-sm86}"
N="${2:-1024}"
MODEL="${MODEL:-$ROOT/models/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
REF="$TOOLS/ref/build/bin"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/inkcheck.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

# texto de prueba: documentación del propio árbol de fuentes
cat "$ROOT/README.md" "$ROOT/src/llama.cpp/docs/build.md" "$ROOT/src/llama.cpp/README.md" "$ROOT/src/llama.cpp/tools/server/README.md" > "$OUT/corpus.txt"

run() { # <bindir> <salida> <args...>
  local bin="$1" out="$2"; shift 2
  LD_LIBRARY_PATH="$bin" "$TOOLS/inkcheck" dump --model "$MODEL" --text "$OUT/corpus.txt" --out "$out" "$@" 2>&1 | grep -E "^\[inkcheck\]|failed|error" || true
}
cmp() { LD_LIBRARY_PATH="$BIN" "$TOOLS/inkcheck" cmp "$1" "$2" | grep -v "fila "; }

echo "== referencia, 1 token por paso";   run "$REF" "$OUT/ref.bin"   --ctx 4096 --prefill 0 --decode "$N"
echo "== referencia, lotes de 512";       run "$REF" "$OUT/ref_b.bin" --ctx 4096 --prefill "$N" --prefill-rows "$N" --ubatch 512 --decode 0
echo "== build a probar ($BIN)";          run "$BIN" "$OUT/new.bin"   --ctx 4096 --prefill 0 --decode "$N"
echo
echo "--- piso de ruido de la propia referencia (1 token vs lotes de 512) ---"; cmp "$OUT/ref.bin" "$OUT/ref_b.bin"
echo "--- referencia vs build a probar ---";                                     cmp "$OUT/ref.bin" "$OUT/new.bin"
echo
echo "Criterio: la KLD y el % de top-1 del build deben quedar al nivel del piso de ruido, y el z de la NLL dentro de +-3."
