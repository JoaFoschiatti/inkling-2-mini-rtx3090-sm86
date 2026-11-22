#!/usr/bin/env bash
# A/B intercalado de llama-bench con temperatura inicial igualada (espera a que la GPU baje a T0 antes de cada corrida).
# Uso: ab-bench.sh <rondas> <T0_C> "<args de llama-bench>" etiqueta=bindir[,VAR=valor...] ...
# Ej.: ./ab-bench.sh 3 55 "-p 0 -n 256 -r 2" actual=../bin-sm86 ref=ref/build/bin
#      ./ab-bench.sh 3 55 "-p 0 -n 256 -r 2" con=../bin-sm86 sin=../bin-sm86,W6_NO_HORIZ=1
# Imprime cada corrida y al final mediana [min..max] por etiqueta. Solo cabe una instancia del modelo en VRAM: parar el servidor antes.
set -uo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TOOLS")"
MODEL="${MODEL:-$ROOT/models/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
ROUNDS="$1"; T0="$2"; BARGS="$3"; shift 3
cool() {
  for _ in $(seq 1 240); do
    t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    [[ "$t" -le "$T0" ]] && return 0
    sleep 1
  done
}
declare -A RES
for r in $(seq 1 "$ROUNDS"); do
  for spec in "$@"; do
    label="${spec%%=*}"; rest="${spec#*=}"
    IFS=',' read -r -a parts <<< "$rest"
    bindir="$(cd "${parts[0]}" && pwd)"; envs=("${parts[@]:1}")
    cool
    out=$(env "${envs[@]}" LD_LIBRARY_PATH="$bindir" "$bindir/llama-bench" -m "$MODEL" -ngl 99 -fa on $BARGS -o jsonl 2>/dev/null | python3 -c '
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line.startswith("{"): continue
    d = json.loads(line)
    kind = "pp" if d["n_gen"] == 0 else "tg"
    print("%s%d=%.2f" % (kind, d["n_prompt"] if kind == "pp" else d["n_gen"], d["avg_ts"]), end=" ")
')
    tend=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
    echo "ronda $r  $label: $out (T fin ${tend}C)"
    RES[$label]+="$out|"
  done
done
python3 - <<PY
import statistics as st
res = {}
raw = """$(for k in "${!RES[@]}"; do echo "$k ${RES[$k]}"; done)"""
for line in raw.strip().splitlines():
    label, rest = line.split(" ", 1)
    for chunk in rest.split("|"):
        for kv in chunk.split():
            k, v = kv.split("=")
            res.setdefault(label, {}).setdefault(k, []).append(float(v))
print("--- resumen (mediana [min..max]) ---")
for label in sorted(res):
    print("  %-14s" % label, "  ".join("%s: %.2f [%.2f..%.2f]" % (k, st.median(v), min(v), max(v)) for k, v in sorted(res[label].items())))
PY
