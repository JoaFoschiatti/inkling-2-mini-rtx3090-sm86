#!/usr/bin/env bash
# Decode sostenido con telemetría térmica: t/s por tramo, temperatura, potencia, reloj y % de tiempo en throttling.
# Sirve para comparar topes de potencia:  sudo ../gpu-power-cap.sh 280 && ./sustained-bench.sh
#   ./sustained-bench.sh [tramos=7] [tokens_por_tramo=2048]     (7 x 2048 tokens son ~2 minutos)
# Corta sola si la GPU pasa de TMAX (por defecto 89 °C) o aparece throttling térmico por hardware.
set -uo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TOOLS")"
BIN="${BIN_DIR:-$ROOT/bin-sm86}"
MODEL="${MODEL:-$ROOT/models/Inkling-2-Mini-UD-Q4_K_XL.gguf}"
REPS="${1:-7}"
NGEN="${2:-2048}"
TMAX="${TMAX:-89}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/sustained.XXXXXX")"
trap 'kill "${MON:-0}" "${BP:-0}" 2>/dev/null; rm -rf "$OUT"' EXIT

echo "Tope de potencia: $(nvidia-smi --query-gpu=power.limit --format=csv,noheader) | temperatura inicial: $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader) C"

nvidia-smi --query-gpu=timestamp,temperature.gpu,fan.speed,power.draw,clocks.gr,clocks_event_reasons.active,utilization.gpu \
  --format=csv,noheader,nounits -lms 250 > "$OUT/mon.csv" & MON=$!
LD_LIBRARY_PATH="$BIN" "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa on -p 0 -n "$NGEN" -r "$REPS" -o jsonl > "$OUT/bench.jsonl" 2>/dev/null & BP=$!

while kill -0 "$BP" 2>/dev/null; do
  read -r t fl < <(nvidia-smi --query-gpu=temperature.gpu,clocks_event_reasons.active --format=csv,noheader,nounits | tr -d ',')
  if [[ "$t" -ge "$TMAX" ]] || (( (fl & 0x40) != 0 )); then
    echo "!! corte de seguridad: ${t} C, flags ${fl}"
    kill "$BP"
    break
  fi
  sleep 1
done
wait "$BP" 2>/dev/null
kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null

python3 - "$OUT/bench.jsonl" "$OUT/mon.csv" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    if line.startswith('{'):
        d = json.loads(line)
        s = d["samples_ts"]
        print("t/s por tramo: " + " ".join("%.1f" % x for x in s))
        print("ráfaga (1er tramo): %.1f t/s | sostenido (últimos 3): %.1f t/s" % (s[0], sum(s[-3:]) / len(s[-3:])))
rows = []
for line in open(sys.argv[2]):
    p = [x.strip() for x in line.split(',')]
    if len(p) >= 7:
        try:
            rows.append((float(p[1]), float(p[2]), float(p[3]), float(p[4]), int(p[5], 16), float(p[6])))
        except ValueError:
            pass
busy = [r for r in rows if r[5] > 50]
if busy:
    n = len(busy)
    tail = busy[-max(1, n // 3):]
    print("GPU: T máx %.0f C | ventilador máx %.0f %% | potencia media %.0f W (último tercio %.0f W) | reloj medio %.0f MHz (último tercio %.0f MHz)" % (
        max(r[0] for r in busy), max(r[1] for r in busy), sum(r[2] for r in busy) / n, sum(r[2] for r in tail) / len(tail),
        sum(r[3] for r in busy) / n, sum(r[3] for r in tail) / len(tail)))
    print("tiempo en throttling térmico: %.0f %% | en tope de potencia: %.0f %%" % (
        100.0 * sum(1 for r in busy if r[4] & 0x60) / n, 100.0 * sum(1 for r in busy if r[4] & 0x4) / n))
PY
