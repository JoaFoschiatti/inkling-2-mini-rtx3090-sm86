#!/usr/bin/env bash
# Monitor térmico de la GPU: temperatura, ventilador, potencia, reloj y motivo de throttling.
# Uso: ./gpu-watch.sh [intervalo_s]      (Ctrl+C para salir)
set -uo pipefail
INT="${1:-1}"

decode_flags() {
  local f=$(( $1 )) out=""
  (( f & 0x1  )) && out+="idle "
  (( f & 0x4  )) && out+="TOPE-POTENCIA "
  (( f & 0x20 )) && out+="THROTTLE-TERMICO(sw) "
  (( f & 0x40 )) && out+="THROTTLE-TERMICO(HW) "
  (( f & 0x8  )) && out+="HW-SLOWDOWN "
  (( f & 0x80 )) && out+="POWER-BRAKE "
  echo "${out:-ok}"
}

printf '%-8s %5s %5s %8s %8s %6s %6s  %s\n' hora "T(C)" "fan%" "P(W)" "tope(W)" "MHz" "util%" estado
while true; do
  IFS=',' read -r t fan p pl clk util fl < <(nvidia-smi \
    --query-gpu=temperature.gpu,fan.speed,power.draw,power.limit,clocks.gr,utilization.gpu,clocks_event_reasons.active \
    --format=csv,noheader,nounits | tr -d ' ')
  printf '%-8s %5s %5s %8s %8s %6s %6s  %s\n' "$(date +%T)" "$t" "$fan" "$p" "$pl" "$clk" "$util" "$(decode_flags "$fl")"
  sleep "$INT"
done
