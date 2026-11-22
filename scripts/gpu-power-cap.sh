#!/usr/bin/env bash
# Tope de potencia de la GPU. Baja temperatura y ruido con poca pérdida de t/s,
# porque el decode está limitado por memoria y no por reloj.
#
#   ./gpu-power-cap.sh            muestra el tope actual y los límites de la placa
#   sudo ./gpu-power-cap.sh 280   fija 280 W (se pierde al reiniciar)
#   sudo ./gpu-power-cap.sh reset vuelve al valor de fábrica
set -euo pipefail

read -r CUR DEF MIN MAX < <(nvidia-smi --query-gpu=power.limit,power.default_limit,power.min_limit,power.max_limit \
  --format=csv,noheader,nounits | tr -d ',')

if [[ $# -eq 0 ]]; then
  echo "Tope actual: ${CUR} W  (fábrica ${DEF} W, rango permitido ${MIN}-${MAX} W)"
  exit 0
fi

if [[ "$1" == "reset" ]]; then
  TARGET="${DEF%.*}"
else
  TARGET="$1"
  if ! [[ "$TARGET" =~ ^[0-9]+$ ]] || (( TARGET < ${MIN%.*} || TARGET > ${MAX%.*} )); then
    echo "Valor fuera de rango: usar entre ${MIN%.*} y ${MAX%.*} W" >&2
    exit 1
  fi
fi

if [[ $EUID -ne 0 ]]; then
  echo "Hace falta root: sudo $0 $1" >&2
  exit 1
fi

nvidia-smi -pl "$TARGET"
