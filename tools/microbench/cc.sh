#!/usr/bin/env bash
# Compila un micro-benchmark CUDA suelto con el toolchain del proyecto (el gcc del sistema es demasiado nuevo para nvcc 12.8).
# Uso: ./cc.sh fuente.cu salida   &&   ./salida
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
E="$ROOT/toolchain/env"
export PATH="$E/bin:$PATH"
nvcc -O3 -arch=sm_86 --use_fast_math -Xptxas=-O3 -ccbin "$E/bin/x86_64-conda-linux-gnu-g++" -o "$2" "$1" \
  -L"$E/lib" -L"$E/targets/x86_64-linux/lib" -Xlinker -rpath,"$E/lib:$E/targets/x86_64-linux/lib"
