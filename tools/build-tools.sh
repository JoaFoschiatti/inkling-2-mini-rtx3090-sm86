#!/usr/bin/env bash
# Compila inkcheck y, si se pide, la referencia sin modificar (commit base de src/llama.cpp).
#   ./build-tools.sh          -> tools/inkcheck
#   ./build-tools.sh ref      -> además tools/ref/build/bin (tarda ~5 min)
set -euo pipefail
TOOLS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$TOOLS")"
S="$ROOT/src/llama.cpp"
E="$ROOT/toolchain/env"
export PATH="$E/bin:$PATH"

"$E/bin/x86_64-conda-linux-gnu-g++" -O2 -std=c++17 -Wno-deprecated-declarations \
  -I"$S/include" -I"$S/ggml/include" "$TOOLS/inkcheck.cpp" -o "$TOOLS/inkcheck" \
  -L"$ROOT/bin-sm86" -l:libllama.so.1 -l:libggml.so.1 -l:libggml-base.so.1 \
  -Wl,--unresolved-symbols=ignore-in-shared-libs -Wl,--enable-new-dtags
echo "ok: $TOOLS/inkcheck"

if [[ "${1:-}" == "ref" ]]; then
  REF="$TOOLS/ref"
  rm -rf "$REF/src" && mkdir -p "$REF/src"
  git -C "$S" archive HEAD | tar -x -C "$REF/src"
  # El RPATH va por CMake: escrito a mano en los flags, $ORIGIN llega vacío al enlazador y deja el directorio actual en la ruta de búsqueda.
  LF="-Wl,-rpath-link,$E/lib:$E/targets/x86_64-linux/lib"
  cmake -S "$REF/src" -B "$REF/build" -G Ninja \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON "-DCMAKE_INSTALL_RPATH=\$ORIGIN;$E/lib;$E/targets/x86_64-linux/lib" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_CUDA_F16=ON -DGGML_NATIVE=ON \
    -DCMAKE_CUDA_FLAGS="--use_fast_math -Xptxas=-O3" \
    -DCMAKE_C_COMPILER="$E/bin/x86_64-conda-linux-gnu-gcc" -DCMAKE_CXX_COMPILER="$E/bin/x86_64-conda-linux-gnu-g++" \
    -DCMAKE_CUDA_COMPILER="$E/bin/nvcc" -DCMAKE_CUDA_HOST_COMPILER="$E/bin/x86_64-conda-linux-gnu-g++" \
    -DCMAKE_EXE_LINKER_FLAGS="$LF" -DCMAKE_SHARED_LINKER_FLAGS="$LF" \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_UI=OFF
  ninja -C "$REF/build" llama-bench
  echo "ok: $REF/build/bin"
fi
