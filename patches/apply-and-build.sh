#!/usr/bin/env bash
# Reproducible build script for Inkling-2-Mini Custom SM 8.6 CUDA (Wave 10).
# Clones unslothai/llama.cpp at base commit 8aa2f23 (branch pr390), applies
# patches/inkling2-sm86-wave10.patch, builds SM 8.6 CUDA targets, and copies
# binaries + shared libraries into bin-sm86/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/unslothai/llama.cpp.git}"
UPSTREAM_BRANCH="${UPSTREAM_BRANCH:-pr390}"
BASE_COMMIT="8aa2f23f6bbe0b2d23eb3ff676f318f96111d591"
PATCH_FILE="$REPO_ROOT/patches/inkling2-sm86-wave10.patch"
SRC_DIR="${SRC_DIR:-$REPO_ROOT/src/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build-sm86}"
BIN_OUT="${BIN_OUT:-$REPO_ROOT/bin-sm86}"
CUDA_ARCH="${CUDA_ARCH:-86}"

echo "==> 1. Fetching upstream unslothai/llama.cpp (${UPSTREAM_BRANCH} @ ${BASE_COMMIT:0:7})"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  mkdir -p "$(dirname "$SRC_DIR")"
  git clone --branch "$UPSTREAM_BRANCH" --single-branch "$UPSTREAM_URL" "$SRC_DIR"
fi
git -C "$SRC_DIR" fetch origin "$BASE_COMMIT" --depth 1 2>/dev/null || true
git -C "$SRC_DIR" checkout --force "$BASE_COMMIT"

echo "==> 2. Applying Wave 10 SM 8.6 patch"
git -C "$SRC_DIR" apply --check "$PATCH_FILE"
git -C "$SRC_DIR" apply "$PATCH_FILE"

echo "==> 3. Configuring CMake (CUDA SM ${CUDA_ARCH}, shared libs, Release)"
CMAKE_ARGS=(
  -S "$SRC_DIR"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  -DGGML_CUDA_F16=ON
  -DGGML_NATIVE=ON
  -DCMAKE_CUDA_FLAGS="--use_fast_math -Xptxas=-O3"
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
)
if [[ -n "${CUDA_HOST_COMPILER:-}" ]]; then
  CMAKE_ARGS+=(-DCMAKE_CUDA_HOST_COMPILER="$CUDA_HOST_COMPILER")
fi
cmake "${CMAKE_ARGS[@]}"

echo "==> 4. Building llama-server, llama-cli, llama-bench, llama-perplexity, llama-completion"
ninja -C "$BUILD_DIR" llama-server llama-cli llama-bench llama-perplexity llama-completion

echo "==> 5. Installing built artifacts into $BIN_OUT"
mkdir -p "$BIN_OUT"
cp -a "$BUILD_DIR/bin"/lib*.so* \
      "$BUILD_DIR/bin"/llama-server \
      "$BUILD_DIR/bin"/llama-cli \
      "$BUILD_DIR/bin"/llama-bench \
      "$BUILD_DIR/bin"/llama-perplexity \
      "$BUILD_DIR/bin"/llama-completion \
      "$BIN_OUT/"

echo "Done! Binaries ready in $BIN_OUT"
