#!/usr/bin/env bash
# Reproducible build script for Inkling-2-Mini Custom SM 8.6 CUDA (Waves 1–14).
# Fetches unslothai/llama.cpp at base commit 8aa2f23 (head of pull request #390), applies
# patches/inkling2-sm86-wave14.patch, builds SM 8.6 CUDA targets, and copies
# binaries + shared libraries into bin-sm86/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/unslothai/llama.cpp.git}"
# 8aa2f23 is the head of pull request #390, not a branch: it is fetched by commit, then by pull ref.
UPSTREAM_REF="${UPSTREAM_REF:-refs/pull/390/head}"
BASE_COMMIT="8aa2f23f6bbe0b2d23eb3ff676f318f96111d591"
PATCH_FILE="$REPO_ROOT/patches/inkling2-sm86-wave14.patch"
SRC_DIR="${SRC_DIR:-$REPO_ROOT/src/llama.cpp}"
BUILD_DIR="${BUILD_DIR:-$SRC_DIR/build-sm86}"
BIN_OUT="${BIN_OUT:-$REPO_ROOT/bin-sm86}"
CUDA_ARCH="${CUDA_ARCH:-86}"

echo "==> 1. Fetching upstream unslothai/llama.cpp (${UPSTREAM_REF} @ ${BASE_COMMIT:0:7})"
if [[ ! -d "$SRC_DIR/.git" ]]; then
  git init -q "$SRC_DIR"
  git -C "$SRC_DIR" remote add origin "$UPSTREAM_URL"
fi
# A source tree with local changes is never reset: it may be a development tree (it lives at the
# default SRC_DIR). If it is exactly the base commit plus this patch, it is built as it is.
PATCHED=0
if [[ -n "$(git -C "$SRC_DIR" status --porcelain)" ]]; then
  if [[ "$(git -C "$SRC_DIR" rev-parse HEAD 2>/dev/null)" == "$BASE_COMMIT" ]] &&
     git -C "$SRC_DIR" apply --reverse --check "$PATCH_FILE" 2>/dev/null; then
    echo "    $SRC_DIR already carries the Wave 14 patch: building it as it is"
    PATCHED=1
  else
    echo "error: $SRC_DIR has local changes that are not this patch; resetting it would discard them." >&2
    echo "       Point SRC_DIR at a fresh directory (and BIN_OUT elsewhere if bin-sm86/ is in use)." >&2
    exit 1
  fi
fi

if [[ "$PATCHED" == "0" ]]; then
  if ! git -C "$SRC_DIR" cat-file -e "$BASE_COMMIT^{commit}" 2>/dev/null; then
    git -C "$SRC_DIR" fetch --depth 1 origin "$BASE_COMMIT" ||
      git -C "$SRC_DIR" fetch --depth 1 origin "$UPSTREAM_REF"
  fi
  git -C "$SRC_DIR" checkout --force --detach "$BASE_COMMIT"

  echo "==> 2. Applying Wave 14 SM 8.6 patch"
  git -C "$SRC_DIR" apply --check "$PATCH_FILE"
  git -C "$SRC_DIR" apply "$PATCH_FILE"
fi

echo "==> 3. Configuring CMake (CUDA SM ${CUDA_ARCH}, shared libs, Release)"
# RPATH $ORIGIN: every binary folder loads its own libraries. With CMake's default build RPATH the
# copies in bin-sm86/ would keep loading the libraries of the build tree.
CMAKE_ARGS=(
  -S "$SRC_DIR"
  -B "$BUILD_DIR"
  -G Ninja
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_SHARED_LIBS=ON
  -DGGML_CUDA=ON
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"
  -DGGML_NATIVE=ON
  -DCMAKE_CUDA_FLAGS="--use_fast_math -Xptxas=-O3"
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  "-DCMAKE_INSTALL_RPATH=\$ORIGIN"
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
