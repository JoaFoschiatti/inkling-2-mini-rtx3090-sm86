#!/usr/bin/env bash
# Regenerates the published patch and the browsable kernel copy from the development tree.
#   patches/inkling2-sm86-wave<N>.patch   working-tree diff of src/llama.cpp over the base commit,
#                                         untracked files (inkling-fused.cuh) included
#   kernels/inkling-fused.cuh             copy of ggml/src/ggml-cuda/inkling-fused.cuh
# Then proves that the patch applied to the base commit gives exactly the working tree.
# Usage: tools/export-patch.sh <wave>          e.g. tools/export-patch.sh 14
# Nothing in src/llama.cpp is modified: the diff goes through a scratch index, not the real one.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC_DIR:-$ROOT/src/llama.cpp}"
BASE_COMMIT="8aa2f23f6bbe0b2d23eb3ff676f318f96111d591"
KERNEL="ggml/src/ggml-cuda/inkling-fused.cuh"
WAVE="${1:?usage: tools/export-patch.sh <wave>}"
OUT="$ROOT/patches/inkling2-sm86-wave$WAVE.patch"
IDX="$(git -C "$SRC" rev-parse --absolute-git-dir)/inkling-export.index"

if [[ "$(git -C "$SRC" rev-parse HEAD)" != "$BASE_COMMIT" ]]; then
  echo "error: HEAD of $SRC is not the base commit ${BASE_COMMIT:0:7}" >&2
  exit 1
fi

# Scratch index = HEAD plus every untracked, non-ignored file as intent-to-add.
GIT_INDEX_FILE="$IDX" git -C "$SRC" read-tree HEAD
git -C "$SRC" ls-files --others --exclude-standard -z |
  GIT_INDEX_FILE="$IDX" xargs -0 -r git -C "$SRC" add -N --
GIT_INDEX_FILE="$IDX" git -C "$SRC" diff --binary HEAD > "$OUT"
cp "$SRC/$KERNEL" "$ROOT/kernels/inkling-fused.cuh"

# Check: base commit + patch must equal the working tree, file by file.
GIT_INDEX_FILE="$IDX" git -C "$SRC" read-tree "$BASE_COMMIT"
GIT_INDEX_FILE="$IDX" git -C "$SRC" apply --cached "$OUT"
if ! GIT_INDEX_FILE="$IDX" git -C "$SRC" diff --quiet; then
  echo "error: the patch does not reproduce the working tree:" >&2
  GIT_INDEX_FILE="$IDX" git -C "$SRC" diff --stat >&2
  exit 1
fi
if [[ -n "$(git -C "$SRC" ls-files --others --exclude-standard | grep -vxF "$KERNEL" || true)" ]]; then
  echo "note: untracked files other than $KERNEL went into the patch:"
  git -C "$SRC" ls-files --others --exclude-standard | grep -vxF "$KERNEL"
fi

echo "wrote $OUT"
echo "  $(grep -c '^diff --git' "$OUT") files, $(wc -l < "$OUT") lines, sha256 $(sha256sum "$OUT" | cut -c1-16)"
echo "  verified: ${BASE_COMMIT:0:7} + patch == working tree of $SRC"
echo "remember to point patches/apply-and-build.sh and the READMEs at wave $WAVE and to drop the previous patch"
