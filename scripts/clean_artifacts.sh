#!/usr/bin/env bash
#
# clean_artifacts.sh - remove generated/build/runtime artifacts from the source tree.
#
# This script is intended to be run by a developer or CI before packaging/submitting.
# It removes:
#   - _bin, _lib
#   - all build-* / cmake-build-* directories
#   - logs
#   - Web Console backend build directory
#
# It does NOT remove downloaded model weights under weights/ because they may be
# intentionally kept locally, but they are already ignored by .gitignore.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TARGETS=(
  "_bin"
  "_lib"
  "build"
  "build-ci"
  "build-werror"
  "build-gate"
  "build-tidy"
  "cmake-build-debug"
  "cmake-build-release"
  "logs"
  "src/apps/web_console/backend/build"
)

for target in "${TARGETS[@]}"; do
  if [ -e "$target" ]; then
    echo "Removing $target"
    rm -rf "$target"
  fi
done

echo "Artifacts cleaned."
