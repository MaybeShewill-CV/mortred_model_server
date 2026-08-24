#!/usr/bin/env bash
# Run ctest against a configured build directory with the vendored shared
# libraries on the loader path (tests link libMNN/libonnxruntime/libnvinfer
# from 3rd_party/libs, which is not on the system library path).
#
# Usage: scripts/run_tests.sh [build-dir] [ctest args...]
#   scripts/run_tests.sh build/full                      # all tests
#   scripts/run_tests.sh build/full -R model_golden_test # subset
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT/build/full}"
if [ $# -gt 0 ]; then
    shift
fi

if [ ! -f "$BUILD_DIR/CTestTestfile.cmake" ]; then
    echo "[ERROR] $BUILD_DIR is not a configured CMake build directory" >&2
    exit 1
fi

export LD_LIBRARY_PATH="$ROOT/3rd_party/libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ctest --test-dir "$BUILD_DIR" "$@"
