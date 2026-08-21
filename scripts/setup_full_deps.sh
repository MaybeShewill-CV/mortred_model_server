#!/usr/bin/env bash
#
# setup_full_deps.sh - idempotently verify/fill in the vendored third-party deps needed for a Mortred full build.
# Target platform: Linux (the project's only supported platform).
#
# Usage:
#   ./scripts/setup_full_deps.sh
# or explicitly point at each dependency's source root via env vars (the script only fills in missing files, never overwrites existing ones):
#   MNN_ROOT_DIR=... WORKFLOW_ROOT_DIR=... ONNXRUNTIME_ROOT_DIR=... \
#   TENSORRT_ROOT_DIR=... \
#   ./scripts/setup_full_deps.sh
#
# Verification scope:
#   MNN / WORKFLOW / ONNXRUNTIME / TensorRT headers and .so,
#   plus the CUDA toolchain (nvcc + libcudart). Missing items produce clear fill-in guidance and a non-zero exit code.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="${PROJECT_ROOT}/3rd_party/include"
LIB_DIR="${PROJECT_ROOT}/3rd_party/libs"
mkdir -p "${INCLUDE_DIR}" "${LIB_DIR}"

MISSING=0

announce() {
    echo ""
    echo "==> $*"
}

# ensure_any <label> <file>...
# Passes if any of the files exists; otherwise records the missing items.
ensure_any() {
    local label="$1"
    shift
    for f in "$@"; do
        if [ -e "$f" ]; then
            return 0
        fi
    done
    echo "[ERROR] ${label}: all of the following files are missing: $*"
    MISSING=1
    return 1
}

# copy_if_missing <source> <destination> <label>
# Copies only when the destination is missing (-n semantics plus an explicit check; idempotent, never overwrites).
copy_if_missing() {
    local src="$1"
    local dst="$2"
    local label="$3"
    if [ -e "$dst" ]; then
        return 0
    fi
    if [ ! -e "$src" ]; then
        echo "[ERROR] ${label}: source path does not exist: ${src}"
        MISSING=1
        return 1
    fi
    mkdir -p "$(dirname "$dst")"
    cp -rn "$src" "$dst"
    echo "[OK] ${label}: filled in ${dst}"
}

copy_glob_if_missing() {
    local src_glob="$1"
    local dst_dir="$2"
    local label="$3"
    mkdir -p "$dst_dir"
    local copied=0
    for src in ${src_glob}; do
        [ -e "$src" ] || continue
        local dst="${dst_dir}/$(basename "$src")"
        if [ ! -e "$dst" ]; then
            cp -n "$src" "$dst"
            echo "[OK] ${label}: filled in ${dst}"
        fi
        copied=1
    done
    if [ "$copied" -eq 0 ]; then
        echo "[ERROR] ${label}: no matching files found in the source dir: ${src_glob}"
        MISSING=1
        return 1
    fi
    return 0
}

# Helpers that fill in missing files from each dependency's source root. Each function only handles the
# "source dir available" branch; if an env var is unset or the source dir is missing, the later ensure_any reports it.

setup_mnn() {
    local root="${MNN_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "filling in MNN (from ${root})"
    copy_if_missing "${root}/include/MNN" "${INCLUDE_DIR}/MNN" "MNN headers"
    copy_glob_if_missing "${root}/build/libMNN*.so*" "${LIB_DIR}" "MNN libs"
}

setup_workflow() {
    local root="${WORKFLOW_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "filling in WORKFLOW (from ${root})"
    copy_if_missing "${root}/_include/workflow" "${INCLUDE_DIR}/workflow" "WORKFLOW headers"
    copy_glob_if_missing "${root}/_lib/libworkflow.so*" "${LIB_DIR}" "WORKFLOW libs"
}

setup_onnxruntime() {
    local root="${ONNXRUNTIME_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "filling in ONNXRUNTIME (from ${root})"
    copy_if_missing "${root}/include/onnxruntime" "${INCLUDE_DIR}/onnxruntime" "ONNXRUNTIME headers"
    copy_glob_if_missing "${root}/lib/libonnxruntime*.so*" "${LIB_DIR}" "ONNXRUNTIME libs"
}

setup_tensorrt() {
    local root="${TENSORRT_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "filling in TensorRT (from ${root})"
    copy_if_missing "${root}/include" "${INCLUDE_DIR}/TensorRT-8.6.1.6" "TensorRT headers"
    copy_glob_if_missing "${root}/lib/libnvinfer*.so*" "${LIB_DIR}" "TensorRT core libs"
    copy_glob_if_missing "${root}/lib/libnvonnxparser*.so*" "${LIB_DIR}" "TensorRT onnx parser libs"
}

echo "Mortred full-build dependency check (automatically tries to fill in missing items from the source roots)"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

setup_mnn
setup_workflow
setup_onnxruntime
setup_tensorrt

# ---- existence checks (always executed, whether or not source roots are configured) ----
announce "checking MNN"
ensure_any "MNN headers" "${INCLUDE_DIR}/MNN/MNNForwardType.h"
ensure_any "MNN libs" "${LIB_DIR}"/libMNN*.so* || true

announce "checking WORKFLOW"
ensure_any "WORKFLOW headers" "${INCLUDE_DIR}/workflow/CommRequest.h"
ensure_any "WORKFLOW libs" "${LIB_DIR}"/libworkflow*.so* || true

announce "checking ONNXRUNTIME"
ensure_any "ONNXRUNTIME headers" "${INCLUDE_DIR}/onnxruntime/onnxruntime_cxx_api.h"
ensure_any "ONNXRUNTIME libs" "${LIB_DIR}"/libonnxruntime*.so* || true

announce "checking TensorRT"
ensure_any "TensorRT headers" "${INCLUDE_DIR}/TensorRT-8.6.1.6/NvInfer.h"
ensure_any "TensorRT core libs" "${LIB_DIR}"/libnvinfer*.so* || true
ensure_any "TensorRT onnx parser libs" "${LIB_DIR}"/libnvonnxparser*.so* || true

announce "checking CUDA toolchain"
if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] nvcc: $(command -v nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    echo "[OK] nvcc: /usr/local/cuda/bin/nvcc"
else
    echo "[ERROR] CUDA: nvcc not found (install the CUDA Toolkit, or add nvcc to PATH)"
    MISSING=1
fi
ensure_any "CUDA runtime (libcudart)" \
    "${LIB_DIR}"/libcudart*.so* \
    /usr/local/cuda/lib64/libcudart.so* || true

echo ""
if [ "$MISSING" -ne 0 ]; then
    echo "===== result: some dependencies are still missing ====="
    echo "please provide the matching source root via env vars and retry, e.g.:"
    echo "  MNN_ROOT_DIR=/path/to/MNN \\"
    echo "  WORKFLOW_ROOT_DIR=/path/to/workflow \\"
    echo "  ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime \\"
    echo "  TENSORRT_ROOT_DIR=/path/to/TensorRT-8.6.1.6 \\"
    echo "  ./scripts/setup_full_deps.sh"
    exit 1
fi

echo "===== result: all dependencies ready; full build can proceed ====="
