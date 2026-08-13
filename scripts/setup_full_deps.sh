#!/usr/bin/env bash
#
# setup_full_deps.sh - 幂等校验/补齐 Mortred full-build 所需的 vendored 第三方依赖。
# 目标平台：Linux（项目唯一支持平台）。
#
# 用法：
#   ./scripts/setup_full_deps.sh
# 或通过环境变量显式指定各依赖源码根目录（脚本只会补齐缺失文件，不会覆盖已有文件）：
#   MNN_ROOT_DIR=... WORKFLOW_ROOT_DIR=... ONNXRUNTIME_ROOT_DIR=... \
#   TENSORRT_ROOT_DIR=... LLAMA_CPP_ROOT_DIR=... FAISS_ROOT_DIR=... \
#   ./scripts/setup_full_deps.sh
#
# 校验范围：
#   MNN / WORKFLOW / ONNXRUNTIME / TensorRT / llama.cpp(ggml) / faiss 的头文件与 .so，
#   以及 CUDA 工具链（nvcc + libcudart）。缺失时给出明确的补齐指引并返回非零退出码。

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
# 任意一个文件存在即通过，否则记录缺失。
ensure_any() {
    local label="$1"
    shift
    for f in "$@"; do
        if [ -e "$f" ]; then
            return 0
        fi
    done
    echo "[ERROR] ${label}: 以下文件均缺失：$*"
    MISSING=1
    return 1
}

# copy_if_missing <source> <destination> <label>
# 目标不存在时才复制（-n 语义 + 显式判断，保证幂等且不覆盖）。
copy_if_missing() {
    local src="$1"
    local dst="$2"
    local label="$3"
    if [ -e "$dst" ]; then
        return 0
    fi
    if [ ! -e "$src" ]; then
        echo "[ERROR] ${label}: 源路径不存在：${src}"
        MISSING=1
        return 1
    fi
    mkdir -p "$(dirname "$dst")"
    cp -rn "$src" "$dst"
    echo "[OK] ${label}: 已补齐 ${dst}"
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
            echo "[OK] ${label}: 已补齐 ${dst}"
        fi
        copied=1
    done
    if [ "$copied" -eq 0 ]; then
        echo "[ERROR] ${label}: 源目录中未找到匹配文件：${src_glob}"
        MISSING=1
        return 1
    fi
    return 0
}

# 从各依赖源码根目录补齐缺失文件的辅助函数。每个函数只处理"源目录可用"的分支；
# 若环境变量未设置或源目录不存在，则由后面的 ensure_any 兜底报错。

setup_mnn() {
    local root="${MNN_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "补齐 MNN（来自 ${root}）"
    copy_if_missing "${root}/include/MNN" "${INCLUDE_DIR}/MNN" "MNN headers"
    copy_glob_if_missing "${root}/build/libMNN*.so*" "${LIB_DIR}" "MNN libs"
}

setup_workflow() {
    local root="${WORKFLOW_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "补齐 WORKFLOW（来自 ${root}）"
    copy_if_missing "${root}/_include/workflow" "${INCLUDE_DIR}/workflow" "WORKFLOW headers"
    copy_glob_if_missing "${root}/_lib/libworkflow.so*" "${LIB_DIR}" "WORKFLOW libs"
}

setup_onnxruntime() {
    local root="${ONNXRUNTIME_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "补齐 ONNXRUNTIME（来自 ${root}）"
    copy_if_missing "${root}/include/onnxruntime" "${INCLUDE_DIR}/onnxruntime" "ONNXRUNTIME headers"
    copy_glob_if_missing "${root}/lib/libonnxruntime*.so*" "${LIB_DIR}" "ONNXRUNTIME libs"
}

setup_tensorrt() {
    local root="${TENSORRT_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "补齐 TensorRT（来自 ${root}）"
    copy_if_missing "${root}/include" "${INCLUDE_DIR}/TensorRT-8.6.1.6" "TensorRT headers"
    copy_glob_if_missing "${root}/lib/libnvinfer*.so*" "${LIB_DIR}" "TensorRT core libs"
    copy_glob_if_missing "${root}/lib/libnvonnxparser*.so*" "${LIB_DIR}" "TensorRT onnx parser libs"
}

setup_llama_cpp() {
    local root="${LLAMA_CPP_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "补齐 llama.cpp / ggml（来自 ${root}）"
    copy_if_missing "${root}/include/llama.h" "${INCLUDE_DIR}/llama_cpp/llama.h" "llama headers"
    copy_glob_if_missing "${root}/include/ggml*.h" "${INCLUDE_DIR}/llama_cpp/" "ggml headers"
    copy_glob_if_missing "${root}/build/src/libllama.so*" "${LIB_DIR}" "llama libs"
    copy_glob_if_missing "${root}/build/ggml/src/libggml*.so*" "${LIB_DIR}" "ggml libs"
}

setup_faiss() {
    local root="${FAISS_ROOT_DIR:-}"
    [ -n "$root" ] && [ -d "$root" ] || return 0
    announce "补齐 faiss（来自 ${root}）"
    copy_if_missing "${root}/include/faiss" "${INCLUDE_DIR}/faiss" "faiss headers"
    copy_glob_if_missing "${root}/lib/libfaiss*.so*" "${LIB_DIR}" "faiss libs"
}

echo "Mortred full-build 依赖校验（缺失时自动尝试从源码根目录补齐）"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"

setup_mnn
setup_workflow
setup_onnxruntime
setup_tensorrt
setup_llama_cpp
setup_faiss

# ---- 存在性校验（无论是否配置了源码根目录都会执行） ----
announce "校验 MNN"
ensure_any "MNN headers" "${INCLUDE_DIR}/MNN/MNNForwardType.h"
ensure_any "MNN libs" "${LIB_DIR}"/libMNN*.so* || true

announce "校验 WORKFLOW"
ensure_any "WORKFLOW headers" "${INCLUDE_DIR}/workflow/CommRequest.h"
ensure_any "WORKFLOW libs" "${LIB_DIR}"/libworkflow*.so* || true

announce "校验 ONNXRUNTIME"
ensure_any "ONNXRUNTIME headers" "${INCLUDE_DIR}/onnxruntime/onnxruntime_cxx_api.h"
ensure_any "ONNXRUNTIME libs" "${LIB_DIR}"/libonnxruntime*.so* || true

announce "校验 TensorRT"
ensure_any "TensorRT headers" "${INCLUDE_DIR}/TensorRT-8.6.1.6/NvInfer.h"
ensure_any "TensorRT core libs" "${LIB_DIR}"/libnvinfer*.so* || true
ensure_any "TensorRT onnx parser libs" "${LIB_DIR}"/libnvonnxparser*.so* || true

announce "校验 llama.cpp / ggml"
ensure_any "llama headers" "${INCLUDE_DIR}/llama_cpp/llama.h"
ensure_any "llama libs" "${LIB_DIR}"/libllama*.so* || true
ensure_any "ggml libs" "${LIB_DIR}"/libggml*.so* || true

announce "校验 faiss"
ensure_any "faiss headers" "${INCLUDE_DIR}/faiss/Index.h"
ensure_any "faiss libs" "${LIB_DIR}"/libfaiss*.so* || true

announce "校验 CUDA 工具链"
if command -v nvcc >/dev/null 2>&1; then
    echo "[OK] nvcc: $(command -v nvcc)"
elif [ -x /usr/local/cuda/bin/nvcc ]; then
    echo "[OK] nvcc: /usr/local/cuda/bin/nvcc"
else
    echo "[ERROR] CUDA: 未找到 nvcc（请安装 CUDA Toolkit，或将 nvcc 加入 PATH）"
    MISSING=1
fi
ensure_any "CUDA runtime (libcudart)" \
    "${LIB_DIR}"/libcudart*.so* \
    /usr/local/cuda/lib64/libcudart.so* || true

echo ""
if [ "$MISSING" -ne 0 ]; then
    echo "===== 结论：仍有依赖缺失 ====="
    echo "请通过环境变量提供对应源码根目录后重试，例如："
    echo "  MNN_ROOT_DIR=/path/to/MNN \\"
    echo "  WORKFLOW_ROOT_DIR=/path/to/workflow \\"
    echo "  ONNXRUNTIME_ROOT_DIR=/path/to/onnxruntime \\"
    echo "  TENSORRT_ROOT_DIR=/path/to/TensorRT-8.6.1.6 \\"
    echo "  LLAMA_CPP_ROOT_DIR=/path/to/llama.cpp \\"
    echo "  FAISS_ROOT_DIR=/path/to/faiss \\"
    echo "  ./scripts/setup_full_deps.sh"
    exit 1
fi

echo "===== 结论：全部依赖就绪，可以执行 full build ====="
