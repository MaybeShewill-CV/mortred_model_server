#!/usr/bin/env bash
# install_deps.sh - 一键构建/安装 Mortred 全部第三方依赖到 3rd_party/{include,libs}
#
# 目标：替代"手动编译第三方库 + 手工拷贝进 3rd_party"的流程。默认对齐本仓库
# 已验证的 CUDA 11 / TensorRT 8.6 基线线；`--cuda-version 12` 切换到
# CUDA 12 / TensorRT 10 线（注意：TRT 10 需要先完成源码迁移，见
# docs/deployment-and-deps-plan.md 工作包 P0）。
#
# 用法:
#   ./scripts/install_deps.sh --check            # 校验 3rd_party 完整性并打印版本
#   ./scripts/install_deps.sh --all              # 安装全部（默认）
#   ./scripts/install_deps.sh --workflow         # 仅构建安装 workflow
#   ./scripts/install_deps.sh --mnn              # 仅构建安装 MNN
#   ./scripts/install_deps.sh --onnxruntime      # 仅下载安装 onnxruntime
#   ./scripts/install_deps.sh --nvidia           # CUDA/TensorRT/cuDNN（需 root + NVIDIA apt）
#   ./scripts/install_deps.sh --cuda-version 12  # 切换到 CUDA 12 / TRT 10 线
#   ./scripts/install_deps.sh --offline DIR      # 使用预下载包目录（离线）
#
# 幂等：每个依赖安装成功后在 3rd_party/.install-stamp/ 留下 stamp；重跑自动跳过。

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="$ROOT/3rd_party/include"
LIB_DIR="$ROOT/3rd_party/libs"
STAMP_DIR="$ROOT/3rd_party/.install-stamp"
BUILD_DIR="${MORTRED_DEPS_BUILD_DIR:-$ROOT/.deps-build}"

# ---- 版本矩阵（两条线）----
CUDA_VERSION="${CUDA_VERSION:-11}"
if [ "$CUDA_VERSION" = "12" ]; then
    TRT_VER="10.3.0.26"
    TRT_INCLUDE_DIR="TensorRT-10.3.0"
    CUDNN_VER="9"
    MNN_CUDA_FLAGS="-DMNN_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12"
else
    TRT_VER="8.6.1.6"
    TRT_INCLUDE_DIR="TensorRT-8.6.1.6"
    CUDNN_VER="8"
    MNN_CUDA_FLAGS="-DMNN_CUDA=ON"
fi
MNN_TAG="${MNN_TAG:-v2.7.0}"
WORKFLOW_TAG="${WORKFLOW_TAG:-v0.10.9}"
ONNXRUNTIME_VER="${ONNXRUNTIME_VER:-1.18.0}"
ONNXRUNTIME_SHA256="${ONNXRUNTIME_SHA256:-}"
OFFLINE_DIR=""
JOBS="$(nproc 2>/dev/null || echo 4)"

fail() { echo "[ERROR] $*" >&2; exit 1; }
info() { echo "[INFO] $*"; }
announce() { echo ""; echo "==> $*"; }

stamp() { test -f "$STAMP_DIR/$1"; }
mark() { mkdir -p "$STAMP_DIR"; touch "$STAMP_DIR/$1"; info "stamped: $1"; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || fail "missing command: $1 (apt install $2)"
}

# 拷贝头/库的通用助手：目标已存在同内容则跳过
copy_tree() { # src_dir dst_dir label
    [ -d "$1" ] || fail "copy_tree: source missing $1"
    mkdir -p "$2"
    cp -rn "$1"/* "$2"/ 2>/dev/null || true
    info "$3: copied into $2"
}
copy_libs() { # src_glob dst_dir label
    local found=0
    for f in $1; do
        [ -e "$f" ] || continue
        cp -n "$f" "$2"/ 2>/dev/null || true
        found=1
    done
    [ "$found" -eq 1 ] || fail "copy_libs: no files matched $1 ($3)"
    info "$3: copied into $2"
}

# ============ header-only 小库（缺失时从上游拉取，钉版本） ============
install_header_only() {
    local name="$1" url="$2" subdir="$3" stamp_name="$4"
    if stamp "$stamp_name"; then
        info "$name: already installed (stamp)"
        return
    fi
    announce "install $name"
    require_cmd curl "curl"
    local tmp="$BUILD_DIR/$stamp_name"
    mkdir -p "$tmp"
    curl -fsSL "$url" -o "$tmp/pkg.tar.gz"
    tar -xzf "$tmp/pkg.tar.gz" -C "$tmp"
    # 找到实际解压目录（release tarball 通常带前缀目录）
    local src
    src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n1)"
    [ -n "$src" ] || fail "$name: unpack failed"
    if [ -n "$subdir" ]; then
        src="$src/$subdir"
    fi
    mkdir -p "$INCLUDE_DIR/$(basename "$src")"
    cp -rn "$src"/* "$INCLUDE_DIR/$(basename "$src")"/ 2>/dev/null || true
    mark "$stamp_name"
}

# ============ fmt（源码构建，钉 tag） ============
install_fmt() {
    if stamp fmt; then info "fmt: already installed"; return; fi
    announce "build fmt 9.1.1"
    require_cmd git "git"
    require_cmd cmake "cmake"
    local src="$BUILD_DIR/fmt-src"
    mkdir -p "$src"
    if [ ! -d "$src/.git" ]; then
        git clone --depth 1 --branch 9.1.1 https://github.com/fmtlib/fmt.git "$src"
    fi
    cmake -S "$src" -B "$src/build-mortred" -DCMAKE_BUILD_TYPE=Release -DFMT_TEST=OFF -DFMT_DOC=OFF
    cmake --build "$src/build-mortred" -j"$JOBS"
    mkdir -p "$INCLUDE_DIR/fmt"
    cp -rn "$src/include/fmt"/*.h "$INCLUDE_DIR/fmt"/ 2>/dev/null || true
    copy_libs "$src/build-mortred/libfmt.so*" "$LIB_DIR" "fmt libs"
    mark fmt
}

# ============ workflow（源码构建，钉 tag） ============
install_workflow() {
    if stamp workflow; then info "workflow: already installed"; return; fi
    announce "build workflow ${WORKFLOW_TAG}"
    require_cmd git "git"
    require_cmd make "build-essential"
    local src="$BUILD_DIR/workflow-src"
    mkdir -p "$src"
    if [ ! -d "$src/.git" ]; then
        git clone --depth 1 --branch "$WORKFLOW_TAG" https://github.com/sogou/workflow.git "$src"
    fi
    (cd "$src" && make -j"$JOBS" >/dev/null)
    mkdir -p "$INCLUDE_DIR/workflow"
    cp -rn "$src/_include/workflow"/* "$INCLUDE_DIR/workflow"/ 2>/dev/null || true
    copy_libs "$src/_lib/libworkflow.so*" "$LIB_DIR" "workflow libs"
    mark workflow
}

# ============ MNN（源码构建，钉 tag，CUDA 后端） ============
install_mnn() {
    if stamp mnn; then info "MNN: already installed"; return; fi
    announce "build MNN ${MNN_TAG} (cuda=${CUDA_VERSION})"
    require_cmd git "git"
    require_cmd cmake "cmake"
    local src="$BUILD_DIR/mnn-src"
    mkdir -p "$src"
    if [ ! -d "$src/.git" ]; then
        git clone --depth 1 --branch "$MNN_TAG" https://github.com/alibaba/MNN.git "$src"
    fi
    local build="$src/build-mortred"
    cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TRAIN=OFF -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_TOOLS=OFF \
        -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TEST=OFF -DMNN_BUILD_BENCHMARK=OFF \
        $MNN_CUDA_FLAGS
    cmake --build "$build" -j"$JOBS"
    mkdir -p "$INCLUDE_DIR/MNN"
    cp -rn "$src/include/MNN"/* "$INCLUDE_DIR/MNN"/ 2>/dev/null || true
    copy_libs "$build/libMNN*.so*" "$LIB_DIR" "MNN libs"
    mark mnn
}

# ============ onnxruntime（官方 release tarball + sha256） ============
install_onnxruntime() {
    if stamp onnxruntime; then info "onnxruntime: already installed"; return; fi
    announce "install onnxruntime ${ONNXRUNTIME_VER}"
    require_cmd curl "curl"
    local tgz="onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VER}.tgz"
    local dst="$BUILD_DIR/onnxruntime-pkg"
    mkdir -p "$dst"
    local pkg_path="$dst/$tgz"
    if [ -n "$OFFLINE_DIR" ] && [ -f "$OFFLINE_DIR/$tgz" ]; then
        cp "$OFFLINE_DIR/$tgz" "$pkg_path"
    else
        curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VER}/${tgz}" -o "$pkg_path"
    fi
    if [ -n "$ONNXRUNTIME_SHA256" ]; then
        echo "$ONNXRUNTIME_SHA256  $pkg_path" | sha256sum -c - || fail "onnxruntime sha256 mismatch"
    fi
    tar -xzf "$pkg_path" -C "$dst"
    local src="$dst/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VER}"
    mkdir -p "$INCLUDE_DIR/onnxruntime"
    cp -rn "$src/include/onnxruntime"/* "$INCLUDE_DIR/onnxruntime"/ 2>/dev/null || true
    copy_libs "$src/lib/libonnxruntime*.so*" "$LIB_DIR" "onnxruntime libs"
    mark onnxruntime
}

# ============ CUDA / TensorRT / cuDNN（NVIDIA apt，需 root） ============
install_nvidia() {
    if stamp nvidia; then info "nvidia stack: already installed"; return; fi
    announce "install CUDA ${CUDA_VERSION} / TensorRT ${TRT_VER} / cuDNN ${CUDNN_VER} (needs root)"
    [ "$(id -u)" -eq 0 ] || fail "nvidia install requires root: run 'sudo ./scripts/install_deps.sh --nvidia'"
    require_cmd apt-get "apt"
    # NVIDIA apt 仓库（Ubuntu 20.04/22.04）
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update
    if [ "$CUDA_VERSION" = "12" ]; then
        apt-get install -y --no-install-recommends \
            cuda-toolkit-12-4 tensorrt-10.3.0.26 libcudnn9-dev-cuda-12
    else
        apt-get install -y --no-install-recommends \
            cuda-toolkit-11-8 tensorrt-8.6.1.6-1+cuda11.8 libcudnn8-dev-cuda-11
    fi
    # 拷入 3rd_party（头 + 库）
    local trt_inc=/usr/include/x86_64-linux-gnu
    mkdir -p "$INCLUDE_DIR/$TRT_INCLUDE_DIR"
    cp -rn "$trt_inc/NvInfer"*.h "$INCLUDE_DIR/$TRT_INCLUDE_DIR"/ 2>/dev/null || true
    cp -rn "$trt_inc/NvOnnxParser.h" "$INCLUDE_DIR/$TRT_INCLUDE_DIR"/ 2>/dev/null || true
    copy_libs "/usr/lib/x86_64-linux-gnu/libnvinfer*.so*" "$LIB_DIR" "TensorRT libs"
    copy_libs "/usr/lib/x86_64-linux-gnu/libnvonnxparser*.so*" "$LIB_DIR" "TensorRT onnx parser"
    copy_libs "/usr/local/cuda/lib64/libcudart*.so*" "$LIB_DIR" "CUDA runtime"
    copy_libs "/usr/lib/x86_64-linux-gnu/libcudnn*.so*" "$LIB_DIR" "cuDNN libs"
    mark nvidia
}

# ============ 校验 ============
check() {
    local rc=0
    local -a problems=()
    echo "== Mortred 3rd_party 完整性校验 =="
    echo "  ROOT         : $ROOT"
    echo "  目标版本线    : CUDA $CUDA_VERSION / TRT $TRT_VER / cuDNN $CUDNN_VER / MNN $MNN_TAG / workflow $WORKFLOW_TAG / onnxruntime $ONNXRUNTIME_VER"
    echo ""

    # 1) 工具链
    for c in git cmake make g++ curl; do
        if command -v "$c" >/dev/null 2>&1; then
            echo "  [ok] tool: $c ($(command -v "$c"))"
        else
            echo "  [!!] tool: $c MISSING"
            problems+=("tool $c")
        fi
    done
    if command -v nvcc >/dev/null 2>&1; then
        echo "  [ok] nvcc: $(nvcc --version | grep -oP 'release \K[0-9.]+' | head -n1)"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        echo "  [ok] nvcc: $(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9.]+' | head -n1) (/usr/local/cuda)"
    else
        echo "  [!!] nvcc MISSING (full build 需要 CUDA 工具链)"
        problems+=("nvcc")
    fi

    # 2) vendored 头
    local -a headers=(
        "MNN/MNNForwardType.h:MNN"
        "workflow/CommRequest.h:workflow"
        "onnxruntime/onnxruntime_cxx_api.h:onnxruntime"
        "$TRT_INCLUDE_DIR/NvInfer.h:TensorRT"
        "rapidjson/document.h:rapidjson"
        "toml/toml.hpp:toml11"
        "stb_image/stb_image.h:stb_image"
        "stl_container/concurrentqueue.h:moodycamel"
        "fmt/format.h:fmt"
        "indicators/indicators.hpp:indicators"
    )
    local entry hdr label
    for entry in "${headers[@]}"; do
        hdr="${entry%%:*}"; label="${entry##*:}"
        if [ -f "$INCLUDE_DIR/$hdr" ]; then
            echo "  [ok] header: $label ($hdr)"
        else
            echo "  [!!] header: $label MISSING ($hdr)"
            problems+=("header $label")
        fi
    done

    # 3) vendored 动态库
    local -a libs=(
        "libMNN.so:MNN"
        "libMNN_Cuda_Main.so:MNN CUDA"
        "libworkflow.so:workflow"
        "libonnxruntime.so:onnxruntime"
        "libnvinfer.so:TensorRT"
        "libnvonnxparser.so:TensorRT parser"
        "libcudart.so:CUDA runtime"
        "libcudnn.so:cuDNN"
        "libfmt.so:fmt"
        "libOpenCL.so:OpenCL"
        "libssl.so:openssl"
    )
    local lib name found
    for entry in "${libs[@]}"; do
        lib="${entry%%:*}"; name="${entry##*:}"
        found=0
        for f in "$LIB_DIR"/${lib}*; do
            [ -e "$f" ] && found=1 && break
        done
        if [ "$found" -eq 1 ]; then
            echo "  [ok] lib: $name ($(ls "$LIB_DIR"/${lib}* 2>/dev/null | head -n1 | xargs -n1 basename))"
        else
            echo "  [!!] lib: $name MISSING (${lib}*)"
            problems+=("lib $name")
        fi
    done

    echo ""
    if [ "${#problems[@]}" -eq 0 ]; then
        echo "== 校验通过：3rd_party 完整 =="
        return 0
    fi
    echo "== 校验失败，缺失项："
    for p in "${problems[@]}"; do echo "   - $p"; done
    echo "== 修复：./scripts/install_deps.sh --all（或对应子命令）"
    return 1
}

usage() {
    sed -n '2,18p' "$0"
    exit 0
}

# ============ main ============
MODE="all"
while [ $# -gt 0 ]; do
    case "$1" in
        --check) MODE="check"; shift ;;
        --all) MODE="all"; shift ;;
        --workflow) MODE="workflow"; shift ;;
        --mnn) MODE="mnn"; shift ;;
        --onnxruntime) MODE="onnxruntime"; shift ;;
        --nvidia) MODE="nvidia"; shift ;;
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --offline) OFFLINE_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done

case "$CUDA_VERSION" in
    11|12) ;;
    *) fail "unsupported --cuda-version $CUDA_VERSION (11 or 12)" ;;
esac

mkdir -p "$BUILD_DIR"

case "$MODE" in
    check) check ;;
    workflow) install_workflow ;;
    mnn) install_mnn ;;
    onnxruntime) install_onnxruntime ;;
    nvidia) install_nvidia ;;
    all)
        install_header_only rapidjson \
            "https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz" "include" "rapidjson"
        install_header_only toml11 \
            "https://github.com/ToruNiina/toml11/archive/refs/tags/v3.7.1.tar.gz" "toml11" "toml11"
        install_header_only stb_image \
            "https://github.com/nothings/stb/archive/refs/tags/0a538a1a2f0e4d166c6e5d3e1e9a1d8b0e1f2a3b.tar.gz" "" "stb_image"
        install_header_only indicators \
            "https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz" "include" "indicators"
        install_header_only moodycamel \
            "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.4.tar.gz" "concurrentqueue" "moodycamel"
        install_fmt
        install_workflow
        install_onnxruntime
        install_mnn
        # nvidia 需要 root，单独执行（避免 --all 在无 root 时报错）
        if [ "$(id -u)" -eq 0 ]; then
            install_nvidia
        else
            info "跳过 nvidia（需要 root）：sudo ./scripts/install_deps.sh --nvidia"
        fi
        echo ""
        echo "== 全部完成。校验：./scripts/install_deps.sh --check"
        ;;
esac
