#!/usr/bin/env bash
# install_deps.sh - one-shot build/install of all Mortred third-party deps into 3rd_party/{include,libs}
#
# Goal: replace the "manually compile third-party libs + hand-copy into 3rd_party" flow. Defaults to this repo's
# verified CUDA 11 / TensorRT 8.6 baseline; `--cuda-version 12` switches to the
# CUDA 12 / TensorRT 10 line (engines must be rebuilt with a matching trtexec version).
#
# Usage:
#   ./scripts/install_deps.sh --check            # verify 3rd_party completeness and print versions
#   ./scripts/install_deps.sh --all              # install everything (default)
#   ./scripts/install_deps.sh --workflow         # build and install only workflow
#   ./scripts/install_deps.sh --mnn              # build and install only MNN
#   ./scripts/install_deps.sh --onnxruntime      # download and install only onnxruntime
#   ./scripts/install_deps.sh --nvidia           # CUDA/TensorRT/cuDNN/trtexec (needs root + NVIDIA apt)
#   ./scripts/install_deps.sh --cuda-version 12  # switch to the CUDA 12 / TRT 10 line
#   ./scripts/install_deps.sh --offline DIR      # use a pre-downloaded package dir (offline)
#   Environment variables:
#   ONNXRUNTIME_SHA256=<hex>  explicitly pin the onnxruntime tarball hash (highest priority);
#                             if unset, consult the ONNXRUNTIME_SHA256S table in this script, else the
#                             official release API asset digest; refuse to install if none is available.
#
# Idempotent: each dependency leaves a stamp in 3rd_party/.install-stamp/ after a successful install; reruns skip automatically.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INCLUDE_DIR="$ROOT/3rd_party/include"
LIB_DIR="$ROOT/3rd_party/libs"
STAMP_DIR="$ROOT/3rd_party/.install-stamp"
BUILD_DIR="${MORTRED_DEPS_BUILD_DIR:-$ROOT/.deps-build}"

# ---- Version matrix (two lines; tags verified to exist via git ls-remote/HTTP HEAD) ----
CUDA_VERSION="${CUDA_VERSION:-11}"
if [ "$CUDA_VERSION" = "12" ]; then
    TRT_VER="10.3.0.26"
    TRT_INCLUDE_DIR="TensorRT-10.3.0"
    CUDNN_VER="9"
    MNN_TAG="${MNN_TAG:-2.9.6}"      # CUDA 12 requires the MNN 2.9+ backend
    MNN_CUDA_FLAGS="-DMNN_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12"
else
    TRT_VER="8.6.1.6"
    TRT_INCLUDE_DIR="TensorRT-8.6.1.6"
    CUDNN_VER="8"
    MNN_TAG="${MNN_TAG:-2.7.0}"
    MNN_CUDA_FLAGS="-DMNN_CUDA=ON"
fi
WORKFLOW_TAG="${WORKFLOW_TAG:-v0.10.9}"
ONNXRUNTIME_VER="${ONNXRUNTIME_VER:-1.18.0}"
ONNXRUNTIME_SHA256="${ONNXRUNTIME_SHA256:-}"
# Pinned sha256 table for the official onnxruntime tarballs (version -> hash). Verification priority:
#   1) env var ONNXRUNTIME_SHA256 (highest; for offline/CI use)
#   2) this table (for offline/air-gapped environments)
#   3) official release API asset digest (automatic when online; GitHub publishes sha256 per release asset)
#   none available -> refuse to install (never silently skip verification).
# To obtain a hash (run once with network access, verify against the official release assets, then fill this table for offline installs):
#   curl -fsSL -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-gpu-1.18.0.tgz
#   sha256sum /tmp/ort.tgz
# Example (replace <64-hex> with the output of the command above):
# declare -A ONNXRUNTIME_SHA256S=(
#     ["1.18.0"]="<64-hex>"
# )
declare -A ONNXRUNTIME_SHA256S=()
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

# Generic helper for copying headers/libs: skips when the destination already has identical content
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

# ============ header-only small libs (fetched from upstream when missing, versions pinned) ============
# install_header_only <name> <url> <archive_subdir> <dst_subdir> <stamp_name>
# archive_subdir is the header dir inside the tarball relative to its top (empty=top level); dst_subdir is the
# target dir under 3rd_party/include (must match the vendored layout, e.g. toml/stb_image/stl_container).
install_header_only() {
    local name="$1" url="$2" archive_subdir="$3" dst_subdir="$4" stamp_name="$5"
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
    # Find the actual unpacked dir (release tarballs usually carry a prefix dir)
    local src
    src="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n1)"
    [ -n "$src" ] || fail "$name: unpack failed"
    if [ -n "$archive_subdir" ]; then
        src="$src/$archive_subdir"
    fi
    [ -d "$src" ] || fail "$name: archive subdir '$archive_subdir' missing"
    mkdir -p "$INCLUDE_DIR/$dst_subdir"
    cp -rn "$src"/* "$INCLUDE_DIR/$dst_subdir"/ 2>/dev/null || true
    mark "$stamp_name"
}

# ============ fmt (built from source, pinned tag; git tag is 9.1.0, no 9.1.1) ============
install_fmt() {
    if stamp fmt; then info "fmt: already installed"; return; fi
    announce "build fmt 9.1.0"
    require_cmd git "git"
    require_cmd cmake "cmake"
    local src="$BUILD_DIR/fmt-src"
    mkdir -p "$src"
    if [ ! -d "$src/.git" ]; then
        git clone --depth 1 --branch 9.1.0 https://github.com/fmtlib/fmt.git "$src"
    fi
    cmake -S "$src" -B "$src/build-mortred" -DCMAKE_BUILD_TYPE=Release -DFMT_TEST=OFF -DFMT_DOC=OFF
    cmake --build "$src/build-mortred" -j"$JOBS"
    mkdir -p "$INCLUDE_DIR/fmt"
    cp -rn "$src/include/fmt"/*.h "$INCLUDE_DIR/fmt"/ 2>/dev/null || true
    copy_libs "$src/build-mortred/libfmt.so*" "$LIB_DIR" "fmt libs"
    mark fmt
}

# ============ workflow (built from source, pinned tag) ============
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

# ============ MNN (built from source, pinned tag, CUDA backend) ============
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

# ============ onnxruntime (official release tarball + sha256) ============
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
    # ---- fail-closed sha256 verification (never silently skipped) ----
    # Priority: explicit env ONNXRUNTIME_SHA256 > pinned table ONNXRUNTIME_SHA256S > official release API digest
    local expect_sha=""
    if [ -n "$ONNXRUNTIME_SHA256" ]; then
        expect_sha="$ONNXRUNTIME_SHA256"
    elif [ -n "${ONNXRUNTIME_SHA256S[$ONNXRUNTIME_VER]:-}" ]; then
        expect_sha="${ONNXRUNTIME_SHA256S[$ONNXRUNTIME_VER]}"
    else
        # GitHub publishes sha256 for each release asset (assets[].digest); official channel + TLS
        require_cmd jq "jq"
        expect_sha="$(curl -fsSL "https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v${ONNXRUNTIME_VER}" \
            | jq -r --arg name "$tgz" '.assets[] | select(.name == $name) | .digest' \
            | sed 's/^sha256://' || true)"
    fi
    [ -n "$expect_sha" ] || fail "cannot get sha256 for onnxruntime ${ONNXRUNTIME_VER}: set ONNXRUNTIME_SHA256 or fill the ONNXRUNTIME_SHA256S table (to get it: curl -fsSL -o /tmp/ort.tgz https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VER}/${tgz} && sha256sum /tmp/ort.tgz)"
    echo "$expect_sha  $pkg_path" | sha256sum -c - || fail "onnxruntime sha256 mismatch"
    tar -xzf "$pkg_path" -C "$dst"
    local src="$dst/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VER}"
    mkdir -p "$INCLUDE_DIR/onnxruntime"
    cp -rn "$src/include/onnxruntime"/* "$INCLUDE_DIR/onnxruntime"/ 2>/dev/null || true
    copy_libs "$src/lib/libonnxruntime*.so*" "$LIB_DIR" "onnxruntime libs"

    # Record: tgz checksum + installed lib hashes for --check re-verification (anti-tamper/corruption)
    echo "$expect_sha  $pkg_path" > "$STAMP_DIR/onnxruntime.sha256"
    (cd "$LIB_DIR" && find . -maxdepth 1 -name 'libonnxruntime.so*' -type f -print0 | sort -z | xargs -0 -r sha256sum > "$STAMP_DIR/onnxruntime.libs.sha256")
    mark onnxruntime
}

# ============ CUDA / TensorRT / cuDNN (NVIDIA apt, needs root) ============
install_nvidia() {
    if stamp nvidia; then info "nvidia stack: already installed"; return; fi
    announce "install CUDA ${CUDA_VERSION} / TensorRT ${TRT_VER} / cuDNN ${CUDNN_VER} (needs root)"
    [ "$(id -u)" -eq 0 ] || fail "nvidia install requires root: run 'sudo ./scripts/install_deps.sh --nvidia'"
    require_cmd apt-get "apt"
    # NVIDIA apt repo (Ubuntu 20.04/22.04)
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update
    if [ "$CUDA_VERSION" = "12" ]; then
        # devel images/machines with CUDA already installed skip the toolkit; only install TRT/cuDNN (avoid package conflicts)
        if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
            apt-get install -y --no-install-recommends cuda-toolkit-12-4
        else
            info "nvcc already present, skipping cuda-toolkit"
        fi
        # TRT 10 / cuDNN 9 dev packages (-dev needed to copy headers and libs into 3rd_party);
        # the tensorrt meta-package provides /usr/src/tensorrt/bin/trtexec (external engine conversion CLI)
        apt-get install -y --no-install-recommends \
            libnvinfer-dev libnvinfer-plugin-dev libnvonnxparser-dev \
            libcudnn9-dev-cuda-12 tensorrt
    else
        if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda/bin/nvcc ]; then
            apt-get install -y --no-install-recommends cuda-toolkit-11-8
        else
            info "nvcc already present, skipping cuda-toolkit"
        fi
        apt-get install -y --no-install-recommends \
            libnvinfer-dev libnvinfer-plugin-dev libnvonnxparser-dev \
            libcudnn8-dev-cuda-11 tensorrt
    fi
    # trtexec: the official TensorRT CLI (provided by the tensorrt meta-package at /usr/src/tensorrt/bin/trtexec);
    # copied into 3rd_party/bin for scripts/convert_trt_engines.sh (it ships in bin/ with the install tree)
    if [ -x /usr/src/tensorrt/bin/trtexec ]; then
        mkdir -p "$ROOT/3rd_party/bin"
        cp -n /usr/src/tensorrt/bin/trtexec "$ROOT/3rd_party/bin/trtexec"
        info "trtexec: copied into 3rd_party/bin"
    else
        info "trtexec: not found in /usr/src/tensorrt/bin (before converting, make sure the tensorrt package is installed)"
    fi
    # Copy into 3rd_party (headers + libs)
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

# ============ verification ============
check() {
    local rc=0
    local -a problems=()
    echo "== Mortred 3rd_party integrity check =="
    echo "  ROOT         : $ROOT"
    echo "  target version line: CUDA $CUDA_VERSION / TRT $TRT_VER / cuDNN $CUDNN_VER / MNN $MNN_TAG / workflow $WORKFLOW_TAG / onnxruntime $ONNXRUNTIME_VER"
    echo ""

    # 1) toolchain
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
        echo "  [!!] nvcc MISSING (full build needs the CUDA toolchain)"
        problems+=("nvcc")
    fi
    if [ -x "$ROOT/3rd_party/bin/trtexec" ] || command -v trtexec >/dev/null 2>&1; then
        echo "  [ok] tool: trtexec"
    else
        echo "  [!!] tool: trtexec MISSING (run sudo ./scripts/install_deps.sh --nvidia or install the system TensorRT package)"
        problems+=("trtexec")
    fi

    # 2) vendored headers
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

    # 3) vendored dynamic libs
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

    # 4) onnxruntime artifact hash re-check (anti-tamper/corruption; recorded at install time)
    if [ -f "$STAMP_DIR/onnxruntime.libs.sha256" ]; then
        if (cd "$LIB_DIR" && sha256sum -c "$STAMP_DIR/onnxruntime.libs.sha256" >/dev/null 2>&1); then
            echo "  [ok] onnxruntime lib hash (stamp re-check)"
        else
            echo "  [!!] onnxruntime lib hash MISMATCH (libs tampered/corrupted; please reinstall onnxruntime)"
            problems+=("onnxruntime lib hash")
        fi
    else
        echo "  [warn] onnxruntime has no recorded hash (run install_deps.sh --onnxruntime first to generate it)"
    fi

    echo ""
    if [ "${#problems[@]}" -eq 0 ]; then
        echo "== verification passed: 3rd_party complete =="
        return 0
    fi
    echo "== verification failed, missing items:"
    for p in "${problems[@]}"; do echo "   - $p"; done
    echo "== fix: ./scripts/install_deps.sh --all (or the matching subcommand)"
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
            "https://github.com/Tencent/rapidjson/archive/refs/tags/v1.1.0.tar.gz" "include" "rapidjson" "rapidjson"
        install_header_only toml11 \
            "https://github.com/ToruNiina/toml11/archive/refs/tags/v3.7.1.tar.gz" "toml" "toml" "toml11"
        install_header_only stb_image \
            "https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz" "" "stb_image" "stb_image"
        install_header_only indicators \
            "https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz" "include" "indicators" "indicators"
        install_header_only moodycamel \
            "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.4.tar.gz" "" "stl_container" "moodycamel"
        install_fmt
        install_workflow
        install_onnxruntime
        install_mnn
        # nvidia needs root; run it separately (so --all doesn't error out without root)
        if [ "$(id -u)" -eq 0 ]; then
            install_nvidia
        else
            info "skipping nvidia (needs root): sudo ./scripts/install_deps.sh --nvidia"
        fi
        echo ""
        echo "== all done. verify: ./scripts/install_deps.sh --check"
        ;;
esac
