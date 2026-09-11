#!/usr/bin/env bash
# install_deps.sh - one-shot build/install of all Mortred third-party deps into 3rd_party/{include,libs}
#
# GPU line is CUDA 12 + TensorRT 10.3 + cuDNN 9 + MNN 3.6.1 + ORT 1.29 cuda12.
# CUDA 11 / TensorRT 8 are deleted. Engines must be rebuilt with matching trtexec.
#
# Usage:
#   ./scripts/install_deps.sh --check            # verify 3rd_party completeness and print versions
#   ./scripts/install_deps.sh --all              # install everything (default)
#   ./scripts/install_deps.sh --cpu --all        # cpu profile: MNN without CUDA, ORT cpu tarball,
#                                                # no NVIDIA/TRT stack at all (GPU-less machines)
#   ./scripts/install_deps.sh --workflow         # build and install only workflow
#   ./scripts/install_deps.sh --mnn              # build and install only MNN
#   ./scripts/install_deps.sh --onnxruntime      # download and install only onnxruntime
#   ./scripts/install_deps.sh --nvidia           # CUDA/TensorRT/cuDNN/trtexec (needs root + NVIDIA apt)
#   ./scripts/install_deps.sh --cuda-version 12  # accepted no-op (12 is the only GPU line)
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

# ---- Version matrix (one GPU line; tags verified to exist via git ls-remote/HTTP HEAD) ----
CUDA_VERSION="${CUDA_VERSION:-12}"
# NVIDIA base images (nvidia/cuda:*) export CUDA_VERSION="12.6.0" style values;
# normalize to the major line so the env bleed cannot fail the 12-only validation
CUDA_VERSION="${CUDA_VERSION%%.*}"
TRT_VER="10.3.0.26"
TRT_INCLUDE_DIR="TensorRT-10.3.0"
CUDNN_VER="9"
MNN_TAG="${MNN_TAG:-3.6.1}"
MNN_CUDA_FLAGS="-DMNN_CUDA=ON"
WORKFLOW_TAG="${WORKFLOW_TAG:-v0.10.9}"
ONNXRUNTIME_VER="${ONNXRUNTIME_VER:-1.29.0}"
NVIDIA_STAMP="nvidia-cuda12-trt10.3"
# deployment profile: gpu (default, full CUDA/TRT stack) | cpu (MNN-CPU +
# ORT-CPU, no NVIDIA). Drives MNN build flags, the ORT tarball flavor and the
# --check expectations; stamps are profile-specific so trees never mix.
DEP_PROFILE="${DEP_PROFILE:-gpu}"
MNN_STAMP="mnn-3.6.1-cuda12"
MNN_BUILD_DIR_NAME="build-mortred"
ORT_FLAVOR="-gpu_cuda12"
ORT_STAMP="onnxruntime-gpu-1.29.0"
ONNXRUNTIME_SHA256="${ONNXRUNTIME_SHA256:-}"
# Pinned sha256 table for the official onnxruntime tarballs (asset filename -> hash). Verification priority:
#   1) env var ONNXRUNTIME_SHA256 (highest; for offline/CI use)
#   2) this table (for offline/air-gapped environments)
#   3) official release API asset digest (automatic when online; GitHub publishes sha256 per release asset)
#   none available -> refuse to install (never silently skip verification).
# GPU package is the CUDA 12 artifact (not the CUDA 13 tarball, not the unsuffixed
# -gpu- name removed in 1.27).
# Digests from GitHub release assets v1.29.0.
declare -A ONNXRUNTIME_SHA256S=(
    ["onnxruntime-linux-x64-gpu_cuda12-1.29.0.tgz"]="4ca594a0da83927befbd73fe020d7f569be151d70bb4fe9741ad405f4882e2ad"
    ["onnxruntime-linux-x64-1.29.0.tgz"]="c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba"
)
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

apply_profile() {
    if [ "$DEP_PROFILE" = "cpu" ]; then
        MNN_CUDA_FLAGS=""
        MNN_STAMP="mnn-3.6.1-cpu"
        MNN_BUILD_DIR_NAME="build-mortred-cpu"
        ORT_FLAVOR=""
        ORT_STAMP="onnxruntime-cpu-1.29.0"
    else
        local cuda_home=""
        if [ -x /usr/local/cuda/bin/nvcc ]; then
            cuda_home=/usr/local/cuda
        elif [ -d /usr/local/cuda-12 ]; then
            cuda_home=/usr/local/cuda-12
        fi
        MNN_CUDA_FLAGS="-DMNN_CUDA=ON"
        if [ -n "$cuda_home" ]; then
            MNN_CUDA_FLAGS="$MNN_CUDA_FLAGS -DCUDA_TOOLKIT_ROOT_DIR=$cuda_home"
        fi
        MNN_STAMP="mnn-3.6.1-cuda12"
        MNN_BUILD_DIR_NAME="build-mortred"
        ORT_FLAVOR="-gpu_cuda12"
        ORT_STAMP="onnxruntime-gpu-1.29.0"
    fi
    if [ -e /usr/lib/x86_64-linux-gnu/libcrypto.so.3 ]; then
        RUNTIME_STAMP="runtime-libs-ssl3"
    else
        RUNTIME_STAMP="runtime-libs-ssl1.1"
    fi
}

wipe_stale_gpu_leftovers() {
    rm -rf "$INCLUDE_DIR"/TensorRT-8*
    rm -f "$LIB_DIR"/libnvinfer.so.8* \
          "$LIB_DIR"/libnvinfer_plugin.so.8* \
          "$LIB_DIR"/libnvonnxparser.so.8* \
          "$LIB_DIR"/libnvparsers.so.8* \
          "$LIB_DIR"/libcudart.so.11* \
          "$LIB_DIR"/libcudart.so.13* \
          "$LIB_DIR"/libcudnn.so.8* \
          "$LIB_DIR"/libonnxruntime.so.1.18.0*
}

wipe_stale_nvidia_tree() {
    wipe_stale_gpu_leftovers
    rm -f "$ROOT/3rd_party/bin/trtexec"
}

leftover_gpu_line() {
    local -a hits=()
    local f
    for f in "$LIB_DIR"/libnvinfer.so.8* "$LIB_DIR"/libnvinfer_plugin.so.8* \
             "$LIB_DIR"/libnvonnxparser.so.8* "$LIB_DIR"/libcudart.so.11* \
             "$LIB_DIR"/libcudnn.so.8* "$LIB_DIR"/libonnxruntime.so.1.18.0* \
             "$LIB_DIR"/libcudart.so.13*; do
        [ -e "$f" ] && hits+=("$(basename "$f")")
    done
    for f in "$INCLUDE_DIR"/TensorRT-8*; do
        [ -e "$f" ] && hits+=("$(basename "$f")")
    done
    if [ ${#hits[@]} -gt 0 ]; then
        printf '%s\n' "${hits[@]}"
        return 0
    fi
    return 1
}

# Generic helper for copying headers/libs: skips when the destination already has identical content
copy_tree() { # src_dir dst_dir label
    [ -d "$1" ] || fail "copy_tree: source missing $1"
    mkdir -p "$2"
    cp -rn "$1"/* "$2"/ 2>/dev/null || true
    info "$3: copied into $2"
}
copy_libs() { # src_glob dst_dir label
    local found=0 dest
    # dst dir may not exist yet in fresh trees/containers (3rd_party/libs is
    # gitignored, so `COPY . /src/` never ships it) - cp would silently fail
    mkdir -p "$2"
    for f in $1; do
        [ -e "$f" ] || continue
        dest="$2/$(basename "$f")"
        # Upgrade leftover: libfoo.so often stays as a dangling symlink after
        # its old soname (e.g. libonnxruntime.so.1.18.0) was rm'd. GNU cp -f
        # refuses to write through that ("not writing through dangling symlink").
        rm -f "$dest"
        cp -f "$f" "$dest" || fail "copy_libs: failed to copy $f into $2 ($3)"
        found=1
    done
    [ "$found" -eq 1 ] || fail "copy_libs: no files matched $1 ($3)"
    info "$3: copied into $2"
}

# ============ header-only small libs (fetched from upstream when missing, versions pinned) ============
# install_header_only <name> <url> <archive_subdir> <dst_subdir> <stamp_name> [amalgamate_header]
# archive_subdir is the header dir inside the tarball relative to its top (empty=top level); dst_subdir is the
# target dir under 3rd_party/include (must match the vendored layout, e.g. toml/stb_image/stl_container).
# When amalgamate_header is set, the individual headers are concatenated into that single header (upstream
# indicators never shipped its amalgamated indicators.hpp; the pieces carry their own include guards and the
# internal #include "x.hpp" lines resolve against the copied sibling files).
install_header_only() {
    local name="$1" url="$2" archive_subdir="$3" dst_subdir="$4" stamp_name="$5"
    local amalgamate="${6:-}"
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
    # Some release tarballs nest the headers one level deeper
    # (include/<proj>/... pattern): if $src holds no files directly, descend
    # into its single subdirectory so the headers land where --check expects
    # them (rapidjson/indicators both use include/<proj>/...).
    if [ -z "$(find "$src" -maxdepth 1 -type f -print -quit)" ]; then
        local inner
        inner="$(find "$src" -mindepth 1 -maxdepth 1 -type d | head -n1)"
        [ -n "$inner" ] && src="$inner"
    fi
    mkdir -p "$INCLUDE_DIR/$dst_subdir"
    cp -rn "$src"/* "$INCLUDE_DIR/$dst_subdir"/ 2>/dev/null || true
    [ -n "$(ls -A "$INCLUDE_DIR/$dst_subdir" 2>/dev/null)" ] \
        || fail "$name: no headers copied (src=$src)"
    if [ -n "$amalgamate" ]; then
        cat "$src"/*.hpp > "$INCLUDE_DIR/$dst_subdir/$amalgamate" 2>/dev/null \
            || fail "$name: amalgamation of $amalgamate failed"
        info "$name: amalgamated $amalgamate"
    fi
    mark "$stamp_name"
}

# ============ fmt (built from source; 9.1.0 is the closest upstream tag to the
# tracked 3rd_party fmt headers - upstream never released a 9.1.1, so cloning
# it fails; 9.1.0 keeps the libfmt.so.9 soname vendored::fmt imports) ============
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
    # BUILD_SHARED_LIBS=ON: fmt defaults to a static lib, but the vendored tree
    # and vendored::fmt expect libfmt.so.9 - fresh containers proved the default
    # produces only libfmt.a and copy_libs then finds nothing
    cmake -S "$src" -B "$src/build-mortred" -DCMAKE_BUILD_TYPE=Release -DFMT_TEST=OFF -DFMT_DOC=OFF -DBUILD_SHARED_LIBS=ON
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

# ============ system runtime libs that must live in the vendored tree ============
# libssl/libcrypto: workflow links against them and mortred-gateway's API-key
# SHA-256 path links libcrypto explicitly (vendored::crypto). libOpenCL: MNN's
# OpenCL schedule backend dlopens it (gpu profile). Fresh containers have no
# legacy hand-copied files, so --all installs them like any other dependency
# and --check keeps demanding them (install/check contract stays in sync).
install_system_runtime_libs() {
    if stamp "$RUNTIME_STAMP"; then info "runtime libs: already installed"; return; fi
    announce "install system runtime libs (ssl/crypto$( [ "$DEP_PROFILE" != "cpu" ] && echo /OpenCL ))"
    local sys=/usr/lib/x86_64-linux-gnu
    copy_libs "$sys/libssl.so*"    "$LIB_DIR" "libssl runtime"
    copy_libs "$sys/libcrypto.so*" "$LIB_DIR" "libcrypto runtime"
    if [ "$DEP_PROFILE" != "cpu" ]; then
        copy_libs "$sys/libOpenCL.so*" "$LIB_DIR" "OpenCL runtime"
    fi
    mark "$RUNTIME_STAMP"
}
# ============ MNN (built from source, pinned tag, CUDA backend) ============
install_mnn() {
    if stamp "$MNN_STAMP"; then info "MNN (${DEP_PROFILE}): already installed"; return; fi
    announce "build MNN ${MNN_TAG} (profile=${DEP_PROFILE})"
    require_cmd git "git"
    require_cmd cmake "cmake"
    local src="$BUILD_DIR/mnn-src"
    mkdir -p "$src"
    if [ ! -d "$src/.git" ]; then
        git clone --depth 1 --branch "$MNN_TAG" https://github.com/alibaba/MNN.git "$src"
    fi
    local build="$src/$MNN_BUILD_DIR_NAME"
    cmake -S "$src" -B "$build" -DCMAKE_BUILD_TYPE=Release \
        -DMNN_BUILD_TRAIN=OFF -DMNN_BUILD_DEMO=OFF -DMNN_BUILD_TOOLS=OFF \
        -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TEST=OFF -DMNN_BUILD_BENCHMARK=OFF \
        $MNN_CUDA_FLAGS
    cmake --build "$build" -j"$JOBS"
    # MNN 3.6.1 still registers the CUDA backend as a separate loadable lib
    # (libMNN_Cuda_Main.so). Ensure it is built even if the default target set
    # skipped it; fail loudly if it can't be. CUTLASS is fetched at configure.
    if [ "$DEP_PROFILE" != "cpu" ]; then
        cmake --build "$build" --target MNN_Cuda_Main -j"$JOBS" \
            || fail "MNN CUDA backend build failed (target MNN_Cuda_Main)"
        # cuda_add_library puts the backend lib in the cuda SUBDIR's binary
        # dir ($build/source/backend/cuda), not $build root - the copy_libs
        # glob below only sees the root, so locate and copy it explicitly
        local mnn_cuda
        mnn_cuda="$(find "$build" -name 'libMNN_Cuda_Main.so*' -type f | head -n1)"
        [ -n "$mnn_cuda" ] || fail "libMNN_Cuda_Main.so not found under $build"
        rm -f "$LIB_DIR/$(basename "$mnn_cuda")"
        cp -f "$mnn_cuda" "$LIB_DIR"/
        info "MNN CUDA lib: copied into $LIB_DIR"
    fi
    mkdir -p "$INCLUDE_DIR/MNN"
    cp -rn "$src/include/MNN"/* "$INCLUDE_DIR/MNN"/ 2>/dev/null || true
    copy_libs "$build/libMNN*.so*" "$LIB_DIR" "MNN libs"
    mark "$MNN_STAMP"
}

# ============ onnxruntime (official release tarball + sha256) ============
curl_retry() { # url dest
    local extra=()
    curl --help 2>&1 | grep -q -- '--retry-all-errors' && extra+=(--retry-all-errors)
    curl -fL --retry 8 --retry-delay 3 --connect-timeout 30 -C - "${extra[@]}" -o "$2" "$1"
}

install_onnxruntime() {
    if stamp "$ORT_STAMP"; then info "onnxruntime (${DEP_PROFILE}): already installed"; return; fi
    announce "install onnxruntime ${ONNXRUNTIME_VER} (${DEP_PROFILE})"
    require_cmd curl "curl"
    local tgz="onnxruntime-linux-x64${ORT_FLAVOR}-${ONNXRUNTIME_VER}.tgz"
    local dst="$BUILD_DIR/onnxruntime-pkg"
    mkdir -p "$dst"
    local pkg_path="$dst/$tgz"
    local url="https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VER}/${tgz}"
    # ---- fail-closed sha256 verification (never silently skipped) ----
    # Priority: explicit env ONNXRUNTIME_SHA256 > pinned table ONNXRUNTIME_SHA256S > official release API digest
    local expect_sha=""
    if [ -n "$ONNXRUNTIME_SHA256" ]; then
        expect_sha="$ONNXRUNTIME_SHA256"
    elif [ -n "${ONNXRUNTIME_SHA256S[$tgz]:-}" ]; then
        expect_sha="${ONNXRUNTIME_SHA256S[$tgz]}"
    else
        # GitHub publishes sha256 for each release asset (assets[].digest); official channel + TLS
        require_cmd jq "jq"
        # `// empty`: GitHub leaves digest null on older release assets; without
        # it jq -r prints the STRING "null", which then confuses sha256sum with
        # "no properly formatted checksum lines" instead of failing cleanly
        expect_sha="$(curl -fsSL "https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v${ONNXRUNTIME_VER}" \
            | jq -r --arg name "$tgz" '.assets[] | select(.name == $name) | .digest // empty' \
            | sed 's/^sha256://' || true)"
    fi
    [ -n "$expect_sha" ] || fail "cannot get sha256 for onnxruntime ${ONNXRUNTIME_VER}: set ONNXRUNTIME_SHA256 or fill the ONNXRUNTIME_SHA256S table (to get it: curl -fsSL -o /tmp/ort.tgz ${url} && sha256sum /tmp/ort.tgz)"

    local have_pkg=0
    if [ -f "$pkg_path" ] && echo "$expect_sha  $pkg_path" | sha256sum -c - >/dev/null 2>&1; then
        info "onnxruntime tarball already present and hash-ok, skipping download"
        have_pkg=1
    elif [ -n "$OFFLINE_DIR" ] && [ -f "$OFFLINE_DIR/$tgz" ]; then
        cp "$OFFLINE_DIR/$tgz" "$pkg_path"
        have_pkg=1
    fi
    if [ "$have_pkg" -ne 1 ]; then
        info "downloading ${tgz} (~400MB for gpu_cuda12; GitHub TLS drops are retried)"
        curl_retry "$url" "$pkg_path" || true
        if ! echo "$expect_sha  $pkg_path" | sha256sum -c - >/dev/null 2>&1; then
            info "incomplete or corrupt tarball, retrying a full download"
            rm -f "$pkg_path"
            curl_retry "$url" "$pkg_path" || fail "onnxruntime download failed (GitHub TLS/network). Download ${url} elsewhere and rerun: ./scripts/install_deps.sh --onnxruntime --offline DIR"
        fi
    fi
    echo "$expect_sha  $pkg_path" | sha256sum -c - || fail "onnxruntime sha256 mismatch"
    tar -xzf "$pkg_path" -C "$dst"
    local src="$dst/onnxruntime-linux-x64${ORT_FLAVOR}-${ONNXRUNTIME_VER}"
    [ -d "$src" ] || fail "onnxruntime unpack dir missing: $src"
    mkdir -p "$INCLUDE_DIR/onnxruntime"
    rm -f "$LIB_DIR"/libonnxruntime.so*
    # 1.18 tarballs were a flat include/; 1.29 may nest include/onnxruntime/.
    if [ -f "$src/include/onnxruntime/onnxruntime_cxx_api.h" ]; then
        cp -rf "$src/include/onnxruntime/." "$INCLUDE_DIR/onnxruntime"/
    elif [ -f "$src/include/onnxruntime_cxx_api.h" ]; then
        cp -rf "$src"/include/. "$INCLUDE_DIR/onnxruntime"/
    else
        fail "onnxruntime headers not found under $src/include"
    fi
    copy_libs "$src/lib/libonnxruntime*.so*" "$LIB_DIR" "onnxruntime libs"

    # Record: tgz checksum + installed lib hashes for --check re-verification (anti-tamper/corruption)
    echo "$expect_sha  $pkg_path" > "$STAMP_DIR/${ORT_STAMP}.sha256"
    (cd "$LIB_DIR" && find . -maxdepth 1 -name 'libonnxruntime.so*' -type f -print0 | sort -z | xargs -0 -r sha256sum > "$STAMP_DIR/${ORT_STAMP}.libs.sha256")
    mark "$ORT_STAMP"
}

# ============ CUDA / TensorRT / cuDNN (NVIDIA apt, needs root) ============
install_nvidia() {
    if stamp "$NVIDIA_STAMP"; then info "nvidia stack: already installed"; return; fi
    announce "install CUDA ${CUDA_VERSION} / TensorRT ${TRT_VER} / cuDNN ${CUDNN_VER} (needs root)"
    [ "$(id -u)" -eq 0 ] || fail "nvidia install requires root: run 'sudo ./scripts/install_deps.sh --nvidia'"
    [ "$CUDA_VERSION" = "12" ] || fail "GPU line is CUDA 12 only (got $CUDA_VERSION)"
    require_cmd apt-get "apt"
    local nvcc_bin=""
    if command -v nvcc >/dev/null 2>&1; then
        nvcc_bin="$(command -v nvcc)"
    elif [ -x /usr/local/cuda/bin/nvcc ]; then
        nvcc_bin=/usr/local/cuda/bin/nvcc
    fi
    if [ -n "$nvcc_bin" ]; then
        local nvcc_major
        nvcc_major="$("$nvcc_bin" --version | grep -oP 'release \K[0-9]+' | head -n1 || true)"
        [ "$nvcc_major" = "12" ] || fail "nvcc major must be 12 (got ${nvcc_major:-unknown} from $nvcc_bin)"
        info "nvcc already present ($nvcc_bin release $nvcc_major), skipping cuda-toolkit"
    fi
    # NVIDIA apt repo: nvidia/cuda base images already configure it; bare machines
    # need the matching keyring. Pick the keyring by OS release - hardcoding one
    # distro (e.g. the jammy keyring on a focal image) mixes repos whose package
    # builds can mismatch the local CUDA.
    if ! grep -rqs 'developer.download.nvidia.com' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null; then
        local os_id os_ver repo_dir
        os_id="$(. /etc/os-release && echo "$ID")"
        os_ver="$(. /etc/os-release && echo "$VERSION_ID")"
        case "$os_id-$os_ver" in
            ubuntu-20.04) repo_dir=ubuntu2004 ;;
            ubuntu-22.04) repo_dir=ubuntu2204 ;;
            *) fail "unsupported OS for the NVIDIA apt repo: $os_id $os_ver" ;;
        esac
        curl -fsSL "https://developer.download.nvidia.com/compute/cuda/repos/${repo_dir}/x86_64/cuda-keyring_1.1-1_all.deb" -o /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
    else
        info "NVIDIA apt repo already configured; skipping keyring install"
    fi
    apt-get update
    # Deterministic install: download the EXACT versioned .debs and dpkg -i them.
    # apt's resolver cannot be trusted here. Direct .deb install leaves no
    # freedom to wander into unversioned nvinfer. All versions below verified
    # against the repo Packages.gz indexes; the focal and jammy repos both
    # carry them.
    local repo_dir2 nv_repo deb_dir d
    repo_dir2="$(grep -rhoE 'repos/ubuntu[0-9]+/x86_64' /etc/apt/sources.list /etc/apt/sources.list.d/ 2>/dev/null | head -n1 | sed 's#repos/##; s#/x86_64##')"
    [ -n "$repo_dir2" ] || repo_dir2=ubuntu2204
    nv_repo="https://developer.download.nvidia.com/compute/cuda/repos/${repo_dir2}/x86_64"
    deb_dir="$BUILD_DIR/nvidia-debs"
    mkdir -p "$deb_dir"
    fetch_deb() { # <deb filename>
        local d="$1"
        [ -f "$deb_dir/$d" ] || curl -fSL "$nv_repo/$d" -o "$deb_dir/$d"
        dpkg-deb --info "$deb_dir/$d" >/dev/null 2>&1 || fail "bad NVIDIA deb: $d"
    }
    if [ -z "$nvcc_bin" ]; then
        apt-get install -y --no-install-recommends cuda-toolkit-12-4
    fi
    # TRT 10.3.0.26 + cuDNN 9 (9.10.2.21 = highest version in BOTH the focal
    # and jammy repos). libnvinfer-bin is NOT installed (its libnvparsers10
    # 10.3.0.26 build is absent from the repos) - trtexec is extracted from
    # its deb below. libnvparsers10 is skipped for the same reason.
    local -a DEBS=(
        libcudnn9-cuda-12_9.10.2.21-1_amd64.deb
        libcudnn9-dev-cuda-12_9.10.2.21-1_amd64.deb
        libnvinfer10_10.3.0.26-1+cuda12.5_amd64.deb
        libnvinfer-plugin10_10.3.0.26-1+cuda12.5_amd64.deb
        libnvonnxparsers10_10.3.0.26-1+cuda12.5_amd64.deb
        libnvinfer-headers-dev_10.3.0.26-1+cuda12.5_amd64.deb
        libnvinfer-headers-plugin-dev_10.3.0.26-1+cuda12.5_amd64.deb
        libnvinfer-dev_10.3.0.26-1+cuda12.5_amd64.deb
        libnvinfer-plugin-dev_10.3.0.26-1+cuda12.5_amd64.deb
        libnvonnxparsers-dev_10.3.0.26-1+cuda12.5_amd64.deb
    )
    for d in "${DEBS[@]}"; do fetch_deb "$d"; done
    # dpkg unpacks/installs the set; it may exit non-zero when a system dep
    # (e.g. protobuf) is missing - the -f pass below pulls it and configures.
    dpkg -i "${DEBS[@]/#/$deb_dir/}" >/dev/null || true
    apt-get install -f -y --no-install-recommends
    local expect_pkg=libnvinfer10 expect_ver=10.3.0.26-1+cuda12.5
    [ "$(dpkg-query -W -f='${Version}' "$expect_pkg" 2>/dev/null)" = "$expect_ver" ] \
        || fail "TensorRT version mismatch: expected $expect_ver for $expect_pkg"
    local bin_deb="libnvinfer-bin_10.3.0.26-1+cuda12.5_amd64.deb"
    fetch_deb "$bin_deb"
    local trt_bin_dir="$deb_dir/bin-extract"
    rm -rf "$trt_bin_dir"
    dpkg-deb -x "$deb_dir/$bin_deb" "$trt_bin_dir"
    mkdir -p "$ROOT/3rd_party/bin"
    wipe_stale_nvidia_tree
    cp -f "$trt_bin_dir/usr/src/tensorrt/bin/trtexec" "$ROOT/3rd_party/bin/trtexec" \
        || fail "trtexec extraction failed"
    chmod +x "$ROOT/3rd_party/bin/trtexec"
    info "trtexec: copied into 3rd_party/bin"
    local trt_inc=/usr/include/x86_64-linux-gnu
    mkdir -p "$INCLUDE_DIR/$TRT_INCLUDE_DIR"
    cp -rf "$trt_inc"/NvInfer*.h "$INCLUDE_DIR/$TRT_INCLUDE_DIR"/ 2>/dev/null || true
    cp -f "$trt_inc/NvOnnxParser.h" "$INCLUDE_DIR/$TRT_INCLUDE_DIR"/ 2>/dev/null || true
    copy_libs "/usr/lib/x86_64-linux-gnu/libnvinfer*.so.10*" "$LIB_DIR" "TensorRT nvinfer"
    copy_libs "/usr/lib/x86_64-linux-gnu/libnvonnxparser*.so.10*" "$LIB_DIR" "TensorRT onnx parser"
    if [ -d /usr/local/cuda/lib64 ]; then
        copy_libs "/usr/local/cuda/lib64/libcudart.so.12*" "$LIB_DIR" "CUDA runtime"
    else
        copy_libs "/usr/local/cuda-12/lib64/libcudart.so.12*" "$LIB_DIR" "CUDA runtime"
    fi
    copy_libs "/usr/lib/x86_64-linux-gnu/libcudnn*.so*" "$LIB_DIR" "cuDNN libs"
    wipe_stale_gpu_leftovers
    leftover="$(leftover_gpu_line || true)"
    [ -z "$leftover" ] || fail "leftover CUDA 11 / TensorRT 8 / ORT 1.18 after nvidia install: $leftover"
    local need found f
    for need in libnvinfer.so.10 libcudart.so.12 libcudnn.so.9; do
        found=0
        for f in "$LIB_DIR"/${need}*; do
            [ -e "$f" ] && found=1 && break
        done
        [ "$found" -eq 1 ] || fail "$need missing after nvidia copy"
    done
    mark "$NVIDIA_STAMP"
}

# ============ verification ============
check() {
    local rc=0
    local -a problems=()
    echo "== Mortred 3rd_party integrity check (profile: $DEP_PROFILE) =="
    echo "  ROOT         : $ROOT"
    if [ "$DEP_PROFILE" = "cpu" ]; then
        echo "  target version line: MNN $MNN_TAG (CPU) / workflow $WORKFLOW_TAG / onnxruntime $ONNXRUNTIME_VER (CPU)"
    else
        echo "  target version line: CUDA $CUDA_VERSION / TRT $TRT_VER / cuDNN $CUDNN_VER / MNN $MNN_TAG / workflow $WORKFLOW_TAG / onnxruntime $ONNXRUNTIME_VER"
    fi
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
    if [ "$DEP_PROFILE" = "cpu" ]; then
        echo "  [--] nvcc: not required (cpu profile)"
        echo "  [--] trtexec: not required (cpu profile)"
    else
        if command -v nvcc >/dev/null 2>&1; then
            local nvcc_rel
            nvcc_rel="$(nvcc --version | grep -oP 'release \K[0-9.]+' | head -n1)"
            echo "  [ok] nvcc: $nvcc_rel"
            [ "${nvcc_rel%%.*}" = "12" ] || { echo "  [!!] nvcc major must be 12"; problems+=("nvcc major"); }
        elif [ -x /usr/local/cuda/bin/nvcc ]; then
            local nvcc_rel
            nvcc_rel="$(/usr/local/cuda/bin/nvcc --version | grep -oP 'release \K[0-9.]+' | head -n1)"
            echo "  [ok] nvcc: $nvcc_rel (/usr/local/cuda)"
            [ "${nvcc_rel%%.*}" = "12" ] || { echo "  [!!] nvcc major must be 12"; problems+=("nvcc major"); }
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
    fi

    # 2) vendored headers
    local -a headers=(
        "MNN/MNNForwardType.h:MNN"
        "workflow/CommRequest.h:workflow"
        "onnxruntime/onnxruntime_cxx_api.h:onnxruntime"
        "rapidjson/document.h:rapidjson"
        "toml/toml.hpp:toml++"
        "stb_image/stb_image.h:stb_image"
        "stl_container/concurrentqueue.h:moodycamel"
        "fmt/format.h:fmt"
        "indicators/indicators.hpp:indicators"
    )
    if [ "$DEP_PROFILE" != "cpu" ]; then
        headers+=("$TRT_INCLUDE_DIR/NvInfer.h:TensorRT")
    fi
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
        "libworkflow.so:workflow"
        "libonnxruntime.so:onnxruntime"
        "libfmt.so:fmt"
        "libssl.so:openssl"
    )
    if [ "$DEP_PROFILE" != "cpu" ]; then
        libs+=(
            "libMNN_Cuda_Main.so:MNN CUDA"
            "libnvinfer.so.10:TensorRT"
            "libnvonnxparser.so.10:TensorRT parser"
            "libcudart.so.12:CUDA runtime"
            "libcudnn.so:cuDNN"
            "libOpenCL.so:OpenCL"
        )
    fi
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

    if [ "$DEP_PROFILE" != "cpu" ]; then
        local leftover
        leftover="$(leftover_gpu_line || true)"
        if [ -n "$leftover" ]; then
            echo "  [!!] leftover CUDA 11 / TensorRT 8 / ORT 1.18 artifacts:"
            while IFS= read -r line; do
                [ -n "$line" ] && echo "       $line"
            done <<<"$leftover"
            problems+=("leftover TRT8/cudart11/ORT1.18")
        else
            echo "  [ok] no leftover TensorRT 8 / cudart 11 / ORT 1.18 artifacts"
        fi
        if ls "$LIB_DIR"/libnvinfer.so.8* >/dev/null 2>&1; then
            problems+=("libnvinfer.so.8 present")
        fi
    fi

    # 4) onnxruntime artifact hash re-check (anti-tamper/corruption; recorded at install time)
    if [ -f "$STAMP_DIR/${ORT_STAMP}.libs.sha256" ]; then
        if (cd "$LIB_DIR" && sha256sum -c "$STAMP_DIR/${ORT_STAMP}.libs.sha256" >/dev/null 2>&1); then
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
        --cpu) DEP_PROFILE="cpu"; shift ;;
        --offline) OFFLINE_DIR="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done

CUDA_VERSION="${CUDA_VERSION%%.*}"
[ "$CUDA_VERSION" = "12" ] || fail "unsupported --cuda-version $CUDA_VERSION (GPU line is CUDA 12 only)"
apply_profile

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
        # toml++ v3.4.0 (marzer/tomlplusplus) - the app uses toml++'s API
        # (table::contains / value::value_or / as_array returning pointers).
        # The single header ships in the repo at the tag; toml11 (ToruNiina)
        # is a DIFFERENT library whose header broke the models build.
        announce "install toml++ v3.4.0 (single header)"
        require_cmd curl "curl"
        mkdir -p "$INCLUDE_DIR/toml"
        [ -f "$INCLUDE_DIR/toml/toml.hpp" ] || \
            curl -fsSL "https://raw.githubusercontent.com/marzer/tomlplusplus/v3.4.0/toml.hpp" -o "$INCLUDE_DIR/toml/toml.hpp"
        # Pin TOML_EXCEPTIONS=0 (the project-wide exceptions-free contract,
        # matching the tracked 3rd_party/include/toml/toml.hpp): upstream ships
        # exceptions enabled by default, so toml::parse_file returns toml::table
        # and the parse_result-based callers (benchmark_runner.h,
        # generic_cv_server.h, ...) fail to compile.
        if ! grep -q "^#ifndef TOML_EXCEPTIONS" "$INCLUDE_DIR/toml/toml.hpp"; then
            { printf '#ifndef TOML_EXCEPTIONS\n#define TOML_EXCEPTIONS 0\n#endif\n\n'; \
              cat "$INCLUDE_DIR/toml/toml.hpp"; } > "$INCLUDE_DIR/toml/toml.hpp.tmp" \
                && mv "$INCLUDE_DIR/toml/toml.hpp.tmp" "$INCLUDE_DIR/toml/toml.hpp"
        fi
        mark tomlpp
        install_header_only stb_image \
            "https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz" "" "stb_image" "stb_image"
        install_header_only indicators \
            "https://github.com/p-ranav/indicators/archive/refs/tags/v2.3.tar.gz" "include" "indicators" "indicators" "indicators.hpp"
        install_header_only moodycamel \
            "https://github.com/cameron314/concurrentqueue/archive/refs/tags/v1.0.4.tar.gz" "" "stl_container" "moodycamel"
        install_fmt
        install_workflow
        install_onnxruntime
        install_mnn
        install_system_runtime_libs
        if [ "$DEP_PROFILE" = "cpu" ]; then
            info "cpu profile: nvidia/TensorRT stack skipped entirely"
        elif [ "$(id -u)" -eq 0 ]; then
            # nvidia needs root; run it separately (so --all doesn't error out without root)
            install_nvidia
        else
            info "skipping nvidia (needs root): sudo ./scripts/install_deps.sh --nvidia"
        fi
        echo ""
        echo "== all done. verify: ./scripts/install_deps.sh --check"
        ;;
esac
