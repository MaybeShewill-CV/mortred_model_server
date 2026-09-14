#!/usr/bin/env bash
# convert_trt_engines.sh - use external trtexec (TensorRT official CLI) to generate
# hardware-adapted TensorRT engines from conf/trt_engines.json.
#
# Background: the .engine files config expects depend on the user's GPU arch / TRT version,
# so they must be built locally (engines in the shipped weights may mismatch the current TRT).
# This script converts onnx sources into engines at config-referenced paths, replacing
# mismatched files. The in-house converter has been removed.
#
# Usage (run from the repo root):
#   ./scripts/convert_trt_engines.sh                  # convert missing engines
#   ./scripts/convert_trt_engines.sh --force          # reconvert all (overwrite existing)
#   ./scripts/convert_trt_engines.sh --list           # print manifest (no trtexec needed)
#   ./scripts/convert_trt_engines.sh --only yolov8    # convert only entries whose path contains yolov8
#   ./scripts/convert_trt_engines.sh --strict         # exit on first failure (CI-friendly)
#   ./scripts/convert_trt_engines.sh --check-engines  # only verify existing engines (exist + non-empty)
#   ./scripts/convert_trt_engines.sh --dry-run        # only print the commands that would run
#   ./scripts/convert_trt_engines.sh --trtexec /path/to/trtexec
#   TRT_VERSION_MAJOR=10 ./scripts/convert_trt_engines.sh --dry-run   # when trtexec cannot be probed
# Product line: TensorRT 10.x only (TRT_VERSION_MAJOR < 10 is refused).
#
# trtexec lookup order: $TRTEXEC (env/--trtexec) → 3rd_party/bin/trtexec
#                  (installed by install_deps.sh --nvidia) → PATH → /usr/src/tensorrt/bin/trtexec
# Deps: missing onnx files are downloaded with scripts/fetch_weights.py.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/conf/trt_engines.json"
LIB_DIR="$ROOT/3rd_party/libs"
TRTEXEC="${TRTEXEC:-}"
FORCE=0
ONLY=""
MODE="convert"
STRICT=0
# Match the old in-house converter's 6GB workspace; override with TRTEXEC_WORKSPACE
WORKSPACE_STR="${TRTEXEC_WORKSPACE:-6G}"

usage() {
    sed -n '2,21p' "$0"
    exit 0
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        --list) MODE="list"; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --strict) STRICT=1; shift ;;
        --check-engines) MODE="check-engines"; shift ;;
        --dry-run) MODE="dry-run"; shift ;;
        --trtexec) TRTEXEC="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) fail "unknown argument: $1 (see --help)" ;;
    esac
done

[ -f "$MANIFEST" ] || fail "manifest not found: $MANIFEST"

# ---- Resolve a working python (skip broken PATH stubs like WindowsApps aliases) ----
resolve_python() {
    local cand p
    for cand in python3 python py; do
        if command -v "$cand" >/dev/null 2>&1; then
            p="$(command -v "$cand")"
            if "$p" -c 'import sys' >/dev/null 2>&1; then
                echo "$p"
                return 0
            fi
        fi
    done
    return 1
}
PY="$(resolve_python)" || fail "missing a working python3/python (needed to parse $MANIFEST)"

# ---- Parse manifest + profiles with python, emit TSV: model<TAB>onnx<TAB>engine<TAB>fp<TAB>shape_flags ----
# Write a temp file instead of process substitution: mapfile+heredoc+process substitution is unreliable in Windows Git Bash
TMPLIST="$(mktemp)" || fail "mktemp failed"
if ! "$PY" - "$ROOT" "$MANIFEST" "$ONLY" >"$TMPLIST" <<'PY'
import json, sys
from pathlib import Path
root, manifest_path, only = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))

def dims(v):
    return "x".join(str(d) for d in v)

for e in manifest.get("engines", []):
    path = e.get("engine", "")
    if only and only.lower() not in path.lower():
        continue
    flags = ""
    if e.get("profile"):
        prof = json.loads((root / e["profile"]).read_text(encoding="utf-8-sig"))
        mins, opts, maxs = [], [], []
        for b in prof:  # works for multiple bindings (lightglue matcher has 4 bindings)
            mins.append(f'{b["name"]}:{dims(b["min"])}')
            opts.append(f'{b["name"]}:{dims(b["opt"])}')
            maxs.append(f'{b["name"]}:{dims(b["max"])}')
        flags = ("--minShapes=" + ",".join(mins) + " "
                 "--optShapes=" + ",".join(opts) + " "
                 "--maxShapes=" + ",".join(maxs))
    print("\t".join([e.get("model", ""), e.get("onnx", ""), path,
                     str(e.get("fp", 0)), flags]))
PY
then
    rm -f "$TMPLIST"
    fail "failed to parse manifest: $MANIFEST"
fi
mapfile -t ENTRIES < "$TMPLIST"
rm -f "$TMPLIST"
if [ "${#ENTRIES[@]}" -eq 0 ]; then
    [ -n "$ONLY" ] && fail "no entries matching '$ONLY' (see --list)"
    fail "manifest is empty: $MANIFEST"
fi

# ---- list: read-only manifest, no trtexec needed ----
if [ "$MODE" = "list" ]; then
    printf "%-32s %-70s %s\n" "MODEL" "ONNX" "ENGINE"
    for line in "${ENTRIES[@]}"; do
        IFS=$'\t' read -r model onnx engine fp flags <<<"$line"
        printf "%-32s %-70s %s\n" "$model" "$onnx" "$engine"
    done
    echo "total: ${#ENTRIES[@]}"
    exit 0
fi

# ---- check-engines: only verify existing engines (exist + non-empty), no trtexec needed ----
if [ "$MODE" = "check-engines" ]; then
    bad=0
    for line in "${ENTRIES[@]}"; do
        IFS=$'\t' read -r model onnx engine fp flags <<<"$line"
        if [ -s "$ROOT/$engine" ]; then
            echo "  [ok] $engine"
        else
            echo "  [!!] $model: engine missing or empty $engine"
            bad=$((bad+1))
        fi
    done
    [ "$bad" -eq 0 ] || exit 1
    exit 0
fi

# ---- Resolve trtexec (required for convert; dry-run may omit it only if TRT_VERSION_MAJOR=10 is set) ----
if [ -z "$TRTEXEC" ]; then
    for cand in "$ROOT/3rd_party/bin/trtexec" \
                "$(command -v trtexec 2>/dev/null || true)" \
                "/usr/src/tensorrt/bin/trtexec" \
                "${TENSORRT_ROOT:-/usr/src/tensorrt}/bin/trtexec"; do
        [ -n "$cand" ] && [ -x "$cand" ] && { TRTEXEC="$cand"; break; }
    done
fi
if [ -z "$TRTEXEC" ] && [ "$MODE" != "dry-run" ]; then
    fail "trtexec not found (install via sudo ./scripts/install_deps.sh --nvidia; or pass --trtexec /path/to/trtexec)"
fi

# trtexec AND libnvinfer.so.10 both hard-require libcuda.so.1 (verified via
# DT_NEEDED; TRT 10 bundles everything else statically). On WSL that file is
# injected by the Windows driver into /usr/lib/wsl/lib, which a stale ldconfig
# cache frequently misses - put the directory on the path explicitly.
extra_lib_path=""
[ -d /usr/lib/wsl/lib ] && extra_lib_path="/usr/lib/wsl/lib"
# vendored trtexec dlopens libnvinfer.so.10 at startup (the version banner
# comes from the loaded library), so 3rd_party/libs must already be on the
# library path when the version probe below runs - not only for conversions.
if [[ "$TRTEXEC" == "$ROOT/3rd_party/"* ]]; then
    extra_lib_path="$LIB_DIR${extra_lib_path:+:$extra_lib_path}"
fi
[ -n "$extra_lib_path" ] && export LD_LIBRARY_PATH="${extra_lib_path}:${LD_LIBRARY_PATH:-}"

# ---- Detect TRT major version (product pin: TensorRT 10.x only, SME-16) ----
# TRT_VERSION_MAJOR is an explicit override when the trtexec banner cannot be
# parsed — it does NOT exempt major < 10. There is no silent dry-run fallback.
TRT_MAJOR_OVERRIDE="${TRT_VERSION_MAJOR:-}"
TRT_MAJOR=""
trt_probe=""
if [ -n "$TRT_MAJOR_OVERRIDE" ]; then
    TRT_MAJOR="$TRT_MAJOR_OVERRIDE"
elif [ -n "$TRTEXEC" ]; then
    trt_probe="$("$TRTEXEC" --help 2>&1 || true)"
    TRT_MAJOR="$(printf '%s\n' "$trt_probe" | grep -m1 -oE 'version:?[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' || true)"
fi
if [ -z "$TRT_MAJOR" ]; then
    if [ -n "$TRTEXEC" ]; then
        echo "---- $TRTEXEC --help output (first 8 lines) ----" >&2
        printf '%s\n' "$trt_probe" | head -n 8 >&2
        echo "------------------------------------------------" >&2
    fi
    fail "cannot detect TensorRT major version (need a probeable trtexec, or set TRT_VERSION_MAJOR=10 for dry-run without trtexec). Product line is TensorRT 10.x only — install via: sudo ./scripts/install_deps.sh --nvidia"
fi
case "$TRT_MAJOR" in
    ''|*[!0-9]*)
        fail "invalid TensorRT major '$TRT_MAJOR' (TRT_VERSION_MAJOR must be an integer >= 10)"
        ;;
esac
if [ "$TRT_MAJOR" -lt 10 ]; then
    fail "TensorRT major $TRT_MAJOR is not supported (product line is TensorRT 10.x only; leftover 8/9 are out of scope). Install the pinned stack: sudo ./scripts/install_deps.sh --nvidia"
fi

size_to_bytes() {
    local s="$1" n u
    n="${s%[KkMmGg]}"
    u="${s: -1}"
    case "$u" in
        K|k) echo $((n*1024)) ;;
        M|m) echo $((n*1024*1024)) ;;
        G|g) echo $((n*1024*1024*1024)) ;;
        *) echo "$n" ;;
    esac
}

# trtexec 10.x's --memPoolSize parser only accepts KiB/MiB/GiB base-2 suffixes
# (a bare number means MiB); the 8.x-style "6G" form fails to parse.
ws_to_trt10_units() { # 6G -> 6GiB, 512m -> 512MiB, 1024 -> 1024, 6GiB -> 6GiB
    local s="$1"
    if [[ "$s" =~ ^([0-9]+([.][0-9]+)?)([KkMmGgTt])$ ]]; then
        case "${BASH_REMATCH[3]}" in
            [Kk]) echo "${BASH_REMATCH[1]}KiB" ;;
            [Mm]) echo "${BASH_REMATCH[1]}MiB" ;;
            [Gg]) echo "${BASH_REMATCH[1]}GiB" ;;
            [Tt]) echo "${BASH_REMATCH[1]}TiB" ;;
        esac
    else
        echo "$s"
    fi
}

# Classify an ONNX model's inputs as static or dynamic WITHOUT any python
# packages: parse the protobuf wire format directly (ModelProto.graph=7 →
# GraphProto.input=11 / initializer=5 → ValueInfoProto.type=2 →
# TypeProto.tensor_type=1 → Tensor.shape=2 → TensorShapeProto.dim=1 →
# Dimension dim_value=1 (fixed) / dim_param=2 (symbolic)). Prints "static" or
# "dynamic"; any parse failure exits nonzero so callers can fall back.
onnx_shape_kind() { # <onnx-file>
    python3 - "$1" <<'PY'
import sys

def varint(buf, i):
    v = shift = 0
    while True:
        b = buf[i]; i += 1
        v |= (b & 0x7F) << shift
        if not b & 0x80:
            return v, i
        shift += 7

def fields(buf):
    i, n = 0, len(buf)
    while i < n:
        tag, i = varint(buf, i)
        f, wt = tag >> 3, tag & 7
        if wt == 0:
            v, i = varint(buf, i)
            yield f, wt, v
        elif wt == 2:
            l, i = varint(buf, i)
            yield f, wt, buf[i:i + l]
            i += l
        elif wt == 5:
            yield f, wt, buf[i:i + 4]; i += 4
        elif wt == 1:
            yield f, wt, buf[i:i + 8]; i += 8
        else:
            raise ValueError("wire type %d" % wt)

def input_dims(vi):
    name, shape = None, None
    for f, wt, v in fields(vi):
        if f == 1 and wt == 2:
            name = v.decode("utf-8", "replace")
        elif f == 2 and wt == 2:              # TypeProto
            for f2, w2, v2 in fields(v):
                if f2 == 1 and w2 == 2:       # tensor_type
                    for f3, w3, v3 in fields(v2):
                        if f3 == 2 and w3 == 2:
                            shape = v3        # TensorShapeProto
    if shape is None:
        return name, None
    dims = []
    for f, wt, v in fields(shape):
        if f == 1 and wt == 2:                # Dimension
            val = None
            for fd, wd, vd in fields(v):
                if fd == 1 and wd == 0:
                    val = vd                  # dim_value: fixed
                elif fd == 2 and wd == 2:
                    val = -1                  # dim_param: symbolic
            dims.append(val)
    return name, dims

def fail(msg):
    sys.stderr.write("onnx scan: %s\n" % msg)
    sys.exit(1)

try:
    data = open(sys.argv[1], "rb").read()
except OSError as e:
    fail(str(e))
try:
    graph = None
    for f, wt, v in fields(data):
        if f == 7 and wt == 2:                    # ModelProto.graph
            graph = v
            break
    if graph is None:
        fail("no graph field - not an ONNX file?")
    inits, inputs = set(), []
    for f, wt, v in fields(graph):
        if f == 5 and wt == 2:                    # initializer (weights)
            for fi, wi, vi in fields(v):
                if fi == 1 and wi == 2:           # TensorProto.name
                    inits.add(vi.decode("utf-8", "replace"))
        elif f == 11 and wt == 2:                 # graph.input
            inputs.append(v)
    real = [vi for vi in inputs
            if (input_dims(vi)[0] or "") not in inits]
    if not real:
        fail("no non-initializer graph inputs found")
    dynamic = False
    for vi in real:
        name, dims = input_dims(vi)
        if dims is None or any(d is None or d < 0 for d in dims):
            dynamic = True
    print("dynamic" if dynamic else "static")
except (ValueError, IndexError) as e:
    fail("cannot parse (%r)" % (e,))
PY
}

# Flags for TensorRT 10.x only (--buildOnly / bare --workspace are TRT 8).
# Banner of trtexec 10.3 lists: --skipInference / KiB|MiB|GiB.
WS_FLAG="--memPoolSize=workspace:$(ws_to_trt10_units "$WORKSPACE_STR")"
BUILD_FLAG="--skipInference"

converted=0; skipped=0; missing_onnx=0; failed=0
declare -a failed_models=()
for line in "${ENTRIES[@]}"; do
    IFS=$'\t' read -r model onnx engine fp flags <<<"$line"
    onnx_path="$ROOT/$onnx"
    engine_path="$ROOT/$engine"
    if [ ! -f "$onnx_path" ]; then
        echo "[skip] $model: onnx missing $onnx (run ./scripts/fetch_weights.py --only $model first)"
        missing_onnx=$((missing_onnx+1))
        continue
    fi
    if [ -f "$engine_path" ] && [ "$FORCE" -eq 0 ]; then
        echo "[skip] $model: engine already exists $engine (add --force to reconvert)"
        skipped=$((skipped+1))
        continue
    fi
    case "$fp" in
        0) fp_flag="" ;;
        1) fp_flag="--fp16" ;;
        *)
            echo "[FAIL] $model: unknown fp=$fp (only 0=FP32 / 1=FP16 supported)"
            failed=$((failed+1)); failed_models+=("$model")
            [ "$STRICT" -eq 1 ] && exit 1
            continue ;;
    esac
    # Pre-scan the ONNX: a static-shaped model rejects explicit
    # --min/--opt/--maxShapes, so drop the profile flags up front instead of
    # failing over via the trtexec retry below. Scan failure (no python3,
    # exotic file) keeps the flags and lets the retry handle it.
    if [ -n "$flags" ]; then
        shape_kind="$(onnx_shape_kind "$onnx_path" 2>/dev/null || true)"
        if [ "$shape_kind" = "static" ]; then
            echo "[info] $model: static-shape ONNX; dropping shape profile flags"
            flags=""
        fi
    fi
    # flags are derived from the profile (space-separated --minShapes/--optShapes/--maxShapes)
    # shellcheck disable=SC2206
    args=(--onnx="$onnx_path" --saveEngine="$engine_path" "$BUILD_FLAG")
    [ -n "$fp_flag" ] && args+=("$fp_flag")
    args+=($flags)
    args+=("$WS_FLAG")
    echo "[convert] $model: fp=$fp${flags:+ profile=$flags}"
    if [ "$MODE" = "dry-run" ]; then
        echo "  cmd: $TRTEXEC ${args[*]}"
        continue
    fi
    mkdir -p "$(dirname "$engine_path")"
    run_trtexec() {
        # shellcheck disable=SC2034
        out="$("$TRTEXEC" "$@" 2>&1)"
    }
    converted_ok=0
    if run_trtexec "${args[@]}"; then
        converted_ok=1
    elif [ -n "$flags" ] && echo "$out" | grep -q "Static model does not take explicit shapes"; then
        echo "[warn] $model: ONNX inputs are static; retrying without min/opt/maxShapes"
        args=(--onnx="$onnx_path" --saveEngine="$engine_path" "$BUILD_FLAG")
        [ -n "$fp_flag" ] && args+=("$fp_flag")
        args+=("$WS_FLAG")
        if run_trtexec "${args[@]}"; then
            converted_ok=1
        fi
    fi
    if [ "$converted_ok" -eq 1 ]; then
        if [ -s "$engine_path" ]; then
            converted=$((converted+1))
            echo "  -> $engine"
        else
            failed=$((failed+1)); failed_models+=("$model")
            echo "[FAIL] $model: trtexec returned 0 but engine is missing or empty"
            [ "$STRICT" -eq 1 ] && exit 1
        fi
    else
        rc=$?
        failed=$((failed+1)); failed_models+=("$model")
        echo "[FAIL] $model: trtexec failed${rc:+ (exit code $rc)}"
        echo "$out" | tail -n 15
        [ "$STRICT" -eq 1 ] && exit 1
    fi
done

echo ""
echo "== done: converted $converted, skipped (existing) $skipped, missing onnx $missing_onnx, failed $failed"
if [ "$failed" -gt 0 ]; then
    echo "== failed entries:"
    for m in "${failed_models[@]}"; do
        echo "   - $m"
    done
    echo "== tip: add --strict to stop at the first failure; if dynamic-input models fail, add a profile under conf/trt_profiles/ and record it in conf/trt_engines.json"
    exit 1
fi
[ "$missing_onnx" -eq 0 ] || echo "== tip: missing onnx files: run scripts/fetch_weights.py to download them"
exit 0
