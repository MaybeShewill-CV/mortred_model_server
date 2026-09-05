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

# ---- Resolve trtexec (required for convert mode; dry-run falls back to TRT 8 syntax if missing) ----
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

# ---- Detect TRT major version: 8.x uses --workspace=<bytes>; 9+/10 use --memPoolSize=workspace:<size> ----
TRT_MAJOR="${TRT_VERSION_MAJOR:-}"
if [ -z "$TRT_MAJOR" ] && [ -n "$TRTEXEC" ]; then
    TRT_MAJOR="$("$TRTEXEC" --help 2>&1 | grep -m1 -oE 'version:?[[:space:]]*[0-9]+' | grep -oE '[0-9]+$' || true)"
fi
if [ -z "$TRT_MAJOR" ]; then
    if [ "$MODE" = "dry-run" ]; then
        TRT_MAJOR=8
        echo "[warn] dry-run: cannot detect TRT version, emitting 8.x syntax (override with TRT_VERSION_MAJOR)" >&2
    else
        fail "cannot detect TRT version (set TRT_VERSION_MAJOR or use the correct trtexec)"
    fi
fi

# vendored trtexec needs 3rd_party/libs on the dynamic library path
if [[ "$TRTEXEC" == "$ROOT/3rd_party/"* ]]; then
    export LD_LIBRARY_PATH="$LIB_DIR:${LD_LIBRARY_PATH:-}"
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

if [ "$TRT_MAJOR" -ge 9 ]; then
    WS_FLAG="--memPoolSize=workspace:$WORKSPACE_STR"
else
    WS_FLAG="--workspace=$(size_to_bytes "$WORKSPACE_STR")"
fi

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
    # flags are derived from the profile (space-separated --minShapes/--optShapes/--maxShapes)
    # shellcheck disable=SC2206
    args=(--onnx="$onnx_path" --saveEngine="$engine_path" --buildOnly)
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
        args=(--onnx="$onnx_path" --saveEngine="$engine_path" --buildOnly)
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
