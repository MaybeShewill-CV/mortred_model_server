#!/usr/bin/env bash
# build_trt_engine.sh - profile-driven TRT engine build without trtexec.
#
# Wrapper around scripts/trt_engine_builder.cc: compiles the builder once
# (cached in 3rd_party/bin) and translates a conf/trt_profiles/*.json profile
# into --min/--opt/--max flags. Drop-in for boxes lacking the TensorRT CLI;
# scripts/convert_trt_engines.sh stays the primary path when trtexec exists.
#
# Usage:
#   ./scripts/build_trt_engine.sh --onnx weights/x.onnx --save weights/x.engine \
#       [--fp16] --profile conf/trt_profiles/x.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/3rd_party/bin"
BUILDER="$BIN/trt_engine_builder"
ONNX=""; SAVE=""; PROFILE=""; FP16=""

while [ $# -gt 0 ]; do
    case "$1" in
        --onnx) ONNX="$2"; shift 2 ;;
        --save) SAVE="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --fp16) FP16="--fp16"; shift ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
    esac
done
if [ -z "$ONNX" ] || [ -z "$SAVE" ] || [ -z "$PROFILE" ]; then
    sed -n '2,13p' "$0"
    exit 1
fi

if [ ! -x "$BUILDER" ]; then
    TRT_INC="$(ls -d "$ROOT"/3rd_party/include/TensorRT-* 2>/dev/null | sort | head -1)"
    if [ -z "$TRT_INC" ]; then
        echo "[ERROR] no vendored TensorRT headers under 3rd_party/include" >&2
        exit 1
    fi
    mkdir -p "$BIN"
    g++ -std=c++17 -O2 "$ROOT/scripts/trt_engine_builder.cc" -o "$BUILDER" \
        -I"$ROOT/3rd_party/include" -I"$TRT_INC" -I/usr/local/cuda/include \
        -L"$ROOT/3rd_party/libs" -lnvinfer -lnvinfer_plugin -lnvonnxparser \
        -Wl,-rpath-link,"$ROOT/3rd_party/libs"
fi

# profile json -> trtexec-style name:d1xd2x... flags (one line per flag pair)
SHAPE_ARGS="$(python3 - "$PROFILE" <<'PY'
import json, sys
items = json.loads(open(sys.argv[1], encoding="utf-8-sig").read())
for key in ("min", "opt", "max"):
    for item in items:
        dims = "x".join(str(d) for d in item[key])
        print(f"--{key} {item['name']}:{dims}")
PY
)"

cd "$ROOT"
# shellcheck disable=SC2086 # SHAPE_ARGS is a controlled flag list
LD_LIBRARY_PATH="$ROOT/3rd_party/libs" "$BUILDER" --onnx "$ONNX" --save "$SAVE" $FP16 $SHAPE_ARGS
