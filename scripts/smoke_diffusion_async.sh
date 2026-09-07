#!/usr/bin/env bash
# smoke_diffusion_async.sh - end-to-end smoke test for async diffusion sampling.
# Requires: GPU + weights + built model server binary.
#
# Usage:
#   ./scripts/smoke_diffusion_async.sh                    # default: ddpm, 10 timesteps
#   ./scripts/smoke_diffusion_async.sh --model ddim --timestep 20
#   MORTRED_SERVER_BIN=/path/to/mortred-model-server.out ./scripts/smoke_diffusion_async.sh
set -euo pipefail

MODEL="ddpm"
TIMESTEP=10
PORT=""
SERVER_PID=""

while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --timestep) TIMESTEP="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_BIN="${MORTRED_SERVER_BIN:-$ROOT/_bin/mortred-model-server.out}"
ASAN_SO=""

# ASan ELFs abort if libasan is not first on the load list. Prepending
# _lib / 3rd_party to LD_LIBRARY_PATH is enough to trigger that. Resolve
# the soname with a clean ldd; preload only the server process so curl and
# python3 are not intercepted (CPython "leaks" under LSan are noise here).
if command -v ldd >/dev/null 2>&1 && [ -e "$SERVER_BIN" ]; then
    ASAN_SO="$(env -u LD_PRELOAD LD_LIBRARY_PATH= ldd "$SERVER_BIN" 2>/dev/null | awk '/libasan/{print $3; exit}')"
    if [ -n "${ASAN_SO:-}" ] && [ -e "$ASAN_SO" ]; then
        echo "[smoke] ASan ELF: will LD_PRELOAD=$ASAN_SO for the server only"
    else
        ASAN_SO=""
    fi
fi
# Prefer the lib/ next to the chosen ELF (build-cpu/bin -> build-cpu/lib).
# Default BUILD_RPATH and a naive `_lib:` prefix both load repo `_lib`, which
# on a mixed tree is often an older libmodels.so without ImagePipeline::letterbox.
BIN_DIR="$(cd "$(dirname "$SERVER_BIN")" && pwd)"
NEAR_LIB="$(cd "$BIN_DIR/.." && pwd)/lib"
LIB_PATHS=""
if [ -d "$NEAR_LIB" ]; then
    LIB_PATHS="$NEAR_LIB"
    echo "[smoke] lib search: $NEAR_LIB (alongside $SERVER_BIN)"
fi
export LD_LIBRARY_PATH="${LIB_PATHS:+$LIB_PATHS:}$ROOT/_lib:$ROOT/3rd_party/libs:${LD_LIBRARY_PATH:-}"

if command -v ldd >/dev/null 2>&1 && command -v nm >/dev/null 2>&1 && [ -e "$SERVER_BIN" ]; then
    MODELS_SO="$(ldd "$SERVER_BIN" 2>/dev/null | awk '/libmodels/{print $3; exit}')"
    if [ -n "${MODELS_SO:-}" ] && [ -e "$MODELS_SO" ]; then
        echo "[smoke] libmodels=$MODELS_SO"
        # grep -q closes the pipe on the first match; with pipefail, nm then
        # exits 141 (SIGPIPE) and this check falsely reports a stale .so.
        if ! nm -D "$MODELS_SO" 2>/dev/null | grep 'ImagePipeline9letterbox' >/dev/null; then
            echo "[FAIL] $MODELS_SO has no ImagePipeline::letterbox (stale .so)."
            echo "  Rebuild models against the current tree, then re-run:"
            echo "    touch src/models/backend/model_runtime.cpp"
            echo "    cmake --build <build-dir> --target models mortred-model-server.out -j\"\$(nproc)\""
            echo "    nm -D $MODELS_SO | grep letterbox   # must print a T symbol"
            exit 1
        fi
    fi
fi

case "$MODEL" in
    ddpm) CONFIG="conf/server/diffusion/ddpm/ddpm_server_config.toml"; MODEL_ID="DDPM"; PARAM_KEY="timesteps" ;;
    ddim) CONFIG="conf/server/diffusion/ddim/ddim_server_config.toml"; MODEL_ID="DDIM"; PARAM_KEY="sample_steps" ;;
    *) echo "unsupported model: $MODEL (use ddpm or ddim)"; exit 1 ;;
esac

if [ -z "$PORT" ]; then
    PORT="$(awk -F= '/^port=/ { gsub(/[[:space:]]/, "", $2); print $2; exit }' "$ROOT/$CONFIG")"
fi
if [ -z "$PORT" ]; then
    echo "could not read port from $CONFIG (pass --port)"; exit 1
fi

BODY="$(python3 -c "import json,sys; print(json.dumps({'images':['aGVsbG8='],'req_id':'smoke-test','params':{sys.argv[1]: int(sys.argv[2])}}))" "$PARAM_KEY" "$TIMESTEP")"

if [ ! -x "$SERVER_BIN" ]; then
    echo "[FAIL] missing executable $SERVER_BIN (set MORTRED_SERVER_BIN or build mortred-model-server.out)"
    exit 1
fi

echo "[smoke] model=$MODEL $PARAM_KEY=$TIMESTEP port=$PORT bin=$SERVER_BIN"

# start the model server with async enabled
echo "[smoke] starting server..."
SERVER_LOG="$(mktemp /tmp/mortred-diffusion-smoke.XXXXXX.log)"
SERVER_CMD=( "$SERVER_BIN" --model "$MODEL_ID" "$ROOT/$CONFIG" )
if [ -n "$ASAN_SO" ]; then
    # detect_leaks=0: this smoke SIGTERM-kills the process; LSan on ONNX/glog
    # teardown is not the generate-path gate.
    env LD_PRELOAD="${ASAN_SO}${LD_PRELOAD:+:$LD_PRELOAD}" \
        ASAN_OPTIONS="${ASAN_OPTIONS:+${ASAN_OPTIONS}:}detect_leaks=0" \
        "${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
else
    "${SERVER_CMD[@]}" >"$SERVER_LOG" 2>&1 &
fi
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true; rm -f "$SERVER_LOG"' EXIT

ready=0
for _ in $(seq 1 30); do
    if curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[FAIL] server exited before healthz"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 1
done
if [ "$ready" -ne 1 ]; then
    echo "[FAIL] server did not become ready on :$PORT within 30s"
    cat "$SERVER_LOG"
    exit 1
fi
echo "[smoke] server ready"

# submit async job (unified envelope: params.<key>, not a root timestep)
echo "[smoke] submitting async job..."
SUBMIT=$(curl -s -w '\n%{http_code}' -X POST "http://127.0.0.1:$PORT/jobs" \
    -H "Content-Type: application/json" \
    -d "$BODY")
SUBMIT_CODE=$(echo "$SUBMIT" | tail -1)
SUBMIT_BODY=$(echo "$SUBMIT" | head -n -1)

if [ "$SUBMIT_CODE" != "202" ]; then
    echo "[FAIL] submit returned $SUBMIT_CODE: $SUBMIT_BODY"
    exit 1
fi
JOB_ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['job_id'])" "$SUBMIT_BODY")
echo "[smoke] submitted: job_id=$JOB_ID (HTTP 202)"

# poll until done (max 10 minutes)
echo "[smoke] polling..."
START=$(date +%s)
for i in $(seq 1 600); do
    STATUS=$(curl -s "http://127.0.0.1:$PORT/jobs/$JOB_ID")
    STATE=$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('state',''))" "$STATUS")
    ELAPSED=$(( $(date +%s) - START ))

    if [ "$STATE" = "done" ]; then
        echo "[smoke] done in ${ELAPSED}s"
        RESULT=$(curl -s "http://127.0.0.1:$PORT/jobs/$JOB_ID/result")
        python3 - "$RESULT" <<'PY'
import json, sys
doc = json.loads(sys.argv[1])
status = doc.get("status")
results = doc.get("results") or []
image = ""
if results and isinstance(results[0], dict):
    data = results[0].get("data") or {}
    image = data.get("image") or ""
print("status=%s image_len=%d" % (status, len(image)))
if status != 0:
    sys.stderr.write("[FAIL] result status=%s (expected 0)\n%s\n" % (status, sys.argv[1]))
    sys.exit(1)
if not image:
    sys.stderr.write("[FAIL] results[0].data.image missing\n%s\n" % sys.argv[1])
    sys.exit(1)
PY
        echo "[PASS] $MODEL async smoke test: submit->poll->result OK (${ELAPSED}s)"
        exit 0
    elif [ "$STATE" = "failed" ] || [ "$STATE" = "timeout" ]; then
        echo "[FAIL] job $STATE: $STATUS"
        exit 1
    fi

    if [ $((i % 30)) -eq 0 ]; then
        echo "[smoke] still $STATE after ${ELAPSED}s..."
    fi
    sleep 1
done

echo "[FAIL] job did not complete within 10 minutes"
exit 1
