#!/usr/bin/env bash
# smoke_diffusion_async.sh - end-to-end smoke test for async diffusion sampling.
# Requires: GPU + weights + built model server binary.
#
# Usage:
#   ./scripts/smoke_diffusion_async.sh                    # default: ddpm, 100 timesteps
#   ./scripts/smoke_diffusion_async.sh --model ddim --timestep 50
set -euo pipefail

MODEL="ddpm"
TIMESTEP=100
PORT=9070
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
export LD_LIBRARY_PATH="$ROOT/_lib:$ROOT/3rd_party/libs:${LD_LIBRARY_PATH:-}"

case "$MODEL" in
    ddpm) CONFIG="conf/model/diffusion/ddpm/ddpm_celeba-hq.toml" ;;
    ddim) CONFIG="conf/model/diffusion/ddpm/ddim_celeba-hq.toml" ;;
    *) echo "unsupported model: $MODEL (use ddpm or ddim)"; exit 1 ;;
esac

echo "[smoke] model=$MODEL timestep=$TIMESTEP port=$PORT"

# start the model server with async enabled
echo "[smoke] starting server..."
"$ROOT/_bin/${MODEL}_server.out" "$ROOT/$CONFIG" &
SERVER_PID=$!
sleep 3

# check server is up
if ! curl -sf "http://127.0.0.1:$PORT/healthz" > /dev/null 2>&1; then
    echo "[FAIL] server did not start (check logs)"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo "[smoke] server ready"

# submit async job
echo "[smoke] submitting async job..."
SUBMIT=$(curl -s -w '\n%{http_code}' -X POST "http://127.0.0.1:$PORT/jobs" \
    -H "Content-Type: application/json" \
    -d "{\"img_data\":\"aGVsbG8=\",\"req_id\":\"smoke-test\",\"timestep\":$TIMESTEP}")
SUBMIT_CODE=$(echo "$SUBMIT" | tail -1)
SUBMIT_BODY=$(echo "$SUBMIT" | head -n -1)

if [ "$SUBMIT_CODE" != "202" ]; then
    echo "[FAIL] submit returned $SUBMIT_CODE: $SUBMIT_BODY"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
JOB_ID=$(echo "$SUBMIT_BODY" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
echo "[smoke] submitted: job_id=$JOB_ID (HTTP 202)"

# poll until done (max 10 minutes)
echo "[smoke] polling..."
START=$(date +%s)
for i in $(seq 1 600); do
    STATUS=$(curl -s "http://127.0.0.1:$PORT/jobs/$JOB_ID")
    STATE=$(echo "$STATUS" | grep -o '"state":"[^"]*"' | cut -d'"' -f4)
    ELAPSED=$(( $(date +%s) - START ))

    if [ "$STATE" = "done" ]; then
        echo "[smoke] done in ${ELAPSED}s"
        RESULT=$(curl -s "http://127.0.0.1:$PORT/jobs/$JOB_ID/result")
        CODE=$(echo "$RESULT" | grep -o '"code":[0-9]*' | cut -d: -f2)
        echo "[smoke] result code=$CODE"
        if [ "$CODE" = "0" ]; then
            echo "[PASS] $MODEL async smoke test: submit->poll->result OK (${ELAPSED}s)"
            kill $SERVER_PID 2>/dev/null || true
            exit 0
        else
            echo "[FAIL] result code=$CODE (expected 0)"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
    elif [ "$STATE" = "failed" ] || [ "$STATE" = "timeout" ]; then
        echo "[FAIL] job $STATE: $STATUS"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi

    if [ $((i % 30)) -eq 0 ]; then
        echo "[smoke] still $STATE after ${ELAPSED}s..."
    fi
    sleep 1
done

echo "[FAIL] job did not complete within 10 minutes"
kill $SERVER_PID 2>/dev/null || true
exit 1
