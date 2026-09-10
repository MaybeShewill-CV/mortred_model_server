#!/usr/bin/env bash
# mortredctl_doctor.sh - live deployment acceptance (invoked by
# `mortredctl doctor`). Prints security warnings (never fail unless
# --strict), then wraps verify_deployment.sh --live; falls back to --basic
# when the supervisor is not reachable so the output still tells the operator
# WHAT is wrong instead of an empty failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADDR="${MORTREDCTL_ADDR:-http://127.0.0.1:8787}"

STRICT_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT_ARGS=(--strict) ;;
        -h|--help)
            echo "usage: mortredctl doctor [--strict]"
            echo "  --strict  fail on security warnings, missing pack TRT engines,"
            echo "            missing occupancy stamps, or occupancy_policy=off / ENFORCE=0"
            exit 0
            ;;
    esac
done

echo "== Mortred doctor =="
PACK="${MORTRED_PACK:-$ROOT/conf/packs/demo.toml}"
GATE_FAIL=0

echo "== pack occupancy / identity ($PACK) =="
if [ -f "$PACK" ]; then
    if python3 "$ROOT/scripts/pack_occupancy.py" --project-root "$ROOT" --pack "$PACK" --check; then
        echo "  [ok] occupancy stamps present or pack has no TensorRT backends"
    else
        echo "  [WARN] pack occupancy gate failed; stop supervisor, then: mortredctl calibrate --pack $PACK --write-pack"
        GATE_FAIL=1
    fi
else
    echo "  [WARN] pack file not found: $PACK"
fi

echo "== pack TensorRT files ($PACK) =="
if [ -f "$PACK" ]; then
    if python3 "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --check; then
        echo "  [ok] pack TRT engines present or pack has no TensorRT backends"
    else
        echo "  [WARN] pack TensorRT engine missing/empty; run mortredctl prepare --pack $PACK"
        GATE_FAIL=1
    fi
fi

if [ ${#STRICT_ARGS[@]} -gt 0 ] && [ "$GATE_FAIL" -ne 0 ]; then
    echo "[FAIL] mortredctl doctor --strict: pack TRT engines or occupancy stamps missing (or occupancy disabled)"
    exit 1
fi

"$ROOT/scripts/security_warn.sh" "${STRICT_ARGS[@]}"

if command -v curl >/dev/null 2>&1 && curl -fs --max-time 5 "$ADDR/api/v1/health" >/dev/null 2>&1; then
    echo "  supervisor: reachable ($ADDR)"
    "$ROOT/scripts/verify_deployment.sh" --live
else
    echo "  supervisor: NOT reachable at $ADDR -> falling back to static checks"
    "$ROOT/scripts/verify_deployment.sh" --basic
    echo ""
    echo "[FAIL] live checks skipped: start the service first"
    exit 1
fi
