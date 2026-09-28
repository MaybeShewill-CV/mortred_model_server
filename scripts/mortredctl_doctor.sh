#!/usr/bin/env bash
# mortredctl_doctor.sh - live deployment acceptance (invoked by
# `mortredctl doctor`). Pack occupancy + pack TRT gates fail-closed by
# default (name the missing engine/stamp). Security warnings never fail
# unless --strict. Then wraps verify_deployment.sh --live; falls back to
# --basic when the supervisor is not reachable so the output still tells
# the operator WHAT is wrong instead of an empty failure.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADDR="${MORTREDCTL_ADDR:-http://127.0.0.1:8787}"

STRICT_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --strict) STRICT_ARGS=(--strict) ;;
        -h|--help)
            echo "usage: mortredctl doctor [--strict]"
            echo "  (default) fail on missing pack TRT engines or occupancy stamps"
            echo "  --strict  also fail on security warnings, or occupancy_policy=off / ENFORCE=0"
            exit 0
            ;;
    esac
done

echo "== Mortred doctor =="
PACK="${MORTRED_PACK:-$ROOT/conf/packs/demo.toml}"
GATE_FAIL=0

echo "== GPU server ELF (ASan) =="
ASAN_HIT=0
for bin in \
    "$ROOT/_bin/mortred-model-server.out" \
    "$ROOT/bin/mortred-model-server.out" \
    /opt/mortred/bin/mortred-model-server.out; do
    [ -x "$bin" ] || continue
    command -v ldd >/dev/null 2>&1 || continue
    if ldd "$bin" 2>/dev/null | grep -Eq 'libasan\.|libclang_rt\.asan'; then
        echo "  [FAIL] AddressSanitizer-linked GPU server: $bin"
        echo "         Rebuild Release without -fsanitize=address:"
        echo "           cmake --preset full && cmake --build --preset full"
        ASAN_HIT=1
    fi
done
if [ "$ASAN_HIT" -eq 1 ]; then
    exit 1
fi
echo "  [ok] no libasan on mortred-model-server.out (or binary not present)"

echo "== pack occupancy / identity ($PACK) =="
if [ -f "$PACK" ]; then
    set +e
    OCC_OUT="$(python3 "$ROOT/scripts/pack_occupancy.py" --project-root "$ROOT" --pack "$PACK" --check 2>&1)"
    OCC_RC=$?
    set -e
    if [ "$OCC_RC" -eq 0 ]; then
        echo "  [ok] occupancy stamps present or pack has no TensorRT backends"
    else
        printf '%s\n' "$OCC_OUT"
        echo "  [FAIL] pack occupancy gate failed; stop supervisor, then calibrate"
        echo "next: mortredctl calibrate --pack $PACK --write-pack"
        GATE_FAIL=1
    fi
else
    echo "  [FAIL] pack file not found: $PACK"
    GATE_FAIL=1
fi

echo "== pack TensorRT files ($PACK) =="
if [ -f "$PACK" ]; then
    set +e
    TRT_OUT="$(python3 "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --check 2>&1)"
    TRT_RC=$?
    set -e
    if [ "$TRT_RC" -eq 0 ]; then
        echo "  [ok] pack TRT engines present or pack has no TensorRT backends"
    else
        printf '%s\n' "$TRT_OUT"
        echo "  [FAIL] pack TensorRT engine missing/empty"
        echo "next: mortredctl prepare --pack $PACK"
        GATE_FAIL=1
    fi
fi

if [ "$GATE_FAIL" -ne 0 ]; then
    echo "[FAIL] mortredctl doctor: pack TRT engines or occupancy stamps missing"
    echo "next: mortredctl next"
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
    echo "next: mortredctl next"
    exit 1
fi
