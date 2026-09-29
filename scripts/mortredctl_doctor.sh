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

    # Prefer trust.env so status can be fetched without a pre-exported shell.
    TRUST="${MORTRED_TRUST_ENV:-$ROOT/conf/local/trust.env}"
    if [ -z "${MORTREDCTL_TOKEN:-}${MORTRED_API_TOKEN:-}" ] && [ -f "$TRUST" ]; then
        # shellcheck disable=SC1090
        set -a && . "$TRUST" && set +a
    fi
    TOKEN="${MORTREDCTL_TOKEN:-${MORTRED_API_TOKEN:-}}"
    STATUS_HDR=()
    if [ -n "$TOKEN" ]; then
        STATUS_HDR=(-H "Authorization: Bearer ${TOKEN}")
    fi

    echo "== supervisor autostart / gateway gate =="
    set +e
    STATUS_JSON="$(curl -fsS --max-time 5 "${STATUS_HDR[@]}" "$ADDR/api/v1/status" 2>/dev/null)"
    STATUS_RC=$?
    set -e
    if [ "$STATUS_RC" -ne 0 ] || [ -z "$STATUS_JSON" ]; then
        echo "  [FAIL] cannot read $ADDR/api/v1/status (need MORTREDCTL_TOKEN / MORTRED_API_TOKEN)"
        echo "next: set -a && . $TRUST && set +a"
        exit 1
    fi

    set +e
    GATE_EVAL="$(STATUS_JSON="$STATUS_JSON" python3 - <<'PY'
import json, os, sys
d = json.loads(os.environ["STATUS_JSON"])
gate = d.get("autostart") or {}
gw = d.get("gateway") or {}
blocked = bool(gate.get("blocked")) or bool(gate.get("models_skipped"))
models_failed = bool(gate.get("models_failed"))
reason = (gate.get("reason") or "").strip()
gw_err = (gw.get("error") or "").strip()
gw_state = (gw.get("state") or "").strip()
text = reason or gw_err
fail = blocked or models_failed or gw_state == "failed" or (
    text and ("scrape" in text.lower() or "deadline" in text.lower()
              or "binary missing" in text.lower() or "not ready" in text.lower())
)
print("blocked=%s" % ("1" if blocked else "0"))
print("models_failed=%s" % ("1" if models_failed else "0"))
print("gw_state=%s" % gw_state)
print("reason=%s" % text)
sys.exit(1 if fail else 0)
PY
)"
    GATE_RC=$?
    set -e
    printf '%s\n' "$GATE_EVAL" | sed 's/^/  /'
    if [ "$GATE_RC" -ne 0 ]; then
        REASON_LINE="$(printf '%s\n' "$GATE_EVAL" | sed -n 's/^reason=//p' | head -1)"
        MODELS_FAILED_LINE="$(printf '%s\n' "$GATE_EVAL" | sed -n 's/^models_failed=//p' | head -1)"
        if [ "$MODELS_FAILED_LINE" = "1" ]; then
            echo "  [FAIL] model wave failed after gateway was admitted"
        else
            echo "  [FAIL] gateway/autostart blocked; models were not started in this wave"
        fi
        if [ -n "$REASON_LINE" ]; then
            echo "  reason: $REASON_LINE"
        fi
        echo "  do not expect routed inference until gateway is running"
        echo "next: mortredctl init-trust  # if scrape token missing/colliding"
        echo "next: mortredctl next"
        exit 1
    fi
    echo "  [ok] gateway running and autostart not blocked"

    "$ROOT/scripts/verify_deployment.sh" --live
else
    echo "  supervisor: NOT reachable at $ADDR -> falling back to static checks"
    "$ROOT/scripts/verify_deployment.sh" --basic
    echo ""
    echo "[FAIL] live checks skipped: start the service first"
    echo "next: mortredctl next"
    exit 1
fi
