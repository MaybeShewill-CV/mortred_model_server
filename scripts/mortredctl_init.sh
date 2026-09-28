#!/usr/bin/env bash
# mortredctl_init.sh - first-hour wizard core (invoked by `mortredctl init`).
#
# Runs ON the target machine / inside the installed tree. Detects hardware,
# recommends a deployment profile, fetches the matching weight subset, and
# verifies the result. Idempotent. Exit 0 = ready to start.
#
#   mortredctl init            # detect + fetch + verify
#   mortredctl init --profile cpu   # force a profile
#
# Fail-closed: weight fetch failure → non-zero (no "init done").
# GPU profile: current MORTRED_PACK missing product TRT engines → non-zero.
# CPU profile never requires TensorRT engines.
# Product GPU engines: mortredctl prepare (product_static_bs1 manifest).
# Bare convert_trt_engines.sh (default legacy trt_engines.json) is NOT the
# product OOB path.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_PROFILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --profile) FORCE_PROFILE="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "== Mortred init =="
echo "  root: $ROOT"

# ---- 1) profile detection ----
PROFILE="$FORCE_PROFILE"
if [ -z "$PROFILE" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        PROFILE="gpu"
        echo "  hardware: NVIDIA GPU detected -> profile gpu"
    else
        PROFILE="cpu"
        echo "  hardware: no NVIDIA GPU -> profile cpu"
    fi
fi
[ "$PROFILE" = "cpu" ] || [ "$PROFILE" = "gpu" ] || { echo "[ERROR] profile must be cpu or gpu" >&2; exit 1; }
echo "  profile: $PROFILE"
echo "$PROFILE" > "$ROOT/PROFILE" 2>/dev/null || true

PACK="${MORTRED_PACK:-$ROOT/conf/packs/demo.toml}"
WEIGHTS_FAIL=0
ENGINE_FAIL=0

# ---- 2) weights subset ----
if command -v python3 >/dev/null 2>&1; then
    echo ""
    echo "== weights (profile: $PROFILE) =="
    set +e
    python3 "$ROOT/scripts/fetch_weights.py" --profile "$PROFILE"
    FETCH_RC=$?
    set -e
    if [ "$FETCH_RC" -eq 0 ]; then
        echo "  weights: ok"
    else
        echo "  [FAIL] weight fetch failed (offline?); rerun:" >&2
        echo "         python3 $ROOT/scripts/fetch_weights.py --profile $PROFILE" >&2
        WEIGHTS_FAIL=1
    fi
else
    echo "  [FAIL] python3 not found; fetch weights manually:" >&2
    echo "         python3 scripts/fetch_weights.py --profile $PROFILE" >&2
    WEIGHTS_FAIL=1
fi

# ---- 3) gpu: product pack TRT engines (fail-closed; prepare is the OOB path) ----
if [ "$PROFILE" = "gpu" ]; then
    echo ""
    echo "== TensorRT engines (gpu pack: $PACK) =="
    echo "  product path: mortredctl prepare --pack $PACK"
    echo "  (bare scripts/convert_trt_engines.sh defaults to legacy conf/trt_engines.json;"
    echo "   that is NOT the product OOB entry — pass --manifest from product_static_bs1)"
    if [ ! -f "$PACK" ]; then
        echo "  [FAIL] pack file not found: $PACK" >&2
        ENGINE_FAIL=1
    elif ! command -v python3 >/dev/null 2>&1; then
        echo "  [FAIL] python3 required to check pack TRT engines" >&2
        ENGINE_FAIL=1
    else
        set +e
        TRT_OUT="$(python3 "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --check 2>&1)"
        TRT_RC=$?
        set -e
        if [ "$TRT_RC" -eq 0 ]; then
            echo "  engines: ok for pack"
        else
            printf '%s\n' "$TRT_OUT" >&2
            echo "  [FAIL] pack TensorRT engine missing/empty" >&2
            echo "next: mortredctl prepare --pack $PACK" >&2
            ENGINE_FAIL=1
        fi
    fi
fi

# ---- 4) verify (only when weights/engines gates pass) ----
if [ "$WEIGHTS_FAIL" -ne 0 ] || [ "$ENGINE_FAIL" -ne 0 ]; then
    echo ""
    if [ "$WEIGHTS_FAIL" -ne 0 ]; then
        echo "[FAIL] mortredctl init: weights not ready (profile: $PROFILE)"
    fi
    if [ "$ENGINE_FAIL" -ne 0 ]; then
        echo "[FAIL] mortredctl init: pack TRT engines not ready (pack: $PACK)"
    fi
    echo "next: mortredctl next"
    exit 1
fi

echo ""
echo "== verify =="
if [ -x "$ROOT/scripts/verify_deployment.sh" ]; then
    "$ROOT/scripts/verify_deployment.sh" --basic
fi

cat <<EOF

== init done (profile: $PROFILE) ==
why: first-hour weights/profile ready; follow the single OOB lane
next: mortredctl next
EOF
