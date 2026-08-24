#!/usr/bin/env bash
# mortredctl_init.sh - first-hour wizard core (invoked by `mortredctl init`).
#
# Runs ON the target machine / inside the installed tree. Detects hardware,
# recommends a deployment profile, fetches the matching weight subset, and
# verifies the result. Idempotent. Exit 0 = ready to start.
#
#   mortredctl init            # detect + fetch + verify
#   mortredctl init --profile cpu   # force a profile
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORCE_PROFILE=""

while [ $# -gt 0 ]; do
    case "$1" in
        --profile) FORCE_PROFILE="$2"; shift 2 ;;
        -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
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

# ---- 2) weights subset ----
if command -v python3 >/dev/null 2>&1; then
    echo ""
    echo "== weights (profile: $PROFILE) =="
    if python3 "$ROOT/scripts/fetch_weights.py" --profile "$PROFILE"; then
        echo "  weights: ok"
    else
        echo "  [WARN] weight fetch failed (offline?); rerun later:" >&2
        echo "         python3 $ROOT/scripts/fetch_weights.py --profile $PROFILE" >&2
    fi
else
    echo "  [WARN] python3 not found; fetch weights manually:" >&2
    echo "         python3 scripts/fetch_weights.py --profile $PROFILE" >&2
fi

# ---- 3) gpu extras: TRT engines are per-machine artifacts ----
if [ "$PROFILE" = "gpu" ]; then
    echo ""
    echo "== TensorRT engines (gpu only, optional but recommended) =="
    if [ -x "$ROOT/scripts/convert_trt_engines.sh" ]; then
        "$ROOT/scripts/convert_trt_engines.sh" --list || true
        echo "  convert missing engines:  $ROOT/scripts/convert_trt_engines.sh"
        echo "  (or set MORTRED_AUTO_BUILD_ENGINES=true before starting the supervisor)"
    fi
fi

# ---- 4) verify ----
echo ""
echo "== verify =="
if [ -x "$ROOT/scripts/verify_deployment.sh" ]; then
    "$ROOT/scripts/verify_deployment.sh" --basic
fi

cat <<EOF

== init done (profile: $PROFILE) ==
next:
  1. set tokens (MORTRED_API_TOKEN / MORTRED_GATEWAY_AUTH_TOKEN)
  2. start the supervisor (systemctl start mortred-supervisor, or docker compose)
  3. mortredctl doctor    # live acceptance
EOF
