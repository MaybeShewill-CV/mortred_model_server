#!/usr/bin/env bash
# mortredctl_next.sh - `mortredctl next`: print exactly ONE next OOB command.
#
# Probes trust (three tokens) → listen tip → supervisor up → pack TRT/occupancy
# → doctor. Exit 0 always when a next step (or done) is printed; exit 2 on usage.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="${MORTRED_PACK:-$ROOT/conf/packs/demo.toml}"
TRUST="${MORTRED_TRUST_ENV:-$ROOT/conf/local/trust.env}"
ADDR="${MORTREDCTL_ADDR:-http://127.0.0.1:8787}"

while [ $# -gt 0 ]; do
    case "$1" in
        --pack) PACK="$2"; shift 2 ;;
        --trust) TRUST="$2"; shift 2 ;;
        -h|--help)
            echo "usage: mortredctl next [--pack FILE] [--trust FILE]"
            echo "  prints one next out-of-box command (docs/oob-main-path.md)"
            exit 0
            ;;
        *) echo "[ERROR] unknown argument: $1" >&2; exit 2 ;;
    esac
done

say_next() {
    # $1 = command line; $2 = short why
    echo "why: $2"
    echo "next: $1"
}

is_loopback_host() {
    local h
    h="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]' | tr -d '\r')"
    case "$h" in
        ''|localhost|127.*|::1|'[::1]') return 0 ;;
    esac
    return 1
}

# ---- 1) three tokens / trust.env ----
need_tokens=0
for k in MORTRED_GATEWAY_AUTH_TOKEN MORTRED_API_TOKEN MORTRED_METRICS_TOKEN; do
    if [ -z "${!k:-}" ]; then
        need_tokens=1
        break
    fi
done

if [ "$need_tokens" -eq 1 ]; then
    if [ ! -f "$TRUST" ]; then
        say_next "mortredctl init-trust" \
            "missing three tokens (no $TRUST); generate machine-local trust.env"
        exit 0
    fi
    say_next "set -a && . $TRUST && set +a" \
        "trust.env exists but the three MORTRED_*_TOKEN vars are not in this shell"
    exit 0
fi

# ---- 2) listen (non-loopback tip; does not block cpu demo on 127.0.0.1) ----
gw_host="${MORTRED_GATEWAY_HOST:-}"
api_host="${MORTRED_API_HOST:-}"
if ! is_loopback_host "$gw_host" || ! is_loopback_host "$api_host"; then
    say_next "mortredctl init-edge --mode lan" \
        "gateway/supervisor listen is not loopback (plain HTTP); terminate TLS at Nginx or bind 127.0.0.1"
    exit 0
fi

# ---- 3) supervisor reachable ----
sup_ok=0
if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 3 "${ADDR%/}/api/v1/health" >/dev/null 2>&1; then
        sup_ok=1
    fi
fi
if [ "$sup_ok" -ne 1 ]; then
    if [ -x "$ROOT/_bin/mortred-supervisor.out" ]; then
        say_next "export MORTRED_PACK=\"$PACK\" MORTRED_PROFILE=\"\${MORTRED_PROFILE:-cpu}\" && \"$ROOT/_bin/mortred-supervisor.out\"" \
            "supervisor not reachable at $ADDR; start it (source-tree)"
    elif [ -x "$ROOT/bin/mortred-supervisor.out" ]; then
        say_next "export MORTRED_PACK=\"$PACK\" MORTRED_PROFILE=\"\${MORTRED_PROFILE:-cpu}\" && \"$ROOT/bin/mortred-supervisor.out\"" \
            "supervisor not reachable at $ADDR; start it (installed tree)"
    elif command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files mortred-supervisor.service >/dev/null 2>&1; then
        say_next "sudo systemctl start mortred-supervisor" \
            "supervisor not reachable at $ADDR; start the systemd unit"
    elif [ -f "$ROOT/docker-compose.yml" ] || [ -f "$ROOT/compose.yaml" ]; then
        say_next "docker compose --profile \${MORTRED_PROFILE:-cpu} up -d" \
            "supervisor not reachable at $ADDR; start compose"
    else
        say_next "start mortred-supervisor (see docs/oob-main-path.md)" \
            "supervisor not reachable at $ADDR"
    fi
    exit 0
fi

# ---- 4) pack TensorRT engines (no-op OK for cpu-only packs) ----
if [ -f "$PACK" ]; then
    if ! python3 "$ROOT/scripts/pack_trt.py" --project-root "$ROOT" --pack "$PACK" --check >/dev/null 2>&1; then
        say_next "mortredctl prepare --pack $PACK" \
            "pack TensorRT engines missing/empty for $PACK"
        exit 0
    fi
    if ! python3 "$ROOT/scripts/pack_occupancy.py" --project-root "$ROOT" --pack "$PACK" --check >/dev/null 2>&1; then
        say_next "mortredctl calibrate --pack $PACK --write-pack" \
            "pack occupancy stamps missing/stale for $PACK (stop supervisor first if it holds ports)"
        exit 0
    fi
else
    say_next "export MORTRED_PACK=$ROOT/conf/packs/demo.toml" \
        "pack file not found: $PACK"
    exit 0
fi

# ---- 5) doctor --strict is the acceptance gate ----
say_next "mortredctl doctor --strict" \
    "trust + loopback listen + supervisor + pack gates look ready; run acceptance"
exit 0
