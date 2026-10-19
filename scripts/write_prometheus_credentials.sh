#!/usr/bin/env bash
# Write the Prometheus scrape credentials file (raw token, no "Bearer " prefix).
# Used by deploy/docker-compose.monitoring.yml and bare-metal Prometheus.
#
#   set -a && . conf/local/trust.env && set +a
#   ./scripts/write_prometheus_credentials.sh
#   sudo ./scripts/write_prometheus_credentials.sh /etc/prometheus/mortred_metrics_token
#
# Does not mint a token. MORTRED_METRICS_TOKEN must already be set and distinct.
#
# The file is mode 600. The reader is not the writer:
#   default path (compose bind-mount) -> uid 65534 (prom/prometheus nobody)
#   /etc/prometheus/*                 -> user prometheus (apt systemd unit)
# Override with MORTRED_METRICS_TOKEN_OWNER=user[:group] (or uid:gid).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="write"
case "${1:-}" in
    --check) MODE="check"; shift ;;
    --self-test) MODE="self-test"; shift ;;
    --help|-h)
        cat <<'USAGE'
usage: write_prometheus_credentials.sh [--check|--self-test] [OUT]
  (no flag)   write OUT from $MORTRED_METRICS_TOKEN (mode 600, owner adjusted)
  --check     verify OUT matches $MORTRED_METRICS_TOKEN (exit 1 when stale)
  --self-test exercise --check against matching/mismatching/missing files
USAGE
        exit 0 ;;
esac
OUT="${1:-$ROOT/conf/local/mortred_metrics_token}"

check_only() {
    : "${MORTRED_METRICS_TOKEN:?set MORTRED_METRICS_TOKEN (mortredctl init-trust) before --check}"
    if [ ! -f "$OUT" ]; then
        echo "[ERROR] $OUT does not exist; run ./scripts/write_prometheus_credentials.sh" >&2
        exit 1
    fi
    local have
    have="$(tr -d '\r\n' < "$OUT")"
    if [ "$have" = "$MORTRED_METRICS_TOKEN" ]; then
        echo "metrics credentials match MORTRED_METRICS_TOKEN ($OUT)"
        exit 0
    fi
    echo "[ERROR] $OUT does not match MORTRED_METRICS_TOKEN" >&2
    echo "  the Prometheus scrape credential is stale: scrapes would get 401 with no other signal" >&2
    echo "  fix: set -a && . conf/local/trust.env && set +a && ./scripts/write_prometheus_credentials.sh $OUT" >&2
    exit 1
}

self_test() {
    local fails=0 tmp f
    tmp="$(mktemp -d)"
    f="$tmp/metrics_token"
    if ( MORTRED_METRICS_TOKEN=abc "$0" --check "$f" >/dev/null 2>&1 ); then
        echo "FAIL: missing file accepted"; fails=$((fails + 1))
    fi
    printf 'abc\n' > "$f"
    if ! ( MORTRED_METRICS_TOKEN=abc "$0" --check "$f" >/dev/null 2>&1 ); then
        echo "FAIL: matching file rejected"; fails=$((fails + 1))
    fi
    printf 'def\n' > "$f"
    if ( MORTRED_METRICS_TOKEN=abc "$0" --check "$f" >/dev/null 2>&1 ); then
        echo "FAIL: stale file accepted"; fails=$((fails + 1))
    fi
    if ( unset MORTRED_METRICS_TOKEN; "$0" --check "$f" >/dev/null 2>&1 ); then
        echo "FAIL: unset token accepted"; fails=$((fails + 1))
    fi
    rm -rf "$tmp"
    echo "SUMMARY FAILS=$fails"
    return "$fails"
}

case "$MODE" in
    self-test) self_test; exit $? ;;
    check) check_only ;;
esac

: "${MORTRED_METRICS_TOKEN:?set MORTRED_METRICS_TOKEN (mortredctl init-trust)}"

file_uid() {
    if stat -c %u "$1" >/dev/null 2>&1; then
        stat -c %u "$1"
    else
        stat -f %u "$1"
    fi
}

owner_uid() {
    local user="${1%%:*}"
    if [[ "$user" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$user"
        return
    fi
    id -u "$user"
}

resolve_owner() {
    if [ -n "${MORTRED_METRICS_TOKEN_OWNER:-}" ]; then
        printf '%s\n' "$MORTRED_METRICS_TOKEN_OWNER"
        return
    fi
    case "$OUT" in
        /etc/prometheus|/etc/prometheus/*)
            if id prometheus >/dev/null 2>&1; then
                printf '%s\n' "prometheus:prometheus"
                return
            fi
            echo "[ERROR] $OUT is a systemd Prometheus path but user prometheus is missing." >&2
            echo "  apt install prometheus, or set MORTRED_METRICS_TOKEN_OWNER=user:group" >&2
            exit 1
            ;;
        *)
            printf '%s\n' "65534:65534"
            ;;
    esac
}

mkdir -p "$(dirname "$OUT")"
umask 077
# Prometheus reads the file as the Bearer secret; a trailing newline is fine.
printf '%s\n' "$MORTRED_METRICS_TOKEN" > "$OUT"
chmod 600 "$OUT"

wanted="$(resolve_owner)"
wanted_uid="$(owner_uid "$wanted")"
current_uid="$(file_uid "$OUT")"

if [ "$current_uid" = "$wanted_uid" ]; then
    echo "wrote $OUT (mode 600, uid $wanted_uid)"
    exit 0
fi

if chown "$wanted" "$OUT" 2>/dev/null; then
    echo "wrote $OUT (mode 600, owner $wanted)"
    exit 0
fi

if [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1; then
    echo "Prometheus reads $OUT as $wanted; requesting sudo chown"
    sudo chown "$wanted" "$OUT"
    echo "wrote $OUT (mode 600, owner $wanted)"
    exit 0
fi

echo "[ERROR] $OUT is uid $current_uid mode 600; reader $wanted cannot open it." >&2
echo "  sudo chown $wanted $OUT && sudo chmod 600 $OUT" >&2
exit 1
