#!/usr/bin/env bash
# Write the Prometheus scrape credentials file (raw token, no "Bearer " prefix).
# Used by deploy/docker-compose.monitoring.yml and bare-metal Prometheus.
#
#   set -a && . conf/local/trust.env && set +a
#   ./scripts/write_prometheus_credentials.sh
#
# Does not mint a token. MORTRED_METRICS_TOKEN must already be set and distinct.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/conf/local/mortred_metrics_token}"

: "${MORTRED_METRICS_TOKEN:?set MORTRED_METRICS_TOKEN (mortredctl init-trust)}"

mkdir -p "$(dirname "$OUT")"
umask 077
# Prometheus reads the file as the Bearer secret; a trailing newline is fine.
printf '%s\n' "$MORTRED_METRICS_TOKEN" > "$OUT"
chmod 600 "$OUT"
echo "wrote $OUT (mode 600)"
