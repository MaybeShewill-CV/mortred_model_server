#!/usr/bin/env bash
# mortredctl_calibrate.sh - `mortredctl calibrate` → pack worker_nums report.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$ROOT/scripts/calibrate_pack.py" "$@"
