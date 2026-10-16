#!/usr/bin/env bash
# Regenerate Round 5 progress board every 60 seconds from ledger merges.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/update_round005_progress.py"
LOG="$ROOT/progress_watch.log"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] progress_watch start pid=$$ root=$ROOT" | tee -a "$LOG"
while true; do
  if ! python3 "$PY" >>"$LOG" 2>&1; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] update failed (exit $?)" >>"$LOG"
  fi
  sleep 60
done
