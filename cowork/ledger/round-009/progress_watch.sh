#!/usr/bin/env bash
# Regenerate Round 9 progress board every 60s; exit when 100% done.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/update_round009_progress.py"
LOG="$ROOT/progress_watch.log"
PIDFILE="$ROOT/progress_watch.pid"
echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] progress_watch start pid=$$ root=$ROOT" | tee -a "$LOG"
echo $$ > "$PIDFILE"
while true; do
  out="$(python3 "$PY" 2>&1)" || true
  echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] $out" >> "$LOG"
  echo "$out"
  pct="$(python3 -c "import json; print(json.load(open('$ROOT/progress_data.json')).get('percent',0))" 2>/dev/null || echo 0)"
  status="$(python3 -c "import json; print(json.load(open('$ROOT/progress_data.json')).get('round_status',''))" 2>/dev/null || echo "")"
  if [ "$pct" = "100" ] || [ "$status" = "done" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S %Z')] all issues done; stopping watch" | tee -a "$LOG"
    rm -f "$PIDFILE"
    exit 0
  fi
  sleep 60
done
