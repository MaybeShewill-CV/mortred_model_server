#!/usr/bin/env bash
# Detached A-set driver. Usage: restart_from.sh <id>
set -u
ROOT=/mnt/g/Codex/mortred_model_server
cd "$ROOT"
FROM="${1:?id}"
mkdir -p logs/bench/set_a
killall -q mortred-model-server.out 2>/dev/null || true
echo "=== restart all --from ${FROM} $(date '+%Y-%m-%d %H:%M:%S') ===" >> logs/bench/set_a/nohup.out
export PYTHONUNBUFFERED=1
setsid python3 -u scripts/bench_set_a/run.py all --from "$FROM" \
  </dev/null >>logs/bench/set_a/nohup.out 2>&1 &
echo "STARTED:$!"
sleep 2
pgrep -af 'scripts/bench_set_a/run.py' || echo NO_DRIVER
