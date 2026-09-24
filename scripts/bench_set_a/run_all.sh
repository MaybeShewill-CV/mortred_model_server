#!/usr/bin/env bash
set -u
ROOT=/mnt/g/Codex/mortred_model_server
cd "$ROOT"
mkdir -p logs/bench/set_a
export PYTHONUNBUFFERED=1
exec python3 -u scripts/bench_set_a/run.py all
