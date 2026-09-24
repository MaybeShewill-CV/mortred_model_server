#!/usr/bin/env bash
# Redo failed HTTP rows first (msocrnet 1024x2048 convert is n/a-bench and blocks the GPU).
set -u
ROOT=/mnt/g/Codex/mortred_model_server
cd "$ROOT"
mkdir -p logs/bench/set_a
killall -q mortred-model-server.out 2>/dev/null || true
echo "=== continue $(date '+%Y-%m-%d %H:%M:%S') ===" >> logs/bench/set_a/nohup.out
export PYTHONUNBUFFERED=1
REDO='yolov8l,yolov6s,yolov5l,yolov7x,nanodet_1x5,nanodet_416,mobilenetv2,resnet,densenet,libface,modnet,ppmatting_512,ppmatting_1024,ppmatting_resnet34,ppmatting_v2,enlightengan,realesrgan,metric3d_512,metric3d_1088,hrnet'
setsid bash -c "cd '$ROOT'; export PYTHONUNBUFFERED=1; python3 -u scripts/bench_set_a/run.py all --only $REDO; python3 -u scripts/bench_set_a/run.py all --only enlightengan; python3 -u scripts/bench_set_a/run.py all --only msocrnet" \
  </dev/null >>logs/bench/set_a/nohup.out 2>&1 &
echo "STARTED:$!"
sleep 2
pgrep -af 'scripts/bench_set_a/run.py' || echo NO_DRIVER
