#!/usr/bin/env bash
# microbench - the three microbenchmarks behind the measured downgrades in
# cowork/progress-card.md. Each was the acceptance gate for a proposed change;
# the numbers below are the recorded baselines (RTX 2070 / WSL2 / 16 threads,
# 2026-10-21). Re-run before touching the corresponding code.
#
#   metrics_bench  prometheus_metrics.h per-request cost
#                  baseline: 30.8 ns/call single-thread; ~1.02M request-mixes/s
#                  under 8-thread contention -> the global mutex is NOT a
#                  bottleneck at realistic QPS; do not rewrite to atomics
#                  without new evidence.
#   nchw_bench     models/backend/nchw_host.h NHWC->NCHW permute
#                  baseline: v0 element-wise 2.4ms @1x640x640x3 f32; channel-
#                  vector memcpy / 32x32 tiling / dst-major candidates all
#                  <=1.1x -> the existing implementation is already optimal
#                  among simple forms; the real lever is MNN-native NHWC.
#   envelope_bench request-envelope parse pipeline cost
#                  baseline: current 5.07ms vs ParseInsitu+views 4.38ms
#                  (-13%, below the >=30% gate) -> the in-situ single-buffer
#                  surgery was rejected; raw path got the move instead.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

g++ -O2 -std=c++17 -I "$ROOT/src" -I "$ROOT/3rd_party/include" \
    "$ROOT/scripts/microbench/metrics_bench.cc" -o "$OUT/metrics_bench" -lglog -lpthread
g++ -O2 -std=c++17 -I "$ROOT/src" -I "$ROOT/3rd_party/include" \
    "$ROOT/scripts/microbench/envelope_bench.cc" -o "$OUT/envelope_bench" -lglog
g++ -O2 -std=c++17 "$ROOT/scripts/microbench/nchw_bench.cc" -o "$OUT/nchw_bench"

img="${1:-$ROOT/demo_data/model_test_input/mot/2.jpg}"
"$OUT/metrics_bench"
echo ---
"$OUT/nchw_bench"
echo ---
"$OUT/envelope_bench" "$img" 200
