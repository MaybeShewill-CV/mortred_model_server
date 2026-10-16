# Round 13 queue (locked)

> Baseline HEAD: `fb982277` on `sub-agent-co-work`.
> Source: `/workspace/triage-defect-cards.md` + user-locked order.
> Packing/Docker OUT OF SCOPE except systemd KillMode under deploy/.
> P0-1 (dual-upstream CMake) already closed at tip — skipped.

## Locked order (11 cards)

1. **P0-2** TRT StagingSlotGuard/D2HSlotGuard: abandon or sync-then-release on error paths
2. **P1-5** superpoint sort >= is not strict weak ordering
3. **P0-3** http_infer_rps.py default warmup crashes; self-test after parse
4. **P1-1** positive max_queue_depth default; fill missing tomls; 429+Retry-After
5. **P1-4** resolve relative MORTRED_CONTROL_CONFIG via project_root
6. **P1-6** size cap / header probe BEFORE full imdecode
7. **P1-7** systemd KillMode=control-group + orphan self-exit
8. **P0-4** capacity: lower handler_threads; calibrate/sheet; RSS occupancy
9. **P1-2** unify YOLO unmap↔NMS order (v5/v6/v7 vs v8)
10. **P1-3** accept MORTRED_METRICS_TOKEN on model /metrics
11. **P1-8** residual: presets/labels SSOT; golden skip exit 77 locally
