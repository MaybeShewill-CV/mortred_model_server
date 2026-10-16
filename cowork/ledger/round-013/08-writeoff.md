# 关闭记录

- 轮次: 13
- 已关闭: P0-2、P1-5、P0-3、P1-1、P1-4、P1-6、P1-7、P0-4、P1-2、P1-3、P1-8（**11/11**；P0-1 跳过已在 tip 关闭）
- 阻塞: 无
- 结果: **done**
- HEAD: `e200ab9998355eb4a75869a3c36d42896c31f685`

## 合入表

| id | 问题分支 | merge | 一句话 |
| --- | --- | --- | --- |
| P0-2 | `8453c81a` | `514b044d` | TRT H2D/D2H sync-then-release on abandon |
| P1-5 | `0a36f1c7` | `f0bfef79` | superpoint sort `>=` → `>` |
| P0-3 | `aad2bc46` | `e1550d2f` | http_infer_rps allow warmup 0; self-test after parse |
| P1-1 | `f915f595` | `a9ac51c2` | max_queue_depth default 32; fill 5 tomls; 429 path kept |
| P1-4 | `ab777c17` | `e804f862` | relative CONTROL_CONFIG via project_root |
| P1-6 | `0409d986` | `c14255a9` | JPEG SOF + PNG IHDR before imdecode |
| P1-7 | `11bf407f` | `7bf9755a` | KillMode=control-group + PDEATHSIG note |
| P0-4 | `0ede69ea` | `2a1f070d` | handler_threads 8/16 + capacity sheet |
| P1-2 | `09b0ed2b` | `fec578f3` | yolov8 unmap then NMS |
| P1-3 | `cc3a8ff8` | `cf322469` | MORTRED_METRICS_TOKEN on model /metrics |
| P1-8 | `c363ea97` | `e200ab99` | golden skip 77 docs + sanitizer SSOT comment |

## Residuals (honest)

- P0-2: no GPU live fault-injection of mid-copy failure.
- P0-4: process RSS not sampled in occupancy (documented out-of-band measured JSON only).
- P1-2: broader matting/seg/libface family audit parked (`issues/p1-2/06-residual-family.md` if present).
- P1-6: non-JPEG/PNG formats still defer size check until after imdecode.
- P1-8: behavior already exit-77; this round is visibility/docs polish.
- Packing/Docker still out of scope per user.
