# 关闭记录

- 轮次: 14
- 已关闭: P0-1、P1-1、P0-2、P1-2、P1-3、P1-4（**6/6**）
- 阻塞: 无
- 结果: **done**
- HEAD: `ff78c4852a1815700132d3f37c04eacc461a3f04`
- Generated: 2026-10-16 22:36:44 CST

## 合入表

| id | 问题分支 | merge | 一句话 |
| --- | --- | --- | --- |
| P0-1 | `3aee1cef` | `2fc9be64` | TRT buffer.ensure fail → abandon + structural inject check |
| P1-1 | `d8cae7aa` | `65217ace` | TSan SSOT script + golden empty/mixed exit 77 |
| P0-2 | `a6edbdae` | `b1eb6562` | calibrate RSS/thread/fd floors; occupancy RAM gate |
| P1-2 | `ba312dd2` | `60d9026a` | BMP/GIF/WebP/TIFF pre-decode size gate |
| P1-3 | `ac515586` | `ff78c485` | same-family libface/score/matting/seg/GPU note |
| P1-4 | `6ef9bfc8` | `93787cc1` | supervisor child env allowlist (+ METRICS_TOKEN) |

## Residuals (honest)

- P0-1: structural `check_trt_abandon_paths.py` only; no live GPU mid-copy fault injection on this host.
- P0-2: thread_count/fd_count are calibrated floors (recorded); spawn hard-gates RSS/host RAM, not thread/fd ceilings.
- P1-2: unknown containers still defer size check until after imdecode (BMP/GIF/WebP/TIFF covered).
- P1-3: GPU vs CPU preprocess remains nearest vs INTER_LINEAR by design (documented); no numeric harness yet.
- P1-4: allowlist is fixed runtime keys + injected MORTRED_*; new CUDA/runtime keys may need allowlist updates.
- Packing/Docker still out of scope.
- Git history note: P1-4 merge commit is older than P1-3 merge (parallel worktree finish order); locked queue was P1-3 then P1-4.
