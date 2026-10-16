# Round 8 discovered

## 进入本轮队列（用户锁定 v2）

- P0-1: check_consistency 与薄 openapi_doc.h loader 不对齐。
- P1-1: worker_budget 在 max_inflight<=0 时硬门失效。
- P1-2: max_queue/worker_budget TOML load 缺回归单测。
- P1-3: gateway e2e 缺 queue residency 与 worker_budget 429/layer。
- P1-4: NvmlGpuProbe 每 sample dlopen/dlclose。
- P1-5: CI evidence 可改进（无 GPU runner）；headers.h layer 注释重复。

## 未入列

- 外部审计候选 v1 全表：用户改锁，见 `02-p0-p1-final.md` 附录「未采用」。
