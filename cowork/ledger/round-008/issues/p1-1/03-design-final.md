# 技术方案
- 问题: P1-1
- 版本: v1
## 目标
worker_budget 硬门在 max_inflight<=0 时仍有效（始终 track inflight）。
## 边界
- 不改 packing/Docker。
- max_inflight 语义不变。
## 实现要点
1. try_acquire_inflight / release_inflight 在 worker_budget>0 时始终增减 inflight。
2. gateway 现有 worker_budget 429 + layer 分支保持。
3. 单测 + smoke 覆盖 max_inflight=0, worker_budget=2。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 8 队列锁定 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
