# 技术方案
- 问题: P1-3
- 版本: v1
## 目标
Gateway e2e queue residency + worker_budget 429/layer
## 边界
- 不改 packing/Docker。
## 实现要点
见问题分支 diff。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 8 队列锁定 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
