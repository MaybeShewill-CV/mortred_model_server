# 技术方案
- 问题: P1-2
- 版本: v1
## 目标
max_queue 真正驻留至 upstream callback release；worker_budget 改为硬限制（429），不再仅 soft metric。
## 边界
- 不改 packing/Docker。
- max_inflight 语义保持；与 worker_budget 同时设置时取更紧的硬上限（两者都 enforce）。
## 实现要点
1. 去掉 inflate 成功后立刻 release_queue；QueueGuard 与 InflightGuard 同生命周期（upstream 回调析构释放）。
2. worker_budget>0 时：inflight 达预算则 429 + layer=api-key-worker-budget；释放已占 queue。
3. 更新 api_keys example 注释：worker_budget 为 hard。
## 确认
| 角色 | 结论 | 方案版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | Round 7 队列锁定 |
