# 技术方案
- 问题: P0-1
- 版本: v1
## 目标
`/ready` 仅表示进程已成功初始化且健康，不再要求 `available_approx()>0`。满载时 LB 探针仍得 200；饱和由 `mortred_workers_busy` / `mortred_workers_available` 暴露。
## 边界
- 不改 packing/Docker。
- 不改 worker 调度与 429 背压语义。
- `/healthz` 不变。
## 实现要点
1. `base_server_impl.h` `/ready`：`ready = _m_successfully_initialized`（去掉 available_approx 合取）。
2. 继续 `set_ready(ready)`；worker 水位仍由既有 `set_workers_available` / `set_workers_busy` 更新。
3. 单测：就绪谓词或探针契约覆盖「init 且 available==0 → ready」。
## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | Round 9 队列锁定文案即方案 |
| cv-pm | 同意 | 无 | v1 | 用户锁定 |
| cv-tech-lead | 同意 | 无 | v1 | 用户锁定 |
| cv-rd | 同意 | 无 | v1 | 用户锁定 |
