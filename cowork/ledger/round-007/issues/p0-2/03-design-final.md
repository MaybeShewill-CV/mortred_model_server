# 技术方案
- 问题: P0-2
- 版本: v1
## 目标
ApiKeyManager::load 解析 max_queue 与 worker_budget，消除死字段。
## 边界
- 不改 packing/Docker。
- residency/硬预算见 P1-2。
## 实现要点
1. load() 解析 max_queue、worker_budget（缺省 0）。
2. 更新 api_keys.toml.example 注释。
3. 单测证明解析非零。
## 确认
| 角色 | 结论 | 方案版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | Round 7 队列锁定 |
