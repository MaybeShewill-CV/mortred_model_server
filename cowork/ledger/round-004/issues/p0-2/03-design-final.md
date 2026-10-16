# 技术方案

- 问题: P0-2
- 版本: v1

默认 enabled=true；公网 listen 且限流全关时拒绝启动，除非 MORTRED_RATE_LIMIT_ALLOW_DISABLE=1。

## 确认
| 角色 | 结论 | 条件 | 方案版本 |
| --- | --- | --- | --- |
| cv-pm/rd/tech-lead | 同意 | 无 | v1 |
