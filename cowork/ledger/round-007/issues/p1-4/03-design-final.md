# 技术方案
- 问题: P1-4
- 版本: v1
## 目标
删除产品路径残留 FixedWindow 成员与冒充产品 layer 的陈旧注释；FixedWindowRateLimiter 类可保留给单测。
## 边界
- 不改 packing/Docker。
- 不删除 src/server/rate_limiter.h（单测仍可用）。
## 实现要点
1. base_server_impl.h：删除 `_m_rate_limiter` 成员及 set_max_qps/(void) 占位。
2. 去掉产品路径对 FixedWindowRateLimiter 的 include（若仅因此引入）。
3. 清理 gcra.h / api_key_manager.h / headers 中 model-fixed-window 产品 layer 叙述。
## 确认
| 角色 | 结论 | 方案版本 | 记录 |
| --- | --- | --- | --- |
| 用户 | 同意 | v1 | Round 7 队列锁定 |
