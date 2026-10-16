# 技术方案

- 问题: P1-1
- 版本: v1

## 目标
Rate-limit authority：Gateway 为唯一公网 limiter；全量 GCRA；去掉模型产品路径 FixedWindow；统一 429 头；RateLimiter 抽象。

## 边界
- 不改 packing/max_batch_size；不改 Docker。
- FixedWindowRateLimiter 类可保留给单测；产品路径（base_server_impl）删除 peer FixedWindow 计量。
- API-key QPS 改为 GCRA（gateway）。
- MORTRED_RATE_LIMIT_AUTHORITY 文档化：默认 unified；legacy 不再恢复模型 FixedWindow（产品路径已删）。

## 实现要点
1. 新增 control/rate_limit/rate_limiter.h（RateLimiter / GcraFlowLimiter）与 headers.h（统一 429 头）。
2. ApiKeyManager::allow_rate_limit 改为 gcra_admit。
3. base_server_impl 删除 FixedWindow allow 产品路径。
4. gateway layer 标签 api-key-gcra；http_wire 默认层更新。
5. 单测 rate_limit_authority_unittest。

## 确认
| 角色 | 结论 | 条件 | 方案版本 | 记录 |
| --- | --- | --- | --- | --- |
| 用户 | 同意 | 无 | v1 | 用户指令 Round 6 队列锁定 |
