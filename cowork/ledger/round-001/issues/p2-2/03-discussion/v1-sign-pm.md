# 第3步确认

- 问题: P2-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 5 已编号，业务后端对 gateway 发一组串行请求就能判定。标准 1 打开仓库 `conf/mortred.toml`，`[gateway.rate_limit]` 仍是 `enabled = false`、`shadow = false`、`rate_per_sec = 100000`、`burst = 200000`、`trusted_proxies = "127.0.0.1,::1"`。标准 2 在计数关闭、且没有 `MORTRED_RATE_LIMIT` 时，同一 `127.0.0.1` 的 3 次错误 Bearer 为 401 且没有 `Retry-After`，随后 4 次正确推理 token 为 200，`X-Mortred-Model` 为 `p2_2_fake`，正文为 `ok`，这 7 次没有 429。标准 3、4 打开同一开关：临时 toml `enabled = true`，或 toml 仍关闭并由 `MORTRED_RATE_LIMIT=enforce` 打开。`rate_per_sec = 1`、`burst = 3` 时，发射间隔为 1000 毫秒，瞬时名额为 3。3 次错误 token 之后仍有 3 次 200，下一次合法请求才是 429，`Retry-After`、`RateLimit-Limit`、`RateLimit-Remaining`、`RateLimit-Reset` 为 `1`、`1`、`0`、`1`。标准 5 先成功 1 次，再 3 次 401，随后还能成功 2 次，下一次才是同样的 429。401 没有吃掉剩余名额，也没有把已经用掉的名额还回来。各序列墙钟小于 900 毫秒，下一次请求在上一次状态行收到之后才发出。错误 token 仍是 401，到不了推理。这是现有 gateway 来源限流的扩展。健康检查一次 `GET /healthz` 不计入次数。按 key 的配额仍只在 key 已允许推理之后扣减。临时上游对任意 POST 回 200 和正文 `ok`，不启动 supervisor，不要求模型 JSON 的 `status`。
