# 第3步确认

- 问题: P2-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 方案只覆盖锁定句。打开 `[gateway.rate_limit]` 或设置 `MORTRED_RATE_LIMIT=enforce` 后，同一 `127.0.0.1` 用错误 token 打满 burst，紧接着的合法推理仍然成功，401 不改变该来源的剩余配额。验收标准 1 至 5 已编号，读产品 toml 或对 gateway 发一组串行请求即可判定。标准 1 要求 `[gateway.rate_limit]` 仍为 `enabled = false`、`shadow = false`、`rate_per_sec = 100000`、`burst = 200000`、`trusted_proxies = "127.0.0.1,::1"`。标准 2 至 5 各自使用新进程和空的来源表，临时 toml 为 `rate_per_sec = 1`、`burst = 3`，发射间隔 1000 毫秒，序列墙钟小于 900 毫秒。3 次错误 Bearer 为 401 且没有 `Retry-After`；随后 3 次 `infer-p2-2` 为 200，`X-Mortred-Model` 为 `p2_2_fake`，正文为 `ok`；再一次为 429，`Retry-After`、`RateLimit-Limit`、`RateLimit-Remaining`、`RateLimit-Reset` 为 `1`、`1`、`0`、`1`。先成功 1 次时，后面的 401 只撤回自己那一次。`check_ip_rate_limit` 在路由和鉴权之前调用 `gcra_admit`，模型路由的 401 在 `forward_to_model` 之前返回，归还写在该返回前，于来源分片锁内减去一个发射间隔。`/metrics` 的 401 在 admit 之前。404、405、502、503 和按 key 的 429 不归还。shadow 与 enforce 都归还这种 401。catalog 会收录 `profile = any`、带 `model` 且 `server_exe` 以 `.out` 结尾的临时 server toml，gateway 单独进程即可转发。三个互不相同的 token 能通过现有启动检查并听端口。不改 supervisor 的 fork/exec，不改 TensorRT execution context，不改 `worker_nums`，不把 P2-5 的启动拒绝拉进来。PM 与 RD 对 v1 的结论都是同意，条件都是无。
