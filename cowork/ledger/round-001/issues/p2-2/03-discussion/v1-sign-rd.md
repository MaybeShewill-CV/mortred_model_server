# 第3步确认

- 问题: P2-2
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `check_ip_rate_limit` 已在路由和鉴权之前调用 `gcra_admit`。允许时 TAT 增加一个发射间隔，拒绝时不写 TAT。模型路由的 401 在 `forward_to_model` 之前返回，可在该返回前于来源分片锁内减去一个发射间隔；未建表的来源不新建表项。`/metrics` 的 401 在 admit 之前返回，404、405、502、503 和按 key 的 429 都不经过这条归还。`enabled` 与 `shadow` 都为假时仍在 admit 之前返回。`rate_per_sec = 1`、`burst = 3` 的发射间隔是 1000 毫秒；序列墙钟小于 900 毫秒时，3 次错误 Bearer 归还后仍剩 3 次瞬时名额，随后 3 次合法推理为 200，再一次为 429，`Retry-After`、`RateLimit-Limit`、`RateLimit-Remaining`、`RateLimit-Reset` 分别为 1、1、0、1。先有一次已计入的 200 时，后面的 401 只撤回自己的那一次。catalog 会收录 `profile = any` 的临时 server toml，gateway 单独进程就能把 `/v1/models/p2_2_fake/infer` 转到临时上游并带上 `X-Mortred-Model`。自测的 CMake 目标会落到 `_bin`，不改 supervisor 进程树，也不改 TensorRT context。
