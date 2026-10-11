# 第3步签字

- 问题: D-9
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `process` 先 `check_ip_rate_limit` 再鉴权；`init` 已经把 `rate_limit_enabled_` 设成 `enabled || shadow`，但这道门仍无条件 `admit`。两者都为假时直接放行、不调用 `admit`，验收 1 的 401 不再吃掉 `burst = 1`，下一次合法请求会进 `forward_to_model`，上游拒绝连接时现有路径就是 HTTP 503 和 `service not ready` 信封；验收 2 的两次合法请求因此都是 503，而不是 429。`allow_rate_limit` 只在 `authenticate` 里、且在 `key_may_infer` 之前给 `rate_window_count` 加一；把这一次增加挪到 `key_may_infer` 为真之后，scope `none` 连续两次仍走现有 401，scope `inference` 同一秒内第二次仍走现有 `api key rate limit exceeded`。`authenticate` 只有网关在用，模型进程不用改。401、503、429 的 JSON 原文已经由 `reply_unified_error` 打出，临时 catalog 只改副本端口即可转发。两处一起改，四条黑盒一次能看见冻结的三句。
