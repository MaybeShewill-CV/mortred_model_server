# 第3步签字

- 问题: D-1 models deadline 到期后 child 可能仍停在 starting
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收第 4 条只保留三条既有用例已经断言的结果。`autostart_models_failed_on_spawn_failure` 判定 `missing_model` 的 `pid <= 0`、`reason` 含 `missing_model` 与 `failed`、`models_failed` 为 true。`empty_server_uri_fails_readiness_immediately` 判定 `state` 为 `failed` 且 `error` 含 `server_uri`。`autostart_skips_models_when_gateway_deadline_missed` 判定 `models_skipped` 为 true 且 `fake_model_server` 的 `pid <= 0`。这三条不新增 `status().state` 断言。兄弟 child 写成验收第 6 条 `autostart_running_sibling_stays_running_on_peer_deadline`：`start_concurrency` 只在该用例设为 2，返回后 `status("ready_sibling").state` 仍为 `running`，该用例期望退出码 0。终态仍是 `failed`。deadline 分支不发信号。外层 SIGKILL 与 `kStopGraceMs` 保持不变。
