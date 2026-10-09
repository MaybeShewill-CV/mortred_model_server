# 第3步签字

- 问题: D-1 models deadline 到期后 child 可能仍停在 starting
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的条件已经写进正文。验收第 4 条只保留三条既有用例已有断言。`autostart_models_failed_on_spawn_failure` 只判定 `missing_model` 的 `pid <= 0`、`reason` 含 `missing_model` 与 `failed`、`models_failed` 为 true。`empty_server_uri_fails_readiness_immediately` 判定 `fake_model_server` 的 `state` 为 `failed` 且 `error` 含 `server_uri`。`autostart_skips_models_when_gateway_deadline_missed` 判定 `models_skipped` 为 true 且 `fake_model_server` 的 `pid <= 0`。已 `running` 的兄弟 child 已列为验收第 6 条 `autostart_running_sibling_stays_running_on_peer_deadline`，用 `status("ready_sibling").state` 判定。
