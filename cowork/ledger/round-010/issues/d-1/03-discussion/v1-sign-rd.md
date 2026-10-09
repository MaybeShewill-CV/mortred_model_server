# 第3步签字

- 问题: D-1 models deadline 到期后 child 可能仍停在 starting
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `autostart_all` 的 models deadline 分支今天只在持有 `_mu` 时调用 `mark_models_failed_locked` 然后 return，已 launch 的 child 会留在 `starting`。同一把锁里可以先看 `engine.state()`：仍为 `starting` 的非网关 child 走已有的 `note_permanent_failure()`，状态变为 `failed`。`note_exit` 见到 `kFailed` 且 `stopping == false` 时直接返回，再配合 `wanted = false`，不会按 `on-failure` 或 `always` 再拉起。`pid > 0` 时 `start_server` 拒绝再 spawn。`stop_server` 与 `perform_ordered_shutdown` 的 SIGINT、`kStopGraceMs`、外层 SIGKILL 不用动。`autostart_models_respects_deadline` 可以补上 `state == failed` 与超过探测间隔后的复查。健康就绪、spawn 失败、空 `server_uri`、网关 deadline 四条不经过这条新终态。
