# 第3步签字

- 问题: D-1 models deadline 到期后 child 可能仍停在 starting
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `autostart_all` 的 models deadline 分支现在持有 `_mu` 时只调用 `mark_models_failed_locked` 然后 return。同一临界区里可以按锁内 `engine.state()` 处理：已 launch 且仍为 `starting` 的非网关 child 调用已有的 `note_permanent_failure()`，并写 `wanted = false`、`ready = false`、`backoff_due_ms = 0`，`stopping` 保持 false。`note_exit` 见到 `kFailed` 直接返回，`pid > 0` 时 `start_server` 不再 spawn。`probe_readiness` 只在锁内仍为 `starting` 时 `note_ready`。验收 4 的三条用例已有对应断言。验收 6 用第二个 catalog id `ready_sibling`，`start_concurrency` 只在该用例设为 2，产品默认仍是 1。`stop_server` 与 `perform_ordered_shutdown` 的 SIGINT、`kStopGraceMs`、外层 SIGKILL 不用动。
