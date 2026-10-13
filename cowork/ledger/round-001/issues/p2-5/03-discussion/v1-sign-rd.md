# 第3步确认

- 问题: P2-5
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `GatewayApp::init` 在 `listen()` 之前返回失败时，`run` 已是非 0 且不会打印监听日志；API key 是否加载不参与这次判定。`spawn_locked` 已在父进程、`executable not found` 和 `fork`/`exec` 之前拒绝 `__gateway`。错误串带上现有的 `gateway scrape token rejected:` 前缀后，`is_permanent_spawn_error` 会把它标成 `failed` 而不是 `backoff`。`SupervisorApp` 对空 `MORTRED_API_TOKEN` 的退出保持不动；新的三种失败只让 `__gateway` 永久失败，管理面仍按现有路径启动。`scrape_token_usable` 对空推理 token、空管理 token 的碰撞规则可以原样保留。不改 TensorRT context 归属，也不改监听之后的鉴权。
