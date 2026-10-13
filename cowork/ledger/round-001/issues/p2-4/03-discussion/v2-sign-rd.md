# 第3步确认

- 问题: P2-4
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 现有 supervisor 先 listen 再 `autostart_all()`。`MORTRED_PACK` 指向只含 `[pack.yolov8s]` 与 `[pack.libface]` 的临时 pack 时，`apply_pack` 把这两节标成 autostart，其余 catalog id 仍出现在 `GET /api/v1/status` 的 `servers` 里且不会被拉起。该响应已有键 `gateway`（`id` 为 `__gateway`）以及 `servers[]` 的 `id`、`state`、`pid`、`ready`、`started_at_ms`。`start_concurrency = 1` 只限制启动宽度，两者 `ready` 之后会同时留在这一份状态里。gateway 按 catalog id 精确把 `POST /v1/models/{id}/infer` 转到 9056 与 9051；`images` 长度为 1 的 base64 成功响应是 HTTP 200、根上 `status` 为 0、`results` 长度为 1、`results[0].status` 为 0。管理面用现有 `MORTRED_API_TOKEN`，推理用现有 `MORTRED_GATEWAY_AUTH_TOKEN`。不写 `gpu_memory_total_mib` 时，stamp 之和相对总显存的检查不启用，`nvidia-smi` 空闲显存低于 1800 MiB 仍拒绝 spawn。pack 的 `worker_nums = 1` 与两份 server toml、`gpu_mem_at_workers = 1` 一致，每个 worker 仍在 `TrtSession::init` 里各自 `createExecutionContext()`。重叠判定、单调时钟采样和 `--check` 都在新脚本内完成。ctest 按现有无 valgrind 标签的 `add_test` 进入默认 `check`。
