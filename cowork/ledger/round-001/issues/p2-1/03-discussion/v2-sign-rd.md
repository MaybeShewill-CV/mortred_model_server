# 第3步确认

- 问题: P2-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `max_batch_size<=1` 的同步请求已经走 `schedule_sync_request` → `start_sync_item` → 逐张 `run_one`。`commit_workers` 只在 `max_batch_size>1` 时启动 `BatchCollector`，`start()` 在上限不超过 1 时直接返回。`on_sync_item_done` 的耗时门闩看 `work->ran`：`run_one` 返回后才置位，回调里的 `observed` 保证同一张只进一次；请求已经回复后 `run_one` 返回仍走这道门闩，并且不再启动下一张。在这道门闩里调用现有 `observe_batch_size(1)`，每张已返回的图记一次值为 1 的观测，没进入或尚未返回的张不记。现有直方图对值 1 会让 `le="1"` 和更高的累计桶各加 1，`_count` 与 `_sum` 各加 1；没有观测时只输出 HELP 和 TYPE。默认流格式把桶上界 1 打成 `le="1"`、16 打成 `le="16"`。v2 按 `le` 标签值完整比较，`le="16"` 不会进入 `le="1"` 那一条；两次观测都是 1 时，这一条等于 `mortred_batch_size_count`。`model` 标签仍是 `set_model` 写入的 server URI，`results` 仍按 `n_items` 组装。契约测试的 `start_server` 固定 `worker_nums=1`，不写 `max_batch_size` 时运行配置为 1。两个新用例以及已有的 `batch_collects_and_distributes_per_request_results`、`model_timeout_n_items_keeps_results_length` 都可以落在 `server_e2e_contract_test` 上。TensorRT 仍是 static shape、batch=1，每个 worker 仍各自持有 execution context。
