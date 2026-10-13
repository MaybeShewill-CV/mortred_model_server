# 第3步确认

- 问题: P2-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `max_batch_size<=1` 的同步请求已经走 `schedule_sync_request` → `start_sync_item` → 逐张 `run_one`。`commit_workers` 只在 `max_batch_size>1` 时启动 `BatchCollector`，`start()` 在上限不超过 1 时直接返回。`on_sync_item_done` 里已有逐张耗时门闩：`work->ran` 为真才记，回调内的 `observed` 保证同一张只进一次；请求已回复后 `run_one` 返回仍走这道门闩，并且不再启动下一张。在这道门闩里调用现有 `observe_batch_size(1)`，就能让每张已返回的图记一次值为 1 的观测，未进入或尚未返回的张不记。直方图桶上界、空序列只输出 HELP/TYPE、`model` 标签取 `set_model` 写入的 server URI、`results` 按 `n_items` 组装，以及每个 worker 自己的 TensorRT execution context，都是现况。契约测试的 `start_server` 固定 `worker_nums=1`，不写 `max_batch_size` 时运行配置为 1；两个新用例和 `batch_collects_and_distributes_per_request_results`、`model_timeout_n_items_keeps_results_length` 都可以落在 `server_e2e_contract_test` 上。
