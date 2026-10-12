# 第3步确认

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: v1 把锁定变化收成一次外部验收。开始和结束时 `/tmp/mortred_stage_timing` 都不存在，gateway 不创建、不写入、不删除该路径。对已能成功推理的 `yolov8s` 发一次请求，响应里读到 `server_time_ms`，紧接着的 gateway `/metrics` 给出 `gw_in`、`route`、`auth`、`fwd`、`upstream` 五段且 count 都是 1。count 为 1 时 p50 取该段 `_sum`，五段之和严格大于 `0.5 * server_time_ms` 时，同一份正文用 gauge 写出主导段，并列取该列表中靠后的名字。模型进程 `/metrics` 不出现这两条序列。自测沿用现有 `worker_nums=1`，每个 worker 仍各自持有 TensorRT execution context，不共享 context，也不加 context 池。TensorRT 仍只覆盖 static shape、batch=1。
