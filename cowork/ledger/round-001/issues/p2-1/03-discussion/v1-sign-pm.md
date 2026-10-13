# 第3步确认

- 问题: P2-1
- 方案版本: v1
- 结论: 不同意
- 条件: 验收标准 4 与第二条自测命令只计入 `le` 标签值完整等于 `1` 的那一条 `mortred_batch_size_bucket`，`le="16"` 不计入；这一条的值等于 `mortred_batch_size_count`。按这个规则，两次观测值都是 1 时该命令退出码为 0。
- 理由: 验收标准 1、2、3、5 一次就能判定。新拉起的 `yolov8s` 使用现有 `yolov8s_server_config.toml`：`max_batch_size=1`、`worker_nums=1`、端口 9056。推理前 `GET /metrics` 没有 `mortred_batch_size_count{`。POST 两张 `bus.jpg` 后 HTTP 200，`status` 为 0，`results` 长度为 2，两条 `results[].status` 都为 0。随后 count 与 sum 都是 2，正文没有 `mortred_batch_window_wait_ms_count{`。这复用现有直方图，桶上界仍是 `1, 2, 4, 8, 16, 32`。验收标准 4 不能判定。值为 1 时，`le="1"` 与 `le="16"` 同时出现，后者含有字符序列 `le="1"`。标准写「恰好一条带 `le="1"`」，自测命令用 `'le="1"' in line` 后要求列表等于 `[2.0]`。正确正文会收集到两条值为 2 的 bucket，退出码不是 0。一次验收确认不了这条通过。
