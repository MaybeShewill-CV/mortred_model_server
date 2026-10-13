# 第3步确认

- 问题: P2-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的条件已经写进验收标准 4 和第二条自测命令。`mortred_batch_size_bucket` 按 `le` 标签的完整取值解析，只保留等于 `1` 的那一条；`le="16"` 与 `le="+Inf"` 不计入，这一条的值必须等于 `mortred_batch_size_count`。两次观测值都是 1 时，count、sum 和这一条都是 2，`le1 == counts`，该命令退出码为 0。改动仍只在 `max_batch_size<=1` 的同步逐张完成路径上对每张已返回的图记一次 `observe_batch_size(1)`。验收标准 1 至 5 都能一次判定。
