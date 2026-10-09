# 第3步签字

- 问题: D-2 attentive_gan.toml 输入尺寸仍 stale
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的条件已写入验收第 4 条、契约和自测。`onnx_source.notes` 单独要求 ASCII `240x360` 与 `[1,240,360,3]`；`contract.preprocess` 单独要求现有乘号 `240×360`，不含 ASCII `240x360` 仍通过。自测对两段分别断言，没有拼接后再搜一次。三份 toml 现为 `[240, 320]`，session 为 `[1, 240, 360, 3]`，导出脚本为 `HEIGHT, WIDTH = 240, 360`，`on_init` 已按 session 的 `shape[1]`、`shape[2]` 做 resize 与解码预算。推荐做法只把手写尺寸改成 `[240, 360]` 并删掉同一句 stale 注记，不改 backend、`server_uri`、进程或 engine。
