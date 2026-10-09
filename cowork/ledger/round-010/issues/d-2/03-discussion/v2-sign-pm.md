# 第3步签字

- 问题: D-2
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的条件已写入验收第 4 条、契约和自测。`onnx_source.notes` 单独含 ASCII `240x360` 和 `[1,240,360,3]`；`contract.preprocess` 单独含乘号 `240×360` 即算写明高 240、宽 360，不含 ASCII `240x360` 仍通过。自测对两段分别检查，拼接后只出现一次 ASCII `240x360` 不能让 preprocess 通过。第 1 至第 4 条都能对照三份 toml、server 配置和 `onnx_sources.json` 用退出码判定。
