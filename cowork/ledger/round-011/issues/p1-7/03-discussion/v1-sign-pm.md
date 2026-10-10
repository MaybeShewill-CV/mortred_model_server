# 第3步签字

- 问题: P1-7
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 上表 39 个 path 的请求示例和 200 示例都写在该 path 的 `application/json.example` 上。键、数组长度、`model.name`、data `$ref` 和字段都能直接读出来判定，并按该 path 现有 schema 校验。`yolov8n`、`yolov8l`、`yolov8x` 的 `model.name` 分别是这三个 id。`libface` 与 `centerface` 的 `landmarks`，以及 `superpoint` 的 `score`、`location`、`descriptor`，都有单独条目。`dbnet` 示例没有 `text`。`info.x-contract-hash` 仍是给定值。两条自测命令的期望退出码都是 0；表外 path 没有示例时，这两条命令仍以退出码 0 为通过。
