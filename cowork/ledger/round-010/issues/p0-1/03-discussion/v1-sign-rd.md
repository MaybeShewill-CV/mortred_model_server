# 第3步签字

- 问题: P0-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 现有 StatusCode 已有契约失败码。encoder 与 decoder 可在现有分支上把空输出与非空不合约分开，并按 LightGlue 方式加无权重契约测。
