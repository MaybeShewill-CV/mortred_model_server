# 第3步签字

- 问题: P1-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `MODEL_OUTPUT_CONTRACT_FAILED` 已存在。lightglue 与 clip 的 EMPTY 与契约失败分支可拆开；契约入口升到 protected 后可无权重测试。
