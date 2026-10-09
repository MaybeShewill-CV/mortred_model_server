# 第3步签字

- 问题: D-4
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 冻结句收成 LightGlue 的 `scores`、`descriptors`、`matches0`、`mscores0` 与 CLIP `"output"` 上的错 dtype、`quiet_NaN`、`+infinity`。这些格子回 `MODEL_OUTPUT_CONTRACT_FAILED`，并且不是 `MODEL_EMPTY_OUTPUT`。对照样本与现有尺寸关系一致；非有限检查落在已有 classify 与新建 protected 入口，位于阈值比较和 `normalize_embedding` 之前。缺名与空文本仍是 `MODEL_EMPTY_OUTPUT`。验收 1 至 16 都能用同一次 `-V` 日志判定。自测命令由 `scripts/run_tests.sh` 交给 ctest，期望退出码 0。没有上一版审查条件，也没有留给 RD 的未决事项。
