# 第3步签字

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 核对 `lightglue.inl`：输出名缺失仍为 `MODEL_EMPTY_OUTPUT`；dtype 不符与相对尺寸不符仍为 `MODEL_EMPTY_OUTPUT`。核对 `openai_clip.inl`：空文本路径仍为 `MODEL_EMPTY_OUTPUT`；`embeddings_compatible` 失败仍为 `MODEL_EMPTY_OUTPUT`。两处契约相关入口均在 `private`，可升至 `protected`；`MODEL_OUTPUT_CONTRACT_FAILED` 已存在。方案可落地。
