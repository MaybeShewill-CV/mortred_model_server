# 第3步签字

- 问题: D-4
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收 1 至 16 已编号。1 至 15 各对应 `LightGlueOutputContract` 或 `OpenAiClipOutputContract` 里的一个用例名，同一次 `-V` 日志里该名有 `[       OK ]` 即通过；该名未执行或出现 `SKIPPED` 即不通过。第 2 至 15 条与契约表逐格对齐：错 dtype 标本、`quiet_NaN`、`+infinity` 都是 `MODEL_OUTPUT_CONTRACT_FAILED`，并且不是 `MODEL_EMPTY_OUTPUT`；CLIP 失败格的 embedding 长度为 0；有限对照为 `OK`，长度为 4，首元素为 2。第 1 条的七个已有用例名与现行套件一致，缺名与空文本由这些用例是否通过判定。第 16 条看自测退出码 0、通过数大于 0，且输出不是 `No tests were found`。命令只跑这两个已有 CTest 别名。这些句子都可以黑盒判定。
