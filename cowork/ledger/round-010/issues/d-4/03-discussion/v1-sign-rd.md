# 第3步签字

- 问题: D-4 LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `classify_extractor_outputs` 与 `classify_matcher_outputs` 已在分数阈值循环之前被两条生产路径调用，缺名仍是 `MODEL_EMPTY_OUTPUT`，错 dtype 与元素个数不符已是 `MODEL_OUTPUT_CONTRACT_FAILED`。`Tensor::make` 把缓冲清零，现有七个用例里的 f32 都是有限值。对照样本满足现有尺寸关系：1 个点对应 keypoints 2、scores 1、descriptors 256；matcher 的 matches0 为 2、mscores0 为 1。错 dtype 走现有分支；`quiet_NaN` 与 `+infinity` 在这两个函数返回 `OK` 之前扫描 `scores`、`descriptors`、`mscores0` 即可，测试直接注入假 tensor，不加载权重、不跑 session。CLIP 新增一个 protected 入口，现有派生类用 `using` 调用，不要求 init，也不打开 session。`encode_image` 与 `encode_text` 在 session 成功之后、写入 embedding 之前调用它；`normalize_embedding` 仍留在后面。有限的 `[1, 4]` 样本 `2, 0, 0, 0` 原样写入，长度为 4、首元素为 2。错 dtype、`quiet_NaN`、`+infinity` 回 `MODEL_OUTPUT_CONTRACT_FAILED` 且不写 embedding。全零仍由归一化回到 `MODEL_EMPTY_OUTPUT`。缺输出名仍可由现有 `OutputReader` 在该入口之前返回 `MODEL_EMPTY_OUTPUT`。验收第 1 条到第 16 条都已编号，同一次 `-V` 日志用对应的 `[       OK ]` 判定；出现 `SKIPPED`、名字未执行或 `No tests were found` 即不通过。两个 CTest 名已在 `test/CMakeLists.txt`，`scripts/run_tests.sh` 会把 `-R`、`--output-on-failure` 和 `-V` 交给 `ctest --test-dir`，该正则能同时选中这两个套件。
