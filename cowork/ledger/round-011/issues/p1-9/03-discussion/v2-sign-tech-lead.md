# 第3步签字

- 问题: P1-9
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的硬件条件已经写进验收 2。采集用的 `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` 第一行就是 GPU 名称、带 `MiB` 的显存和驱动版本；现有「机器」节表头按这条通不过。其余仍只覆盖冻结句 H3：39 个 engine 与 `conf/product_static_bs1.json` 一致，四条 shapes 是 Set A profile 的 opt，`libface` 单独一行。指标只取本次 `Throughput:`，失败不发表表 5。验收 1 到 9 都能黑盒判定，自测命令写了期望退出码。
