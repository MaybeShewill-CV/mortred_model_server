# 第3步签字

- 问题: P1-9
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收 2 要的硬件三段，就是本机 `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` 的第一行。现有机器行的数据部分是 `NVIDIA GeForce RTX 2070 SUPER, 8192 MiB, 6924 MiB, 616.92, 7.5`，去掉 `memory.free` 和 `compute_cap` 后恰好是 GPU 名称、含 `MiB` 的显存、驱动版本，且不含表头里的 `memory.total`、`driver_version`、`compute_cap` 或 `N/A`。39 条命令沿用 `profile_one` 的 `3rd_party/bin/trtexec --loadEngine --warmUp=500 --duration=15 --avgRuns=100 --useSpinWait`，fp16 再加 `--fp16`。`--loadEngine` 与 `conf/product_static_bs1.json` 的 engine 一致；只有 `libface`、`centerface`、`enlightengan`、`realesrgan` 的 Set A profile 非空，四条 `--shapes` 就是这四份 profile 的 opt。`libface` 与 `libface.toml` 的 `model_file_path` 是同一份 engine，单独占一行。`write_docs` 重写英文文档时把已有 `## 表 5 — 一行基准` 到文末原样接回，中文副本仍由不含该节的正文生成。指标只取本次进程标准输出里最后一条 `Throughput:` 与 `qps` 之间的数字原样；退出码非 0、没有该行、`nvidia-smi` 失败、输出含 `N/A` 或 engine 不在时不写表 5。采集不拉起 gateway 与 supervisor，也不改推理热路径。
