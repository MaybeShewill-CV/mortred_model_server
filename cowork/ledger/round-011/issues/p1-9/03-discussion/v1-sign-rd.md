# 第3步签字

- 问题: P1-9
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 表 5 的 39 条命令落在现有 `scripts/bench_set_a/run.py` 的 `profile_one` 上：仓库根目录调用已存在的 `3rd_party/bin/trtexec`，参数为 `--loadEngine`、`--warmUp=500`、`--duration=15`、`--avgRuns=100`、`--useSpinWait`，fp16 engine 再加 `--fp16`。39 个 `--loadEngine` 路径与 `conf/product_static_bs1.json` 的 `engine` 一致；只有 `libface`、`centerface`、`enlightengan`、`realesrgan` 的 profile 非空，方案里的四条 `--shapes` 就是这四份 Set A profile 的 opt。`libface` 使用 `libface.toml` 的同一份 engine，单独占一行。`write_docs` 重写英文文档时，把已有 `## 表 5 — 一行基准` 到文末原样接回，中文副本仍由不含该节的正文生成。指标只取本次进程输出里最后一条 `Throughput:`；退出码非 0、没有该行、`nvidia-smi` 失败、输出含 `N/A` 或 engine 不在时不写表 5。采集不拉起 gateway 与 supervisor，也不改推理热路径。
