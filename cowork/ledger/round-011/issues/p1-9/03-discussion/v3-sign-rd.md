# 第3步签字

- 问题: P1-9
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: 本机 `centerface.engine.static_bs1.fp16` 只有一个输入，剖面是 `input.1`，`opt=1x3x640x640`。v3 的 shapes 单元格 `input.1:1x3x640x640` 与这一行一致。同一份 engine 加上 `--fp16 --useSpinWait --shapes=input.1:1x3x640x640` 时，trtexec 打印 `Input inference shape : input.1=1x3x640x640` 和 `Throughput: 677.863 qps`，进程退出 0。v2 里按名字 `input` 查找失败的那条路径不再出现。其余 38 条命令、engine 路径、`--warmUp=500 --duration=15 --avgRuns=100` 和 fp16 的 `--fp16` 都不变；上一轮这 38 个 id 已经打出过 `Throughput:`。`libface`、`enlightengan`、`realesrgan` 的 shapes 仍是各自 Set A profile 的 name 和 opt。39 个 `--loadEngine` 与 `conf/trt_engines.set_a.json` 一致，`libface` 与 `libface.toml` 的 `model_file_path` 是同一文件。采集仍是仓库根目录串行调用已有的 `3rd_party/bin/trtexec`，硬件行来自 `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader` 的第一行。指标只取本次标准输出里最后一条 `Throughput:`；失败、没有该行、硬件探测失败或 engine 不在时不写表 5。`write_docs` 重写英文文档时把已有表 5 原样接回，中文副本不含该节。不拉起 gateway 与 supervisor，也不改推理热路径。
