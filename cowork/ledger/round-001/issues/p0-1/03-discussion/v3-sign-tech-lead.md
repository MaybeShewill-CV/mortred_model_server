# 第3步确认

- 问题: P0-1
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: v3 只改 QA 打回的验收标准 7 和对应自测。`--check` 的非 0 来自 `mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 的 MNN on cpu 输出，不来自 libface 高宽。验收标准 7 改为：守卫条件要求 toml 含 `model_input_image_size = [480, 640]`，且该命令的 stderr 不含 `libface params are not 640x480`；不采用 `--check` 自身退出码，上述两份 cpu 配置的输出不使本条失败。高 480、宽 640、profile 三档同为 `[1, 3, 480, 640]`、`worker_nums` 与 `max_batch_size` 为 1 都还在。每个 worker 仍各自 `init` 并 `createExecutionContext`，不共享 context，也不加 context 池。TensorRT 仍只覆盖 static shape、batch=1。
