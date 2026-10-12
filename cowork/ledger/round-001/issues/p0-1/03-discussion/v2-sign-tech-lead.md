# 第3步确认

- 问题: P0-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 基准文件只按一组结果通过：命令含 `--shapes=input:1x3x480x640`，同一文件有硬件行、`YYYY-MM-DD` 和 `Throughput:`，高 480、宽 640 与 `model_input_image_size = [480, 640]`、profile 里 `input` 的 `[1, 3, 480, 640]` 相同。自测从仓库 `_bin` 启动 `_bin/mortred-model-server.out`，推理请求带该进程非空的 `MORTRED_AUTH_TOKEN`，成功状态是 HTTP 200、`status` 为 0、`status_str` 为 `OK`、`results[0].status` 为 0。`min`、`opt`、`max` 同为 `[1, 3, 480, 640]`，`worker_nums` 与 `max_batch_size` 保持 1。每个 worker 继续各自 `init` 并 `createExecutionContext`。TensorRT 只覆盖 static shape、batch=1。
