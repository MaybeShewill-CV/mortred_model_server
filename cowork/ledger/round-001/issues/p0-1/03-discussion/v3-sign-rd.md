# 第3步确认

- 问题: P0-1
- 方案版本: v3
- 结论: 同意
- 条件: 无
- 理由: `parse_model_input_size` 把 `model_input_image_size` 读成 `[height, width]`。改成 `[480, 640]` 后，CPU 的 `cv::resize` 和 GPU 的 `DIRECT_RESIZE_PAD_TO_MULTIPLE` 都用这个尺寸。480 与 640 能被 32 整除，右/下补齐不改变形状，送入会话的是名为 `input` 的 `1×3×480×640`，batch=1。静态引擎拒绝 `setInputShape` 时，`TrtSession::run` 返回 `MODEL_RUN_SESSION_FAILED`。`docs/benchmark-h3/libface.txt` 已经含 `--shapes=input:1x3x480x640`、硬件行、日期和 `Throughput:`；这次探测退出 0 时不重编、不改该文件。`product_static_bs1.py` 里只需把 libface 那一句守卫改成要求 `model_input_image_size = [480, 640]`。验收标准 7 的命令只核对这句守卫，并确认 `--check` 的 stderr 不含 `libface params are not 640x480`；它不采用 `--check` 的退出码，`mobilenetv2_cpu.toml` 与 `resnet_cpu.toml` 的 MNN on cpu 输出不会使这条失败。非空 `MORTRED_AUTH_TOKEN` 覆盖空的 `auth_token`；推理 URI 在进入会话前检查 Bearer，`/ready` 不查这个头。从仓库 `_bin` 启动 `_bin/mortred-model-server.out` 时，`../conf` 与 `../weights` 按该目录打开。`generic_cv_server.h` 里每个 worker 仍在自己的 `init` 中创建 `TrtSession` 并 `createExecutionContext`。profile 的 `min`、`opt`、`max` 同为 `[1, 3, 480, 640]`，`worker_nums` 与 `max_batch_size` 保持 1。这些行为用方案里的两份配置和那一句守卫就能完成。
