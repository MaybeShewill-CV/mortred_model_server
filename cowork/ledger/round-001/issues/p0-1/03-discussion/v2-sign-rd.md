# 第3步确认

- 问题: P0-1
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: `parse_model_input_size` 把 `model_input_image_size` 读成 `[height, width]`。CPU 的 `cv::resize` 和 GPU 的 `pre_crop_size` 用这一尺寸，再按 32 向右、向下补齐。480 和 640 都能被 32 整除，补齐不改变形状，送入会话的是名为 `input` 的 `1×3×480×640`，batch=1。静态引擎拒绝 `setInputShape` 时，`TrtSession::run` 返回 `MODEL_RUN_SESSION_FAILED`。`docs/benchmark-h3/libface.txt` 已经用 `--shapes=input:1x3x480x640` 写出 Throughput；这次探测退出 0 时不重编、不改该文件。探测失败时，ONNX 合同是 N=1、高宽仍为动态维，`convert_trt_engines.sh` 不会丢掉 profile。`mortredctl prepare --id libface --force` 会按改后的三档相同 `[1, 3, 480, 640]` 编一次静态引擎。`product_static_bs1.py --check` 对 libface 尺寸只断言那一句守卫。进程工作目录是 `_bin` 时，`../conf` 与 `../weights` 按该目录打开。非空 `MORTRED_AUTH_TOKEN` 覆盖空的 `auth_token`；推理 URI 在进入会话前检查 Bearer，`/ready` 不查这个头。`generic_cv_server.h` 里每个 worker 仍在自己的 `init` 中创建 `TrtSession` 并 `createExecutionContext`。这些行为用方案里的配置和那一句守卫就能完成，不必改检测器、会话或按 worker 建会话的循环。
