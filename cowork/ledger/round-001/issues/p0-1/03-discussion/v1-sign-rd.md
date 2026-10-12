# 第3步确认

- 问题: P0-1
- 方案版本: v1
- 结论: 不同意
- 条件: 这一版交付不了验收标准 4 的成功 JPEG。自测从仓库根目录执行 `build/full/bin/mortred-model-server.out`。`cmake --preset full` 把该可执行文件写到 `_bin/mortred-model-server.out`。`libface_server_config.toml` 的 `model_config_file_path` 是 `../conf/model/object_detection/libfacedetection/libface.toml`，产品 toml 的 `model_file_path` 是 `../weights/...`。这两条路径按进程工作目录解析。supervisor 在 `execve` 之前 `chdir` 到 `<project_root>/_bin`，`scripts/prepare_pack.sh` 也先进入二进制目录再启动。工作目录是仓库根时，这两条路径打不开，`generic_cv_server.h` 在 listen 之前以 `SERVER_INIT_FAILED` 退出。同一条请求没有 `Authorization`。该 server toml 没有 `auth_token`。`is_bearer_authorized` 在配置 token 为空时返回 false。推理 URI 会走鉴权，`/ready` 才豁免。响应是 HTTP 401。
- 理由: 高宽对齐可以落在现有检测器和 TensorRT 会话上。`detection_params.h` 把 `model_input_image_size` 的第一项当高、第二项当宽。改成 `[480, 640]` 后，CPU resize 和 GPU `DIRECT_RESIZE_PAD_TO_MULTIPLE` 都用这个尺寸。480 与 640 能被 32 整除，右/下补齐不会再放大，送入 `setInputShape` 的张量是 `1×3×480×640`、batch=1。`scripts/product_static_bs1.py --check` 对 libface 尺寸的硬检查只有 `model_input_image_size = [640, 480]` 那一句；`--check` 比对的 `conf/product_static_bs1.json` 不含这个尺寸。profile 的 `min`、`opt`、`max` 改成同一个 `[1, 3, 480, 640]` 后，仍是静态 batch=1。worker 循环继续为每个 worker 单独 `init`，`TrtSession` 继续 `createExecutionContext`。现有 engine 若已接受基准文件里的 `input:1x3x480x640`，不需要改会话结构。这一版过不了的是拉起进程和拿到 HTTP 200 的那一段命令。
