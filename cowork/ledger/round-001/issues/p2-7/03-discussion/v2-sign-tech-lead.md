# 第3步确认

- 问题: P2-7
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: v1 的两条不同意已经写进 v2 正文。验收标准 1 至 5 都有编号，并且都能在外部判定。标准 1 看启动前 `127.0.0.1:9012/ready` 连不上，随后进程是仓库 `_bin/mortred-model-server.out`、cwd 是 `_bin`，环境有非空 `MORTRED_AUTH_TOKEN`，同时没有 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_IMAGE_DECODE_BACKEND`、`MORTRED_MODEL_CONFIG_FILE`，90 秒内 `/ready` 返回 HTTP 200 且不带 Authorization。标准 2 用 OpenCV 判定一张无方向标记的彩色 JPEG，解码后高 240、宽 320、通道 3。标准 3 把这一次 POST 发到该进程的 `http://127.0.0.1:9012/mortred_ai_server_v1/enhancement/realesrgan`，请求头带同一个 `Authorization: Bearer`，`Content-Type` 为 `image/jpeg`；响应是 HTTP 400，JSON `status` 为 8，`status_str` 全文等于 `尺寸须为 256×256`（乘号 U+00D7，`为` 与 `256` 之间一个空格），`partial` 为 false，`model.name` 为 `realesrgan`，`results` 长度为 1，`results[0].status` 为 8，`results[0].data` 为 null。标准 4 在同一次响应上确认 HTTP 不是 401 也不是 500，JSON `status` 不是 0 也不是 2，`results[0].data` 里没有增强图。标准 5 确认产品 toml 仍含 `realesr-general-x4v3.engine.static_bs1.fp32`，基准文件仍含 `--shapes=input:1x256x256x3`。HTTP 401 不算通过。自测第二条含 `export P2_7_PID="$model_pid"`，进程带非空 `MORTRED_AUTH_TOKEN`，POST 带同一个 Bearer，`status_str` 只做全文相等，两条命令的期望退出码都是 0。状态 8 目前空着，HTTP 映到 400，句子放在现有 `status_str`。尺寸判断发生在组 NHWC 和 `session().run` 之前，只针对 `realesrgan` 这一张已解码工作图。静态引擎、每个 worker 自己的 execution context、鉴权代码和其他 id 保持不动。没有还要另加的条件。
