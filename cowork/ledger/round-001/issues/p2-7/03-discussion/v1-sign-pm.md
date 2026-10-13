# 第3步确认

- 问题: P2-7
- 方案版本: v1
- 结论: 不同意
- 条件: 验收标准 1 写明该进程环境设有非空的 `MORTRED_AUTH_TOKEN`，同时没有 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、`MORTRED_IMAGE_DECODE_BACKEND`、`MORTRED_MODEL_CONFIG_FILE`。验收标准 3 写明这次 POST 发往该进程的 `http://127.0.0.1:9012/mortred_ai_server_v1/enhancement/realesrgan`，请求头带 `Authorization: Bearer` 与同一个 token，`Content-Type` 仍为 `image/jpeg`。HTTP 401 不算通过。自测命令按这两句发送，不再去掉 `MORTRED_AUTH_TOKEN`，也不再发送不带 `Authorization` 的 POST。自测对 `status_str` 只做全文等于 `尺寸须为 256×256`（乘号 U+00D7），去掉下标 5 等于 U+00D7 的断言。
- 理由: 锁定的变化是用一张解码后宽或高不是 256 的合法 JPEG 调用已拉起的 `realesrgan`，返回输入错误，并写明尺寸须为 256×256。验收标准 2、4、5 可以外部判定：320×240 的合法 JPEG、响应不是成功也不是会话失败、产品文件仍指向 `realesr-general-x4v3.engine.static_bs1.fp32` 且基准仍含 `--shapes=input:1x256x256x3`。成功增强图留在附录。标准 1 要求进程没有 `MORTRED_AUTH_TOKEN`，标准 3 要求这次 POST 不带 `Authorization`。`docs/api-contract.md` 写明 token 为空时模型推理返回 401；`realesrgan` 的 server toml 没有 `auth_token`。这次请求在解码前被拒绝，调用方看到 401，看不到 `尺寸须为 256×256`。方案把鉴权留在现有行为。自测在全文相等之外还断言 `status_str` 的下标 5 为 U+00D7，该下标是字符 `2`，正确正文也会被这条自测判失败。
