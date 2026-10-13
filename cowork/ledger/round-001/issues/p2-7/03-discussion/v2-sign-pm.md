# 第3步确认

- 问题: P2-7
- 方案版本: v2
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 5 已编号，调用方一次请求就能判定。标准 1 看启动前 `127.0.0.1:9012/ready` 连不上，进程是仓库 `_bin/mortred-model-server.out`、工作目录是 `_bin`，环境有非空 `MORTRED_AUTH_TOKEN` 且没有那四个覆盖变量，90 秒内 `/ready` 返回 HTTP 200 且不带 `Authorization`。标准 2 用 OpenCV 解码确认 320×240、3 通道、两边都不是 256 且都不小于 10。标准 3 看同一次 POST 发往该进程的 realesrgan 地址，带同一个 `Authorization: Bearer` 和 `Content-Type: image/jpeg`，响应为 HTTP 400、`status` 与 `results[0].status` 为 8、`status_str` 全文等于 `尺寸须为 256×256`（乘号 U+00D7）、`partial` 为 false、`model.name` 为 `realesrgan`、`results` 长度为 1、`results[0].data` 为 null，并写明 HTTP 401 不算通过。标准 4 看同一次响应不是 401 或 500，`status` 不是 0 或 2，正文里没有增强图。标准 5 看产品 toml 仍含静态引擎文件名、基准文件仍含 `--shapes=input:1x256x256x3`。成功增强图和送入会话的 `1×256×256×3` 不作为通过。两条自测是完整 bash：进程带非空 token，POST 带同一个 Bearer，`status_str` 只做全文相等。现有非空 `MORTRED_AUTH_TOKEN` 会放行这个 Bearer，`image/jpeg` 作为 raw body 进入模型，统一响应已有这些字段；状态 8 当前空着。不要求动态 shape，也不突破不支持边界。
