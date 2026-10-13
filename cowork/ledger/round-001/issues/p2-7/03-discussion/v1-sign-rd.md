# 第3步确认

- 问题: P2-7
- 方案版本: v1
- 结论: 不同意
- 条件: 验收标准 3 与自测命令第二条要求这一次不带 `Authorization` 的 POST 得到 HTTP 400、JSON `status` 8。现有 `serve_process` 在进入模型前调用 `is_bearer_authorized`；`realesrgan_server_config.toml` 没有 `auth_token`，该命令又清掉 `MORTRED_AUTH_TOKEN`。空 token 在 `is_bearer_authorized` 里直接返回 false，这次响应是 HTTP 401、`status` 401。方案写明不改鉴权，也不能补 `Authorization`。同一段 Python 还有 `payload["status_str"][5] == "\u00d7"`。契约文案 `尺寸须为 256×256` 的 U+00D7 在下标 8，下标 5 是字符 `2`。
- 理由: 整数 8 目前空着（7 之后是 11），文案可以放进 `status_code_to_str`，`http_status_of` 可以把 8 映到 400。统一响应已有 `status`、`status_str`、`results[0].status`、`results[0].data` 为 null、`partial`、`model.name`。`prepare_inputs` 可以在派生类里、在空张量收成状态 3 之前返回 8。验收请求到不了这条路径：鉴权先把无 `Authorization` 的 POST 挡成 401。自测第二条的代码块还会先执行占位 Python `int(open("/proc/self/status").read())`，并且块内没有 `export P2_7_PID`。按后面的完整顺序补上 export、跳过占位段之后，仍停在 HTTP 401 和错误下标。两条自测里第二条按原文不能以退出码 0 结束。
