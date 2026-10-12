# 第3步确认

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 验收标准 1 至 7 已编号，业务后端一次调用就能判定。验收前与验收后 `/tmp/mortred_stage_timing` 都不存在。对已能成功推理的 `yolov8s`（`conf/server/object_detection/yolov8/yolov8s_server_config.toml` 端口 9056，`worker_nums=1`）发一次 `POST /v1/models/yolov8s/infer`，复用现有 `Content-Type: image/jpeg` 原始字节和响应信封里的顶层 `server_time_ms`。HTTP 200 且 `status` 为 0 时，该字段是大于等于 0 的有限数字。紧接着用 `MORTRED_METRICS_TOKEN` 读一次该 gateway 的 `GET /metrics`，这是现有 scrape 的扩展。正文里 `gw_in`、`route`、`auth`、`fwd`、`upstream` 各有 `mortred_gateway_stage_duration_ms_count{model="gateway",stage="该名字"}` 为 1，并各有 `_sum`；`mortred_http_request_duration_ms` 仍在。count 为 1 时该段 p50 取该段 `_sum`。G 为五段 `_sum` 之和，S 为这次响应的 `server_time_ms`。G 大于 `0.5 * S` 时，同一次正文出现契约中的 `mortred_gateway_stage_dominant{model="gateway",stage="STAGE"} 1`，STAGE 为 `_sum` 最大的名字，并列时按 `gw_in`、`route`、`auth`、`fwd`、`upstream` 取靠后的名字。G 小于或等于 `0.5 * S` 时，正文不出现 `mortred_gateway_stage_dominant`。用模型进程的 Bearer 读 `127.0.0.1:9056/metrics`，这两条序列都不出现。四份指南和契约写明上述直方图、五个 stage、count 为 1 的 p50、比较式、并列规则，以及不创建该 flag 文件也能读到；中文指南含「时间主要花在」。
