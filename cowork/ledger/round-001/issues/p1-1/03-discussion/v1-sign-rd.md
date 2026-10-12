# 第3步确认

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: 现有 `process()` 与 `forward_to_model()` 已有 `gw_in`、`route`、`auth`、`fwd`、`upstream` 五个 mark，`upstream` 位于把正文拷回客户端之前。五段可用 `steady_clock` 在这些位置另记，不调用 `stage_timing::enabled()`，也不创建 `/tmp/mortred_stage_timing`；flag 仍只决定原有 stderr `stage_trace`。成功回调里用 gateway 已通过 `common` 带上的 rapidjson 解析上游正文副本的顶层 `server_time_ms`，不改转发字节。直方图和条件 gauge 追加在 gateway `GET /metrics` 现有正文之后，不必改 `prometheus_metrics.h` 或模型进程的 `/metrics`。统一信封已有顶层 `server_time_ms`。`yolov8s` 仍走现有 static shape、batch=1、每个 worker 独立 TensorRT execution context 的路径。`--mode envelope` 与五个 e2e 用例可以加在现有 fake 与 `gateway_e2e_test` 上。
