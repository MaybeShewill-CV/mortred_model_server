# 模块评分

- id: src-server-apps
- 权重: 20
- 分数: 9.4
- 读过的路径: `cowork/ledger/rubric.md`；`src/apps/model_server_main.cpp`；`src/apps/common/model_server_main.h`；`src/apps/product_index.h`；`src/apps/product_index.cpp`（`catalog()` 投影入口）；`src/server/abstract_server.h`；`src/server/base_server_impl.h`；`src/server/async_endpoints.h`；`src/server/item_exec.h`；`src/server/sync_request_graph.h`；`src/server/prometheus_metrics.h`；`src/server/parsed_request.h`；`src/server/request_admission.h`；`src/server/http_wire.h`；`src/server/http_status.h`；`src/server/backpressure.h`；`src/server/batch_collector.h`（排队、推理、batch 观测）；`src/server/server_runtime_config.h`（`max_queue_depth` / `async_enabled` / `rate_limit_qps` 默认值）；`src/common/request_envelope.h`；`src/common/response_envelope.h`；`src/common/auth_token.h`；`src/common/stage_timing.h`；`src/control/gateway/gateway_app.cpp`（`note_upstream_stage` / `record_stage_sample` / `render_stage_metrics`）；`docs/perf/coverage_set_http_rps.md`；`docs/benchmark-h3/` 下九个覆盖集 id 的 Throughput 行；`docs/api-contract.zh-cn.md`（过载与 `/jobs`）；`scripts/server/http_infer_rps.py`（状态分类与 `server_time_ms`）；`scripts/coverage_http_report.py`（`error_rate = (errors - http_429) / requests`）；`test/server_e2e_contract_test.cc`；覆盖集九份 `conf/server` toml 的 `max_queue_depth`、`max_batch_size`

## 已具备

- 一个进程入口。`model_server_main.cpp` 用 `--list` 或 `--model <id>` 启动；未知 id、非 HTTP id、toml 对不上都在监听前返回 -1。`product_index.cpp` 从各家族 `catalog()` 投影。`run_model_server_main` 在 `init` 非 OK 时返回 -1，发生在 `start` 之前；`SIGINT`/`SIGTERM` 经 `ProcessStop` 进 `stop()`。析构先停 batch，再在同一截止时间内 drain async 与 worker。
- PF2。`max_batch_size<=1` 时，`observe_queue_wait_ms` 记 checkout 等待，`observe_inference_duration_ms` 只记随后 `run_one` 的 `run_ms`。两张直方图都带 `model` 标签。`docs/perf/coverage_set_http_rps.md` 九行的 `mortred_queue_wait_duration_ms_count_delta` 与 `mortred_inference_duration_ms_count_delta` 成对出现且数值相同，模型标签是该 id 的 `server_uri`。
- PF3。九份覆盖集 server toml 都是 `max_batch_size=1`。`on_sync_item_done` 在 `run_one` 返回后对每张已执行的图 `observe_batch_size(1)`。`assemble_published` 按 `n_items` 组装。`inference_result_to_unified` 只给 status 为 OK 的项填 `data`。e2e `batch_size_one_counts_each_executed_image`：两张图，`results` 长度为 2，`mortred_batch_size` 的 count、sum、`le="1"` 都是 2。`multi_image_failure_is_isolated_per_item`：中间项失败为 HTTP 500 且 `data` 为 null，两侧成功项仍是 status 0 并带各自的 `data`。
- PF4。响应 `server_time_ms` 由 `item_exec.h` 写成排队等待加本次 run。压测脚本从响应体取 p50。gateway 在 HTTP 200 且读到 `server_time_ms` 时记 `gw_in`、`route`、`auth`、`fwd`、`upstream`，不依赖 `/tmp/mortred_stage_timing`（该文件只控制 glog `stage_trace`）。同一份清单九行都有五段 p50 和 `server_time_ms_p50`。九行的五段 p50 之和都大于模型 `server_time_ms` p50 的一半，并且都写了 `dominant_stage=upstream`。
- PF1、PF5、TP1。清单九行都是 `checked=yes`，硬件 RTX 2070 SUPER 8192 MiB，日期 2026-10-14 或 2026-10-15。`conf/server` 的 `worker_nums` 仍是 1；pack 结论分别是 4、4、4、2、2、2、1、4、1，`decode_auto` 与 `jpeg_landing` 一致，进程 worker 数等于该 `worker_nums`。命令都是 `http_infer_rps.py --raw`，`-c` 等于该 `worker_nums` 的两倍（8、8、8、4、4、4、2、8、2），时长 60 秒。成功 RPS、p50、p99、`http_429=0`、`error_rate=0.0` 都在。`throughput_qps` 与 `docs/benchmark-h3/<id>.txt` 的 Throughput 一致，并写了比值。`coverage_http_report.py` 用 `(errors - http_429) / requests` 算错误率，429 单独计数。没有看到同命令复测掉到已记录成功 RPS 的 85% 以下，或 p99 高到已记录值的 150% 以上。
- ST1。未知根字段、空 `images`、空 base64、`img_data` 在 `request_envelope.h` 记为非法请求。空 raw body 在 `parsed_request.h` 标 `MODEL_EMPTY_INPUT_IMAGE`，`is_valid` 为假，`http_status.h` 映射为 400。未知 `params` 在 `param_spec.h` 拒绝。错误或缺失 Bearer 在推理前 401；空配置 token 不放行。健康检查、`/ready`、`/openapi.json` 不走这道鉴权。e2e：`strict_envelope_rejects_unknown_fields_and_bad_params` 为 422，`legacy_img_data_returns_422_with_migration_hint` 为 422，`missing_token_returns_401_with_www_authenticate` 为 401，`model_failure_returns_500_with_null_data` 为 500 且 `data` 为 null。
- ST3。九份覆盖集 server toml 都是 `max_queue_depth=32`，没有打开 `async_enabled` 或 `rate_limit_qps`（默认关）。等待队列在入队前 `queue_would_overflow`，立即 429，并用 `compute_retry_after_seconds` 写 `Retry-After`（1–60 秒）；拒绝不增加 `_m_waiting_jobs`。e2e `queue_limit_returns_429_with_retry_after`：占着槽的请求仍 200，后来的请求 429 且带 `Retry-After`，`mortred_queue_rejected_total` 增加。

## 问题

无

## 范围外

- 重复描述，不重新入列。P2-1：覆盖集 `max_batch_size=1` 的路径不记录 batch 观测。当前 `on_sync_item_done` 已对每张执行完的图记 1。
- 重复描述，不重新入列。P2-11：九个 TensorRT id 没有按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告。当前清单九行都在，`-c` 是已部署 `worker_nums` 的两倍。
- 重复描述，不重新入列。P1-1：不创建 `/tmp/mortred_stage_timing` 时 gateway `/metrics` 读不到阶段耗时。当前 `record_stage_sample` 在成功转发路径上记五段，不看这个文件。
- 每个 worker 各自持有 TensorRT execution context。不把共享 context 或 server 内 context 池写成问题。
- 动态 shape、动态 batch，以及 batch 大于 1 的 engine。`max_batch_size>1` 的 `BatchCollector` 仍在；覆盖集 toml 保持 1。
- 一个 catalog id 一个 OS 进程。drain 超时后由外层 SIGKILL。
- MNN 与 ONNX Runtime 的吞吐和延迟。
- 覆盖集以外的 catalog id。
- `docs/benchmark-h3/<id>.txt` 的 Throughput 只作引擎参照。
- ST2、ST4、ST5、ST6、TP2、TP3、OP、UX、EX 的实现不在本模块的开放项里。

## 依据

分数 9.4。落在本模块的 PF1、PF2、PF3、PF4、PF5、ST1、ST3、TP1 都能对上当前文件：等待队列有 429 和 `Retry-After`，未知字段、空图和错误 token 不进成功信封，九个 id 的 `--raw` 报告里排队直方图和推理直方图分开，batch 观测为 1，gateway 五段和 `server_time_ms` 同次可读，且五段之和大于模型 p50 一半时写了 `dominant_stage`。并发按 rubric 取已部署 `worker_nums` 的两倍，不把这一点写成未满足。

不到 10，是因为响应里的 `server_time_ms` 仍是排队等待与本次 run 之和，拆开要读两张直方图；模型侧 IP 限流的 429（`reply_rate_limited`）和 `/jobs` 队列满的 429（`reply_async_error`）不写 `Retry-After`。覆盖集 toml 没有打开这两条，ST3 的等待队列路径已经带该头，所以不单列成未满足行。

未验证判断：这次调用没有重跑压测，也没有重跑 `server_e2e_contract_test`。上列行为来自当前源码、e2e 用例正文和 `docs/perf/coverage_set_http_rps.md`。没有把 AV1 写成已经发生的拉起失败。

## 未覆盖

无
