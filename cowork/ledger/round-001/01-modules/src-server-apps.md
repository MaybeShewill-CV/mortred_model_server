# 模块评分

- id: src-server-apps
- 权重: 20
- 分数: 7.3
- 读过的路径: `cowork/ledger/rubric.md`；`src/apps/model_server_main.cpp`；`src/apps/common/model_server_main.h`；`src/apps/product_index.h`；`src/apps/product_index.cpp`；`src/server/abstract_server.h`；`src/server/base_server_impl.h`；`src/server/async_endpoints.h`；`src/server/parsed_request.h`；`src/server/http_wire.h`；`src/server/item_exec.h`；`src/server/sync_request_graph.h`；`src/server/prometheus_metrics.h`；`src/server/rate_limiter.h`；`src/server/backpressure.h`；`src/server/server_runtime_config.h`；`src/server/batch_collector.h`（`observe_batch_size` 调用点）；`src/server/worker_pool.h`（租约与 drain 说明）；`src/common/auth_token.h`；`src/common/request_envelope.h`；`src/common/stage_timing.h`；`test/server_e2e_contract_test.cc`；`test/CMakeLists.txt`；`docs/benchmark-set-a.md`；`docs/benchmark-h3/yolov8s.txt`；`docs/perf/yolov8s_decode_backend_rps.md`；覆盖集九个产品 toml 的 `type` 与对应 `conf/server` 的 `max_batch_size`；`scripts/server/http_infer_rps.py` 的状态分类与报告字段

## 已具备

- 一个 `mortred-model-server.out`。`src/apps/model_server_main.cpp` 用 `--list` 或 `--model <id>` 启动；`src/apps/product_index.cpp` 从各家族 `catalog()` 投影，没有手写产品清单。
- 初始化失败不监听。`src/apps/common/model_server_main.h` 在 `init` 非 OK 时返回 -1，发生在 `start` 之前。`SIGINT`/`SIGTERM` 由 `ProcessStop` 接到 `server->stop()`；`src/server/base_server_impl.h` 析构先停 batch，再在 3000 ms 预算内 drain async 与 worker。
- 推理路由先鉴权再限流。`base_server_impl.h` 的 `serve_process` 对非 `/healthz`、`/ready`、`/openapi.json` 调用 `is_bearer_authorized`；`src/common/auth_token.h` 里空 token 不放行。非回环且没有 token 时 `server_runtime_config.h` 返回 `SERVER_INIT_FAILED`。`test/server_e2e_contract_test.cc` 的 `unauthenticated_stays_401_under_rate_limit`、`missing_token_returns_401_with_www_authenticate` 覆盖 401 不消耗 QPS、健康检查豁免。
- 未知字段、`img_data`、空图、模型失败不报成空成功。`src/common/request_envelope.h` 拒绝未知字段、空 `images`、空 base64，并给 `img_data` 迁移说明；`parsed_request.h` 对空 raw body 标 `MODEL_EMPTY_INPUT_IMAGE`。e2e：`strict_envelope_rejects_unknown_fields_and_bad_params` 为 422，`legacy_img_data_returns_422_with_migration_hint` 为 422，`model_failure_returns_500_with_null_data` 为 500 且 `results[0].data` 为 null。
- 同步队列打满立即 429 并带 `Retry-After`。`base_server_impl.h` 在入队前调用 `queue_would_overflow`，用 `backpressure.h` 的 `compute_retry_after_seconds`（1–60 秒）写头。e2e `queue_limit_returns_429_with_retry_after`：占着槽的请求仍 200，后来的请求 429，`mortred_queue_rejected_total` 增加。
- 多图响应按张返回。`max_batch_size==1` 时 `start_sync_item` 逐张 `run_one`；`sync_request_graph.h` 的 `assemble_published` 按 `n_items` 组装。e2e `model_timeout_n_items_keeps_results_length` 两张图得到两条 `results`。失败项不填 `data`（`item_exec.h`）。超时且已有成功项时顶层为 `DEADLINE_EXCEEDED_PARTIAL`。
- `/metrics` 把排队和推理拆开。`observe_queue_wait_ms` 记 checkout 等待；`observe_inference_duration_ms` 只记 `run_one` 的 `run_ms`。响应里的 `server_time_ms` 是两者之和（`item_exec.h`）。e2e `metrics_inference_duration_sum_is_positive_after_request` 要求推理直方图 sum 大于 0。JPEG 计数由 `/metrics` 追加 `mortred_jpeg_decode_total`。
- 覆盖集九个 id 的产品 toml 都是 `type=tensorrt`，对应 server toml 都是 `max_batch_size=1`、git 里 `worker_nums=1`。`MORTRED_WORKER_NUMS` 可覆盖进程内 worker 数（`parse_worker_nums`）。

## 问题

1. P1。覆盖集九个 `type=tensorrt` id 没有一份按「本次性能约束」完成的 HTTP 报告，因此也没有同一次压测读到的排队/推理拆分、`server_time_ms` 和 gateway 阶段耗时，也没有可复测的已记录 HTTP RPS（PF1、PF2、PF4、PF5、TP1）。
   - 出处：`docs/benchmark-set-a.md` 表 3、表 4；`docs/perf/yolov8s_decode_backend_rps.md`；`docs/benchmark-h3/yolov8s.txt`；`conf/model/object_detection/yolov8/yolov8s.toml`。
   - 现在的行为：九个覆盖 id 都是 TensorRT。`docs/benchmark-h3/yolov8s.txt` 只有 trtexec Throughput `400.677 qps`（RTX 2070 SUPER，2026-10-10），不是 HTTP。表 3 写明 `--duration 15s`，且文件头写「表 3 不是发布 HTTP 值」。表 4 里 `yolov8s` 的 `w*=8`、`conc=32`，并发不等于 worker 数，也没有 p50、错误率、`decode_auto`、压测命令，以及 HTTP RPS 与 `400.677` 的比值。`docs/perf/yolov8s_decode_backend_rps.md` 虽用 `--raw`，但是 8 worker、`-c 16`、`--duration 15s`，没有 p50/p99/错误率，也不是 `mortredctl calibrate --write-pack` 写下的 `decode_auto`。`/metrics` 里 `mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` 在代码中是分开的，推理直方图不含排队；响应体有 `server_time_ms`。这些数没有出现在任何一份符合约束的压测报告里。模型阶段耗时只在存在 `/tmp/mortred_stage_timing` 时打到 glog（`src/common/stage_timing.h`、`base_server_impl.h`），不是该次压测的读数。
   - 改完后一次外部验收即可确认的变化：对覆盖集中至少一个 TensorRT id（先看 `yolov8s`），仓库里有一份报告，同时写明硬件、日期、pack 的 `worker_nums` 与 `decode_auto`、`http_infer_rps.py --raw` 命令、成功 RPS、p50、p99、错误率、`docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值。并发等于已部署的 `worker_nums`，时长不少于 30 秒。同一份报告引用该次压测中模型进程 `/metrics` 的排队直方图与推理直方图，并引用响应里的 `server_time_ms` 和 gateway 阶段耗时；若 gateway 阶段 p50 大于 `server_time_ms` p50 的一半，报告写明时间主要花在哪一段。429 单独计数，不计入那 1% 错误率。

2. P2-1。覆盖集 server toml 的 `max_batch_size=1` 路径不记录 batch 观测，外部看不到执行 batch 为 1（PF3）。
   - 出处：`src/server/base_server_impl.h` 的 `schedule_sync_request`；`src/server/batch_collector.h` 的 `observe_batch_size`；`src/server/prometheus_metrics.h` 的 `render_histogram_locked`；`conf/server/object_detection/yolov8/yolov8s_server_config.toml` 以及其余八个覆盖集 server toml 的 `max_batch_size=1`。
   - 现在的行为：`max_batch_size<=1` 时走 `start_sync_item`，逐张 `run_one`，不调用 `observe_batch_size`。该调用只出现在 `max_batch_size>1` 的 `BatchCollector`。直方图只渲染已经观测过的 label，所以一次 N 张图（N>1）的请求之后，`/metrics` 没有 `mortred_batch_size` 的 count=1 样本。响应条数本身仍按 `n_items` 组装。
   - 改完后一次外部验收即可确认的变化：用覆盖集任一 id 的现有 server toml（`max_batch_size=1`）提交 N 张合法图片（N>1），响应 `results` 长度为 N；随后 `GET /metrics`，`mortred_batch_size` 的样本都是 1，且 count 等于这次执行的张数（sum 等于 count）。

## 范围外

- 不改「每个 worker 各自持有 TensorRT execution context」，也不在 server 里做跨 worker context 池。`worker_pool.h` 只租赁已有 worker 实例。
- 动态 batch、动态 shape、batch 大于 1 的 engine 不在本次性能范围。`max_batch_size>1` 的 `BatchCollector` 是既有代码；覆盖集 toml 保持 1。不把删掉这条路径写成问题。
- 一个 catalog id 一个 OS 进程。drain 超时后由外层 SIGKILL，见 `base_server_impl.h` 析构注释。
- MNN / ONNX Runtime 的吞吐和延迟优化。
- 覆盖集以外的 catalog id。
- `docs/benchmark-h3/<id>.txt` 的 Throughput 只作引擎参照，不是 HTTP 吞吐。

## 依据

分数 7.3：请求契约、鉴权顺序、队列 429、按张结果和排队/推理直方图拆分都有文件和 `server_e2e_contract_test` 出处，模型进程也能在信号后 `stop` 并 drain。扣分是因为九个 TensorRT id 都给不出符合本次约束的 HTTP 吞吐，现有表 4 和 `yolov8s` 解码笔记的并发与时长不能当作该数；静态 batch=1 路径也没有 batch 观测。本次没有跑压测，缺报告写成未满足，不标阻塞。

未验证判断：没有在本调用里执行 MNN 或 ONNX 的一次 HTTP 成功推理，也不把 AV1 写成已经发生的拉起失败。cpu profile 上 TensorRT 初始化是否失败并拒绝 ready，取决于模型后端；server 只保证 `init` 非 OK 时不 `start`。ST4、ST5、ST6、TP2、TP3、OP1–OP4 以及 gateway 停机顺序的实现在 `src/control` 与运维脚本，不在本模块问题里。EX1 的文档表述不在本模块；`--list` 的代码路径是 `catalog()` 投影。

## 未覆盖

无
