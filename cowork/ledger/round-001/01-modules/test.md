# 模块评分

- id: test
- 权重: 10
- 分数: 6.8
- 读过的路径: `test/` 目录清单，`test/CMakeLists.txt`，根 `CMakeLists.txt` 的 `enable_testing` 与 `add_subdirectory(test)`，`test/golden/mobilenetv2_classification.json`，`test/golden/dbnet_text_detection.json`，`test/golden/realesrgan_enhancement.png`，`test/golden_baseline.json`，`test/contract_baseline/v1_http_contract.md`，`test/contract_baseline/M0_baseline_report.md`，`test/model_golden_test.cc`，`test/model_golden_registry.h`，`test/server_e2e_contract_test.cc`，`test/prometheus_metrics_test.cc`，`test/gateway_e2e_test.cc`，`test/occupancy_gate_unittest.cc`，`scripts/golden_drift_check.py`，`scripts/check_consistency.py` 的 golden 调用，`docs/benchmark-set-a.md`，`docs/benchmark-h3/yolov8s.txt`

## 已具备

构建入口在根 `CMakeLists.txt` 调用 `enable_testing()` 和 `add_subdirectory(test)`。`test/CMakeLists.txt` 用 `TEST_LIST` 注册 gtest，`check` 目标跑 `ctest -LE valgrind`。`model_golden_test` 的 `SKIP_RETURN_CODE` 为 77；`test/model_golden_test.cc` 写明，选中用例全部因缺权重没有做数值比对时进程退出 77，并打印一行 `NUMERICAL_NOT_RUN`。`MORTRED_CI_REQUIRE_WEIGHTS=1` 时缺权重改为失败。

`test/server_e2e_contract_test.cc` 用真实 `BaseAiServer` 加 `FakeModel` 锁住统一信封。`legacy_img_data_returns_422_with_migration_hint` 对 `img_data` 返回 422，错误指针是 `/img_data`。`strict_envelope_rejects_unknown_fields_and_bad_params` 对未知字段返回 422。`multi_image_results_align_with_items` 与 `multi_image_failure_is_isolated_per_item` 要求 N 张图得到 N 条 `results`，失败项的 `data` 为 null，相邻成功项仍带自己的值。`unauthenticated_stays_401_under_rate_limit` 连续 5 次无 token 都是 401，随后带 token 的请求仍是 200。`queue_limit_returns_429_with_retry_after` 在队列打满时要求 429 和 `Retry-After`。`test/prometheus_metrics_test.cc` 的直方图用例分别渲染 `mortred_queue_wait_duration_ms_count` 和 `mortred_inference_duration_ms_count`。

覆盖集九个 id 在 `test/model_golden_test.cc` 里都有用例：`mobilenetv2`、`libface`、`pphuman`（`pphuman_segmentation` / `TrtProductGolden.pphuman_mobile`）、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`，以及 `Yolov8TrtDetection` 读取 `conf/model/object_detection/yolov8/yolov8s.toml`。`#ifdef MORTRED_HAS_TRT` 下还有这九个 id 的 `TRT_PRODUCT_GOLDEN`。标准用例经 `load_model_cfg` 把 backend 改成 cpu。直接读到的样例：`test/golden/mobilenetv2_classification.json` 含 `class_id` 与 `scores`；`test/golden/dbnet_text_detection.json` 含 `regions`；`test/golden/realesrgan_enhancement.png` 在目录里。

`test/contract_baseline/v1_http_contract.md` 写明自己是改造前快照，现行请求是 `images[]`，`img_data` 返回 422。`test/contract_baseline/M0_baseline_report.md` 写明是 2026-08-31 的历史基线，其中 realesrgan 的那次失败记录不作为当前故障。

## 问题

1. 完成标准 PF1、PF2、PF4、PF5、TP1。出处：`test/CMakeLists.txt` 的 `TEST_LIST` 与 `add_test` 没有 `http_infer_rps.py --raw`；`docs/benchmark-set-a.md` 表 3 写明 `--duration 15s`，并写「表 3 不是发布 HTTP 值」；表 4 里 `yolov8s` 的 `w*` 为 8、`conc` 为 32，有 `http_rps_gpu` 与 `p99_gpu`；`docs/benchmark-h3/yolov8s.txt` 写的是 2026-10-10 的 trtexec，`Throughput: 400.677 qps`。现在的行为：覆盖集里没有一份同时写明硬件、日期、pack 的 `worker_nums` 与 `decode_auto`、`--raw`、并发数等于已部署 `worker_nums`、时长不少于 30 秒、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` 的 Throughput 和二者比值的报告。表 4 的并发数是 `w*` 的 4 倍，也没有 p50、错误率和这个比值。因此同一次压测读不到分开的排队等待和推理耗时，也读不到 gateway 阶段耗时与模型 `server_time_ms` 的比较；PF5 没有已记录的成功 RPS。`test/prometheus_metrics_test.cc` 只在单测里分别调用两个直方图，`test/server_e2e_contract_test.cc` 只断言单次响应里 `server_time_ms` 是数字。改完后一次外部验收即可确认的变化：对覆盖集至少一个 `type=tensorrt` 的 id，例如 `yolov8s`，仓库里有一份报告，字段与上面的清单一致，命令是 `scripts/server/http_infer_rps.py --raw`，并发数等于该 pack 里的 `worker_nums`，时长不少于 30 秒；同一份输出里排队等待和推理耗时是分开的计数，并能读到 gateway 阶段耗时与 `server_time_ms`。gateway 阶段 p50 大于模型 `server_time_ms` p50 的一半时，报告写明时间主要花在哪一段。

2. 完成标准 TP2。出处：`test/gateway_e2e_test.cc` 的 `dead_upstream_maps_to_503` 只停一个 fake 上游；`test/` 下没有同时请求 `yolov8s` 与 `libface` 的用例。现在的行为：套件里没有一次在 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求、且两者都有成功响应的记录。改完后一次外部验收即可确认的变化：两者都处于 ready 时各发并发请求，两边的响应都成功。先停掉其中一个再压另一个，不能当作这次验收。

3. `test/golden_baseline.json` 与当前 golden 用例不一致，`scripts/check_consistency.py` 会调用失败的零漂移检查。出处：`test/golden_baseline.json` 的 `case_count` 为 30、`golden_count` 为 26；`test/model_golden_test.cc` 里 `TEST(model_golden,` 与 `GOLDEN_*_CASE(` 共 49 个名字，其中 `dinov2_vits14`、`metric3d_512` 不在 baseline 的 `case_names` 里；`test/golden/dinov2_vits14.json`、`test/golden/metric3d_512.json` 以及多个 `trt_*.json` 不在 baseline 的 `golden_sha256` 里。`scripts/golden_drift_check.py` 在名单或哈希集合不一致时打印 drift 并返回 1。`scripts/check_consistency.py` 的 `check_golden_zero_drift()` 执行 `golden_drift_check.py --check`，`.github/workflows/ci.yml` 会跑 `python3 scripts/check_consistency.py`。现在的行为：按该脚本的比较，会报 cases added 和 golden file added，`--check` 退出码为 1，一致性检查把这条算进失败。覆盖集的 `dbnet_text_detection.json` 与 `realesrgan_enhancement.png` 可以读到，这一条不把它们写成缺失。改完后一次外部验收即可确认的变化：`python3 scripts/golden_drift_check.py --check` 退出码为 0；baseline 的 `case_names` 含 `dinov2_vits14` 与 `metric3d_512`，`golden_sha256` 含 `dinov2_vits14.json` 与 `metric3d_512.json`。

## 范围外

`test/` 里还有 SAM、CLIP、LightGlue、RTDETR、diffusion、`msocrnet`、byte tracker 的用例和 golden。这些模型不进入完成标准，也不单列为问题。MNN 与 ONNX Runtime 的吞吐、TensorRT 动态 batch、多个 worker 共享一个 execution context，不在本模块问题里。

## 依据

分数对照的是：这套测试能不能让一个小组在这一台上说明一次失败的时间构成，并给出符合本次性能约束的 HTTP 吞吐。契约、鉴权和队列打满有真实 server 用例，覆盖集九个 id 有进程内 golden 或 TensorRT product 用例，所以高于只剩空壳的分数。PF1、PF5、TP1 要求的报告不存在，表 4 的数字因为并发数与 `worker_nums` 不一致而不能充当该报告，所以到不了可以报出 HTTP 吞吐的分数。

完成标准逐条：

- PF1、PF2、PF4、PF5、TP1、TP2：未满足，见问题 1、2。这次没有跑 GPU 压测，不把这几行标成阻塞。
- PF3：未验证判断。`multi_image_results_align_with_items` 在 FakeModel 上要求 N 条 `results`。`batch_collects_and_distributes_per_request_results` 断言的是假模型上 4 个独立请求的 `mortred_batch_size_count` 小于 4。没有覆盖集 TensorRT id 在一次提交 N 张图（N>1）时 batch 观测为 1 的记录，也没有相反的失败记录。
- AV1：未验证判断。进程内 golden 在缺权重时跳过，标准路径强制 cpu。没有把「缺少 HTTP RPS」写成这一行未满足。
- ST1、ST2：`server_e2e_contract_test.cc` 已断言未知字段与 `img_data` 为 422、空 raw body 为 400、错误 token 为 401，未鉴权不消耗限流配额。健康检查另有豁免用例。
- ST3：`queue_limit_returns_429_with_retry_after` 断言 429 与 `Retry-After`。该用例没有在拒绝结束后再发一条成功请求。没有把「恢复失败」写成已经发生的故障。
- ST4：未验证判断。`dead_upstream_maps_to_503` 只覆盖一个 fake id 变为 503。
- ST5：抽读 `src/control/supervisor.cpp` 的子进程路径，`fork` 之后到 `execve` 之前调用的是 `dup2`、`close`、`prctl`、`sigaction`、`sigprocmask`、`chdir`。这一行的实现不在 `test/`，不写成测试问题。
- ST6：`occupancy_gate_unittest.cc` 的 `query_nvidia_smi_times_out_on_hang` 对挂起的 `nvidia-smi` 期望 `available` 为 false。读数为 N/A 时监督侧如何记账，未验证。
- TP3：`occupancy_gate_unittest.cc` 对缺戳和过期 `worker_nums` 断言失败文本里有 `mortredctl calibrate --pack ... --write-pack`。`mortredctl doctor --strict` 的退出码不在 `test/`。
- UX1、UX2、UX3、OP1 至 OP5、EX1 至 EX3：证据在 README、`docs/`、`.github/`、`scripts/`、`src/control`，不写入本模块问题。`test/openapi_consistency_test.cc` 检查 OpenAPI 文档要求 `images` 且拒绝未知字段，这是 schema 检查，不代替每个覆盖集 id 的请求响应示例。

## 未覆盖

无
