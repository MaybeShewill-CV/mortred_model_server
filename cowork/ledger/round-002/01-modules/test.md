# 模块评分

- id: test
- 权重: 10
- 分数: 8.8
- 读过的路径: `cowork/ledger/rubric.md`；`test/` 目录清单；`test/CMakeLists.txt`；根 `CMakeLists.txt` 的 `enable_testing()` 与 `add_subdirectory(test)`；`test/golden/yolov8_detection.json`；`test/golden/dbnet_text_detection.json`；`test/golden/realesrgan_enhancement.png`；`test/golden_baseline.json`；`test/contract_baseline/v1_http_contract.md`；`test/model_golden_test.cc`；`test/model_golden_registry.h`；`test/server_e2e_contract_test.cc`；`test/gateway_multiinstance_test.cc`；`test/request_envelope_unittest.cc`；`test/http_contract_test.cc`；`scripts/golden_drift_check.py`；`scripts/resident_pair_record.py`；`docs/perf/coverage_set_http_rps.md`；`docs/benchmark-h3/` 下九个覆盖集 id 的 Throughput 行；九份覆盖集 server toml 的 `max_batch_size`；`src/server/base_server_impl.h` 里 `max_batch_size<=1` 的 `observe_batch_size(1)`

## 已具备

构建入口在根 `CMakeLists.txt` 调用 `enable_testing()` 和 `add_subdirectory(test)`。`test/CMakeLists.txt` 用 `TEST_LIST` 注册 gtest，`check` 跑 `ctest -LE valgrind`。`model_golden_test` 的 `SKIP_RETURN_CODE` 为 77；选中用例全部因缺权重跳过时，进程退出 77 并打印一行 `NUMERICAL_NOT_RUN`。

`test/server_e2e_contract_test.cc` 用真实 `BaseAiServer` 加假模型锁信封。未知字段 `foo` 返回 422，指针 `/foo`。空 raw body 返回 400，指针 `/body`。`images:[]` 与空字符串图片在 `request_envelope_unittest.cc` 里无效。错误 token 在 `gateway_multiinstance_test.cc` 的 `wrong_token_preserves_full_burst` 里是 401。`multi_image_results_align_with_items` 对两张图要求 `results` 长度为 2。`multi_image_failure_is_isolated_per_item` 里失败项 `data` 为 null，相邻成功项仍带自己的值。`unauthenticated_stays_401_under_rate_limit` 连续 5 次无 token 都是 401，随后带 token 的请求是 200。`queue_limit_returns_429_with_retry_after` 在 `max_queue_depth=1` 时要求 429 和 `Retry-After`。`batch_size_one_counts_each_executed_image` 一次提交 2 张图，`results` 长度为 2，`mortred_batch_size_count` 与 `sum` 都是 2，`le="1"` 的桶是 2。九份覆盖集 server toml 的 `max_batch_size` 都是 1。

`test/model_golden_test.cc` 里 `TEST(model_golden,` 与 `GOLDEN_*_CASE(` 共 49 个名字，顺序与 `test/golden_baseline.json` 的 `case_names` 一致，其中有 `dinov2_vits14` 和 `metric3d_512`。`golden_count` 为 62。覆盖集九个 id 都有用例：`mobilenetv2_classification`、`Yolov8TrtDetection`（`yolov8s.toml`）、`libface`、`pphuman_mobile`（对照 `pphuman_segmentation`）、`dbnet_text_detection`、`dinov2_vits14`、`superpoint_feature_point`、`realesrgan_enhancement`、`metric3d_512`。`#ifdef MORTRED_HAS_TRT` 下有对应的 `TRT_PRODUCT_GOLDEN`。直接打开的样例：`yolov8_detection.json` 含 `boxes`；`dbnet_text_detection.json` 含 `regions`；`realesrgan_enhancement.png` 在 `test/golden/`。

`docs/perf/coverage_set_http_rps.md` 有九行，顺序是 `mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。每行都是 `--raw`，`wall_s` 大于 60，`warmup_s` 为 5，`error_rate` 为 0，`http_429` 为 0。pack 的 `worker_nums` 与 `-c` 分别是 4 与 8、4 与 8、4 与 8、2 与 4、2 与 4、2 与 4、1 与 2、4 与 8、1 与 2。`docs/benchmark-h3/<id>.txt` 的 Throughput 与报告里的 `throughput_qps` 一致，比值写在同一行。排队计数与推理计数分开。`jpeg_landing` 与该行 `decode_auto` 一致。gateway 五段和 `server_time_ms_p50` 都在，`dominant_stage` 为 `upstream`。

## 问题

无

## 范围外

重复描述，已关闭 P1-1、P2-11：覆盖集没有一份按本次约束完成的 HTTP 报告。当前报告已在 `docs/perf/coverage_set_http_rps.md`，`-c` 等于已部署 `worker_nums` 的两倍。

重复描述，已关闭 P2-1：`max_batch_size=1` 的路径不记录 batch 观测。当前 `batch_size_one_counts_each_executed_image` 已断言每次执行的样本是 1。

重复描述，已关闭 P2-2：鉴权失败仍消耗来源配额。当前未鉴权与错误 token 的用例在配额打满前保持 401，随后合法请求仍能进入。

重复描述，已关闭 P2-4：没有 `yolov8s` 与 `libface` 同时 ready 时两者并发且都成功的记录。当前工作区没有 `logs/resident_pair_yolov8s_libface.json`，`logs/` 在 `.gitignore`。`resident_pair_record_rules` 只执行 `python3 scripts/resident_pair_record.py --check`，该文件已存在时脚本退出码非 0。这次没有把关闭记录写成这一行已满足，也没有把「实跑失败」写成已经发生的故障。

重复描述，已关闭 P2-9：`golden_baseline.json` 的 `case_count` 为 30、`golden_count` 为 26，且缺少 `dinov2_vits14` 与 `metric3d_512`。当前名单是 49 与 62，这两个名字和对应 golden 文件都在。

P2-8 已在范围外。`test/` 里还有 SAM、CLIP、LightGlue、RTDETR、diffusion、`msocrnet`、byte tracker 的用例和 golden。MNN 与 ONNX Runtime 的吞吐、TensorRT 动态 batch、多个 worker 共享一个 execution context，不在本模块问题里。

## 依据

分数对照的是：这套测试能不能让一个小组在这一台上看住失败语义，并对照已经写下的 HTTP 吞吐。契约、鉴权、队列打满和 batch=1 观测有真实 server 用例，覆盖集九个 id 有进程内 golden 或 TensorRT product 用例，九行 HTTP 报告与 `docs/benchmark-h3` 的 Throughput 对得上，所以高于上一轮缺报告、缺 batch 观测时的分数。默认 `check` 不跑官方压测，缺权重时 `model_golden_test` 以 77 跳过数值比对，所以不到 10。

完成标准里落在本模块的行：

- PF3：`batch_size_one_counts_each_executed_image` 对默认 `max_batch_size=1` 的服务一次提交 2 张图，`results` 长度为 2，batch 直方图的样本都落在 1。`multi_image_failure_is_isolated_per_item` 里单张失败的 `data` 为 null，相邻项仍成功。`mobilenetv2_mixed_batch_isolates_failed_items` 把空图标成 `MODEL_EMPTY_INPUT_IMAGE`，同批其余项与单张结果对齐。
- ST1：未知字段为 422，空 raw body 与空 `images` 不成功，错误 token 为 401。失败项编码是 `data: null`，健康检查另有豁免用例。
- ST2：`unauthenticated_stays_401_under_rate_limit` 与 `wrong_token_preserves_full_burst` 里，未鉴权和错误 token 不消耗随后那次合法请求的配额。未带 token 的推理是 401。
- ST3：`queue_limit_returns_429_with_retry_after` 在队列被占满时要求 429 和 `Retry-After`，占着槽位的那条请求返回 200。该用例没有在拒绝结束后再发一条新的成功请求。没有观察到拒绝结束后合法请求失败、且进程不回到 ready。
- TP1：`docs/perf/coverage_set_http_rps.md` 九行都先写了 `--raw` 的 calibrate，再按 pack 的 `worker_nums` 与 `decode_auto` 部署，然后 `http_infer_rps.py --raw`，并发数是已部署 `worker_nums` 的两倍，时长不少于 60 秒，错误率为 0，429 单独为 0。Throughput 与九份 `docs/benchmark-h3/<id>.txt` 一致。
- TP2：见范围外里已关闭 P2-4 的重复描述。默认 ctest 只跑 `--check`。
- P2-9 那类 golden：见范围外里已关闭 P2-9 的重复描述。这次没有执行 `python3 scripts/golden_drift_check.py --check`，没有把退出码写成已经跑过的 0。

PF1、PF2、PF4、PF5 的字段在同一份九行报告里：硬件、日期、`worker_nums`、`decode_auto`、成功 RPS、p50、p99、错误率、Throughput、比值、分开的排队与推理计数、gateway 阶段和 `server_time_ms_p50`。没有看到同一命令的成功 RPS 低于已记录值 85% 的复测。AV1、ST4、ST5、ST6、TP3、UX、OP、EX 的实现证据在 `src/`、`scripts/`、`docs/` 或 `.github/`，不写入本模块问题。

## 未覆盖

无
