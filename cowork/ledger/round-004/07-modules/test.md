# 模块评分

- id: test
- 权重: 15
- 分数: 8.4
- 读过的路径: `cowork/ledger/rubric.md`；`test/`（目录清单，约 98 项含 `golden/`、`contract_baseline/`、`testdata/`、各 `*_unittest.cc`/`*_test.cc`）；`test/CMakeLists.txt`；`test/mono_depth_output_contract_unittest.cc`；`test/worker_pool_unittest.cc`；`test/async_job_unittest.cc`（含 drain 段）；`test/golden/yolov8_detection.json`；`test/contract_baseline/v1_http_contract.md`；`scripts/test_p0_1_engine_fail_closed.sh`（AC8/退出码段）；`test/ci_require_weights.h`；`test/model_golden_test.cc`（头部行为约定）

## 优势
- 套件入口清楚：`TEST_LIST` + `add_test` + `check`（默认 `-LE valgrind`），FULL 才挂模型契约/golden；本轮 `worker_pool_unittest` / `async_job_unittest` 已在常驻列表，并带 sanitizer label（`test/CMakeLists.txt:42-92,185-186,242-246,306-314`）。
- 关停可定位：`worker_pool.drain_timeout_returns_with_remaining_log`、`async_endpoints.drain_timeout_returns_with_inflight_log` / `shared_deadline_budget_under_350ms` 断言超时返回与 `drain timeout` / `remaining`/`inflight` 日志（`test/worker_pool_unittest.cc:88-105`，`test/async_job_unittest.cc:325-363`）。
- 深度族有无权重后处理契约：`MonoDepthOutputContract` 覆盖空输出与错误 rank，FULL 登记 `mono_depth_output_contract_unittest`（`test/mono_depth_output_contract_unittest.cc:55-79`，`test/CMakeLists.txt:126,223-224`）；现场经验：WSL 上 `ctest -R mono_depth_output_contract` 1/1 Passed。
- P0-1 黑盒可进默认 ctest：`add_test(NAME p0_1_engine_fail_closed …)`，`RUN_SERIAL`、无 valgrind label；脚本无 trtexec 时 AC8 `[SKIP]` 且 `FAILS=0` → exit 0（`test/CMakeLists.txt:317-325`，`scripts/test_p0_1_engine_fail_closed.sh:193-222`）；现场经验：tests-only 与 FULL 的 `ctest -N` 均能列出该名。

## 劣势
- 对交付最有用的推理黑盒默认仍弱：`model_golden_test` 与多数 `*_output_contract` 仅 `MORTRED_BUILD_FULL`；无权重本地 `GTEST_SKIP`（`test/CMakeLists.txt:117-139`，`test/ci_require_weights.h:3-7,31-37`，`test/model_golden_test.cc:18-22`）。
- Golden 仍强制 CPU，GPU/TRT 无同级数值黑盒；`p0_1` 的 AC8 在无 GPU/trtexec/onnx 时仍 SKIP，不构成闭环节回归（`test/model_golden_test.cc:21-22`，`scripts/test_p0_1_engine_fail_closed.sh:193-200`）。
- 深度仅有失败路径契约，无 `test/golden/` 正路径基线；`contract_baseline/v1_http_contract.md` 自承非现行契约（`test/mono_depth_output_contract_unittest.cc` 对照 `test/golden/` 清单，`test/contract_baseline/v1_http_contract.md:1-8`）。

## 依据
- 相对上轮 8.2：drain 超时单测、DepthAnything 无权重契约进 FULL、以及 `p0_1_engine_fail_closed` 挂入 `add_test`（且默认 `check` 可跑）抬升关停/深度失败定位与 fail-closed 可发现性。
- 仍不到 9.0：本机绿跑仍不等于客户 GPU/TRT 与带权重推理被验过；深度无数值 golden。
- 分数 8.4：契约 + 关停/深度失败路径 + ctest 可发现的 P0-1 脚本，足以支撑多数服务面回归；SME 真交付证据链仍有洞。

## 未读
无
