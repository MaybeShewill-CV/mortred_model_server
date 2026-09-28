# 模块评分

- id: test
- 权重: 15
- 分数: 7.8
- 读过的路径: `test/`（含 `golden/`、`contract_baseline/`、`testdata/`、约 70+ `*_unittest.cc`/`*_test.cc`、`CMakeLists.txt`）；`test/golden/yolov8_onnx_detection.json`；`test/contract_baseline/v1_http_contract.md`；`CMakeLists.txt`（`enable_testing` / `add_subdirectory(test)`）；`test/CMakeLists.txt`（`check`/`add_test`）；`test/model_golden_test.cc`；`test/model_golden_registry.h`；`test/http_contract_test.cc`；`test/server_e2e_contract_test.cc`；`test/ci_require_weights.h`；`scripts/run_tests.sh`

## 优势
- 根构建已挂测试：`enable_testing()` + `add_subdirectory(test EXCLUDE_FROM_ALL)`，`test/CMakeLists.txt` 用 GTest/`add_test`/`check`（默认排除 valgrind label）注册大量可执行用例，小团队可用 `scripts/run_tests.sh` / `ctest` 复跑（`CMakeLists.txt:143-144`，`test/CMakeLists.txt:21,300-301`，`scripts/run_tests.sh`）。
- HTTP/进程契约有可执行证据：`http_contract_test.cc` 锁 StatusCode→HTTP 与统一信封字段；`server_e2e_contract_test.cc` 起真实 WFHttpServer+FakeModel 验信封/状态码/headers；full 构建还有 `gateway_e2e_test`/`supervisor_*` + `fake_model_server`（`test/http_contract_test.cc`，`test/server_e2e_contract_test.cc:1-8`，`test/CMakeLists.txt:117-119,259-295`）。
- 客户主任务有 L2 golden 回归：`model_golden_test` + `test/golden/`（检测/分类/分割/抠图/增强/特征点/SAM/CLIP 等 JSON/PNG），七步流程与 `MORTRED_UPDATE_GOLDEN` 写在 `model_golden_registry.h`；另有若干 `*_output_contract_unittest` 用假 tensor 验后处理，不依赖权重（`test/model_golden_test.cc:6-23,53-268`，`test/golden/yolov8_onnx_detection.json`，`test/object_detection_output_contract_unittest.cc`，`test/CMakeLists.txt:117-139`）。

## 劣势
- 对交付最有用的推理证据默认拿不到：`model_golden_test` 与多数 output-contract 仅在 `MORTRED_BUILD_FULL` 进 `TEST_LIST`；无权重时 `MORTRED_SKIP_OR_FAIL_WEIGHTS` 本地 SKIP，只有 CI 设 `MORTRED_CI_REQUIRE_WEIGHTS` 才 FAIL——小团队本机绿跑不等于客户模型行为被验过（`test/CMakeLists.txt:117-139`，`test/ci_require_weights.h:3-7,31-37`，`test/model_golden_test.cc:18-20`）。
- Golden 强制 CPU 后端（注释写明剥 `../`、backend=cpu），GPU/TensorRT 交付路径几乎没有同级黑盒数值回归；TRT 相关更多靠 `MORTRED_HAS_TRT` 编译裁剪与控制面单测（`test/model_golden_test.cc:21-22`，`test/CMakeLists.txt:133-136`）。
- 目录里的 `contract_baseline/v1_http_contract.md` 自承「不是现行契约」（旧 `img_data`/`code`/`msg` 快照），容易和现行统一信封测试混淆；`model_catalog_unittest` 登记了 `mono_depth_estimation`，但 `model_golden_test`/`golden/` 未见深度估计黑盒基线（`test/contract_baseline/v1_http_contract.md:1-8`，`test/model_catalog_unittest.cc` 对照 `test/model_golden_test.cc`/`test/golden/` 列表）。

## 依据
- 构建入口：仓库根 `CMakeLists.txt` L143–144；套件与门控：`test/CMakeLists.txt`（`TEST_LIST`、`MORTRED_BUILD_FULL`/`MORTRED_WORKFLOW_AVAILABLE`、`check`、`add_test`）。
- 黑盒/契约：`http_contract_test.cc`、`server_e2e_contract_test.cc`、gateway/supervisor e2e + `fake_model_server.cc`。
- 回归样例：`test/golden/yolov8_onnx_detection.json`（及同目录多任务基线）；编排：`model_golden_test.cc` + `model_golden_registry.h`。
- 历史快照（非现行）：`test/contract_baseline/v1_http_contract.md`。
- 分数 7.8：契约与多任务 CPU golden 足以定位多数服务/后处理回归，但 SME 真交付（本机无权重、GPU/TRT、深度等）仍缺稳定可执行黑盒证据，达不到 9+。

## 未读
无
