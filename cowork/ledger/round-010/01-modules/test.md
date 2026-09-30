# 模块评分

- id: test
- 权重: 15
- 分数: 7.5
- 读过的路径: `test/`（目录清单与主要用例源）、`test/CMakeLists.txt`、`test/model_golden_test.cc`、`test/model_golden_registry.h`、`test/ci_require_weights.h`、`test/golden/yolov8_detection.json`、`test/contract_baseline/v1_http_contract.md`、`test/object_detection_output_contract_unittest.cc`、`test/lightglue_output_contract_unittest.cc`、`test/openai_clip_output_contract_unittest.cc`、`test/model_contract_test_util.h`、`test/backend_unittest.cc`、`test/rtdetr_detector_output_contract_unittest.cc`

## 优势
- `test/CMakeLists.txt` 在 FULL/workflow 门控下挂起 golden、lifecycle、多模型 output contract、gateway/supervisor e2e，默认 `check` 排除 valgrind，覆盖面可交付定位。
- `test/model_golden_registry.h` + `test/model_golden_test.cc` 统一七步 golden（分类/检测/分割/抠图/增强/关键点/CLIP 等），样例 `test/golden/yolov8_detection.json` 可对 IoU/score。
- `test/object_detection_output_contract_unittest.cc` 已覆盖 YOLOv8 错 dtype、短 buffer、NaN、Inf → `MODEL_OUTPUT_CONTRACT_FAILED`。
- `test/lightglue_output_contract_unittest.cc`、`test/openai_clip_output_contract_unittest.cc` 已有代表性空输出 / 契约失败错码（不得写成「完全没有」）。
- `test/ci_require_weights.h`：CI 设 `MORTRED_CI_REQUIRE_WEIGHTS=1` 时缺权重改 FAIL，推理 job 可 fail-closed。

## 劣势
- 缺权重时 golden 默认仍 `GTEST_SKIP`（`test/ci_require_weights.h` 的 `MORTRED_SKIP_OR_FAIL_WEIGHTS`；`test/model_golden_registry.h` `run_case` / `test/model_golden_test.cc` 各手写用例），本机无权重时套件可全绿却未校验数值结果。
- Golden 强制 CPU（`test/model_golden_registry.h` `force_cpu_backend`）；GPU/TRT 侧 `test/backend_unittest.cc` 的 `TrtSession::*` 只断言 session `run`/形状/dtype 拒绝，无与 `test/golden/` 同级的检测框/分数数值对照。
- LightGlue / CLIP 未接入 `test/model_contract_test_util.h` 的完整 missing/dtype/rank/shape/short-buffer/NaN/Inf 矩阵：`test/lightglue_output_contract_unittest.cc` 仅 extractor/matcher 空输出与部分 dtype/size/layout；`test/openai_clip_output_contract_unittest.cc` 仅空 texts 与 embedding shape，无 NaN/Inf 拒绝用例。

## 不计毛病
- 已有 YOLOv8 dtype/NaN/Inf 与 lightglue/clip 代表性错码，不按「完全没有契约测」计。
- `test/contract_baseline/` 为改造前快照（`v1_http_contract.md` 自述非现行契约），不按现行契约缺口计。

## 依据
- 构建入口：`test/CMakeLists.txt`（`TEST_LIST`、`MORTRED_BUILD_FULL` 追加 `model_golden_test` / `*_output_contract_unittest`、`add_test` / `check`）。
- Golden 权重门与 CPU 强制：`test/model_golden_registry.h`（`weights_available`、`MORTRED_SKIP_OR_FAIL_WEIGHTS`、`force_cpu_backend`）；`test/ci_require_weights.h`。
- YOLOv8 契约：`test/object_detection_output_contract_unittest.cc`（`YoloV8RejectsMalformedOutput`、`YoloV8RejectsInfinity`）。
- LightGlue/CLIP 代表性测 vs 完整矩阵：`test/lightglue_output_contract_unittest.cc`、`test/openai_clip_output_contract_unittest.cc` 对照 `test/model_contract_test_util.h` / `test/rtdetr_detector_output_contract_unittest.cc`。
- TRT 仅 session 级：`test/backend_unittest.cc`（`#ifdef MORTRED_HAS_TRT` 下 `InitAndRunYolov8` 等，无 golden 比对）。

## 未读
无
