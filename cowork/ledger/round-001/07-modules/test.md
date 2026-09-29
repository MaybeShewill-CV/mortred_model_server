# 模块评分

- id: test
- 权重: 15
- 分数: 8.1
- 读过的路径: `test/`；`test/golden/yolov8_onnx_detection.json`；`test/contract_baseline/v1_http_contract.md`；根 `CMakeLists.txt`；`test/CMakeLists.txt`；`test/model_golden_test.cc`；`test/supervisor_unittest.cc`；`test/supervisor_multiinstance_test.cc`；`scripts/test_p0_1_engine_fail_closed.sh`

## 优势
- 根构建仍挂测试：`enable_testing()` + `add_subdirectory(test)`。
- 本轮 supervisor 单测覆盖 scrape 缺失、gateway 截止、缺二进制时跳过模型波。
- P0-1 脚本覆盖产品 engine 单路径与 doctor/init fail-closed（AC1–AC7）。

## 劣势
- 推理黑盒默认仍难拿到：无权重时本地 SKIP。
- Golden 仍强制 CPU；GPU/TRT 无同级数值黑盒。P0-1 脚本 AC8 无 GPU 即 SKIP，且未挂入 `ctest`。
- `contract_baseline/v1_http_contract.md` 自承非现行契约；深度族无 golden。

## 依据
- 分数 8.1：相对上轮 7.8，控制面与 TRT fail-closed 回归更好；推理路径默认证据缺口未消，不到 9.0。

## 未读
无
