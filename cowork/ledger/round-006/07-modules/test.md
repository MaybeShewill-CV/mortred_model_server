# 模块评分

- id: test
- 权重: 15
- 分数: 8.6
- 读过的路径: `test/`、`test/CMakeLists.txt`、`test/realesrgan_output_contract_unittest.cc`、`test/golden/yolov8_detection.json`、`test/contract_baseline/v1_http_contract.md`

## 优势
FULL 已登记 `realesrgan_output_contract_unittest` 与 ctest 名 `RealEsrGanOutputContract`。5 条覆盖空输出、错 channel、错 rank、短 buffer、合法 NCHW。WSL 上 `-R` 为 1/1 Passed。

## 劣势
拒绝矩阵未含 dtype/NaN/Inf。Golden 仍依赖权重。GPU/TRT 无同级数值黑盒。

## 依据
相对上轮 8.4，本轮有专用 RealESRGAN 契约门，上调到 8.6。

## 未读
无
