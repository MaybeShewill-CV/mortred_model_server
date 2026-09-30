# 模块评分

- id: test
- 权重: 15
- 分数: 8.7
- 读过的路径: `test/`、`test/CMakeLists.txt`、`test/attentive_gan_output_contract_unittest.cc`、`test/hrnet_output_contract_unittest.cc`、`test/golden/nanodet_detection.json`、`test/contract_baseline/v1_http_contract.md`

## 优势
FULL 上 `AttentiveGanOutputContract` 与 `HRNetOutputContract` 可被 `-R` 选中。现场 2/2 Passed。

## 劣势
新契约没有 dtype/NaN/Inf 拒绝矩阵。Golden 仍依赖权重。GPU/TRT 无同级数值黑盒。

## 依据
相对上轮 8.6，去雨和分割各增一门，上调到 8.7。

## 未读
无
