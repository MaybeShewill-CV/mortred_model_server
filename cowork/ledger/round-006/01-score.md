# 项目评分

- 轮次: 4
- 自算原始分: 8.2
- 自算加权分: 8.2
- 上限规则: 无
- 相对上轮综合分: 高于 8.1

## 模块
| id | 权重 | 分数 | 记录 |
| --- | --- | --- | --- |
| src-models | 15 | 7.8 | 未改，沿用上轮 |
| src-server-apps | 15 | 8.1 | 本轮复审（drain 3s 截止） |
| src-control | 10 | 8.7 | 未改，沿用上轮 |
| src-factory-common | 5 | 7.6 | 未改，沿用上轮 |
| conf | 10 | 8.0 | 未改，沿用上轮 |
| ops | 15 | 8.7 | 未改，沿用上轮 |
| test | 15 | 8.4 | 本轮复审（深度假 tensor 契约、fail-closed 进 ctest） |
| docs | 10 | 8.7 | 未改，沿用上轮 |
| ci | 5 | 7.2 | 未改，沿用上轮 |

## P0
无

## P1
无

## 附录
本轮 P1-1（drain 时限）、P1-2（深度假 tensor 契约）、P1-3（fail-closed 进 ctest）已关闭。未升：WORKFLOW 首次线程池锁定、分段耗时依赖 touch 文件、golden 缺权重 SKIP、`trt_engines.json` 旧名、OCR 代码无识别、install-host 不 start、FAQ 游离 `</details>`、无 GPU CI、catalog nullptr。残余：P1-3 `RUN_SERIAL` 未用 `ctest -j` 现场复现；未跑完整 `cmake --build --target check`。

## 未读模块
无
