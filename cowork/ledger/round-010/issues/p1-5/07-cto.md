# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、D-1、D-2、D-3
- 挂起: 无

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 8.4 | 无 | 无 |
| src-server-apps | 15 | 9.3 | 无 | 无 |
| src-control | 10 | 8.7 | 无 | 无 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 7.9 | 无 | 无 |
| ops | 15 | 8.6 | 无 | 无 |
| test | 15 | 7.5 | 无与 CPU golden 同级的 GPU/TRT 数值黑盒 | LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 |
| docs | 10 | 9.1 | 无 | deployment.md §15 FAQ 结构损坏；中文 README 路径 B 与现行 full 构建入口不一致 |
| ci | 5 | 8.0 | 无 | 无 |

## 附录
P1-5 合入 `1f267ea20c125f40e674242a5908a2583a18e331`，问题分支提交 `55f812fca57f2744d3350c9018cf38e7c30d2a4c`。QA 结论 pass，偏离无。关闭范围：产品 yolov8s 的 TRT 检测框门，同类且 IoU >= 0.5。CPU 用例 ctest 名 `yolov8_detection` 仍登记 `conf/ci/yolov8_onnx_hosted.toml`。其余家族的 TRT 数值门不另开。diff 为 `test/CMakeLists.txt`、`test/model_golden_test.cc`。

工作树缺 `3rd_party` 是环境，不是产品劣势。缺 engine 的文案是 `tensorrt engine not available`，`golden_process_exit` 不把它当成 `weights not available`，因此这条不会退出 77。这是两侧已签字行为，不另开。

没有重跑 CPU 数值、方案本来不要求 score 近等、只覆盖 `bus.jpg`，都是 QA 残余风险，不另开。

P0-1、P1-1、P1-2、P1-3、P1-4、P1-6、D-1、D-2、D-3 的原句与关闭范围见 `cowork/ledger/round-010/08-writeoff.md`，本轮不重开。任意新架构零 C++ 不另开。src-models、src-server-apps、src-control、src-factory-common、conf、ops、ci 的劣势集合没改，引用该文件与 `01-modules`。docs 的 D-5、D-6 原句保留。test 留下 D-4 原句。九份模块分照抄 `01-modules`，未重算。

## 新准入
无
