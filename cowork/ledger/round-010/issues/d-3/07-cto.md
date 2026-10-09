# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4、P1-6、D-1、D-2、D-3
- 挂起: P1-5：无与 CPU golden 同级的 GPU/TRT 数值黑盒

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 8.4 | 无 | 无 |
| src-server-apps | 15 | 9.3 | 无 | 无 |
| src-control | 10 | 8.7 | 无 | 无 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 7.9 | 无 | 无 |
| ops | 15 | 8.6 | 无 | 无 |
| test | 15 | 7.5 | 缺权重时 golden 默认仍 SKIP | 无与 CPU golden 同级的 GPU/TRT 数值黑盒；LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 |
| docs | 10 | 9.1 | 无 | deployment.md §15 FAQ 结构损坏；中文 README 路径 B 与现行 full 构建入口不一致 |
| ci | 5 | 8.0 | 无 | 无 |

## 附录
D-3 合入 `f43d544810918d9d9b831e0273cf1890951e3a2b`，问题分支提交 `7aad43f8edbad7fbc449672007c3073f912df348`。QA 结论 pass，偏离无。关闭范围：选中用例全部因缺权重跳过时，`model_golden_test` 退出码 77，标准输出恰好一行 `NUMERICAL_NOT_RUN`。单条缺权重仍可以跳过。diff 为 `.github/workflows/ci.yml`、`docs/ci-golden-regression.md`、`docs/model-developer-guide.md`、`test/CMakeLists.txt`、`test/ci_require_weights.h`、`test/model_golden_test.cc`，这些改动属于 D-3 的关闭范围。

权重已存在时的 `NUMERICAL_PASSED` 路径本机没跑，只是 QA 残余风险。RD 新发现为无。这不是 D-3 的同义改写，也不另开劣势。

P0-1、P1-1、P1-2、P1-3、P1-4、P1-6、D-1、D-2 的原句与关闭范围见 `cowork/ledger/round-010/08-writeoff.md`，本轮不重开。src-models、src-server-apps、src-control、src-factory-common、conf、ops、ci 的劣势集合没改，引用该文件与 `01-modules`。docs 的 D-5、D-6 原句保留。九份模块分照抄 `01-modules`，未重算。

## 新准入
无
