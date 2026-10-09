# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、D-1、D-2、D-3、D-4、D-5
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
| test | 15 | 7.5 | 无 | 无 |
| docs | 10 | 9.1 | deployment.md §15 FAQ 结构损坏 | D-6：中文 README 路径 B 与现行 full 构建入口不一致 |
| ci | 5 | 8.0 | 无 | 无 |

## 附录
D-5 合入 `37533364646240c995eb612666a925278974d398`。问题分支提交 `bf03c58a68c28dbdc25b2d0d95bf025e3d4e7e7f`。QA 结论 pass。偏离无。RD 新发现无。QA 残余风险无。diff 为 `docs/deployment.md` 删除 4 行。关闭范围：§15 第六条 FAQ 闭合后的无 summary 段落和多余 `</details>` 已删除。六条既有 FAQ 与 §9.4 不动。

未改模块的分数照 `cowork/ledger/round-010/01-modules/` 九份，与 `cowork/ledger/round-010/08-writeoff.md` 一致，不改分。此前已关闭范围照该文件。

## 新准入
无
