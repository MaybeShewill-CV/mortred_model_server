# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、D-1、D-2、D-3、D-4、D-5、D-6
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
| docs | 10 | 9.1 | 中文 README 路径 B 与现行 full 构建入口不一致 | 无 |
| ci | 5 | 8.0 | 无 | 无 |

## 附录
D-6 合入 `67ebfa2e2df031ac16355d070345f361e2fea800`。问题分支提交 `ee5fc42d5c06e4b3f26c4cd80868d10d4fa06beb`。QA 结论 pass。偏离无。RD 新发现无。QA 残余风险无。diff 只有 `README.zh-cn.md`，13 行增、21 行删，只在路径 B 切片内。关闭范围：路径 B 使用 `install_deps.sh` 与 `cmake --preset full` / `full-cpu`，切片里不再出现 `setup_full_deps.sh` 和手写 `cmake ..`。gpu 与 cpu 仍是两档，要求只执行其中一档。不改 `README.md`、deployment 文档、`CMakePresets.json`、`scripts/setup_full_deps.sh`。

同义句：路径 B 已与现行 full 构建入口对齐。这句只留附录。

未改模块的分数照 `cowork/ledger/round-010/01-modules/` 九份，与 `cowork/ledger/round-010/08-writeoff.md` 一致，不改分。此前已关闭的 P0-1、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、D-1、D-2、D-3、D-4、D-5 原句与关闭范围照该文件，不回到留下的劣势。

## 新准入
无
