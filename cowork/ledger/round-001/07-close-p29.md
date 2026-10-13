# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。产品 diff 只有 `test/golden_baseline.json`，不把 P2-9 记入本模块。P0-1、P2-7 维持 `cowork/ledger/round-001/07-close-p27.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | 无 |
| src-server-apps | 20 | 7.3 | 无新关闭。产品 diff 只有 `test/golden_baseline.json`，不关闭本模块的开放陈述。P2-1 维持此前关闭记录，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1、P2-2、P2-3、P2-5 维持 `cowork/ledger/round-001/07-close-p27.md` 的已关闭，陈述不在此重复。 | 无 |
| src-factory-common | 10 | 9.0 | 无新关闭。产品 diff 只有 `test/golden_baseline.json`，不把 P2-9 记入本模块。 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持此前关闭记录，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| ops | 15 | 6.5 | 无新关闭。P2-4、P2-5、P2-6 维持 `cowork/ledger/round-001/07-close-p27.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| test | 10 | 6.8 | `python3 scripts/golden_drift_check.py --check` 不能以 0 退出。合入 `f79e11ae6ceb167095ccd73483cb654f2032e57c`。问题分支提交 `3123def874691a807e4c5db07bae07e37a4cca35`。产品 diff：`test/golden_baseline.json`。QA 结论 pass。重跑 `--check` 退出码 0，标准输出 `zero drift: 49 cases and 62 golden files unchanged`。`case_names` 含 `dinov2_vits14` 与 `metric3d_512`。`golden_sha256` 含 `dinov2_vits14.json` 与 `metric3d_512.json`，值与文件字节 SHA-256 一致：`dinov2_vits14.json` 为 `c2c7d990d66a69f7af9d6268c46d0f87b376551ed744f61cf9af0b6d6cbc94fa`，`metric3d_512.json` 为 `f243416080350b7cc6308f31aecc6ad417d8b778238fb40d79db7d08ba01aa4a`。推理成功不作为通过。QA：`cowork/ledger/round-001/issues/p2-9/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-9/05-merge.md`。P2-4 维持 `cowork/ledger/round-001/07-close-p27.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| docs | 5 | 6.5 | 无新关闭。P1-2 维持此前关闭记录，陈述不在此重复。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p27.md`；`cowork/ledger/round-001/08-writeoff.md`。 |

## 附录

本步新关闭的只有 P2-9。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7 沿用此前关闭记录，陈述不重复，也不写入仍开放。

P2-9 的陈述只在 test。合入 `f79e11ae6ceb167095ccd73483cb654f2032e57c`。重跑 `python3 scripts/golden_drift_check.py --check` 退出码 0，标准输出 `zero drift: 49 cases and 62 golden files unchanged`。`case_names` 含 `dinov2_vits14` 与 `metric3d_512`。`golden_sha256` 含 `dinov2_vits14.json` 与 `metric3d_512.json`，值与文件字节 SHA-256 一致。推理成功不作为通过。`test/golden_baseline.json` 不关闭 test 的 P2-11，也不关闭 docs 的 P2-10。

九个模块分数沿用 `01-modules` 正文。

上一份关闭记录之后，未关闭且未阻塞的 id 为 3 条。本步关闭 P2-9 后为 2 条：P2-10、P2-11。

P2-8 留在锁定队列附录，不重新入列。

`discovered.md` 的表是无。附录里没有重新跑推理、以及没有改文件制造不一致，都不新入列。

## 新入列

无
