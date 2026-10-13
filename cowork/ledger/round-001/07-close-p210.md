# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。产品 diff 只有 `README.zh-cn.md`，不把 P2-10 记入本模块。P0-1、P2-7 维持 `cowork/ledger/round-001/07-close-p29.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | 无 |
| src-server-apps | 20 | 7.3 | 无新关闭。产品 diff 只有 `README.zh-cn.md`，不关闭本模块的开放陈述。P2-1 维持 `cowork/ledger/round-001/07-close-p29.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1、P2-2、P2-3、P2-5 维持 `cowork/ledger/round-001/07-close-p29.md` 的已关闭，陈述不在此重复。 | 无 |
| src-factory-common | 10 | 9.0 | 无新关闭。产品 diff 只有 `README.zh-cn.md`，不把 P2-10 记入本模块。 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持此前关闭记录，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| ops | 15 | 6.5 | 无新关闭。P2-4、P2-5、P2-6 维持 `cowork/ledger/round-001/07-close-p29.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| test | 10 | 6.8 | 无新关闭。产品 diff 只有 `README.zh-cn.md`，不关闭本模块的开放陈述。P2-9 维持 `cowork/ledger/round-001/07-close-p29.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| docs | 5 | 6.5 | 中文 README 第一小时命令块没有 `mortredctl next`。合入 `471dc4826b51da604f2550bd42b19975bfe4aee4`。问题分支提交 `17c751ce7f66a544e313342e94a0081e33592e8f`。产品 diff 只有 `README.zh-cn.md`，增加一行。QA 结论 pass。重跑三条检查退出码都是 0。中文 README「第一小时：mortredctl」的 bash 块在 `mortredctl doctor` 之前有 `mortredctl next`，其后仍能读到 `mortredctl doctor` 与 `mortredctl doctor --strict`。没有执行 `mortredctl`。QA：`cowork/ledger/round-001/issues/p2-10/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-10/05-merge.md`。P1-2 维持 `cowork/ledger/round-001/07-close-p29.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p29.md`；`cowork/ledger/round-001/08-writeoff.md`。 |

## 附录

本步新关闭的只有 P2-10。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9 沿用 `cowork/ledger/round-001/07-close-p29.md` 与 `cowork/ledger/round-001/08-writeoff.md`，陈述不重复，也不写入仍开放。

P2-10 的陈述只在 docs。合入 `471dc4826b51da604f2550bd42b19975bfe4aee4`。问题分支提交 `17c751ce7f66a544e313342e94a0081e33592e8f`。产品 diff 只有 `README.zh-cn.md`，增加一行。QA 结论 pass。重跑三条检查退出码都是 0。中文 README「第一小时：mortredctl」的 bash 块在 `mortredctl doctor` 之前有 `mortredctl next`，其后仍能读到 `mortredctl doctor` 与 `mortredctl doctor --strict`。没有执行 `mortredctl`。`README.zh-cn.md` 不关闭 docs 及其他模块的 P2-11。

UX2 仍由英文 README 与 `docs/deployment.md` 满足。本条不把 UX2 改成未满足，也不把 UX2 再记一次已满足的新变化。

九个模块分数保持上一份关闭记录，本步不重算。

上一份关闭记录之后，未关闭且未阻塞的 id 为 2 条。本步关闭 P2-10 后为 1 条：P2-11。

P2-8 留在锁定队列附录，不重新入列。

`discovered.md` 的表是无。附录里没有执行 `mortredctl`，不新入列。

## 新入列

无
