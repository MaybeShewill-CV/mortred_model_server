# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。本次产品 diff 没有 `src/`。P0-1、P2-7 维持 `cowork/ledger/round-001/07-close-p210.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| src-server-apps | 20 | 7.3 | 无新关闭。本次 diff 没有 `src/server`、`src/apps`。合入清单的官方压测 `-c` 是该行 `worker_nums` 的两倍，不关闭本模块的开放陈述。P2-1 维持 `cowork/ledger/round-001/07-close-p210.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 `type=tensorrt` 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1、P2-2、P2-3、P2-5 维持 `cowork/ledger/round-001/07-close-p210.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| src-factory-common | 10 | 9.0 | 无新关闭。 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| conf | 5 | 6.0 | 无新关闭。本次 diff 没有 `conf/server`、`conf/packs`。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 `type=tensorrt` 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| ops | 15 | 6.5 | 无新关闭。`scripts/calibrate_pack.py`、`scripts/coverage_http_report.py`、`scripts/server/http_infer_rps.py` 已合入。官方压测 `-c` 仍是该行 `worker_nums` 的两倍，不关闭开放陈述。P2-4、P2-5、P2-6 维持 `cowork/ledger/round-001/07-close-p210.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 `type=tensorrt` 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。 |
| test | 10 | 6.8 | 无新关闭。本次 diff 没有 `test/`。P2-9 维持 `cowork/ledger/round-001/07-close-p210.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 `type=tensorrt` 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| docs | 5 | 6.5 | 无新关闭。`docs/perf/coverage_set_http_rps.md` 已合入，九行 `-c` 都是该行 `worker_nums` 的两倍，不关闭开放陈述。P1-2、P2-10 维持 `cowork/ledger/round-001/07-close-p210.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 `type=tensorrt` 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p210.md`；`cowork/ledger/round-001/08-writeoff.md`。 |

## 附录

本步没有新关闭。合入 `9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32`。问题分支提交 `44b95e6f`、`34bd6bab`、`f9273a9e`。QA 结论 pass。重跑退出码与 `cowork/ledger/round-001/issues/p2-11/05-qa.md` 一致。

P2-11 仍开放。`docs/perf/coverage_set_http_rps.md` 九行都在。每一行官方压测的 `-c` 都是该行 `worker_nums` 的两倍：mobilenetv2 为 4 与 8，yolov8s 为 4 与 8，libface 为 4 与 8，pphuman_mobile 为 2 与 4，dbnet 为 2 与 4，dinov2_vits14 为 2 与 4，superpoint 为 1 与 2，realesrgan 为 4 与 8，metric3d_512 为 1 与 2。rubric「本次性能约束」第 3 步要求并发数等于已部署的 `worker_nums`。PF1 与 TP1 写明并发数不等于已部署的 `worker_nums` 算未满足。这张清单不算按该约束完成。仍开放的是锁定队列原文，同一条 id，不另开问题。

相对合入前的基线：`docs/perf/coverage_set_http_rps.md` 新增 24 行；`scripts/calibrate_pack.py` 为 119 行量级变更；`scripts/coverage_http_report.py` 新增 1429 行；`scripts/server/http_infer_rps.py` 为 118 行量级变更。合计 4 个文件，1634 行新增，56 行删除。产品 diff 没有 `src/`、`conf/server`、`conf/packs`。

P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p210.md` 的已关闭。陈述不重复，也不写入仍开放。

P2-8 留在锁定队列附录，不重新入列。

九个模块分数沿用 `cowork/ledger/round-001/01-modules`，本步不重评。没改到的模块沿用上一轮证据。

上一份关闭记录之后，未关闭且未阻塞的 id 为 1 条。本步仍是这一条：P2-11。

`cowork/ledger/round-001/discovered.md` 的表是无。附录里的句子一次外部验收确认不了，不写入新入列。

## 新入列

无
