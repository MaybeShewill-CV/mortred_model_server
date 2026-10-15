# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。分数沿用 `cowork/ledger/round-001/01-modules/src-models.md`。P0-1、P2-7 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| src-server-apps | 20 | 7.3 | P2-11。覆盖集九个 type=tensorrt 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。分数沿用 `cowork/ledger/round-001/01-modules/src-server-apps.md`。P2-1 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| src-control | 15 | 7.6 | 无新关闭。分数沿用 `cowork/ledger/round-001/01-modules/src-control.md`。P1-1、P2-2、P2-3、P2-5 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| src-factory-common | 10 | 9.0 | 无新关闭。分数沿用 `cowork/ledger/round-001/01-modules/src-factory-common.md`。 | 无 |
| conf | 5 | 6.0 | P2-11。覆盖集九个 type=tensorrt 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。分数沿用 `cowork/ledger/round-001/01-modules/conf.md`。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| ops | 15 | 6.5 | P2-11。覆盖集九个 type=tensorrt 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。分数沿用 `cowork/ledger/round-001/01-modules/ops.md`。P2-4、P2-5、P2-6 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| test | 10 | 6.8 | P2-11。覆盖集九个 type=tensorrt 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。分数沿用 `cowork/ledger/round-001/01-modules/test.md`。P2-9 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| docs | 5 | 6.5 | P2-11。覆盖集九个 type=tensorrt 的 id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1）。分数沿用 `cowork/ledger/round-001/01-modules/docs.md`。P1-2、P2-10 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，陈述不在此重复。 | 无 |
| ci | 5 | 9.0 | 无新关闭。分数沿用 `cowork/ledger/round-001/01-modules/ci.md`。上一轮证据：`cowork/ledger/round-001/07-close-p211.md`；`cowork/ledger/round-001/08-writeoff.md`。 | 无 |

## 附录

本步新关闭的只有 P2-11。合入 `9f1dc0b33b2a6f778e4c305a58e3ad831f1aff32`。QA 结论 pass。重跑退出码与 `cowork/ledger/round-001/issues/p2-11/05-qa.md` 一致。

`docs/perf/coverage_set_http_rps.md` 九行顺序是 mobilenetv2、yolov8s、libface、pphuman_mobile、dbnet、dinov2_vits14、superpoint、realesrgan、metric3d_512。每行 `--raw`，`wall_s` 不少于 60，`warmup_s=5.0`，`error_rate=0.0`，429 为 0。pack `worker_nums` 与 `-c` 分别是 4 与 8、4 与 8、4 与 8、2 与 4、2 与 4、2 与 4、1 与 2、4 与 8、1 与 2。`yolov8s` 的 `throughput_qps=400.677`。每行排队计数与推理计数分开，JPEG 落点与 `decode_auto` 一致。gateway 五段和 `server_time_ms_p50` 都在。五段之和大于模型 p50 一半，`dominant_stage=upstream`。硬件是 NVIDIA GeForce RTX 2070 SUPER，8192 MiB。`conf_server_worker_nums=1`，进程 worker 数等于 pack 的 `worker_nums`。

当前 rubric 要求官方压测带 `--raw`，并发数等于已部署 `worker_nums` 的两倍，时长不少于 30 秒。这张清单对得上 PF1、PF2、PF4、PF5、TP1。`08-writeoff.md` 里多写的 PF4 半边是同一条 P2-11，不另开。

后来的 `--reload` 没有合入，也没有覆盖这张清单，不另开。

相对合入：`docs/perf/coverage_set_http_rps.md` 新增 24 行；`scripts/calibrate_pack.py` 约 119 行变更；`scripts/coverage_http_report.py` 新增 1429 行；`scripts/server/http_infer_rps.py` 约 118 行变更。合计 4 个文件，1634 行新增，56 行删除。没有改 `src/`、`conf/server`、`conf/packs`。

P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10 的陈述是重复描述，维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p211.md`，不回到仍开放。

P2-8 留在锁定队列附录，不重新入列。

九个模块分数沿用 `cowork/ledger/round-001/01-modules`，本步不重评。没改到的模块引用 `cowork/ledger/round-001/07-close-p211.md` 与 `cowork/ledger/round-001/08-writeoff.md`。

上一份关闭记录之后，未关闭且未阻塞的 id 为 1 条。本步关闭 P2-11。

`discovered.md` 的表是无。附录里的句子一次外部验收确认不了，不写入新入列。

## 新入列

无
