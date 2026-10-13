# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-7。解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-server-apps | 20 | 7.3 | 无新关闭。P2-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1、P2-2、P2-3 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p23.md` 的已关闭，陈述不在此重复。 | P2-5。推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-factory-common | 10 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/src-factory-common.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ops | 15 | 6.5 | 没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。合入 `eddcc454a20293ccbfbef0717cb6453032109cb5`（`.git/logs/HEAD`：`merge cowork/r1-p2-4-resident-pair`）。问题分支提交 `992942999c0b0cfc3cf84bbed609915bbd6e4939`（`.git/refs/heads/cowork/r1-p2-4-resident-pair`）。产品 diff：`scripts/resident_pair_record.py`、`test/CMakeLists.txt`。重跑 `python3 scripts/resident_pair_record.py --check` 退出码 0，实跑退出码 0，判定脚本退出码 0。记录里 `before` 的 `yolov8s` 与 `libface` 同时 `ready` 且 `state` 为 `running`；两条路径为 `/v1/models/yolov8s/infer` 与 `/v1/models/libface/infer`，两条 `start` 都大于 `before_sample`，且每条 `start` 小于另一条 `end`；两条 HTTP 都是 200，根上 `status` 与 `results[0].status` 都是 0。QA：`cowork/ledger/round-001/issues/p2-4/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-4/05-merge.md`。 | P2-5。推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。P2-6。停止信号没有收成只发给 supervisor 主进程（OP5）。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| test | 10 | 6.8 | 没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。合入 `eddcc454a20293ccbfbef0717cb6453032109cb5`（`.git/logs/HEAD`：`merge cowork/r1-p2-4-resident-pair`）。问题分支提交 `992942999c0b0cfc3cf84bbed609915bbd6e4939`。产品 diff：`scripts/resident_pair_record.py`、`test/CMakeLists.txt`。`test/CMakeLists.txt` 注册 `resident_pair_record_rules`，命令是 `python3 scripts/resident_pair_record.py --check`。重跑 `--check`、实跑与判定脚本退出码都是 0。记录里两者同时 ready，两条请求时间重叠且都成功。QA：`cowork/ledger/round-001/issues/p2-4/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-4/05-merge.md`。 | P2-9。`python3 scripts/golden_drift_check.py --check` 不能以 0 退出。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| docs | 5 | 6.5 | 无新关闭。P1-2 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/ci.md`。 |

## 附录

本步新关闭的只有 P2-4。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3 沿用 `cowork/ledger/round-001/08-writeoff.md`，陈述不重复，也不写入仍开放。

ops 与 test 上的 P2-4 是同一条已关闭问题。九个模块分数沿用 `cowork/ledger/round-001/01-modules/` 正文。

P2-4 关闭的是 `yolov8s` 与 `libface` 同时 ready 时的并发成功记录。P2-11 仍要求覆盖集的 calibrate 与 `--raw` HTTP 报告。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-8 留在锁定队列附录，不重新入列。

P2-7 的通过结果是输入错误并写明 256×256。成功返回 `1×256×256×3` 不作为通过。

上一份关闭记录之后，未关闭且未阻塞的 id 为 7 条。本步关闭 P2-4 后为 6 条：P2-5、P2-6、P2-7、P2-9、P2-10、P2-11。

`discovered.md` 的表是无。其中附录里的句子，包括这次没有执行 `ctest -R resident_pair_record_rules`，都不新入列。

## 新入列

无
