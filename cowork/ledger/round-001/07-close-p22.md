# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-7。解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-server-apps | 20 | 7.3 | 无新关闭。P2-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-control | 15 | 7.6 | 打开限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。合入 `b44f9cb2afd06269f446c9e4dfe0e04a202dcbd0`（`.git/logs/HEAD`：`merge cowork/r1-p2-2-rate-auth`）。产品 diff：`src/control/gateway/gateway_app.cpp`、`src/control/gateway/gateway_app.h`、`src/control/rate_limit/gcra.h`、`test/gateway_multiinstance_test.cc`、`test/rate_limit_gcra_unittest.cc`。进入转发前的 401 在本次 admit 已允许时归还一个发射间隔。重跑退出码 0。`enabled` 与 `MORTRED_RATE_LIMIT=enforce` 的序列都是 3 次 401、3 次 200、再一次 429。QA：`cowork/ledger/round-001/issues/p2-2/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-2/05-merge.md`。P1-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-3。`fork` 返回之后、随后的 `exec` 之前，库函数跟踪里出现 `prctl`、`execvp` 或 `setenv`（ST5）。P2-5。推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-factory-common | 10 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/src-factory-common.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ops | 15 | 6.5 | 无 | P2-4。没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。P2-5。推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。P2-6。停止信号没有收成只发给 supervisor 主进程（OP5）。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| test | 10 | 6.8 | 无新关闭。两份测试文件在 P2-2 的产品 diff 里，不关闭本模块的开放陈述。 | P2-4。没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。P2-9。`python3 scripts/golden_drift_check.py --check` 不能以 0 退出。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| docs | 5 | 6.5 | 无新关闭。P1-2 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/ci.md`。 |

## 附录

本步新关闭的只有 P2-2。P0-1、P1-1、P1-2、P2-1 沿用 `cowork/ledger/round-001/08-writeoff.md`，陈述不重复，也不写入仍开放。

锁定队列里「打开 gateway 限流后，鉴权失败仍消耗来源 IP 的配额（ST2）」与本步关闭的短句是同一条已关闭陈述。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-8 留在锁定队列附录，不重新入列。

P2-7 的通过结果是输入错误并写明 256×256。成功返回 `1×256×256×3` 不作为通过。

P2-3 的通过条件仍是一次库函数跟踪。绝对路径 `execve` 可以作为随后那次 `exec` 本身出现。只对照源码不算通过。

上一份关闭记录之后，未关闭且未阻塞的 id 为 9 条。本步关闭 P2-2 后为 8 条：P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11。

`discovered.md` 的表是无。shadow 没有单独量剩余配额，以及 admit 到归还之间的暂扣，留在该文件附录，不新入列。两份 CPU toml 的 `--check` 退出码 1，以及 P1-1 外部样本没有覆盖 4096 与并列，都确认不了一次新的外部验收，不入列。

## 新入列

无
