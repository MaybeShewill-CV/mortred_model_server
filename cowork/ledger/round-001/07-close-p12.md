# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-7。解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-server-apps | 20 | 7.3 | 无新关闭。P1-1 已关闭的那一半维持上一份关闭记录，陈述不在此重复。 | P2-1。`max_batch_size=1` 时外部看不到 `mortred_batch_size` 样本为 1（PF3）。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-2。打开限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。P2-3。`fork` 返回之后、随后的 `exec` 之前，库函数跟踪里出现 `prctl`、`execvp` 或 `setenv`（ST5）。P2-5。推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-factory-common | 10 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/src-factory-common.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ops | 15 | 6.5 | 无 | P2-4。没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。P2-5。推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。P2-6。停止信号没有收成只发给 supervisor 主进程（OP5）。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| test | 10 | 6.8 | 无 | P2-4。没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。P2-9。`python3 scripts/golden_drift_check.py --check` 不能以 0 退出。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| docs | 5 | 6.5 | 按文档给已有家族加一个模型时，步骤与当前 `catalog()`、`mortred-model-server.out --list` 使用的 id 不一致（EX1）。合入 `259c11a006826f6a65b6ad24caded3c6c6a12d1d`。产品 diff 只有 `docs/how_to_add_new_server.md`、`README.zh-cn.md`、`README.md`。指南示例的 id、family 和 `model =` 为 `densenet`。`README.zh-cn.md` 的 `--model` 为 `mobilenetv2`。两份 README 的 cpu 精选为 `resnet`。QA 为 pass：`cowork/ledger/round-001/issues/p1-2/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p1-2/05-merge.md`。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/ci.md`。 |

## 附录

本步新关闭的只有 P1-2。P0-1、P1-1 沿用 `cowork/ledger/round-001/08-writeoff.md`，陈述不重复，也不写入仍开放。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-8 留在锁定队列附录，不重新入列。

P2-7 的通过结果是输入错误并写明 256×256。成功返回 `1×256×256×3` 不作为通过。

P2-3 的通过条件仍是一次库函数跟踪。绝对路径 `execve` 可以作为随后那次 `exec` 本身出现。只对照源码不算通过。

上一份关闭记录之后，未关闭且未阻塞的 id 为 11 条。本步关闭 P1-2 后为 10 条：P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9、P2-10、P2-11。

指南示例的 display name 仍是 `densenet classification`，源码注册行是 `Densenet classification`。不新入列。中文 README 第一小时命令块仍没有 `mortredctl next`，已是 P2-10，不新入列。

`discovered.md` 的表是无。两份 CPU toml 的 `--check` 退出码 1，以及 P1-1 外部样本没有覆盖 4096 与并列，都确认不了一次新的外部验收，不入列。

## 新入列

无
