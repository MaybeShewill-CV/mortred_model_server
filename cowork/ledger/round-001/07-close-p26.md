# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-7。解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-server-apps | 20 | 7.3 | 无新关闭。P2-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1、P2-2、P2-3、P2-5 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p25.md` 的已关闭，陈述不在此重复。 | 无 |
| src-factory-common | 10 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/src-factory-common.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ops | 15 | 6.5 | 停止信号没有收成只发给 supervisor 主进程（OP5）。合入 `1175a0cab213499034f33b3af376f6b108fadc68`。问题分支提交 `9bc0154af06976c66ce4328e9fa63374400bc07f`。产品 diff：`deploy/mortred-supervisor.service`（`[Service]` 增加一行 `KillMode=process`）、`scripts/stop_order_record.py`（新增）、`test/CMakeLists.txt`（注册 `stop_order_record_rules`）。本机没有 systemd，验收向这次拉起的 supervisor 主进程发 SIGTERM。一次外部跟踪里，supervisor 先向 yolov8s 和 libface 发 SIGINT 并等到它们退出，再向 gateway 发 SIGINT。gateway 退出后这两个模型进程不再存在。QA 记录：libface kill `82346180587`、wait `82346477685`；yolov8s kill `82346180743`、wait `82346482083`；gateway kill `82346482626`，晚于两个 wait。`after_gateway` 的 `gateway_alive`、`yolov8s_alive`、`libface_alive` 都是 false。重跑 `--check` 退出码 0，实跑退出码 0，记录判定 `JUDGE_OK`。QA：`cowork/ledger/round-001/issues/p2-6/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-6/05-merge.md`。P2-4、P2-5 维持已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 |
| test | 10 | 6.8 | 无新关闭。`test/CMakeLists.txt` 在 P2-6 的产品 diff 里，注册 `stop_order_record_rules`，不关闭本模块的开放陈述。P2-4 维持已关闭，陈述不在此重复。 | P2-9。`python3 scripts/golden_drift_check.py --check` 不能以 0 退出。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 |
| docs | 5 | 6.5 | 无新关闭。P1-2 维持已关闭，陈述不在此重复。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/ci.md`。 |

## 附录

本步新关闭的只有 P2-6。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5 沿用此前关闭记录，陈述不重复，也不写入仍开放。

P2-6 的陈述只在 ops。合入 `1175a0cab213499034f33b3af376f6b108fadc68`。本机没有 systemd，验收没有执行 `systemctl`。一次外部跟踪里，supervisor 先向 yolov8s 和 libface 发 SIGINT 并等到它们退出，再向 gateway 发 SIGINT。gateway 退出后这两个模型进程不再存在。`test/CMakeLists.txt` 注册 `stop_order_record_rules`，不关闭 test 的 P2-9、P2-11。

九个模块分数沿用 `01-modules` 正文。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-4 关闭的是 `yolov8s` 与 `libface` 同时 ready 时的并发成功记录。P2-11 仍要求覆盖集的 calibrate 与 `--raw` HTTP 报告。

P2-5 关闭的是推理 token 缺失、管理 token 缺失，或二者相同而 metrics token 不同时 gateway 仍会监听。三个彼此不同的短 token 可以监听。

P2-8 留在锁定队列附录，不重新入列。

P2-7 的通过结果是输入错误并写明 256×256。成功返回 `1×256×256×3` 不作为通过。

上一份关闭记录之后，未关闭且未阻塞的 id 为 5 条。本步关闭 P2-6 后为 4 条：P2-7、P2-9、P2-10、P2-11。

`discovered.md` 的表是无。其中附录里的句子，包括没有执行 `ctest -R stop_order_record_rules`，都不新入列。

## 新入列

无
