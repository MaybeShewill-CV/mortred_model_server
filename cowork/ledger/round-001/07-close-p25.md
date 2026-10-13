# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 无新关闭。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-7。解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-server-apps | 20 | 7.3 | 无新关闭。P2-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| src-control | 15 | 7.6 | 推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。合入 `525dd8353a3c5056a209d8a6c60e5b0d13cad96d`（`.git/logs/HEAD`：`merge cowork/r1-p2-5-token-listen`）。问题分支提交 `c38bff3a8b69a5449ff332953ce14476f8c2cd21`（`.git/refs/heads/cowork/r1-p2-5-token-listen`）。产品 diff：`conf/api_keys.toml.example`、`src/control/gateway/gateway_app.cpp`、`src/control/supervisor.cpp`、`src/control/supervisor/supervisor_app.cpp`、`src/control/trust_tokens.h`、`test/gateway_e2e_test.cc`、`test/gateway_multiinstance_test.cc`、`test/supervisor_unittest.cc`、`test/trust_tokens_unittest.cc`。`tokens_usable_for_listen` 在 `GatewayApp::init` 里、`listen` 之前拒绝这三种情况，`run` 返回 1。`spawn_locked` 在 `exec` 之前、在「executable not found」之前用同一判定，失败不 `exec`，`__gateway` 为 `failed`。重跑白盒退出码 0，外部脚本退出码 0。推理 token 缺失、管理 token 缺失、二者相同而 metrics token 不同时，gateway 退出码 1，不监听。三个彼此不同的短 token 可以监听，`GET /healthz` 为 200，正文 ok。QA：`cowork/ledger/round-001/issues/p2-5/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-5/05-merge.md`。P1-1、P2-2、P2-3 维持 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p23.md` 的已关闭，陈述不在此重复。 | 无 |
| src-factory-common | 10 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/src-factory-common.md`。 |
| conf | 5 | 6.0 | 无新关闭。`conf/api_keys.toml.example` 在 P2-5 的产品 diff 里，不关闭本模块的开放陈述。P0-1 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ops | 15 | 6.5 | 推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。合入 `525dd8353a3c5056a209d8a6c60e5b0d13cad96d`（`.git/logs/HEAD`：`merge cowork/r1-p2-5-token-listen`）。问题分支提交 `c38bff3a8b69a5449ff332953ce14476f8c2cd21`（`.git/refs/heads/cowork/r1-p2-5-token-listen`）。产品 diff：`conf/api_keys.toml.example`、`src/control/gateway/gateway_app.cpp`、`src/control/supervisor.cpp`、`src/control/supervisor/supervisor_app.cpp`、`src/control/trust_tokens.h`、`test/gateway_e2e_test.cc`、`test/gateway_multiinstance_test.cc`、`test/supervisor_unittest.cc`、`test/trust_tokens_unittest.cc`。直接启动的 gateway 在上述三种情况下退出码 1，不监听。三个彼此不同的短 token 可以监听，`GET /healthz` 为 200，正文 ok。重跑白盒退出码 0，外部脚本退出码 0。QA：`cowork/ledger/round-001/issues/p2-5/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-5/05-merge.md`。P2-4 维持 `cowork/ledger/round-001/07-close-p24.md` 的已关闭，陈述不在此重复。 | P2-6。停止信号没有收成只发给 supervisor 主进程（OP5）。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| test | 10 | 6.8 | 无新关闭。`test/gateway_e2e_test.cc`、`test/gateway_multiinstance_test.cc`、`test/supervisor_unittest.cc`、`test/trust_tokens_unittest.cc` 在 P2-5 的产品 diff 里，不关闭本模块的开放陈述。P2-4 维持 `cowork/ledger/round-001/07-close-p24.md` 的已关闭，陈述不在此重复。 | P2-9。`python3 scripts/golden_drift_check.py --check` 不能以 0 退出。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| docs | 5 | 6.5 | 无新关闭。P1-2 维持 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/02-p0-p1-final.md`。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/08-writeoff.md`；`cowork/ledger/round-001/01-modules/ci.md`。 |

## 附录

本步新关闭的只有 P2-5。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4 沿用 `cowork/ledger/round-001/08-writeoff.md` 与 `cowork/ledger/round-001/07-close-p24.md`，陈述不重复，也不写入仍开放。

src-control 与 ops 上的 P2-5 是同一条已关闭问题。九个模块分数沿用 `cowork/ledger/round-001/01-modules/` 正文。

P2-5 关闭的是推理 token 缺失、管理 token 缺失，或二者相同而 metrics token 不同时 gateway 仍会监听。三个彼此不同的短 token 可以监听。`scripts/security_warn.sh`、`docker-compose.yml`、`deploy/` 不在这次产品 diff 里。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-4 关闭的是 `yolov8s` 与 `libface` 同时 ready 时的并发成功记录。P2-11 仍要求覆盖集的 calibrate 与 `--raw` HTTP 报告。

P2-8 留在锁定队列附录，不重新入列。

P2-7 的通过结果是输入错误并写明 256×256。成功返回 `1×256×256×3` 不作为通过。

上一份关闭记录之后，未关闭且未阻塞的 id 为 6 条。本步关闭 P2-5 后为 5 条：P2-6、P2-7、P2-9、P2-10、P2-11。

`discovered.md` 的表是无。其中附录里的句子，包括这次外部脚本不启动 supervisor、`__gateway` 永久失败和管理面继续监听只在 `supervisor_unittest` 里看到，都不新入列。

## 新入列

无
