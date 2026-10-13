# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7
- 阻塞: 无

## 模块
| id | 权重 | 分数 | 已关闭的问题陈述 | 仍开放的问题 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 6.0 | 解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。合入 `1df0997e561983bc2ddce217e379f10132d38f28`。问题分支提交 `49030aa42cb90a73ae9b643577a23fe6e4310cfb`。产品 diff：`src/common/status_code.h`、`src/server/http_status.h`、`src/models/enhancement/real_esrgan.h`、`src/models/enhancement/real_esrgan.inl`、`test/status_code_unittest.cc`。QA 结论 pass。重跑白盒退出码 0，状态码测试退出码 0（3 项通过），实跑退出码 0 并打印 `p2-7-ok`。一张解码后 320×240 的合法 JPEG 调用已拉起的 `realesrgan`，HTTP 400，`status` 8，`status_str` 全文为「尺寸须为 256×256」。不要求动态 shape。成功返回 `1×256×256×3` 不作为通过。QA：`cowork/ledger/round-001/issues/p2-7/05-qa.md`。合入记录：`cowork/ledger/round-001/issues/p2-7/05-merge.md`。P0-1 维持 `cowork/ledger/round-001/07-close-p26.md` 与 `cowork/ledger/round-001/08-writeoff.md` 的已关闭，陈述不在此重复。 | 无 |
| src-server-apps | 20 | 7.3 | 无新关闭。`src/server/http_status.h` 在 P2-7 的产品 diff 里，不关闭本模块的开放陈述。P2-1 维持此前关闭记录，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p26.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| src-control | 15 | 7.6 | 无新关闭。P1-1、P2-2、P2-3、P2-5 维持 `cowork/ledger/round-001/07-close-p26.md` 的已关闭，陈述不在此重复。 | 无 |
| src-factory-common | 10 | 9.0 | 无新关闭。`src/common/status_code.h` 在 P2-7 的产品 diff 里，不把 P2-7 记入本模块。 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p26.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| conf | 5 | 6.0 | 无新关闭。P0-1 维持此前关闭记录，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。上一轮证据：`cowork/ledger/round-001/07-close-p26.md`；`cowork/ledger/round-001/08-writeoff.md`。 |
| ops | 15 | 6.5 | 无新关闭。P2-4、P2-5、P2-6 维持 `cowork/ledger/round-001/07-close-p26.md` 的已关闭，陈述不在此重复。 | P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 |
| test | 10 | 6.8 | 无新关闭。`test/status_code_unittest.cc` 在 P2-7 的产品 diff 里，不关闭本模块的开放陈述。P2-4 维持此前关闭记录，陈述不在此重复。 | P2-9。`python3 scripts/golden_drift_check.py --check` 不能以 0 退出。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 |
| docs | 5 | 6.5 | 无新关闭。P1-2 维持此前关闭记录，陈述不在此重复。 | P2-10。中文 README 第一小时命令块没有 `mortredctl next`。P2-11。覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 |
| ci | 5 | 9.0 | 无 | 无。上一轮证据：`cowork/ledger/round-001/07-close-p26.md`；`cowork/ledger/round-001/08-writeoff.md`。 |

## 附录

本步新关闭的只有 P2-7。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6 沿用此前关闭记录，陈述不重复，也不写入仍开放。

P2-7 的陈述只在 src-models。合入 `1df0997e561983bc2ddce217e379f10132d38f28`。一张解码后 320×240 的合法 JPEG 调用已拉起的 `realesrgan`，HTTP 400，`status` 8，`status_str` 全文为「尺寸须为 256×256」。不要求动态 shape。成功返回 `1×256×256×3` 不作为通过。`src/server/http_status.h` 不关闭 src-server-apps 的 P2-11。`src/common/status_code.h` 不给 src-factory-common 新增开放项。`test/status_code_unittest.cc` 不关闭 test 的 P2-9、P2-11。

九个模块分数沿用 `01-modules` 正文。

上一份关闭记录之后，未关闭且未阻塞的 id 为 4 条。本步关闭 P2-7 后为 3 条：P2-9、P2-10、P2-11。

P2-8 留在锁定队列附录，不重新入列。

`discovered.md` 的表是无。附录里正好 256×256 未发成功请求、以及引擎被记成动态剖面，都不新入列。

## 新入列

无
