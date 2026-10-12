# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1
- 阻塞: 无
- 未阻塞的开放项: 相对本轮起点减少

核对 `07-close.md`。已关闭陈述没有留在仍开放的问题里。P0-1 的 libface 高宽不一致只出现在已关闭栏。P1-1 的「不创建 `/tmp/mortred_stage_timing` 就读不到 gateway 阶段」只出现在已关闭栏和附录。仍开放栏里的 PF4 指的是同一次压测报告还没有引用这些读数，对应 P2-11，不是已经关闭的那句。P2-8 没有重新入列。新入列为无。九个模块分数与 `01-modules` 正文一致。

本轮起点未关闭且未阻塞的 id 共 13 条。关闭 2 条后剩 11 条。

## 已关闭的问题陈述
| id | 模块 | 问题陈述 |
| --- | --- | --- |
| P0-1 | src-models | 驻留集中的 `libface` 对一张合法 JPEG 完不成成功推理。产品 toml、TensorRT profile 与引擎剖面的高宽不一致。 |
| P0-1 | conf | 产品 toml、TensorRT profile 与引擎剖面三处高宽不一致。 |
| P1-1 | src-control | 不创建 `/tmp/mortred_stage_timing` 时，gateway `/metrics` 读不到阶段耗时（PF4）。 |

## 仍开放的问题
| id | 模块 | 问题陈述 | 状态 |
| --- | --- | --- | --- |
| P1-2 | docs | 按文档给已有家族加一个模型时，步骤与当前 `catalog()` 使用的 id 不一致（EX1）。 | 未做 |
| P2-1 | src-server-apps | `max_batch_size=1` 时外部看不到 `mortred_batch_size` 样本为 1（PF3）。 | 未做 |
| P2-2 | src-control | 打开限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。 | 未做 |
| P2-3 | src-control | `fork` 返回之后、随后的 `exec` 之前，库函数跟踪里出现 `prctl`、`execvp` 或 `setenv`（ST5）。 | 未做 |
| P2-4 | ops、test | 没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。 | 未做 |
| P2-5 | src-control、ops | 推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。 | 未做 |
| P2-6 | ops | 停止信号没有收成只发给 supervisor 主进程（OP5）。 | 未做 |
| P2-7 | src-models | 解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。 | 未做 |
| P2-9 | test | `python3 scripts/golden_drift_check.py --check` 不能以 0 退出。 | 未做 |
| P2-10 | docs | 中文 README 第一小时命令块没有 `mortredctl next`。 | 未做 |
| P2-11 | src-server-apps、conf、ops、test、docs | 覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 | 未做 |

## 附录

P0-1 在 src-models 与 conf 的两条陈述是同一条已关闭问题。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-8 留在锁定队列附录，不重新入列。

P2-7 的通过结果是输入错误并写明 256×256。成功返回 `1×256×256×3` 不作为通过。

P2-3 的通过条件仍是一次库函数跟踪。绝对路径 `execve` 可以作为随后那次 `exec` 本身出现。只对照源码不算通过。

`discovered.md` 的表是无。两份 CPU toml 的 `--check` 退出码 1，以及 P1-1 外部样本没有覆盖 4096 与并列，都确认不了一次新的外部验收，不入列。

## 新入列

无
