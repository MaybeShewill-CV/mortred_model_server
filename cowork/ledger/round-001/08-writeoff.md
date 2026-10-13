# 关闭记录

- 轮次: 1
- 已关闭: P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9
- 阻塞: 无
- 未阻塞的开放项: 相对本轮起点减少

核对 `07-close.md`、`07-close-p12.md`、`07-close-p21.md`、`07-close-p22.md`、`07-close-p23.md`、`07-close-p24.md`、`07-close-p25.md`、`07-close-p26.md`、`07-close-p27.md` 与 `07-close-p29.md`。P0-1、P1-1、P1-2、P2-1、P2-2、P2-3、P2-4、P2-5、P2-6、P2-7、P2-9 的原陈述都没有留在仍开放的问题里。P2-8 没有重新入列。新入列为无。九个模块分数与 `01-modules` 正文一致。

本轮起点未关闭且未阻塞的 id 共 13 条。关闭 11 条后剩 2 条。

## 已关闭的问题陈述
| id | 模块 | 问题陈述 |
| --- | --- | --- |
| P0-1 | src-models | 驻留集中的 `libface` 对一张合法 JPEG 完不成成功推理。产品 toml、TensorRT profile 与引擎剖面的高宽不一致。 |
| P0-1 | conf | 产品 toml、TensorRT profile 与引擎剖面三处高宽不一致。 |
| P1-1 | src-control | 不创建 `/tmp/mortred_stage_timing` 时，gateway `/metrics` 读不到阶段耗时（PF4）。 |
| P1-2 | docs | 按文档给已有家族加一个模型时，步骤与当前 `catalog()`、`mortred-model-server.out --list` 使用的 id 不一致（EX1）。 |
| P2-1 | src-server-apps | 覆盖集 `max_batch_size=1` 的路径不记录 batch 观测，外部看不到执行 batch 为 1（PF3）。 |
| P2-2 | src-control | 打开限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。 |
| P2-3 | src-control | `fork` 返回之后、随后的 `exec` 之前，库函数跟踪里出现 `prctl`、`execvp` 或 `setenv`（ST5）。 |
| P2-4 | ops、test | 没有 `yolov8s` 与 `libface` 同时 ready 时对两者并发请求且两者都成功的记录（TP2）。 |
| P2-5 | src-control、ops | 推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。 |
| P2-6 | ops | 停止信号没有收成只发给 supervisor 主进程（OP5）。 |
| P2-7 | src-models | 解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan` 时，返回的不是写明 256×256 的输入错误。 |
| P2-9 | test | `python3 scripts/golden_drift_check.py --check` 不能以 0 退出。 |

## 仍开放的问题
| id | 模块 | 问题陈述 | 状态 |
| --- | --- | --- | --- |
| P2-10 | docs | 中文 README 第一小时命令块没有 `mortredctl next`。 | 未做 |
| P2-11 | src-server-apps、conf、ops、test、docs | 覆盖集九个 TensorRT id 没有一张按本次性能约束完成的 calibrate 与 `--raw` HTTP 报告清单（PF1、PF2、PF5、TP1；压测报告尚未引用 gateway 阶段与 `server_time_ms`，PF4 的报告半边仍在此）。 | 未做 |

## 附录

P0-1 在 src-models 与 conf 的两条陈述是同一条已关闭问题。

P1-1 只关闭「不创建 flag 文件就读不到 gateway 阶段」。同一次压测报告仍缺失，留在 P2-11。

P2-1 已关闭。合入 `d5c9dffc522802a7e29292c7d8bd90f71fcaa837`。`yolov8s` 提交 2 张合法图片后 `results` 长度为 2；随后 `/metrics` 里 `mortred_batch_size` 的样本都是 1，count 与 sum 都是 2。其余八个覆盖集 id 未单独验收，不新入列。

P2-2 已关闭。合入 `b44f9cb2afd06269f446c9e4dfe0e04a202dcbd0`。打开同一限流开关后，错误 token 打满 burst，同一 IP 紧接着的合法推理仍然成功。401 不改变该来源的剩余配额。shadow 没有单独量剩余配额，以及 admit 到归还之间的暂扣，留在 `discovered.md` 附录，不新入列。

P2-8 留在锁定队列附录，不重新入列。

P2-9 已关闭。合入 `f79e11ae6ceb167095ccd73483cb654f2032e57c`。`python3 scripts/golden_drift_check.py --check` 退出码 0，标准输出 `zero drift: 49 cases and 62 golden files unchanged`。`case_names` 含 `dinov2_vits14` 与 `metric3d_512`。没有重新跑推理，留在 `discovered.md` 附录，不新入列。

P2-7 已关闭。合入 `1df0997e561983bc2ddce217e379f10132d38f28`。一张解码后 320×240 的合法 JPEG 调用已拉起的 `realesrgan`，HTTP 400，`status` 8，`status_str` 全文为「尺寸须为 256×256」。成功返回 `1×256×256×3` 不作为通过。正好 256×256 未发成功请求，以及引擎被记成动态剖面，留在 `discovered.md` 附录，不新入列。

P2-5 已关闭。合入 `525dd8353a3c5056a209d8a6c60e5b0d13cad96d`。推理 token 缺失、管理 token 缺失，或二者相同而 metrics token 不同时，gateway 退出码 1 且不监听。三个彼此不同的短 token 可以监听，`GET /healthz` 为 200，正文为 `ok`。外部脚本不启动 supervisor，留在 `discovered.md` 附录，不新入列。

P2-4 已关闭。合入 `eddcc454a20293ccbfbef0717cb6453032109cb5`。同一份记录里 `yolov8s` 与 `libface` 同时 ready，两条请求时间重叠且都成功。没有执行 `ctest -R resident_pair_record_rules`，留在 `discovered.md` 附录，不新入列。

P2-6 已关闭。合入 `1175a0cab213499034f33b3af376f6b108fadc68`。一次外部跟踪里，supervisor 先向 yolov8s 和 libface 发 SIGINT 并等到它们退出，再向 gateway 发 SIGINT。gateway 退出后这两个模型进程不再存在。没有执行 `ctest -R stop_order_record_rules`，留在 `discovered.md` 附录，不新入列。

P2-3 已关闭。合入 `6031d5fdbe66b217762f25ba51f9a9027cfb8e6e`。同一次启动的库函数跟踪里，libface 在关掉 fd 3 及以上之后写下绝对路径 `execve` 终点，`inh=closed`，终点前没有 `prctl`、`execvp`、`setenv`。进程 `state=running`，采样日志为 `gpu sampler online`。父进程退出后是否收到 `SIGINT`、跟踪不在 CI、终点之后那一条跟踪，留在 `discovered.md` 附录，不新入列。

`discovered.md` 的表是无。两份 CPU toml 的 `--check` 退出码 1，以及 P1-1 外部样本没有覆盖 4096 与并列，都确认不了一次新的外部验收，不入列。

## 新入列

无
