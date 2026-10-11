# 销账

- 轮次: 12
- 已关闭: D-11
- 挂起: P0-2（工作树缺 39 个产品 onnx，且工作树看不见 trtexec。未跑验收 6、7、8、12）。P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）。D-8（本机和 WSL 都没有可执行的 docker。未跑验收 1。未合入）

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 9.3 | 无 | 无 |
| src-server-apps | 15 | 8.2 | 无 | 无 |
| src-control | 10 | 7.4 | 无 | S2：一次拉起后，检查失败并点名 fork 与 exec 之间的调用 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 8.1 | 无 | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:`（挂起） |
| ops | 15 | 7.0 | 无 | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:`（挂起）；S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因（挂起） |
| test | 15 | 6.0 | 无 | H2：点名的 id 没有 CPU 数值用例（挂起） |
| docs | 10 | 7.2 | `deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only | 无 |
| ci | 5 | 9.0 | 无 | 无 |

## 附录
同义句：`deployment.md` 把 cpu 目录写成 `resnet50`，并把 `msocrnet` 写成 Bench-only。关闭范围只含 `docs/deployment.md` 第 108 行与第 253 行这两处替换。D-11 不是双目标行。关闭它不改变 H1–H4、C1、C4、S1、S2、S3、S4、S6 的已满足/未满足。merge commit `a11758bd9201d1a002af2d727a43c876ce8c1943`，问题分支提交 `89d1e8e58dcaa37f0eb4d6c7eb7b554bfb3a8b95`。QA 结论 pass。调度者重跑验收 1 和验收 2，退出码都是 0，标准错误为空，标准输出与方案原文一致。偏离为无。

`docs/deployment.zh-cn.md` 与 README 里仍写 `resnet50` 的句子不新开。

H2 仍未满足，因为 P1-8 仍挂起。H4 仍未满足，因为 P0-2 仍挂起。S2 仍是 D-12。S3 仍是 D-8，状态挂起。S4 与 S6 已满足。

已关闭原句不重开：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、P1-12、D-1 至 D-6、D-9、D-10。

模块分保持第 11 轮记录，未重算。

## 新准入
无
