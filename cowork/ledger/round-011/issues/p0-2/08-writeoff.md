# 销账

- 轮次: 12
- 已关闭: P0-2
- 挂起: P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）。D-8（本机和 WSL 都没有可执行的 docker。未跑验收 1。未合入）
- 非挂起劣势: 相对本轮起点减少

## 删除的原句
| id | 模块 | 原句 |
| --- | --- | --- |
| P0-2 | conf、ops | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` |

## 留下的劣势
| id | 模块 | 原句 | 状态 |
| --- | --- | --- | --- |
| P1-8 | test | H2：点名的 id 没有 CPU 数值用例 | 挂起 |
| D-8 | ops | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因 | 挂起 |

## 附录
同义句：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:`。关闭范围只含这条。H4 改为已满足。merge commit `6389504638dccc00dbae60987c1c4b7fcfb55350`，问题分支提交 `dd1699492955de96347726a5c8b9c914d1109251`。QA 结论 pass。调度者重跑冻结名单 39 个 id 的 `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id <id>`，退出码都是 0。偏离为无。diff 只有五份脚本。

H2 只剩挂起的 P1-8。S3 只剩挂起的 D-8。非挂起劣势为无。

已关闭原句不重开：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、P1-12、D-1 至 D-6、D-9、D-10、D-11、D-12。

模块分保持第 11 轮记录，未重算。

## 新准入
无
