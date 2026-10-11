# 销账

- 轮次: 12
- 已关闭: P0-2
- 挂起: P1-8（msocrnet 在约 19GiB 的 WSL 上被 OOM 杀掉，没有黄金；工作树没有 full 构建的 model_golden_test。未跑验收 3 的 msocrnet 与 19 条同进程过滤，以及验收 11 的 full 列表）。D-8（本机和 WSL 都没有可执行的 docker。未跑验收 1。未合入）

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 9.3 | 无 | 无 |
| src-server-apps | 15 | 8.2 | 无 | 无 |
| src-control | 10 | 7.4 | 无 | 无 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 8.1 | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` | 无 |
| ops | 15 | 7.0 | H4：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:` | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因（挂起） |
| test | 15 | 6.0 | 无 | H2：点名的 id 没有 CPU 数值用例（挂起） |
| docs | 10 | 7.2 | 无 | 无 |
| ci | 5 | 9.0 | 无 | 无 |

## 附录
同义句：不设 `MORTRED_PACK` 时，范围内全部 39 个 id 的产品 engine 缺失，`mortredctl doctor --strict` 仍退出 0，失败输出没有 `next:`。关闭范围只含这条，原先同时记在 conf 与 ops。H4 改为已满足。merge commit `6389504638dccc00dbae60987c1c4b7fcfb55350`，问题分支提交 `dd1699492955de96347726a5c8b9c914d1109251`。QA 结论 pass。调度者重跑冻结名单 39 个 id 的 `env -u MORTRED_PACK bash scripts/mortredctl_prepare.sh --id <id>`，退出码都是 0。偏离为无。diff 只有 `scripts/mortredctl_doctor.sh`、`scripts/mortredctl_prepare.sh`、`scripts/prepare_pack.sh`、`scripts/product_static_bs1.py`、`scripts/test_p0_2_unset_pack_doctor.sh`。

QA 残余风险：39 个 engine 执行前已存在，prepare 打印 skip，这是签字方案允许的结果；`.github` 没有调用该测试脚本。这两句都不是新准入。

H2 只剩挂起的 P1-8。S3 只剩挂起的 D-8。

已关闭原句不重开：P0-1、P0-3、P1-1 至 P1-7、P1-9、P1-10、P1-11、P1-12、D-1 至 D-6、D-9、D-10、D-11、D-12。

模块分保持第 11 轮记录，未重算。

## 新准入
无
