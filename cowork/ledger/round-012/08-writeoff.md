# 销账

- 轮次: 12
- 已关闭: 无。18 个 CPU 黄金已合入，不覆盖 `msocrnet` 的数值通过。D-8 未合入。
- 挂起: P1-8（`msocrnet` 黄金未生成，没有 `NUMERICAL_PASSED`，不再在这台 WSL 上测试）。D-8（未合入，docker 不开发、不测试，验收 1 未跑）

## 模块
| id | 权重 | 分数 | 删除的原句 | 留下的劣势 |
| --- | --- | --- | --- | --- |
| src-models | 15 | 9.3 | 无 | 无 |
| src-server-apps | 15 | 8.2 | 无 | 无 |
| src-control | 10 | 7.4 | 无 | 无 |
| src-factory-common | 5 | 9.3 | 无 | 无 |
| conf | 10 | 8.1 | 无 | 无 |
| ops | 15 | 7.0 | 无 | S3：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因（挂起） |
| test | 15 | 6.0 | 无 | H2：`msocrnet` 没有 CPU 数值通过（挂起）。其余点名 id 的 CPU 黄金已合入 |
| docs | 10 | 7.2 | 无 | 无 |
| ci | 5 | 9.0 | 无 | 无 |

## 附录
用户于 2026-10-11 的原句仍有效。这两条行为没有实现，继续挂起，不写成已满足：

1. msocrnet不需要再 wsl 上测试，机器内存不够。
2. docker相关的不需要进行开发和测试。

同义句：点名的 id 没有 CPU 数值用例。18 个 id 已合入，关闭范围不含 `msocrnet` 的数值通过。merge commit `c6f164dd59796536ee03135fd74966d466751b0f`，问题分支提交 `42cf0f45`。diff 是 17 份 `conf/ci` toml、18 份 `test/golden` json、`test/model_golden_registry.h`、`test/model_golden_test.cc`。没有 `test/golden/msocrnet.json`。18 个 id 的 CPU 黄金已在基线：`nanodet_416`、`yolov7x`、`yolov8n`、`yolov8l`、`yolov8x`、`pphuman_lite`、`pphuman_server`、`hrnet`、`dinov2_vits14`、`dinov2_vitl14`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`、`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。`msocrnet` 的用例登记已在基线。黄金没有生成，测试未跑，不能记成 `NUMERICAL_PASSED`。

同义句：默认发布镜像以 root 运行，边界文档没有写明必须 root 的原因。D-8 仍挂起。用户决定 docker 相关不进行开发和测试。不开发、不测试、不合入。分支 `cowork/r12-d8-gpu-user` 停在 `4a732063`。验收 1 未跑。默认镜像没有被证明为 uid 10001。

已关闭原句不重开：P0-1、P0-2、P0-3、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、P1-7、P1-9、P1-10、P1-11、P1-12、D-1、D-2、D-3、D-4、D-5、D-6、D-9、D-10、D-11、D-12。P1-8 与 D-8 仍挂起。

模块分保持第 11 轮 `cowork/ledger/round-011/01-score.md`，未重算。没改到的模块沿用该记录。`cowork/ledger/round-012/discovered.md` 是无。

## 新准入
无
