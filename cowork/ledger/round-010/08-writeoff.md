# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4、P1-5、P1-6、D-1、D-2、D-3
- 挂起: 无
- 非挂起劣势: 相对本轮起点减少

第 1 步的模块记录和 `02-p0-p1-final.md` 保留作本轮起点，不改写。有效劣势以本文件为准。本次按 2026-10-08 口径销账，不重新打分。P1-5 已合入，并从有效劣势删除。角色原文在 `issues/p1-5/07-cto.md`。

## 删除的原句
| id | 模块 | 原句 |
| --- | --- | --- |
| P0-1 | src-models | SAM 非空契约故障仍报成空输出 |
| P1-1 | src-models | 接新架构/客户自定义导出仍要写 C++ |
| P1-2 | src-models | CLIP/LightGlue/SAM/diffusion 配错时未在启动或契约层点名 |
| P1-3 | conf、ops | conf 遗留 engine 清单与 `onnx_sources` 相对产品 toml 漂移，且对齐检查未进日常门禁 |
| P1-4 | conf、src-control | pack 覆盖偏样板，occupancy 依赖声明的 `gpu_mem_mib` |
| P1-6 | ci | 默认 PR 下 GPU/TRT 不是 fail-closed |
| D-1 | src-control | models deadline 到期后 child 可能仍停在 starting |
| D-2 | conf | attentive_gan.toml 输入尺寸仍 stale |
| D-3 | test | 缺权重时 golden 默认仍 SKIP |
| P1-5 | test | 无与 CPU golden 同级的 GPU/TRT 数值黑盒 |

## 留下的劣势
| id | 模块 | 原句 | 状态 |
| --- | --- | --- | --- |
| D-4 | test | LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 | 未做 |
| D-5 | docs | deployment.md §15 FAQ 结构损坏 | 未做 |
| D-6 | docs | 中文 README 路径 B 与现行 full 构建入口不一致 | 未做 |

## 附录
P1-1 的关闭范围是已合入的可复制模板。任意新架构零 C++ 不另开。P1-5 合入 `1f267ea20c125f40e674242a5908a2583a18e331`。关闭范围：产品 yolov8s 的 TRT 检测框门，同类且 IoU ≥ 0.5。CPU 用例 ctest 名 `yolov8_detection` 仍登记 `conf/ci/yolov8_onnx_hosted.toml`。其余家族的 TRT 数值门不另开。

P1-6 合入 `a9e6284bae832996c0d0c52ec39bd69da1d6fe05`。关闭范围：`cpu=success` 且 `gpu=skipped` 时退出码 2，并打印 `NOT_A_TRT_PROOF`。本条不含 GPU/TRT 数值黑盒。

D-1 合入 `8f3a0d101070737f63694024e94eab968f2c80f4`。关闭范围：deadline 返回后未就绪 child 的 `state` 为 `failed`。deadline 分支不发信号。外层 SIGKILL 仍属不计毛病。

D-2 合入 `03d604d766da55792494c5dadfc8998dbc671a76`。关闭范围：三份 attentive_gan toml 的 `model_input_image_size` 改为 `[240, 360]`，并删掉同一句 stale 注记。preprocess 仍用乘号 `240×360`。不改推理路径。

D-3 合入 `f43d544810918d9d9b831e0273cf1890951e3a2b`。关闭范围：选中用例全部因缺权重跳过时，`model_golden_test` 退出码 77，标准输出恰好一行 `NUMERICAL_NOT_RUN`。单条缺权重仍可以跳过。

src-models、ops、src-server-apps、src-factory-common、src-control、conf、ci 没有留下的劣势。

## 新准入
无
