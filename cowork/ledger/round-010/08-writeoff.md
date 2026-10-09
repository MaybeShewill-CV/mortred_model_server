# 销账

- 轮次: 10
- 已关闭: P0-1、P1-1、P1-2、P1-3、P1-4
- 挂起: P1-5
- 非挂起劣势: 相对本轮起点减少

第 1 步的模块记录和 `02-p0-p1-final.md` 保留作本轮起点，不改写。有效劣势以本文件为准。本次按 2026-10-08 口径销账，不重新打分。

## 删除的原句
| id | 模块 | 原句 |
| --- | --- | --- |
| P0-1 | src-models | SAM 非空契约故障仍报成空输出 |
| P1-1 | src-models | 接新架构/客户自定义导出仍要写 C++ |
| P1-2 | src-models | CLIP/LightGlue/SAM/diffusion 配错时未在启动或契约层点名 |
| P1-3 | conf、ops | conf 遗留 engine 清单与 `onnx_sources` 相对产品 toml 漂移，且对齐检查未进日常门禁 |
| P1-4 | conf、src-control | pack 覆盖偏样板，occupancy 依赖声明的 `gpu_mem_mib` |

## 留下的劣势
| id | 模块 | 原句 | 状态 |
| --- | --- | --- | --- |
| P1-5 | test | 无与 CPU golden 同级的 GPU/TRT 数值黑盒 | 挂起 |
| P1-6 | ci | 默认 PR 下 GPU/TRT 不是 fail-closed | 未做 |
| D-1 | src-control | models deadline 到期后 child 可能仍停在 starting | 未做 |
| D-2 | conf | attentive_gan.toml 输入尺寸仍 stale | 未做 |
| D-3 | test | 缺权重时 golden 默认仍 SKIP | 未做 |
| D-4 | test | LightGlue / CLIP 未接完整 dtype/NaN/Inf 矩阵 | 未做 |
| D-5 | docs | deployment.md §15 FAQ 结构损坏 | 未做 |
| D-6 | docs | 中文 README 路径 B 与现行 full 构建入口不一致 | 未做 |

## 附录
P1-1 的关闭范围是已合入的可复制模板。任意新架构零 C++ 不另开。P1-5 的关闭范围仍是已签字方案里的 yolov8s、同类 IoU ≥ 0.5。其余家族的 TRT 数值门不在本轮劣势里。

src-models、ops、src-server-apps、src-factory-common 没有留下的劣势。

## 新准入
无
