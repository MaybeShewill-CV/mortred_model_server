# 第2步表决

- 轮次: 10
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| P1-1 | 拆开 | 新架构接法与多 session/diffusion 配错点名是两条可独立验收的终点。 |
| P1-2 | 改优先级 | SAM 契约失败标成空输出，现场出错无法定位，升为 P0。 |
| P1-3 | 保留 | 遗留 engine / onnx_sources 漂移且对齐未进日常门禁。 |
| P1-4 | 保留 | pack 偏样板且占用靠声明，换机/共卡易错。 |
| P1-5 | 保留 | 无与 CPU golden 同级的 TRT 数值门。 |
| P1-6 | 改优先级 | 中文路径 B 是一次性文档入口，降为劣势，仍进队列。 |
| P1-7 | 保留 | 默认 PR 下 GPU/TRT skipped 仍绿。 |
| D-1 | 保留 | deadline 后 child 可停在 starting。 |
| D-2 | 保留 | ATTENTIVE_GAN toml 尺寸与 session 不一致。 |
| D-3 | 保留 | 缺权重 golden 默认 SKIP。 |
| D-4 | 保留 | LightGlue/CLIP 契约矩阵不全。 |
| D-5 | 保留 | §15 FAQ 结构损坏。 |

## 不同意的条目
P1-1 拆成两条。P1-2 升为 P0。P1-6 降为劣势。
