# 第2步表决

- 轮次: 7
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| C1 | 保留 | AttentiveGAN 非空输出在 HWC 转换空或体积不符时与空 tensor 同回 `MODEL_EMPTY_OUTPUT`。改完后应可观察为契约失败码，空 outputs 仍 EMPTY。不是已关闭的 RealESRGAN 文件。 |
| C2 | 合并 | 与 C1 同类。HRNet 契约失败已是 `MODEL_OUTPUT_CONTRACT_FAILED`，mask 过小却 EMPTY。并入同一条。 |

## 不同意的条目
无
