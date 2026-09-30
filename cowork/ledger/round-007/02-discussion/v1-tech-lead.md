# 第2步表决

- 轮次: 7
- 针对版本: v1
- 结论: 待修订

## 逐条
| id | 表决 | 理由 |
| --- | --- | --- |
| C1 | 合并 | 与 C2 同属非空 shape/体积故障误标 `MODEL_EMPTY_OUTPUT`。文件是 `attentive_gan_derain_net.inl`，不是已关闭 RealESRGAN 那单的改写。 |
| C2 | 合并 | HRNet 契约失败已是 `MODEL_OUTPUT_CONTRACT_FAILED`，mask 过小却仍回 empty。应并入同一条。 |

## 不同意的条目
无
