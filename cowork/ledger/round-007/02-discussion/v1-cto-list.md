# 列表修订

- 轮次: 7
- 版本: v1
- 针对: 第 6 轮 P0/P1 皆空且综合分 8.3 后，AttentiveGan / HRNet 错码候选

## P0
无

## P1
| id | 问题 | 场景 | 现在的行为 | 改完后能观察到的变化 | 依据 |
| --- | --- | --- | --- | --- | --- |
| P1-1 | AttentiveGan / HRNet 在输出 shape/体积不符（或 HWC 转换失败、mask 小于输入图）时仍回 `MODEL_EMPTY_OUTPUT`，与空 tensor 同码 | 客户换导出权重或后端改 layout 后增强去雨 / HRNet 分割接口失败，运维只看 HTTP/状态码排障 | `outputs.empty()` 已回 `MODEL_EMPTY_OUTPUT`；契约校验失败处已回 `MODEL_OUTPUT_CONTRACT_FAILED`；但 AttentiveGan 的 HWC 转换空结果与 `element_count` 不符、HRNet 的 mask 小于输入图仍回 `MODEL_EMPTY_OUTPUT`（日志已写 invalid shape / smaller） | 上述 shape/体积类失败与空输出返回不同状态码（或契约失败码）；空 `outputs` 仍保持 `MODEL_EMPTY_OUTPUT`；日志能直接区分两类故障 | `src/models/enhancement/attentive_gan_derain_net.inl`（约 76–78、113–114、126–129 行）；`src/models/scene_segmentation/hrnet_segmentation.inl`（约 96、100–102 行） |

## 附录
无

## 未采纳
无
