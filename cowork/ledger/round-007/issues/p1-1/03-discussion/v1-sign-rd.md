# 第3步签字

- 问题: P1-1
- 方案版本: v1
- 结论: 同意
- 条件: 无
- 理由: `attentive_gan_derain_net.h` 与 `hrnet_segmentation.h` 的 `postprocess` 均在 `private:` 下，可升为 `protected` 供子类 `using`。`attentive_gan_derain_net.inl` 约 113–114 与 126–129 仍返回 `MODEL_EMPTY_OUTPUT`；约 76–78 空 outputs 不动。`hrnet_segmentation.inl` 约 100–102 仍返回 `MODEL_EMPTY_OUTPUT`。改动面与 FULL 单测可在现有进程模型内落地。
