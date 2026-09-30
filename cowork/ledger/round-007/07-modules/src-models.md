# 模块评分

- id: src-models
- 权重: 15
- 分数: 8.0
- 读过的路径: `src/models/CMakeLists.txt`、`src/models/base_model.h`、`src/models/backend/model_runtime.h`、各家族公开头、`attentive_gan_derain_net.inl`、`hrnet_segmentation.inl`

## 优势
AttentiveGAN 与 HRNet 的形状/体积失败改为 `MODEL_OUTPUT_CONTRACT_FAILED`。空输出仍是 `MODEL_EMPTY_OUTPUT`。契约单测已通过。

## 劣势
OCR 仍无识别。上新模型仍要写 C++。MOT/diffusion 接法不均。`lightglue.inl` 与 `openai_clip.inl` 仍有非空故障回 `MODEL_EMPTY_OUTPUT`。

## 依据
相对上轮 7.9，本轮关掉点名的两处误报，上调到 8.0。

## 未读
无
