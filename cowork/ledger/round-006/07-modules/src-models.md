# 模块评分

- id: src-models
- 权重: 15
- 分数: 7.9
- 读过的路径: `src/models/CMakeLists.txt`、`src/models/base_model.h`、`src/models/backend/model_runtime.h`、各家族公开头、`src/models/enhancement/real_esrgan.inl`、`test/realesrgan_output_contract_unittest.cc`

## 优势
RealESRGAN 空输出仍是 `MODEL_EMPTY_OUTPUT`，手工 shape/体积失败改为 `MODEL_OUTPUT_CONTRACT_FAILED`。FULL 上 5 条契约单测通过。

## 劣势
OCR 仍无识别。上新模型仍要写 C++。MOT/diffusion 接法不均。其他家族仍有把非空故障报成 `MODEL_EMPTY_OUTPUT` 的路径，例如 `attentive_gan_derain_net.inl` 约 114 与 129 行、`hrnet_segmentation.inl` 约 102 行。

## 依据
相对上轮 7.8 只动了 RealESRGAN 分码，上调到 7.9。结构性短板还在。

## 未读
无
