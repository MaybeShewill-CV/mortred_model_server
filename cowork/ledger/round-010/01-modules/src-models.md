# 模块评分

- id: src-models
- 权重: 15
- 分数: 8.4
- 读过的路径:
  - cowork/ledger/rubric.md
  - cowork/ledger/round-010/00-rule-change.md
  - docs/unsupported-boundaries.md
  - src/models/CMakeLists.txt
  - src/models/base_model.h
  - src/models/backend/model_runtime.h
  - src/models/backend/backend_cv_model.h
  - src/models/backend/multi_session_model.h
  - src/models/backend/f32_output.h
  - src/models/classification/resnet.h
  - src/models/clip/openai_clip.h
  - src/models/clip/openai_clip.inl
  - src/models/diffusion/ddpm_unet.h
  - src/models/enhancement/real_esrgan.h
  - src/models/enhancement/enlightengan.inl
  - src/models/feature_embedding/dinov2.h
  - src/models/feature_point/lightglue.h
  - src/models/feature_point/lightglue.inl
  - src/models/matting/modnet_matting.h
  - src/models/mono_depth_estimation/depth_anything.h
  - src/models/mot/byte_tracker/byte_tracker.h
  - src/models/object_detection/yolov8_detector.h
  - src/models/object_detection/rtdetr_detector.h
  - src/models/object_detection/rtdetr_detector.inl
  - src/models/ocr/db_text_detector.h
  - src/models/io/ocr.h
  - src/models/scene_segmentation/bisenetv2.h
  - src/models/segment_anything/sam_prediction/sam_predictor.h
  - src/models/segment_anything/sam_prediction/sam_prompt_decoder.cpp
  - src/models/segment_anything/sam_prediction/sam_vit_encoder.cpp
  - test/lightglue_output_contract_unittest.cc
  - test/openai_clip_output_contract_unittest.cc

## 优势
- 统一生命周期：`BackendCvModel` 把 backend 建会话、`on_init`、`preprocess`/`postprocess` 串成固定路径；多引擎模型有 `MultiSessionModel` 声明式建会话（`backend_cv_model.h`、`multi_session_model.h`）。
- 三后端可交付：`CMakeLists.txt` 在 cpu profile 整段摘掉 TRT，gpu 才编入 `trt_session`；出错靠 `StatusCode` + `invoke_nothrow`（`base_model.h`）与 `ParamReader`/`SessionIoValidator`/`OutputReader`/`validated_f32_*`（`model_runtime.h`、`f32_output.h`）。
- 主力族覆盖检测/分割/抠图/深度/增强/分类等公开头 + `.inl`；已核对 lightglue / openai clip：非空 dtype/尺寸（及 embedding 形状不兼容）回 `MODEL_OUTPUT_CONTRACT_FAILED`，缺输出/空文本仍是 `MODEL_EMPTY_OUTPUT`（`lightglue.inl`、`openai_clip.inl`），并有对应无权重契约单测。

## 劣势
- 接新架构/客户自定义导出仍要写 C++：实现以公开头 + `.inl` 为主，`CMakeLists.txt` 只编译少量 backend/mot/sam/clip 源文件；常见换图通常要新类或改后处理，不能只改 TOML。
- 多 session / diffusion 比单图 `BackendCvModel` 更难一次配稳：CLIP/LightGlue/SAM 走 `MultiSessionModel` 与多后端表；diffusion（如 `ddpm_unet.h`）拆成多组件，现场比 YOLO/增强更易配错。
- SAM 仍有非空契约故障报成空输出：`sam_prompt_decoder.cpp` 在 full/low-res mask shape 非法时回 `MODEL_EMPTY_OUTPUT`；`sam_vit_encoder.cpp` 把非 F32 / 零元素与空输出捆成同一 `MODEL_EMPTY_OUTPUT`，排障时与真正空输出同码。

## 不计毛病
- HTTP OCR 仅检测：族内只有 `db_text_detector.h`；`io/ocr.h` 的 `text_region` 无识别文本字段；边界见 `docs/unsupported-boundaries.md`。
- RTDETR 脚手架：`rtdetr_detector.inl` 钩子回 `MODEL_NOT_IMPLEMENTED`，未当已上线检测器。
- 无 MOT HTTP：`mot/byte_tracker` 为库内跟踪器，不构成 HTTP 跟踪服务。

## 依据
- 分数 8.4：主力单图 CV 线在统一后端与契约分码下可交付可定位；lightglue/clip 错码已修、OCR/RTDETR/MOT HTTP 按新口径不计；仍因「上新写 C++」、多引擎/diffusion 接法、SAM 残余错码达不到 9.0。
- 必读路径与各家族至少一个公开头均已读；未读：无。

## 未读
无
