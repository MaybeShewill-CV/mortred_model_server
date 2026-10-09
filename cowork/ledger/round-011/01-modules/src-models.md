# 模块评分

- id: src-models
- 权重: 15
- 分数: 9.3
- 读过的路径:
  - cowork/ledger/rubric.md
  - src/models/CMakeLists.txt
  - src/models/base_model.h
  - src/models/backend/model_runtime.h
  - src/models/backend/f32_output.h
  - src/models/backend/backend_cv_model.h
  - src/models/backend/session.cpp
  - src/models/backend/trt_session.cpp
  - src/models/catalog/model_entry.h
  - src/models/classification/resnet.h
  - src/models/classification/resnet.inl
  - src/models/classification/mobilenetv2.h
  - src/models/classification/densenet.h
  - src/models/clip/openai_clip.h
  - src/models/diffusion/ddpm_unet.h
  - src/models/enhancement/real_esrgan.h
  - src/models/enhancement/enlightengan.h
  - src/models/enhancement/attentive_gan_derain_net.h
  - src/models/feature_embedding/dinov2.h
  - src/models/feature_embedding/dinov2.inl
  - src/models/feature_point/superpoint.h
  - src/models/feature_point/lightglue.h
  - src/models/io/classification.h
  - src/models/io/object_detection.h
  - src/models/io/scene_segmentation.h
  - src/models/io/ocr.h
  - src/models/io/enhancement.h
  - src/models/io/feature_point.h
  - src/models/io/feature_embedding.h
  - src/models/io/matting.h
  - src/models/io/mono_depth_estimation.h
  - src/models/matting/modnet_matting.h
  - src/models/matting/pp_matting.h
  - src/models/mono_depth_estimation/depth_anything.h
  - src/models/mono_depth_estimation/depth_anything.inl
  - src/models/mono_depth_estimation/metric3d.h
  - src/models/mot/byte_tracker/byte_tracker.h
  - src/models/object_detection/yolov8_detector.h
  - src/models/object_detection/yolov8_detector.inl
  - src/models/object_detection/yolov5_detector.inl
  - src/models/object_detection/yolov6_detector.inl
  - src/models/object_detection/nano_detector.inl
  - src/models/object_detection/centerface_detector.h
  - src/models/object_detection/centerface_detector.inl
  - src/models/object_detection/libface_detector.h
  - src/models/ocr/db_text_detector.h
  - src/models/scene_segmentation/bisenetv2.h
  - src/models/scene_segmentation/bisenetv2.inl
  - src/models/scene_segmentation/pp_humanseg.h
  - src/models/scene_segmentation/hrnet_segmentation.h
  - src/models/scene_segmentation/msocrnet.h
  - src/models/segment_anything/sam_prediction/sam_predictor.h
  - src/factory/classification_task.h
  - src/factory/obj_detection_task.h
  - src/factory/scene_segmentation_task.h
  - src/factory/ocr_task.h
  - src/factory/enhancement_task.h
  - src/factory/feature_point_task.h
  - src/factory/feature_embedding_task.h
  - src/factory/matting_task.h
  - src/factory/mono_depth_estimate_task.h
  - test/model_golden_test.cc
  - test/model_golden_registry.h

## 优势
- 范围内每个 HTTP catalog id 都有实现，输出带可比较字段：分类 `class_id`（`resnet.inl` argmax）、检测与人脸框加 `class_id`（`yolov8_detector.inl`、`centerface_detector.inl`、`io/object_detection.h`）、分割类别图（`bisenetv2.inl`）、OCR 多边形（`io/ocr.h`）、增强图、SuperPoint 点、DINOv2 向量、抠图 mat、深度 `depth_map`。
- `yolov8n` / `yolov8s` / `yolov8l` / `yolov8x` 共用 `YoloV8Detector`，输入高宽来自会话张量，不是写死 `yolov8s`（`yolov8_detector.inl`）。`yolov8s` 的 IoU 关闭范围没有当成其余三个 id 已证明。
- 失败不按成功返回：未初始化拒绝跑（`base_model.h`）；预处理空张量回 `MODEL_EMPTY_INPUT_IMAGE`（`backend_cv_model.h`）；形状或非有限值回 `MODEL_OUTPUT_CONTRACT_FAILED`（`f32_output.h`）；cpu 构建遇到 `type=tensorrt` 直接 init 失败（`session.cpp`）；engine 缺失或 TensorRT 版本不对回 `MODEL_INIT_FAILED` 并写明路径（`trt_session.cpp`）。

## 劣势
无

## 不计毛病
- SAM、diffusion、CLIP、LightGlue 不在最高标准范围表。`segment_anything`、`diffusion`、`clip`、`feature_point/lightglue.h` 只作家族头核对。
- HTTP OCR 只有 `dbnet` 检测。`io/ocr.h` 的 `text_region` 无识别文本。`dbnet` 仍在范围内，多边形字段在。
- RTDETR 不在范围表。MOT 的 `byte_tracker.h` 不是 HTTP catalog id。
- 一个 catalog id 一个进程、不为对齐企业平台改 HTTP 契约，不记成劣势。

## 依据
- H1：示例是否与 `docs/openapi.json` 一致，不由本模块提供。本模块给出的是各 id 的输出结构。
- H2：逐 id 的 CPU golden 与 `type=tensorrt` 比较在 `test/model_golden_test.cc`。该文件里 TensorRT 同类 IoU 只加载 `conf/model/object_detection/yolov8/yolov8s.toml`。`yolov8n`、`yolov8l`、`yolov8x` 以及其余未单列的 id，缺的是测试比较，不是本模块里一条只能跑 `yolov8s` 的解码。检测器对缺输出名与坏形状分别回空输出码和契约失败码，空框列表仍是 `OK`。
- H3：硬件、复现命令、指标、日期的基准行不在 `src/models`。
- H4：`trt_session.cpp` 只反序列化已有 engine。生成命令和 `mortredctl doctor --strict` 不在本模块。缺 engine 时 init 失败，不会把未加载当成已加载。
- 分数 9.3：范围内家族可以交付，出错能靠状态码定位。不到 10，是因为逐 id 数值证明、示例、基准和换机 engine 命令不在本模块，不能由这里单独宣称 H1–H4 已满足。
- 必读路径与每个家族目录至少一个公开头已读。

## 未读
无

## 附录
- 旧问题改写：上新架构仍要写 C++。现有范围内 id 已有类，一次黑盒看不出这条，且不推翻现有 HTTP。
- 旧问题改写：多 session / diffusion 更难配。CLIP、LightGlue、SAM、diffusion 不在范围表。
- 旧问题改写：SAM 非空契约故障报成空输出。SAM 不在范围表，已关闭范围不重开。
