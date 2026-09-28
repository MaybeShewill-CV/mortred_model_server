# 模块评分

- id: src-models
- 权重: 15
- 分数: 7.8
- 读过的路径:
  - cowork/ledger/rubric.md
  - src/models/CMakeLists.txt
  - src/models/base_model.h
  - src/models/backend/model_runtime.h
  - src/models/backend/session.h
  - src/models/backend/backend_cv_model.h
  - src/models/backend/backend_config.h
  - src/models/backend/multi_session_model.h
  - src/models/catalog/model_entry.h
  - src/models/classification/resnet.h
  - src/models/clip/openai_clip.h
  - src/models/diffusion/ddpm_unet.h
  - src/models/enhancement/real_esrgan.h
  - src/models/feature_embedding/dinov2.h
  - src/models/feature_point/superpoint.h
  - src/models/io/object_detection.h
  - src/models/io/ocr.h
  - src/models/io/classification.h
  - src/models/matting/modnet_matting.h
  - src/models/mono_depth_estimation/depth_anything.h
  - src/models/mot/byte_tracker/byte_tracker.h
  - src/models/object_detection/yolov8_detector.h
  - src/models/object_detection/yolov8_detector.inl
  - src/models/ocr/db_text_detector.h
  - src/models/scene_segmentation/bisenetv2.h
  - src/models/segment_anything/sam_prediction/sam_predictor.h
  - src/models/model_io_define.h

## 优势
- 统一生命周期：`BackendCvModel` 把 TOML `[SECTION.backend]` → 建 session → `on_init(params)` → `preprocess`/`postprocess` 串成固定路径，服务层不用关心后端选型（`backend_cv_model.h`）。
- 三后端可交付：`InferenceSession::create` + `BackendConfig` 支持 mnn/onnx/tensorrt；`CMakeLists.txt` 在 cpu profile 整段摘掉 TRT，避免 CPU 机链进 TRT 头库。
- 出错可定位：`StatusCode` + `invoke_nothrow`（`base_model.h`）；`ParamReader`/`SessionIoValidator`/`OutputReader`（`model_runtime.h`）；YOLOv8 对 shape/params/geometry 有带前缀的 `LOG(ERROR)`（`yolov8_detector.inl`）。
- 业务族覆盖公司主力：检测/分割/抠图/深度/增强/分类/OCR 检测/SAM/CLIP/嵌入/特征点均有公开头与 `.inl`；单图模型免费拿 `run_batch` 隔离合同（`backend_cv_model.h`）。
- 任务 IO 契约清晰：`io/*.h` + `model_io_define.h` 聚合，bbox/分类/深度等输出类型可直接对客户协议。

## 劣势
- OCR 交付不完整：族内仅有 `db_text_detector.h`；`io/ocr.h` 只有 `text_region`（框/多边形/分），无识别文本字段，也无 recognizer 头文件——「OCR 整包」客户项目仍要外接或自写识别。
- 接新架构仍是写 C++：模型实现以头文件 + `.inl` 模板为主，`CMakeLists.txt` 只编译少量 backend/mot/sam/clip cpp；换客户自定义导出图通常要新类/改后处理，小团队无法「只改 TOML」上新模型。
- 交付成熟度不均：`mot/byte_tracker` 独立于 `BackendCvModel`；diffusion（如 `ddpm_unet.h`）拆成多组件，多 session（CLIP/SAM）虽有 `MultiSessionModel`，仍比单图 YOLO/增强更难一次配稳。

## 依据
- 分数 7.8：主力 CV 交付线（检测/分割/抠图/深度/增强/分类）在统一后端与日志下可独立交付并定位；OCR 缺识别、新模型要写代码、MOT/diffusion 非同质接法，达不到 9.0「独立交付全品类」门槛。
- 必读路径均已读完；未读：无。

## 未读
无
