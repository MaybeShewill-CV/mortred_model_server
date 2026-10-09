# 模块评分

- id: src-factory-common
- 权重: 5
- 分数: 9.3
- 读过的路径: `src/factory/model_catalog.h`、`src/factory/cv_catalog.h`、`src/common/status_code.h`、`src/common/response_envelope.h`、`src/factory/classification_task.h`、`src/factory/obj_detection_task.h`、`src/factory/scene_segmentation_task.h`、`src/factory/ocr_task.h`、`src/factory/enhancement_task.h`、`src/factory/feature_point_task.h`、`src/factory/feature_embedding_task.h`、`src/factory/matting_task.h`、`src/factory/mono_depth_estimate_task.h`、`src/apps/product_index.h`、`src/apps/product_index.cpp`

## 优势
- 十类任务的 HTTP catalog id 都在对应 `catalog()` / `face_catalog()` 里，且 `product_index.cpp` 的 `build_index()` 用 `project_http` 逐条投影，不在 `MORTRED_WITH_CUSTOM_DRIVERS` 条件编译之后。39 个 id 的 `model_section` 互不重复，`server_section` 非空，`fill_response` 非空，投影后 `http` 为真。
- 人脸只在 `object_detection::face_catalog()`：`libface`、`centerface`，由 `project_http(..., "face_detection", face_catalog(), ...)` 单独投影，不并进目标检测表。
- `status_code.h` 用同一张 X-macro 绑定枚举、wire 整型与文案，并写明 wire code 属于稳定 HTTP 契约。
- `response_envelope.h` 集中 encode/decode，信封含 `partial` 与 `errors`，未知字段忽略；payload 归 `server/response_serializers.h`。

## 劣势
无

## 不计毛病
- SAM、diffusion、CLIP、LightGlue 不在最高标准范围表。`match_catalog()` 的 `LIGHTGLUE`、`clip::catalog()`、SAM predictor/fast-sam 只在自定义驱动宏下做 bench 投影；AMG 与 diffusion 虽进索引，但不按十类任务计缺口。
- HTTP OCR 只有 `ocr::catalog()` 的 `dbnet`。识别器不因「OCR 要达到最高标准」进入本模块劣势。
- 不因重排状态码或重写信封而推翻现有 HTTP wire 契约。

## 依据
- 分类 `classification::catalog()`：`mobilenetv2`、`resnet`、`densenet`。
- 目标检测 `object_detection::catalog()`：`yolov5l`、`yolov6s`、`nanodet_1x5`、`nanodet_416`、`yolov7`、`yolov7x`、`yolov8n`、`yolov8s`、`yolov8l`、`yolov8x`。`yolov8n`/`yolov8s`/`yolov8l`/`yolov8x` 各占一行，`model_section` 不同。
- 人脸 `object_detection::face_catalog()`：`libface`、`centerface`。
- 分割 `scene_segmentation::catalog()`：`bisenetv2`、`pphuman_mobile`、`pphuman_lite`、`pphuman_server`、`hrnet`、`msocrnet`。同文件 `bench_catalog()` 为空表。
- OCR `ocr::catalog()`：`dbnet`。
- 增强 `enhancement::catalog()`：`attentive_gan`、`enlightengan`、`realesrgan`。
- 特征点 `feature_point::catalog()`：`superpoint`。
- feature embedding `feature_embedding::catalog()`：`dinov2_vits14`、`dinov2_vitb14`、`dinov2_vitl14`。
- matting `matting::catalog()`：`modnet`、`ppmatting_512`、`ppmatting_1024`、`ppmatting_resnet34`、`ppmatting_v2`。
- 深度 `mono_depth_estimation::catalog()`：`depth_vits14`、`depth_vitb14`、`depth_vitl14`、`metric3d_512`、`metric3d_1088`。
- 投影：`src/apps/product_index.cpp` 第 126–143 行对上述十张表调用 `project_http`；`id` 取 `model_section`，`http` 为 `fill_response != nullptr && !server_section.empty()`（`project_http` 第 60–64 行）。重复 id 会被 `push_unique` 丢掉并打到 stderr；上述 39 个 id 无碰撞。
- 双目标范围表逐 id 对照后，本模块没有未投影或缺 id 的行。H1–H4 的示例、数值、基准和换机 engine 不在本模块行为里，本记录不另开句。

## 未读
无
