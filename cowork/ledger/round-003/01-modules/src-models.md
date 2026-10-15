# 模块评分

- id: src-models
- 权重: 15
- 分数: 9.0
- 读过的路径: `src/models/CMakeLists.txt`、`src/models/base_model.h`、`src/models/backend/model_runtime.h`、`src/models/backend/model_runtime.cpp`、`src/models/backend/session.h`、`src/models/backend/session.cpp`、`src/models/backend/trt_session.h`、`src/models/backend/trt_session.cpp`、`src/models/backend/backend_cv_model.h`、`src/models/backend/gpu_jpeg_decoder.h`、`src/models/backend/gpu_jpeg_decoder.cpp`、`src/models/catalog/model_entry.h`、`src/models/classification/mobilenetv2.h`、`src/models/enhancement/real_esrgan.h`、`src/models/feature_embedding/dinov2.h`、`src/models/feature_point/superpoint.h`、`src/models/mono_depth_estimation/metric3d.h`、`src/models/object_detection/yolov8_detector.h`、`src/models/object_detection/libface_detector.h`、`src/models/ocr/db_text_detector.h`、`src/models/scene_segmentation/pp_humanseg.h`、`src/models/io/common_input.h`、`src/common/status_code.h`、覆盖集九份产品 toml、`docs/benchmark-h3/` 九个 id 的 txt、`test/backend_unittest.cc`、`test/base_model_nvi_unittest.cc`、`test/packed_batch_nvi_unittest.cc`、`test/cv_image_input_unittest.cc`、`test/model_golden_test.cc`

直接子目录有头文件的都读过公开头：`backend`、`catalog`、`classification`、`clip`、`diffusion`、`enhancement`、`feature_embedding`、`feature_point`、`io`、`matting`、`mono_depth_estimation`、`mot`、`object_detection`、`ocr`、`scene_segmentation`、`segment_anything`。

## 已具备

- 一个模型实例持有一份 `InferenceSession`。GPU 构建把 `TrtSession` 编进 `models`，`TrtSession::init` 为该实例 `createExecutionContext()`，并打出 `static; batched shapes will be rejected`。`session.h` 写明一个 worker 一份 session。cpu 构建不定义 `MORTRED_HAS_TRT`。`type=tensorrt` 在 cpu 构建上返回空指针；`BackendCvModel::init` 在 session 为空时返回 `MODEL_INIT_FAILED`。
- 单张图进 `ImagePipeline` 时 leading dim 为 1。`run_image_batch` 在引擎拒绝打包形状时退回逐张。空图变成 `MODEL_EMPTY_INPUT_IMAGE`。异常收成 `MODEL_RUN_SESSION_FAILED`。
- MNN 与 ONNX Runtime 能创建会话；缺文件时 `create` 返回空指针。
- 覆盖集产品 toml 都是 `type=tensorrt`，权重文件名带 `static_bs1`。`libface.toml` 为 `[480, 640]`，与 `docs/benchmark-h3/libface.txt` 的 `--shapes=input:1x3x480x640` 一致。`realesrgan` 非 256×256 在进会话前返回 `MODEL_INPUT_SIZE_REJECTED`。
- 每个 `BackendCvModel` 持有自己的 `GpuDecodeSlot`；`resolve_decode_routing` 用 `MORTRED_DECODE_AUTO`，缺省 cpu。

## 问题

无

## 范围外

- 已关闭的 P0-1、P2-7 重复描述，不重新入列。当前 `libface` 与 `realesrgan` 形状契约仍成立。
- P2-8（交换目录与产品 toml）仍在范围外。
- 动态 shape / 动态 batch / batch>1 engine；改写每 worker 一份 TRT context 的架构；MNN/ORT 吞吐；覆盖集以外尺寸与 SAM/diffusion/CLIP 等。

## 依据

分数 9.0：三条后端、静态 batch=1、失败不报成成功、cpu 构建拒绝 TensorRT、覆盖集形状契约都有文件与测试出处。本模块无未关闭问题。AV1/ST1/PF3/EX3 的模型侧对得上当前文件。PF1/TP1 的 HTTP 报告不在本模块。

未验证判断：未打开 engine 二进制核对会话 H/W；未在 WSL 重跑九个 id 推理。不写成故障。

## 未覆盖

无
