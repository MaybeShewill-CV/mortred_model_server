# 模块评分

- id: src-models
- 权重: 15
- 分数: 9.0
- 读过的路径: `src/models/CMakeLists.txt`、`src/models/base_model.h`、`src/models/backend/model_runtime.h`、`src/models/backend/model_runtime.cpp`、`src/models/backend/session.h`、`src/models/backend/session.cpp`、`src/models/backend/trt_session.h`、`src/models/backend/trt_session.cpp`、`src/models/backend/backend_cv_model.h`、`src/models/backend/gpu_jpeg_decoder.h`、`src/models/backend/gpu_jpeg_decoder.cpp`、`src/models/backend/gpu_preprocess.cu`、`src/models/catalog/model_entry.h`、`src/models/classification/mobilenetv2.h`、`src/models/classification/mobilenetv2.inl`、`src/models/clip/openai_clip.h`、`src/models/diffusion/ddpm_unet.h`、`src/models/enhancement/real_esrgan.h`、`src/models/enhancement/real_esrgan.inl`、`src/models/feature_embedding/dinov2.h`、`src/models/feature_embedding/dinov2.inl`、`src/models/feature_point/superpoint.h`、`src/models/feature_point/superpoint.inl`、`src/models/io/common_input.h`、`src/models/matting/pp_matting.h`、`src/models/mono_depth_estimation/metric3d.h`、`src/models/mono_depth_estimation/metric3d.inl`、`src/models/mot/byte_tracker/byte_tracker.h`、`src/models/object_detection/yolov8_detector.h`、`src/models/object_detection/yolov8_detector.inl`、`src/models/object_detection/libface_detector.h`、`src/models/object_detection/libface_detector.inl`、`src/models/object_detection/detection_params.h`、`src/models/ocr/db_text_detector.h`、`src/models/ocr/db_text_detector.inl`、`src/models/scene_segmentation/pp_humanseg.h`、`src/models/scene_segmentation/pp_humanseg.inl`、`src/models/segment_anything/sam_prediction/sam_predictor.h`、`src/common/status_code.h`、覆盖集九份产品 toml、`docs/benchmark-h3/` 里这九个 id 的 txt、`test/backend_unittest.cc`、`test/base_model_nvi_unittest.cc`、`test/packed_batch_nvi_unittest.cc`、`test/cv_image_input_unittest.cc`、`test/model_golden_test.cc`

直接子目录 16 个，有头文件的都读过一个公开头：`backend`、`catalog`、`classification`、`clip`、`diffusion`、`enhancement`、`feature_embedding`、`feature_point`、`io`、`matting`、`mono_depth_estimation`、`mot`、`object_detection`、`ocr`、`scene_segmentation`、`segment_anything`。

## 已具备

- 一个模型实例持有一份 `InferenceSession`。GPU 构建把 `TrtSession` 编进 `models`，`TrtSession::init` 为该实例 `createExecutionContext()`，并打出 `static; batched shapes will be rejected`（`src/models/backend/trt_session.cpp`、`src/models/CMakeLists.txt`）。`session.h` 写明一个 worker 一份 session，session 不要求线程安全。cpu 构建不定义 `MORTRED_HAS_TRT`。`type=tensorrt` 在 cpu 构建上返回空指针，错误含 `not compiled`；`device=cpu` 返回 `configuration error`。`BackendCvModel::init` 在 session 为空时返回 `MODEL_INIT_FAILED`，不把 `_m_successfully_initialized` 设为真。`test/backend_unittest.cc` 的 `BackendSession.UnknownBackendAndMissingFileFail` 锁住这两句错误。
- 单张图进 `ImagePipeline` 时 leading dim 为 1（`model_runtime.cpp` 的 `{1, ...}`）。`run_image_batch` 在引擎拒绝打包形状时退回逐张 `run()`。空图在 `prepare_inputs` 里变成 `MODEL_EMPTY_INPUT_IMAGE`。未初始化的 `run` 返回 `MODEL_INIT_FAILED`。异常被 `invoke_nothrow` 收成 `MODEL_RUN_SESSION_FAILED`；打包路径抛异常时每一项都是 `MODEL_RUN_SESSION_FAILED`。`test/cv_image_input_unittest.cc` 的 `raw_bytes_empty_and_undecodable_yield_clear_status`、`test/base_model_nvi_unittest.cc`、`test/packed_batch_nvi_unittest.cc`、`test/model_golden_test.cc` 的 `mobilenetv2_mixed_batch_isolates_failed_items` 写的是：空图单独失败，同批其余项仍成功。
- MNN 与 ONNX Runtime 能创建会话。缺文件时 `create` 返回空指针。`MnnSession.InitAndRunMobilenetv2` 在权重存在时 `run` 返回 `OK`，dtype 不对时返回 `MODEL_RUN_SESSION_FAILED`。`OrtSession.InitAndRunDdpmUnet` 同样：合法输入 `OK`，int32 timestep 为 `MODEL_RUN_SESSION_FAILED`。
- 覆盖集产品 toml 都是 `type=tensorrt`，权重文件名带 `static_bs1`。`yolov8`、`dbnet` 在 toml 的 `[height, width]` 与会话 H/W 不一致时初始化失败。`dinov2`、`pphuman_mobile`、`mobilenetv2` 用 `static_shape()`。`superpoint`、`metric3d` 要求空间维为正，尺寸来自会话。`libface.toml` 的 `model_input_image_size = [480, 640]` 经 `parse_model_input_size` 得到高 480、宽 640；`docs/benchmark-h3/libface.txt` 是 `--shapes=input:1x3x480x640`。480 与 640 已是 32 的倍数，CPU 与 `DIRECT_RESIZE_PAD_TO_MULTIPLE` 送出的 NCHW 都是 `1×3×480×640`。
- `realesrgan` 的引擎剖面是 `1×256×256×3`。解码后宽或高不是 256 时，`prepare_inputs` 在进会话前返回 `MODEL_INPUT_SIZE_REJECTED`，`status_code.h` 的文案是「尺寸须为 256×256」。`Resize::NONE` 使静态引擎走不上 GPU 零拷贝，请求留在这条 CPU 预处理。
- 每个 `BackendCvModel` 持有自己的 `GpuDecodeSlot`。`gpu_jpeg_decoder.cpp` 写出 `mortred_jpeg_decode_total{backend=...}`。`resolve_decode_routing` 用 `MORTRED_DECODE_AUTO`，缺省为 cpu。

## 问题

无

## 范围外

- P0-1 重复描述，不重新入列。当前 `libface.toml` 与 `docs/benchmark-h3/libface.txt` 的高宽已经同为 480×640，预处理按这个尺寸对齐到 32 后仍是 `1×3×480×640`。
- P2-7 重复描述，不重新入列。非 256 的图在进会话前返回 `MODEL_INPUT_SIZE_REJECTED`，文案写明尺寸须为 256×256。
- P2-8（交换目录与产品 toml 不一致）仍在范围外。本模块没有新证据。
- 其余已关闭 id 的陈述不落在本模块，不重复。
- 动态 shape、动态 batch、batch 大于 1 的 engine。
- 每个 worker 各自一个 TensorRT execution context。`TrtSession` 已按实例持有 context，不写成待改结构。
- MNN 与 ONNX Runtime 的吞吐和延迟。
- 覆盖集以外的尺寸，以及 SAM、diffusion、CLIP、LightGlue、RTDETR、MOT、`msocrnet`。这些目录有实现，不进入本模块问题。

## 依据

分数 9.0：这个模块可以放上这一台 Linux GPU 服务器。三条后端、静态 batch=1、空图和异常不报成成功、cpu 构建拒绝 TensorRT、`libface` 与 `realesrgan` 的当前形状契约，都有上面的文件和测试出处。

落在本模块、且当前文件满足的行：

- AV1 的模型侧：`type=mnn` / `type=onnx` 的会话能创建；初始化失败返回空指针；`run` 失败返回 `MODEL_RUN_SESSION_FAILED`，不返回 `OK`。
- ST1 的空图：`load_image` 与 `prepare_inputs` 给出 `MODEL_EMPTY_INPUT_IMAGE`。未知字段和错误 token 不在 `src/models`。
- PF3 的模型侧：单张 leading dim 为 1；静态引擎拒绝打包后退回逐张；一张失败不改同批成功项的状态；会话级抛异常时每一项都是 `MODEL_RUN_SESSION_FAILED`。`mortred_batch_size` 直方图不在本模块。
- EX3 的模型侧：cpu 构建上 `type=tensorrt` 初始化失败，错误文本写明未编进该构建，或 `device=cpu` 是配置错误。进程是否进入 ready 在 server。

PF1、TP1 不在本模块判成未满足。这两行要的是 calibrate 之后的 HTTP 报告，报告不在 `src/models`。官方压测并发等于已部署 `worker_nums` 的两倍，不据此把这两行写成未满足。模型侧是每个 `TrtSession` 一个 execution context，静态输入会拒绝打包形状并退回逐张，没有把覆盖集改成动态 batch。PF2、PF4 的排队等待、gateway 阶段耗时和 `server_time_ms` 不在 `src/models`。PF5、TP2、TP3、UX、OP、EX1、EX2 的可观察物在文档、脚本、gateway 或 factory。

未验证判断：没有打开 engine 二进制，去核对 `dbnet` 的 `[544, 960]`、`yolov8s` 的 `[640, 640]`，以及 `mobilenetv2`、`dinov2_vits14`、`pphuman_mobile`、`superpoint`、`metric3d_512` 的会话空间尺寸。`dbnet` 与 `yolov8` 在 toml 和会话不一致时初始化失败。其余这些 id 的 H/W 来自会话。`metric3d_512.toml` 的 `canonical_size = [512, 960]` 在 `metric3d.inl` 里没有读取。这次没有在 WSL 里重跑九个 id 的推理。

## 未覆盖

无
