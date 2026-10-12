# 模块评分

- id: src-models
- 权重: 15
- 分数: 6.0
- 读过的路径: `src/models/CMakeLists.txt`、`src/models/base_model.h`、`src/models/backend/model_runtime.h`、`src/models/backend/model_runtime.cpp`、`src/models/backend/session.cpp`、`src/models/backend/trt_session.h`、`src/models/backend/trt_session.cpp`、`src/models/backend/backend_cv_model.h`、`src/models/backend/gpu_jpeg_decoder.cpp`、`src/models/catalog/model_entry.h`、`src/models/io/common_input.h`、`src/models/classification/mobilenetv2.h`、`src/models/classification/mobilenetv2.inl`、`src/models/object_detection/yolov8_detector.h`、`src/models/object_detection/yolov8_detector.inl`、`src/models/object_detection/libface_detector.h`、`src/models/object_detection/libface_detector.inl`、`src/models/object_detection/detection_params.h`、`src/models/scene_segmentation/pp_humanseg.h`、`src/models/scene_segmentation/pp_humanseg.inl`、`src/models/ocr/db_text_detector.h`、`src/models/ocr/db_text_detector.inl`、`src/models/feature_embedding/dinov2.h`、`src/models/feature_embedding/dinov2.inl`、`src/models/feature_point/superpoint.h`、`src/models/feature_point/superpoint.inl`、`src/models/enhancement/real_esrgan.h`、`src/models/enhancement/real_esrgan.inl`、`src/models/mono_depth_estimation/metric3d.h`、`src/models/mono_depth_estimation/metric3d.inl`、`src/models/matting/pp_matting.h`、`src/models/clip/openai_clip.h`、`src/models/diffusion/ddim_sampler.h`、`src/models/mot/byte_tracker/byte_tracker.h`、`src/models/segment_anything/sam_prediction/sam_predictor.h`、`conf/model/object_detection/libfacedetection/libface.toml`、`conf/model/enhancement/real_esrgan/realesrgan.toml`、`conf/model/classification/mobilenetv2/mobilenetv2.toml`、`conf/model/object_detection/yolov8/yolov8s.toml`、`docs/benchmark-h3/libface.txt`、`docs/benchmark-h3/realesrgan.txt`、`test/backend_unittest.cc`、`test/model_golden_test.cc`、`test/packed_batch_nvi_unittest.cc`、`test/cv_image_input_unittest.cc`

## 已具备

- 一个模型实例持有一份 `InferenceSession`。GPU 构建把 `TrtSession` 编进 `models`，`TrtSession::init` 为该实例创建一个 `IExecutionContext` 和一条 CUDA stream（`src/models/backend/trt_session.cpp`、`src/models/CMakeLists.txt`）。cpu 构建不定义 `MORTRED_HAS_TRT`，`InferenceSession::create` 对 `type=tensorrt` 返回空指针，错误文本包含 `not compiled`；`device=cpu` 的 TensorRT 配置返回 `configuration error`。`test/backend_unittest.cc` 的 `BackendSession.UnknownBackendAndMissingFileFail` 锁住这两句。
- 单张图像进 `ImagePipeline::pack` 时 leading dim 固定为 1（`src/models/backend/model_runtime.cpp`）。`BackendCvModel::run_image_batch` 在引擎拒绝打包形状时退回逐张 `run()`（`src/models/backend/backend_cv_model.h`）。`TrtSession::init` 对静态输入打日志 `static; batched shapes will be rejected`。
- 未初始化时 `BaseAiModel::run` 返回 `MODEL_INIT_FAILED`。异常被 `invoke_nothrow` 收成 `MODEL_RUN_SESSION_FAILED`。空图在 `prepare_inputs` 里变成 `MODEL_EMPTY_INPUT_IMAGE`。`test/cv_image_input_unittest.cc`、`test/base_model_nvi_unittest.cc`、`test/packed_batch_nvi_unittest.cc`、`test/model_golden_test.cc` 的 `mobilenetv2_mixed_batch_isolates_failed_items` 写的是：空图单独失败，同批其余项仍为成功，打包路径抛异常时每一项都是 `MODEL_RUN_SESSION_FAILED`。
- 覆盖集里 `mobilenetv2`、`yolov8s`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`metric3d` 的空间尺寸来自会话输入，或在初始化时要求 toml 与会话 H/W 一致（`mobilenetv2.inl`、`yolov8_detector.inl`、`pp_humanseg.inl`、`db_text_detector.inl`、`dinov2.inl`、`superpoint.inl`、`metric3d.inl`）。产品 toml 均为 `type=tensorrt`，权重文件名带 `static_bs1`。
- `gpu_jpeg_decoder.cpp` 写出 `mortred_jpeg_decode_total{backend=...}`。每个 `BackendCvModel` 持有自己的 `GpuDecodeSlot`。

## 问题

P0

- 问题：驻留集中的 `libface` 对一张合法 JPEG 也到不了成功推理。
- 出处：`conf/model/object_detection/libfacedetection/libface.toml` 的 `model_input_image_size = [640, 480]`；`src/models/object_detection/detection_params.h` 把该数组定成 `[height, width]`，得到高 640、宽 480；`src/models/object_detection/libface_detector.inl` 的 `on_init` 只检查 `shape[1] == 3`，不核对 H/W，随后 `preprocess` 用这个 `cv::Size` 做 resize 再打成 NCHW；`docs/benchmark-h3/libface.txt` 对同一份 `face_detection_yunet_2026may.engine.static_bs1.fp16` 使用 `--shapes=input:1x3x480x640`；`src/models/backend/trt_session.cpp` 在 `setInputShape` 失败时返回 `MODEL_RUN_SESSION_FAILED`。GPU 路径的 `DIRECT_RESIZE_PAD_TO_MULTIPLE` 使用同一个 `pre_crop_size`，输出仍是高 640、宽 480，被拒绝后再走 CPU 预处理。
- 现在的行为：预处理送出的输入是 `1×3×640×480`。引擎剖面是 `1×3×480×640`。初始化可以成功。GPU 路径和 CPU 路径都会被静态形状拒绝，请求以 `MODEL_RUN_SESSION_FAILED` 结束。
- 改完后能观察到的变化：对已拉起的 `libface` 提交一张合法 JPEG，响应为成功状态。

P1

无

P2

- 问题：`realesrgan` 的产品引擎只接受 `1×256×256×3`，预处理按原图像素组 NHWC，解码尺寸不是 256×256 时请求以会话失败结束。
- 出处：`conf/model/enhancement/real_esrgan/realesrgan.toml` 指向 `realesr-general-x4v3.engine.static_bs1.fp32`；`docs/benchmark-h3/realesrgan.txt` 的命令含 `--shapes=input:1x256x256x3`；`src/models/enhancement/real_esrgan.inl` 的 `preprocess` 不做 resize，`nhwc` 使用工作图的行数和列数；`on_init` 把 `Resize::NONE` 与 `dynamic_size = true` 一起登记。静态引擎上 `network_blind_resize` 使 GPU 零拷贝路径不合格，请求落到这条 CPU 预处理。`setInputShape` 失败时返回 `MODEL_RUN_SESSION_FAILED`。
- 现在的行为：解码结果正好是 256×256 时，tensor 与引擎一致。宽或高不是 256 时，静态引擎拒绝该形状，调用方得到 `MODEL_RUN_SESSION_FAILED`。
- 改完后能观察到的变化：用一张解码后宽或高不是 256 的合法 JPEG 调用 `realesrgan`，结果是二者之一：成功返回增强图，且送入会话的形状为 `1×256×256×3`；或返回输入错误，并写明尺寸须为 256×256。

## 范围外

- 动态 shape、动态 batch、batch 大于 1 的 engine。上面的 P2 只要求与现有 static 256 对齐，或在进会话前返回标明 256×256 的输入错误。
- 每个 worker 各自一个 TensorRT execution context。`TrtSession` 已按实例持有 context，本记录不把它写成待改结构。
- MNN 与 ONNX Runtime 的吞吐和延迟。`session.cpp` 能创建这两类会话；初始化失败返回空指针。
- 覆盖集以外的尺寸，以及 SAM、diffusion、CLIP、LightGlue、RTDETR、MOT、`msocrnet`。这些目录有实现，不进入本模块问题。

## 依据

分数 6.0：三条后端、单张 batch 维为 1、空图与异常不报成成功、cpu 构建拒绝 TensorRT，这些都有上述文件和测试出处。驻留集要求同时能跑 `yolov8s` 和 `libface`。`libface` 的预处理形状与 `docs/benchmark-h3/libface.txt` 的引擎剖面不一致，合法 JPEG 完不成成功推理，所以这一模块不能按可上线的人脸路径计到 7.0 以上。

PF1、PF2、PF4、PF5、TP1 本轮未写入问题。仓库里没有一份同时含 calibrate、`worker_nums`、`decode_auto`、`http_infer_rps.py --raw`、并发数等于已部署 `worker_nums`、时长不少于 30 秒、成功 RPS、p50、p99、错误率，以及 `docs/benchmark-h3/<id>.txt` Throughput 比值的报告。`docs/benchmark-h3/*.txt` 只有 trtexec。`docs/benchmark-set-a.md` 表 4 里 yolov8s 的 `w*` 为 8、并发为 32，calibrate 小节写的是 15 秒。缺这份报告是仓库现状，本次调用没有跑压测，所以不标阻塞，也不把这些行写成已满足。排队等待直方图 `mortred_queue_wait_duration_ms` 与 `mortred_inference_duration_ms` 不在 `src/models`。

PF3 的模型侧：单张 pack 的 leading dim 为 1；引擎拒绝打包形状时退回逐张。本轮没有做 N>1 的 HTTP 验收，不把「batch 观测不是 1」写成已经发生的故障。

未验证判断：没有打开 engine 二进制核对 `dbnet` 的 `[544, 960]` 与文件内静态 H/W 是否一致。`db_text_detector.inl` 在二者不一致时初始化失败。`metric3d_512.toml` 的 `canonical_size = [512, 960]` 在 `src/models/mono_depth_estimation` 中没有读取，张量 H/W 来自会话输入。

## 未覆盖

无
