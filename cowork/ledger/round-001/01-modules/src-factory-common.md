# 模块评分

- id: src-factory-common
- 权重: 10
- 分数: 9.0
- 读过的路径: `src/factory/model_catalog.h`、`src/factory/cv_catalog.h`、`src/factory/base_factory.h`、`src/factory/CMakeLists.txt`、`src/factory/classification_task.h`、`src/factory/obj_detection_task.h`、`src/factory/scene_segmentation_task.h`、`src/factory/ocr_task.h`、`src/factory/matting_task.h`、`src/factory/enhancement_task.h`、`src/factory/feature_point_task.h`、`src/factory/feature_embedding_task.h`、`src/factory/mono_depth_estimate_task.h`、`src/factory/diffusion_task.h`、`src/factory/diffusion_model_adapter.h`、`src/factory/clip_task.h`、`src/factory/sam_task.h`；`src/common/status_code.h`、`src/common/response_envelope.h`、`src/common/request_envelope.h`、`src/common/auth_token.h`、`src/common/listen_policy.h`、`src/common/process_stop.h`、`src/common/stage_timing.h`、`src/common/request_size_limit.h`、`src/common/base64.h`、`src/common/file_path_util.h`、`src/common/time_stamp.h`、`src/common/time_stamp.cpp`、`src/common/mortred_version.h.in`、`src/common/CMakeLists.txt`、`src/common/cv_utils.h`（调色板开头）；对照用到 `src/apps/product_index.cpp`、`src/control/gateway/gateway_app.cpp`、`src/server/http_status.h`、`src/server/parsed_request.h`、`src/server/item_exec.h`、`test/envelope_codec_unittest.cc`、`test/request_envelope_unittest.cc`、`test/status_code_unittest.cc`、`scripts/check_consistency.py`、覆盖集九份 `conf/server` toml、`docs/how_to_add_new_server.md`

## 已具备

覆盖集九个 id 都在对应家族的 `catalog()` / `face_catalog()` 里，带 `server_section`、worker 工厂和 `fill_response`：`mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。九份 `conf/server` toml 的 `model=` 与这些 id 一致，`server_exe` 为 `mortred-model-server.out`。`cv_catalog.h` 的 `create_server` 组装通用 `CvModelServer`，不按 id 再做一个服务器二进制。

`src/apps/product_index.cpp` 的 `project_http` 遍历已接入家族的 `catalog()`，`id` 取 `model_section`，`http` 在 `fill_response` 和 `server_section` 都存在时为真。`gateway_app.cpp` 的 `resolve_route` 用 `/v1/models/{id}/` 做 `catalog_.find(id)`，没有按 id 写的路由分支。已有家族再加一条 `CvModelEntry`，`--list` 的数据来自这份表，gateway 转发不需要改路由代码。`scripts/check_consistency.py` 的 `parse_cpp_http_models` 用 `*_task.h` 里的 HTTP 行核对 `conf/server` 的 `model=`。

请求契约在 `request_envelope.h`：未知顶层字段记入 `/{field}` 并列出允许的 `req_id, images, params, options`；`images` 缺省、空数组、空字符串都失败；`img_data` 返回指针 `/img_data` 和 “use images” 迁移说明，状态为 `INVALID_REQUEST_PARAMETER`。`status_code.h` 把该码定为 66，`src/server/http_status.h` 将其映射为 422。`test/envelope_codec_unittest.cc` 与 `test/request_envelope_unittest.cc` 覆盖未知字段、`img_data`、空图和 raw 空 body。`status_code_unittest.cc` 锁定 wire 整数和文案。

响应信封 `encode` 写出逐项 `results[].status`、`server_time_ms`、`partial`，有错误时写出 `errors[].pointer`。空 token 在 `auth_token.h` 的 `is_bearer_authorized` 中直接拒绝。默认监听由 `listen_policy.h` 限制在 loopback，除非 `MORTRED_EXPOSE` 为 `docker` 或 `unsafe`。

## 问题

无

## 范围外

`diffusion_task.h`、`clip_task.h`、`sam_task.h`、`feature_point_task.h` 的 `match_catalog` 含 DDPM、DDIM、CLS_COND_DDIM、LDM、OPENAI_CLIP、SAM、FAST_SAM、LIGHTGLUE。`obj_detection_task.h` 里 `yolov8n` / `yolov8l` / `yolov8x`，`mono_depth_estimate_task.h` 里 Depth Anything 的 vit 尺寸和 `metric3d_1088`，`scene_segmentation_task.h` 里的 `msocrnet`，以及同一家族的其他尺寸，不进入本模块问题。一个 catalog id 一个 OS 进程、MNN / ONNX Runtime 的吞吐、TensorRT 动态 batch，保持在范围外。

## 依据

EX2 由上面的 `catalog()`、`project_http` 和 `resolve_route` 对上，当前代码在已有家族加一行后不需要改 gateway 路由函数。EX1 的工厂侧是通用 `CvModelServer`。`docs/how_to_add_new_server.md` 示例 id 写成 `DENSENET`、命令写成 `--model DENSENET`，而 `classification_task.h` 的 id 是 `densenet`，`cv_catalog.h` 写明 catalog id 为小写，gateway 注释写明不做大小写折叠。该行的未满足文本在这两份文档里，不记入本模块问题。

ST1、UX4 在本模块能对上的部分：未知字段、空 `images`、`img_data` 都不会得到 `StatusCode::OK`。错误 Bearer 由 `is_bearer_authorized` 返回 false。空图在模型侧返回 `MODEL_EMPTY_INPUT_IMAGE` 的路径在 `src/models`，HTTP 状态映射在 `src/server/http_status.h`。

PF1、PF5、TP1 要求的 calibrate 加 `--raw` 压测报告，本次在 `docs/benchmark-h3` 没有对上 `worker_nums`、`decode_auto`、`http_infer_rps`。这些行的报告和命令不在 `src/factory` 或 `src/common`，不记入本模块问题，也不标成阻塞。`stage_timing.h` 只在 `/tmp/mortred_stage_timing` 存在时打日志；`/metrics` 里排队与推理分开计数的实现在 `src/server`。`response_envelope.h` 只原样写出调用方传入的 `server_time_ms`。`item_exec.h` 把找 worker 的时间和 worker 运行时间相加后写入该字段，这一加法不在本模块。

`decode_response` 在解析成功后把 `DecodeResult.status` 设为 `StatusCode::OK`，响应体里的 `status` 只复制到 `value.status`。全仓库调用点只有该函数自己和 `test/envelope_codec_unittest.cc`，服务路径用的是 `encode`。一次 HTTP 验收看不到这个解码结果，所以不入列。本次没有执行上述 unittest，它们是否退出 0 是未验证判断，不写成已经失败。

ST2、ST3、ST4、ST5、ST6、OP1 至 OP5、TP2、TP3、UX1 至 UX3、AV1、EX3 的可观察行为在 gateway、supervisor、`conf/`、文档或模型后端，本模块没有对应的未满足实现。

分数 9.0：九个覆盖 id、统一服务器和请求拒绝路径都有文件和测试出处，可以放进这一台机器的目录和契约层。不到 10.0，是因为响应解码把可解析对象标成 OK，以及本模块的阶段计时不是 `/metrics` 上的排队/推理拆分。

## 未覆盖

无
