# 模块评分

- id: src-factory-common
- 权重: 10
- 分数: 9.2
- 读过的路径: `src/factory/model_catalog.h`、`src/factory/cv_catalog.h`、`src/factory/base_factory.h`、`src/factory/CMakeLists.txt`、`src/factory/classification_task.h`、`src/factory/obj_detection_task.h`、`src/factory/scene_segmentation_task.h`、`src/factory/ocr_task.h`、`src/factory/matting_task.h`、`src/factory/enhancement_task.h`、`src/factory/feature_point_task.h`、`src/factory/feature_embedding_task.h`、`src/factory/mono_depth_estimate_task.h`、`src/factory/diffusion_task.h`、`src/factory/diffusion_model_adapter.h`、`src/factory/clip_task.h`、`src/factory/sam_task.h`；`src/common/status_code.h`、`src/common/response_envelope.h`、`src/common/request_envelope.h`、`src/common/auth_token.h`、`src/common/base64.h`、`src/common/listen_policy.h`、`src/common/process_stop.h`、`src/common/stage_timing.h`、`src/common/request_size_limit.h`、`src/common/file_path_util.h`、`src/common/time_stamp.h`、`src/common/cv_utils.h`、`src/common/mortred_version.h.in`、`src/common/CMakeLists.txt`；对照 `src/apps/product_index.h`、`src/apps/product_index.cpp`、`src/apps/model_server_main.cpp`、`src/control/catalog.h`、`src/control/gateway/gateway_app.cpp`、`src/control/http_reply.h`、`src/server/base_server_impl.h`、`src/server/http_wire.h`、`src/server/http_status.h`、`src/server/item_exec.h`、`src/server/parsed_request.h`、`src/models/catalog/model_entry.h`、`src/models/cv_image_input.h`、`test/request_envelope_unittest.cc`、`test/envelope_codec_unittest.cc`、`scripts/check_consistency.py`、覆盖集九份 `conf/server` toml、`docs/how_to_add_new_server.md`、`docs/model-developer-guide.md`

## 已具备

覆盖集九个 id 都在对应家族的 `catalog()` 或 `face_catalog()` 里，带 `server_section`、worker 工厂、`fill_response` 和 family：`mobilenetv2`、`yolov8s`、`libface`、`pphuman_mobile`、`dbnet`、`dinov2_vits14`、`superpoint`、`realesrgan`、`metric3d_512`。九份 `conf/server` toml 的 `model=` 与这些 id 一致，`server_exe` 都是 `mortred-model-server.out`。`cv_catalog.h` 的 `create_server` 组装通用 `CvModelServer`，不按 id 再做一个服务器二进制。

`src/apps/product_index.cpp` 的 `project_http` 遍历已接入家族的 `catalog()` / `face_catalog()`，`id` 取 `model_section`。`fill_response` 和 `server_section` 都存在时 `http` 为真。`print_list` 打印这张表，`model_server_main.cpp` 的 `--list` 走这里。已有家族再加一条带响应填充的 `CvModelEntry`，`--list` 会出现该 `model_section`。`gateway_app.cpp` 的 `resolve_route` 用 `/v1/models/{id}/` 做 `catalog_.find(id)`，没有按 id 写的路由分支。控制面 `Catalog` 从 `conf/server` 的 `model=` 装表。

`docs/how_to_add_new_server.md` 的示例 id、family 和命令是 `densenet`，并写明不要新增按模型拆的 ELF。`classification_task.h` 里 `densenet` 那一行的 `model_section`、`server_section` 和 family 与此一致。`docs/model-developer-guide.md` 要求手动把 `Entry` 加进 `catalog()`，并要求探针 id 不出现在 `--list`，不要求每个模型单独做一个服务器二进制。

未知顶层字段在 `request_envelope.h` 记入 `/{field}`，允许的键是 `req_id, images, params, options`，状态为 `INVALID_REQUEST_PARAMETER`（66）。`img_data` 返回指针 `/img_data` 和 “use images” 说明，即使旁边有 `images` 也提前返回，状态同样是 66。`images` 缺失、空数组、空字符串都失败。`src/server/http_status.h` 把 66 映成 422。`base_server_impl.h` 在 `!parsed.is_valid` 时 `reply_reject`，不进入推理，也不填写 `results[].data`。raw 空 body 在 `parsed_request.h` 里状态为 `MODEL_EMPTY_INPUT_IMAGE`（3），`is_valid` 保持默认 false，HTTP 为 400。解码后的空图在模型侧返回状态 3；`item_exec.h` 只在逐项状态为 OK 时填充 `data`，有错误时顶层状态取第一条错误，不是 0。

错误 Bearer 在 `auth_token.h` 的 `is_bearer_authorized` 中拒绝；配置 token 为空时直接拒绝。模型进程 `reply_unauthorized` 使用 `UNAUTHORIZED`（401）。gateway 的 `reply_error(401)` 经 `status_for_proxy_http` 映到同一码。`/healthz`、`/ready`、`/openapi.json` 在鉴权之前返回。`encode` 写出顶层 `status`、`status_str`、逐项 `results[].status`、`server_time_ms`、`partial`，有错误时写出 `errors[].pointer`。

## 问题

无

## 范围外

`diffusion_task.h` 的 `DDPM`、`DDIM`、`CLS_COND_DDIM`、`LDM`，`clip_task.h` 的 `OPENAI_CLIP`，`sam_task.h` 的 `SAM`、`FAST_SAM`，`feature_point_task.h` 的 `LIGHTGLUE`，`obj_detection_task.h` 里 `yolov8n` / `yolov8l` / `yolov8x`，`mono_depth_estimate_task.h` 里 Depth Anything 的 vit 尺寸和 `metric3d_1088`，`scene_segmentation_task.h` 里的 `msocrnet`，以及 `attentive_gan`、`enlightengan`，不进入本模块问题。一个 catalog id 一个 OS 进程、MNN 与 ONNX Runtime 的吞吐、TensorRT 动态 batch，保持在范围外。P2-8 已在范围外。

重复描述：P1-2。英文 `docs/how_to_add_new_server.md` 的示例 id 已是 `densenet`，与 `classification_task.h` 一致，不再把「指南仍写 `DENSENET`」写成问题。重复描述：P2-7。`status_code.h` 里 `MODEL_INPUT_SIZE_REJECTED` 的整数是 8，文案是「尺寸须为 256×256」，这是已关闭修复的现状，不重开。

## 依据

EX2 由 `catalog()`、`project_http` 和 `resolve_route` 对上。在已有家族的 `catalog()` 增加一条带 `fill_response` 和 `server_section` 的行后，`--list` 的 id 来自 `model_section`，gateway 转发不需要改路由函数。EX1 点名的两份英文文档不要求按模型做一个服务器二进制；`densenet` 的 id、family 和 `--model` 与当前 `catalog()` 一致。示例里的 display name 是 “densenet classification”，源码是 “Densenet classification”，`--list` 的 id 列仍是 `densenet`。`docs/how_to_add_new_server.zh-cn.md` 仍写 `Entry{"DENSENET"...}` 和 `--model DENSENET`，且示例没有 family 字符串 `"densenet"`。EX1 点名的不是这份中文文件，出处在 `docs/`，不记入本模块问题。

ST1 在本模块能对上的部分：未知字段、空 `images`、空字符串、`img_data`、raw 空 body 都不会得到 `StatusCode::OK`，也不会把 `results[].data` 填成成功载荷。错误 token 得到 401。健康检查按 ST1 豁免。空图在进模型之后的状态 3 由 `src/models` 返回，HTTP 映射在 `src/server/http_status.h`。

`decode_response` 在 JSON 对象可解析时把 `DecodeResult.status` 设为 `StatusCode::OK`，响应体里的 `status` 只复制到 `value.status`。全仓库调用点只有该函数自己和 `test/envelope_codec_unittest.cc`，服务路径用的是 `encode`。一次 HTTP 验收看不到这个解码结果，所以不入列。本次没有执行上述 unittest，它们是否退出 0 是未验证判断，不写成已经失败。

PF1、PF5、TP1 的 calibrate 与 `--raw` 压测报告不在 `src/factory` 或 `src/common`。`stage_timing.h` 只在 `/tmp/mortred_stage_timing` 存在时打日志；`/metrics` 里排队与推理分开计数的实现在 `src/server`。`response_envelope.h` 只原样写出调用方传入的 `server_time_ms`。`item_exec.h` 把找 worker 的时间和 worker 运行时间相加后写入该字段，这一加法不在本模块。ST2、ST3、ST4、ST5、ST6、OP1 至 OP5、TP2、TP3、UX1 至 UX4、AV1、EX3 的可观察行为在 gateway、supervisor、`conf/`、文档或模型后端，本模块没有对应的未满足实现。

`constant_time_equals` 把长度异或收进一个字节。内容不同的错误 token 比较结果仍是 false，带错误 token 调用得到 401。长度差为 256 的倍数且多出的字节为 0 时，比较会与原 token 相同；这仍要带上原 token 的字节，不构成「错误 token 得到成功」。不入列。

分数 9.2：九个覆盖 id、统一服务器和请求拒绝路径都有文件出处，可以放进这一台机器的目录和契约层。不到 10.0，是因为响应解码把可解析对象的 `DecodeResult.status` 标成 OK。

## 未覆盖

无
