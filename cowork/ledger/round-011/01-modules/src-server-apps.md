# 模块评分

- id: src-server-apps
- 权重: 15
- 分数: 8.2
- 读过的路径: `src/apps/model_server_main.cpp`、`src/server/abstract_server.h`、`src/server/base_server_impl.h`、`src/server/async_endpoints.h`、`src/server/openapi_doc.h`、`src/server/http_status.h`、`src/server/http_wire.h`、`src/server/item_exec.h`、`src/server/generic_cv_server.h`、`src/server/rate_limiter.h`、`src/common/auth_token.h`、`docs/openapi.json`、`docs/api-contract.md`、`src/factory/classification_task.h`、`src/factory/obj_detection_task.h`、`src/models/backend/f32_output.h`、`src/models/object_detection/yolov7_detector.inl`、`src/models/object_detection/yolov8_detector.inl`

## 优势
- 单进程按 catalog id 启动：`--list` / `--model`、未知模型、bench-only、TOML 缺段、`model` 与 `--model` 不一致都在监听前 stderr 退出（`src/apps/model_server_main.cpp`）。
- 推理路由先鉴权再计 IP QPS。鉴权失败直接 401，不调用 `FixedWindowRateLimiter::allow`。`auth_token` 为空时 `is_bearer_authorized` 返回 false，推理与 `/jobs` 都不能匿名调用。`/healthz`、`/ready` 豁免（`src/server/base_server_impl.h` `serve_process`；`src/common/auth_token.h`）。
- 模型返回的状态原样进入统一信封：`MODEL_OUTPUT_CONTRACT_FAILED` 与 `MODEL_EMPTY_OUTPUT` 都是 HTTP 500，但 `status` 分别为 6 和 5；非 OK 的 item 不填 `data`（`src/server/item_exec.h` `inference_result_to_unified`；`src/server/http_status.h`）。YOLOv8 等走 `validated_f32_named_output` 的 id，非空 dtype/形状错误返回 6。

## 劣势
- H1：范围内每个 HTTP catalog id 都没有一条与 `docs/openapi.json` 一致的请求和响应示例。出处：`docs/openapi.json` 与进程内嵌的 `src/server/openapi_doc.h`（`x-contract-hash` 同为 `b320c08dde6d040146cc649381361fae41acb57ea3ba9e531c842e6b0aece400`）都有对应 path，两份全文都没有 `example`。`docs/api-contract.md` 只给了 `mobilenetv2`、`yolov8s`、`pphuman_server`、`dbnet`、`modnet`、`realesrgan`、`metric3d_512` 的人读片段，不是 OpenAPI 示例，也不覆盖 `yolov8n` / `yolov8l` / `yolov8x`、`libface`、`centerface` 以及其余 id。现在的行为：对任一范围内 id 执行 `GET /openapi.json`，该 path 的 `requestBody` 与 `200` 响应只有 schema，没有可照抄的请求体和响应体。一次黑盒可判定的变化：同一响应里，该 id 的 path 出现一条请求示例和一条 200 响应示例，且能按该 path 的 schema 校验通过。`yolov8n`、`yolov8l`、`yolov8x` 各自都要有，不能用 `yolov8s` 代替。
- S1：`yolov7` 与 `yolov7x` 在输出张量非空、但既没有 `output`/`518`/`532`，也凑不齐 3 个 `[1,3,H,W,C]` 头时，仍报成空输出。出处：`src/models/object_detection/yolov7_detector.inl` 的 `postprocess` 在该分支返回 `MODEL_EMPTY_OUTPUT`；HTTP 由 `src/server/item_exec.h` 原样写成 `status` 5。现在的行为：对这两个 id 的推理 POST，响应是 HTTP 500、`status` 5、`status_str` 为 `model output empty`，`results[].data` 为空。一次黑盒可判定的变化：同一请求变为 HTTP 500、`status` 6、`status_str` 为 `model output contract failed`。已关闭的 SAM 原句不在此条。

## 不计毛病
- 一个 catalog id 一个 OS 进程；超时 drain 之后析构成员，再由外层 SIGKILL（`src/server/base_server_impl.h` 析构；`src/server/async_endpoints.h` `drain`）。
- SAM、diffusion、CLIP、LightGlue 不在本范围。缺名仍按空输出返回的既有关闭范围不重开。

## 依据
- H1 未满足：`docs/openapi.json` 与 `src/server/openapi_doc.h` 无示例字段；服务入口是 `serve_process` 对 `/openapi.json` 的原样下发。范围内 id 的 path 已逐个出现在该文档（含 `yolov8n`/`yolov8s`/`yolov8l`/`yolov8x`、`libface`、`centerface`）。
- S1 对 YOLOv8、分类、以及走 `validated_f32_*` 且在校验失败时返回 `MODEL_OUTPUT_CONTRACT_FAILED` 的路径已满足，证据是 `src/models/backend/f32_output.h` 与 `src/models/object_detection/yolov8_detector.inl`。未满足的是 `yolov7`/`yolov7x` 这一新入口。
- S4 已满足：`serve_process` 在 `allow()` 之前返回 401；健康检查不进入推理路由。未新开 id。
- 分数 8.2：进程可以按 catalog 交付并靠状态码区分多数失败，但范围内 id 都没有可照抄的 OpenAPI 示例，且 `yolov7`/`yolov7x` 的非空头不匹配会被看成空输出。

## 未读
无
