# v1 HTTP 契约基线（改造前快照，M0.2）

> 用途：统一契约改造（Contract v1）迁移对照的"改动前"基准。
> 来源：`src/common/json_request_parser.h`、`src/common/http_response.h`、
> `src/server/base_server_impl.h`、`src/server/response_serializers.h`、
> `src/server/http_status.h`、`test/server_e2e_contract_test.cc`（快照日期：2026-08-31）。

## 1. 同步推理请求

```
POST {server_uri}            # 例 /mortred_ai_server_v1/obj_detection/yolov8
Authorization: Bearer <token>   # 模型端点必填；健康端点白名单除外
Content-Type: application/json  # 缺失或非 JSON → 415

{ "img_data": "<base64>",      # 必填，非空字符串；空/缺失 → 400(MODEL_EMPTY_INPUT_IMAGE)
  "req_id":  "client-id" }     # 可选，字符串；缺省服务端生成 16 位 hex
```

解析语义（`json_request_parser.h`）：未知字段**静默忽略**；畸形 JSON → 400(JSON_DECODE_ERROR)；
`Content-Length` 超限 → 413；队列满 → 429 + Retry-After。

## 2. 响应信封（`http_response.h`，全任务唯一）

```json
{ "req_id": "client-id",
  "code": 0,
  "msg": "OK",
  "data": { …任务相关… } }
```

## 3. data 载荷形态（`response_serializers.h`，逐任务）

| 任务 | data 形态 | 字段 |
|---|---|---|
| classification | object | `class_id:int, category:string, scores:float[]` |
| object_detection | array | `{class_id, score, category, bbox:[x1,y1,x2,y2], detail_infos:{}}` |
| face_detection | array | 同上 + `landmarks:[[x,y]…]` |
| text_regions (OCR) | array | `{score, bbox, polygon:[[x,y]…], detail_infos:{}}` |
| scene_segmentation | object | `image`(PNG b64), `colorized_mask`(PNG b64) |
| matting | object | `image`(PNG b64) |
| enhancement | object | `image`(JPG b64) |
| depth_estimation | object | `image`(PNG b64，着色深度图) |
| feature_points | array | `{score, location:[x,y], descriptor:float[]}` |
| sam_amg | array | `{segmentation(PNG b64), area, bbox, predicted_iou, stability_score}` |
| diffusion（base64_image）| object | `image`(PNG b64) |

## 4. StatusCode → HTTP 映射（`http_status.h`）

| StatusCode | wire | HTTP |
|---|---|---|
| OK | 0 | 200 |
| JSON_DECODE_ERROR / MODEL_EMPTY_INPUT_IMAGE | 50 / 3 | 400 |
| UNSUPPORTED_MEDIA_TYPE | 60 | 415 |
| REQUEST_ENTITY_TOO_LARGE | 61 | 413 |
| METHOD_NOT_ALLOWED | 62 | 405 |
| NOT_FOUND | 63 | 404 |
| UNAUTHORIZED | 401 | 401 |
| RATE_LIMITED | 429 | 429 |
| NOT_READY | 65 | 503 |
| MODEL_RUN_TIMEOUT | 4 | 504 |
| 其余模型/服务错误 | 1/2/5/6/7/11/64/80/9x | 500 |

## 5. 异步作业端点（`async_enabled` 服务）

```
POST /jobs                     → 202 {"job_id","state":"pending","poll_url","result_url"} + Location
GET  /jobs/{id}                → 200 {"job_id","state","elapsed_ms"[,"error"]}
GET  /jobs/{id}/wait?timeout=N → 同上（长轮询，默认 30s，上限 300s）
GET  /jobs/{id}/result         → 200 标准 v1 信封（同 §2）｜409 {"error":"job not finished…"}
```

## 6. 公开端点（免鉴权）

`/healthz` `/ready` `/metrics` `/openapi.json`

## 7. 已知将在统一契约中移除/变更的 v1 行为（迁移对照）

| v1 行为 | 统一契约后 |
|---|---|
| `img_data` 单图字符串 | **移除**：`images: ["<b64>", …]`（恒数组）；旧字段 → 422 + migration 提示 |
| 未知请求字段静默忽略 | 422 + JSON Pointer |
| 无请求级参数 | `params`（按模型 ParamSpec 白名单严格校验） |
| 无输出选项 | `options`（encoding/include_image/max_results/echo_params） |
| 响应 data 平铺/数组 | `results[]` 与 `images[]` 下标对齐，每项独立 status |
| 单请求 = 单图 | 多图；第 k 张损坏仅失败第 k 项 |
