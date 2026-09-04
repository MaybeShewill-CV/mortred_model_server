# HTTP API 契约

| [English](api-contract.md) | [中文](api-contract.zh-cn.md) |
|---|---|

## 拓扑说明：mortred-gateway

生产流量统一经由 **mortred-gateway**（默认 `:8080`）：网关按各模型的
`server_uri` 将请求路由到仅监听环回地址的模型服务器。网关负责外部
Bearer Token 鉴权（`MORTRED_GATEWAY_AUTH_TOKEN`），将上游不可达映射为
`503`、传输失败映射为 `502`；下文所有模型服务器状态码均原样透传。网关的
`GET /healthz` 与 `GET /metrics` 为公开端点。模型端口仅绑定环回地址、
不得对外暴露；对外服务必须由网关前置的反向代理终结 TLS。

监督器（supervisor，`:8787`）在 `/api/v1/` 下提供管理 REST API
（health/catalog/status/生命周期/日志/metrics + 推理测试代理）与内嵌
Web UI；`mortredctl` 是它的命令行客户端。

所有模型服务器遵循统一的 HTTP JSON 契约。权威的机器可读描述是
`docs/openapi.json`（每个模型服务器在 `GET /openapi.json` 提供）；
本文档是它的可读摘要。任何状态码、端点或响应结构的变更必须同时更新
两份文件（用 `python scripts/gen_openapi.py` 重新生成）。

## 鉴权

当服务器配置了 `auth_token` 时，模型推理端点要求携带 `Authorization`
请求头：

```http
Authorization: Bearer <token>
```

- 缺失或错误的 token：`401` + `WWW-Authenticate: Bearer realm="Mortred"`。
- 健康/元数据端点（`/healthz`、`/ready`、`/metrics`、`/openapi.json`）公开。
- `auth_token` 为空时模型端点开放访问，但服务器拒绝在非环回地址上监听
  （fail-closed，配置缺失即拒绝启动）。

## 请求规则

- 模型端点仅接受 `POST`；其他方法返回 `405` 并携带 `Allow: POST`。
- `Content-Type` 必须为 `application/json`（允许 `; charset=` 参数）；
  缺失或为其他媒体类型返回 `415`。
- 请求体上限为 `request_size_limit` MB；显式 `Content-Length` 超限返回
  `413`。

## 通用响应封装

```json
{
  "req_id": "客户端提供或服务器生成",
  "code": 0,
  "msg": "success",
  "data": {}
}
```

错误时：

```json
{
  "req_id": "...",
  "code": 50,
  "msg": "decode json error",
  "data": null
}
```

## HTTP 状态码映射

| 业务码 | 含义 | HTTP 状态码 |
|---:|---|---:|
| 0 | 成功 | 200 |
| 50 | JSON 解析错误 | 400 |
| 3 | 输入图片为空 | 400 |
| 60 | 不支持的媒体类型 | 415 |
| 61 | 请求实体过大 | 413 |
| 62 | 方法不允许 | 405 |
| 63 | 未找到 | 404 |
| 65 | 服务未就绪 | 503 |
| 4 | 模型运行超时 | 504 |
| 6 | ?????????? | 500 |
| 401 | 未授权 | 401 |
| 429 | 触发限流 | 429 |
| 429 | 等待队列已满（超出 `max_queue_depth`；携带 `Retry-After`） | 429 |
| 其他 | 服务器错误 | 500 |

## 通用响应头

```http
Content-Type: application/json; charset=utf-8
X-Request-ID: <req_id>
Cache-Control: no-store
```

## 通用端点

| 端点 | 方法 | 说明 |
|---|---|---|
| `/healthz` | GET | 存活探针 |
| `/ready` | GET | 就绪探针 |
| `/metrics` | GET | Prometheus 指标 |
| `/openapi.json` | GET | OpenAPI 文档（内嵌副本） |

未知路径（含已删除的 `/welcome`、`/hello_world` HTML 探活）返回 `404` 与
进程级 `UnifiedResponse`。

## 模型推理请求

```json
{
  "req_id": "可选",
  "img_data": "base64 编码的图片"
}
```

## 过载行为

当 `max_queue_depth > 0` 且等待队列已满时，模型服务器立即以 `429`
拒绝，并携带 `Retry-After` 响应头（依据队列深度、运行时长 EWMA 与
worker 数量估算排水时间，钳制在 1-60 秒）。网关将两者原样转发。

新增可选服务端配置键：

- `max_queue_depth`：等待队列上限，超过后快速失败（0 = 不限制）。
  调优公式：`深度 ≈ worker_nums × 目标排队秒数 / 单次推理时长（秒）`。
- `max_batch_size`：动态批处理，默认 1（关闭）；大于 1 时在
  `max_batch_delay_ms` 收集窗口内凑批，合成一次 `[N,H,W,3]` 推理。
  仅适用于支持动态 batch 维的模型（当前：mobilenetv2 / resnet50）。

新增指标：`mortred_queue_rejected_total`（过载拒绝计数）、
`mortred_batch_size`（实际批大小直方图）、
`mortred_batch_window_wait_ms`（批收集窗口等待直方图）。

批内逐条失败隔离：一条失败（坏图、解码错误）只返回它自己的错误码，
同批其他条目照常得到结果；仅会话级失败（引擎错误）才会使全部参与
条目失败。

## 模型推理响应

```json
{
  "req_id": "...",
  "code": 0,
  "msg": "success",
  "data": {
    "class_id": 123,
    "category": "tabby cat",
    "scores": [0.1, 0.8, 0.1]
  }
}
```

所有非成功响应的 `data` 均为 `null`
（400/401/404/405/413/415/429/500/503/504）。各任务的 `data` 结构
（分类、检测、人脸、OCR、分割、抠图、增强、深度、特征点）定义于
`docs/openapi.json` 的 `components.schemas`，由
`src/server/response_serializers.h` 实现。
## 异步任务

服务端开启 `async_enabled` 后，长耗时推理可异步提交：

| 端点 | 成功 | 错误 |
|---|---|---|
| `POST /jobs` | `202`，返回 `job_id`、`state`、`poll_url`、`result_url` | 准入队列满时 `429` |
| `GET /jobs/{id}` | `200`，返回 `state`（`pending`/`running`/`done`/`failed`/`timeout`） | 未知 id `404` |
| `GET /jobs/{id}/wait?timeout=N` | 状态变化或终态后 `200` | 未知 id `404` |
| `GET /jobs/{id}/result` | `200` 标准响应封装（可重复读取） | 未知 id `404`，未完成 `409` |

任务账本保存在内存中（重启即失）。组件设计、并发契约与验证门禁见
[async-job-table.zh-cn.md](async-job-table.zh-cn.md)。
