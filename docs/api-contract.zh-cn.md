# HTTP API 契约

| [English](api-contract.md) | [中文](api-contract.zh-cn.md) |
|---|---|

## 拓扑说明：mortred-gateway

生产流量统一经由 **mortred-gateway**（默认 `:8080`）。网关先按 catalog
**id** 匹配 `/v1/models/{id}/…`，再按遗留 `server_uri` 精确匹配，转发到
仅监听环回地址的模型服务器。网关负责外部 Bearer Token 鉴权
（`MORTRED_GATEWAY_AUTH_TOKEN` 或 `MORTRED_API_TOKEN`），将上游不可达映射为
`503`、传输失败映射为 `502`；下文所有模型服务器状态码均原样透传。网关的
`GET /healthz` 为公开端点。`GET /metrics` 在环回上默认公开；非环回网关必须设置
独立的 `MORTRED_METRICS_TOKEN`（不要复用推理 token）。
模型端口仅绑定环回地址、
不得对外暴露。Mortred 自身是明文 HTTP；对外服务必须由网关前置的反向
代理终结 TLS。fail-closed 拒绝非环回且无鉴权，以及非环回且无独立 scrape token。

监督器（supervisor，`:8787`）在 `/api/v1/` 下提供管理 REST API
（health/catalog/status/生命周期/日志/metrics）与内嵌 Web UI；
`mortredctl` 是它的命令行客户端。**推理冒烟**（控制台发送按钮和
`mortredctl infer`）把数据面信封 POST 到网关的
`/v1/models/{id}/infer`，Bearer 与管理 API 相同（`MORTRED_API_TOKEN`）。
监督进程只做管理（catalog / 启停 / 日志 / UI）。推理和异步 jobs
走网关。`:8080` 上的遗留 `{server_uri}` 仍然可用。

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
- 健康/元数据端点（`/healthz`、`/ready`、`/openapi.json`）公开。
  监督器 `GET /api/v1/metrics` 需要管理 token。网关 `GET /metrics` 在环回上默认公开；
  非环回必须设置独立的 `MORTRED_METRICS_TOKEN`。模型在配置了 `auth_token` /
  `MORTRED_AUTH_TOKEN` 时，`GET /metrics` 也要 Bearer（`/healthz` 仍公开）。
  不要把推理 token 当作 scrape 密钥。
- `auth_token` 为空时模型端点开放访问，但服务器拒绝在非环回地址上监听
  （fail-closed，配置缺失即拒绝启动）。该门闩不是 TLS、不是指标保密、
  也不是 token 强度检查。

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

服务端开启 `async_enabled` 后，长耗时推理可异步提交。**模型端口**上的路径不变
（`POST /jobs`、`GET /jobs/{id}` 等）。经 **网关** 时同一套处理带 catalog id
前缀；网关无状态，并把 `Location` / `poll_url` / `result_url` 改写成该前缀：

| 网关 | 方法 | 模型端口上游 |
|---|---|---|
| `/v1/models/{id}/infer` | POST | `{server_uri}` |
| `/v1/models/{id}/jobs` | POST | `/jobs` |
| `/v1/models/{id}/jobs/{job}` | GET | `/jobs/{job}` |
| `/v1/models/{id}/jobs/{job}/wait` | GET | `/jobs/{job}/wait` + query |
| `/v1/models/{id}/jobs/{job}/result` | GET | `/jobs/{job}/result` |
| `{server_uri}` | POST | `{server_uri}`（遗留） |

`GET /v1/models/{id}/infer` 与 `GET {server_uri}` 返回 `405`。未知 `{id}` 与未知
`server_uri` 使用同一 404 信封。模型未开异步时，上游 `404` 原样透传。

| 端点（模型端口） | 成功 | 错误 |
|---|---|---|
| `POST /jobs` | **准入时** `202`，返回 `job_id`、`state: pending`、`poll_url`、`result_url` | 准入队列满时 `429` |
| `GET /jobs/{id}` | `200`，返回 `state`（`pending`/`running`/`done`/`failed`/`timeout`） | 未知 id `404` |
| `GET /jobs/{id}/wait?timeout=N` | job **进入终态**或 wait 预算耗尽时 `200`（`timeout` 单位毫秒；默认 30000，上限 300000）。预算耗尽时 state 仍可能是 `pending`/`running` | 未知 id `404` |
| `GET /jobs/{id}/result` | `200` 标准响应封装（可重复读取） | 未知 id `404`，未完成 `409`（含 `pending`/`running`/`failed`/`timeout`） |

`202` 表示服务器接受了该 job，不表示推理已经完成。正确的客户端不要把
`POST /jobs` 当成阻塞的 `/infer`。逐步验收步骤见
[async-jobs-customer-test.zh-cn.md](async-jobs-customer-test.zh-cn.md)。

任务账本保存在内存中（重启即失）。组件设计、并发契约与验证门禁见
[async-job-table.zh-cn.md](async-job-table.zh-cn.md)。
