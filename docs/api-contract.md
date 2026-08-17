# HTTP API Contract

All model servers follow a unified HTTP JSON contract.

## Common response envelope

```json
{
  "req_id": "client-provided-or-server-generated",
  "code": 0,
  "msg": "success",
  "data": {}
}
```

On error:

```json
{
  "req_id": "...",
  "code": 50,
  "msg": "decode json error",
  "data": null
}
```

## HTTP status mapping

| Business code | Meaning | HTTP status |
|---:|---|---:|
| 0 | OK | 200 |
| 50 | JSON decode error | 400 |
| 3 | Empty input image | 400 |
| 60 | Unsupported media type | 415 |
| 61 | Request entity too large | 413 |
| 62 | Method not allowed | 405 |
| 63 | Not found | 404 |
| 4 | Model run timeout | 504 |
| 401 | Unauthorized | 401 |
| 429 | Rate limited | 429 |
| others | Server error | 500 |

## Common headers

```http
Content-Type: application/json; charset=utf-8
X-Request-ID: <req_id>
Cache-Control: no-store
```

## Common endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/openapi.json` | GET | OpenAPI document (static file provided under docs/) |

## Model inference request

```json
{
  "req_id": "optional",
  "img_data": "base64 encoded image"
}
```

## Model inference response

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
