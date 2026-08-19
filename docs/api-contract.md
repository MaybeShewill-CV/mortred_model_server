# HTTP API Contract

All model servers follow a unified HTTP JSON contract. The authoritative machine-readable
description is `docs/openapi.json` (served by every model server at `GET /openapi.json`);
this document is the human-readable summary. Any change to status codes, endpoints or
response schemas must update both files (regenerate with `python scripts/gen_openapi.py`).

## Authentication

Model inference endpoints require an `Authorization` header when the server is configured
with `auth_token`:

```http
Authorization: Bearer <token>
```

- Missing or invalid token: `401` + `WWW-Authenticate: Bearer realm="Mortred"`.
- Health/metadata endpoints (`/healthz`, `/ready`, `/metrics`, `/openapi.json`) are public.
- When `auth_token` is empty, model endpoints are open, but the server refuses to listen
  on a non-loopback host (fail-closed).

## Request rules

- Model endpoints accept `POST` only; any other method returns `405` with `Allow: POST`.
- `Content-Type` must be `application/json` (a `; charset=` parameter is allowed); a
  missing or different media type returns `415`.
- The request body is limited to `request_size_limit` MB; an explicit `Content-Length`
  above the limit returns `413`.

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
| 65 | Service not ready | 503 |
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
| `/openapi.json` | GET | OpenAPI document (served from the embedded copy) |

`/welcome` and `/hello_world` are legacy HTML endpoints kept for the web console health
check; they are marked `deprecated` in the OpenAPI document.

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

`data` is `null` for every non-OK response (400/401/404/405/413/415/429/500/503/504).
Per-task `data` schemas (classification, detection, face, OCR, segmentation, matting,
enhancement, depth, feature point) are defined in `docs/openapi.json`
`components.schemas` and implemented by `src/server/response_serializers.h`.
