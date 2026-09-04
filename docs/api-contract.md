# HTTP API Contract

| [English](api-contract.md) | [涓枃](api-contract.zh-cn.md) |
|---|---|

## Topology note: mortred-gateway

Production traffic goes through **mortred-gateway** (default `:8080`): each
model `server_uri` is routed to its loopback model server. The gateway enforces
the external Bearer token (`MORTRED_GATEWAY_AUTH_TOKEN`), maps a dead upstream
to `503` and transport failures to `502`; all model-server status codes below
pass through unchanged. `GET /healthz` and `GET /metrics` are public on the
gateway. Model ports are loopback-only and must not be exposed; TLS must be
terminated by a reverse proxy in front of the gateway.

The supervisor (`:8787`) exposes the management REST API under `/api/v1/`
(health/catalog/status/lifecycle/logs/metrics) and the embedded web UI;
`mortredctl` is its CLI client. **Inference smoke tests** (the Web UI send
button and `mortredctl infer`) POST the data-plane envelope to the gateway
using the model's `server_uri`, with the same Bearer token as the management
API (`MORTRED_API_TOKEN`). The supervisor `/api/v1/infer` proxy remains for
compatibility in this release but is no longer the UI/CLI path.

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
| 6 | Model output contract failed | 500 |
| 401 | Unauthorized | 401 |
| 429 | Rate limited | 429 |
| 429 | Queue full (`max_queue_depth` exceeded; carries `Retry-After`) | 429 |
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

Unknown paths, including the removed `/welcome` and `/hello_world` HTML
probes, answer `404` with process-level `UnifiedResponse`.

## Model inference request

```json
{
  "req_id": "optional",
  "img_data": "base64 encoded image"
}
```

## Overload behaviour

When `max_queue_depth > 0` and the waiting queue is full, the model server
rejects immediately with `429` and a `Retry-After` header (estimated drain
time from queue depth, run-time EWMA and worker count, clamped to 1-60s). The
gateway forwards both verbatim. New optional server keys: `max_queue_depth`
(0 = unlimited), `max_batch_size` (default 1; >1 enables dynamic batching with
a `max_batch_delay_ms` collection window), plus the
`mortred_queue_rejected_total` / `mortred_batch_size` /
`mortred_batch_window_wait_ms` metrics.

Per-item failure isolation: within a batch, a failing item (bad image,
decode error) returns its own error code while its batch mates keep their
results; only session-level failures (engine errors) fail every participating
item.

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

## Async jobs

Long-running inference can be submitted asynchronously when the server enables
`async_enabled`:

| Endpoint | Success | Errors |
|---|---|---|
| `POST /jobs` | `202` with `job_id`, `state`, `poll_url`, `result_url` | `429` when the admission queue is full |
| `GET /jobs/{id}` | `200` with `state` (`pending`/`running`/`done`/`failed`/`timeout`) | `404` unknown id |
| `GET /jobs/{id}/wait?timeout=N` | `200` after a state change or terminal state | `404` unknown id |
| `GET /jobs/{id}/result` | `200` standard envelope (repeatable) | `404` unknown id, `409` not finished |

The job ledger is in-memory (lost on restart). The component design, concurrency
contract and verification gates are documented in
[async-job-table.md](async-job-table.md).
