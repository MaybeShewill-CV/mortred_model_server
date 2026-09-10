# HTTP API Contract

| [English](api-contract.md) | [中文](api-contract.zh-cn.md) |
|---|---|

The machine-readable contract is `docs/openapi.json` (also served at
`GET /openapi.json` on every model process). This page is the human summary.
Do not copy the removed `{req_id, code, msg, data}` envelope or the `img_data`
field from older blog posts — those requests answer **422**.

## Topology note: mortred-gateway

Production traffic goes through **mortred-gateway** (default `:8080`). The
gateway looks up the catalog **id** first (`/v1/models/{id}/…`), then the
legacy `server_uri`, and forwards to the model's loopback port. The gateway
enforces the external Bearer token (`MORTRED_GATEWAY_AUTH_TOKEN` or
`MORTRED_API_TOKEN`), maps a dead upstream to `503` and transport failures to
`502`; all model-server status codes below pass through unchanged. `GET
/healthz` is public on the gateway. `GET /metrics` requires
`MORTRED_METRICS_TOKEN` on every listen, including loopback. A gateway
refuses to start without a distinct scrape Bearer (not the inference token).
Model ports are loopback-only and must not be exposed. Mortred itself is
plain HTTP; TLS is terminated by Nginx on the host network
(`mortredctl init-edge`). Fail-closed startup refuses a listener with no
auth, a missing metrics token, and a wildcard bind unless
`MORTRED_EXPOSE=docker` or `unsafe`.

The supervisor (`:8787`) exposes the management REST API under `/api/v1/`
(health/catalog/status/lifecycle/logs/metrics) and the embedded web UI;
`mortredctl` is its CLI client. **Inference smoke tests** (the Web UI send
button and `mortredctl infer`) POST the data-plane envelope to
`POST /v1/models/{id}/infer` on the gateway, with the same Bearer token as
the management API (`MORTRED_API_TOKEN`). The supervisor is management only
(catalog / lifecycle / logs / UI). Inference and async jobs go through
the gateway. Legacy `{server_uri}` on `:8080` is still accepted.

Any change to status codes, endpoints or response schemas must update
`docs/openapi.json` (regenerate with `python scripts/gen_openapi.py`).

## Authentication

Model inference endpoints require an `Authorization` header when the server is
configured with `auth_token`:

```http
Authorization: Bearer <token>
```

- Missing or invalid token: `401` + `WWW-Authenticate: Bearer realm="Mortred"`.
- Health/metadata endpoints (`/healthz`, `/ready`, `/openapi.json`) are public.
  Supervisor `GET /api/v1/metrics` requires the management token. Gateway
  `GET /metrics` requires `MORTRED_METRICS_TOKEN` on every listen, including
  loopback. Model `GET /metrics` requires the process auth token (supervisor
  children always have `MORTRED_AUTH_TOKEN`); empty token yields 401, not a
  public scrape. `/healthz` stays public. Do not reuse the inference token as
  the scrape secret.
- When `auth_token` is empty, model inference and `/metrics` return 401.
  Health/metadata stay public. The server still refuses a non-loopback listen
  without a token, and refuses a wildcard bind unless `MORTRED_EXPOSE=docker|unsafe`.

## Request rules

- Model endpoints accept `POST` only; any other method returns `405` with `Allow: POST`.
- `Content-Type` must be `application/json` (a `; charset=` parameter is allowed); a
  missing or different media type returns `415`.
- The request body is limited to `request_size_limit` MB; an explicit `Content-Length`
  above the limit returns `413`.
- Allowed JSON keys are only `req_id`, `images`, `params`, `options`. Unknown keys
  and the removed `img_data` field answer **422**.

## Common response envelope

```json
{
  "status": 0,
  "status_str": "OK",
  "task_id": "client-provided-or-server-generated",
  "model": { "name": "MOBILENETV2", "version": "" },
  "results": [
    {
      "status": 0,
      "data": {
        "class_id": 123,
        "category": "tabby cat",
        "scores": [0.1, 0.8, 0.1]
      }
    }
  ],
  "server_time_ms": 41.2,
  "partial": false
}
```

On a contract violation (HTTP 422):

```json
{
  "status": 66,
  "status_str": "invalid request parameter",
  "task_id": "",
  "results": [],
  "server_time_ms": 0.0,
  "partial": false,
  "errors": [
    {
      "pointer": "/img_data",
      "message": "field 'img_data' was removed; use images: [\"<base64>\"] (migration: img_data -> images[0])"
    }
  ]
}
```

Read HTTP status first, then top-level `status`, then each `results[i].status`.
`results[i].data` is `null` when that item failed. Ignore unknown response
fields. Per-task payloads live under `results[].data` and are defined in
`docs/openapi.json` `components.schemas` (`src/server/response_serializers.h`).

## HTTP status mapping

The JSON field is **`status`** (not `code`). Mapping is `src/server/http_status.h`.

| `status` | Meaning | HTTP |
|---:|---|---:|
| 0 | OK | 200 |
| 68 | Deadline exceeded, partial results | 200 |
| 50 | JSON decode error | 400 |
| 3 | Empty input image | 400 |
| 66 | Invalid request (`img_data`, unknown key, bad `params`) | 422 |
| 60 | Unsupported media type | 415 |
| 61 | Request entity too large | 413 |
| 67 | Too many items in one request | 413 |
| 62 | Method not allowed | 405 |
| 63 | Not found | 404 |
| 65 | Service not ready | 503 |
| 4 | Model run timeout | 504 |
| 6 | Model output contract failed | 500 |
| 401 | Unauthorized | 401 |
| 429 | Rate limited or queue full (`Retry-After`) | 429 |
| others | Server error | 500 |

Admitted model requests always return `results[]` of length `N = images[]`.
HTTP **504** (`status` 4) means zero items had been published when the
deadline fired; each slot is timeout with `data: null`. HTTP **200** +
`status` 68 + `partial: true` means at least one item completed in time —
clients must not retry that response as a 5xx. Rejection envelopes (401/422/…)
may still use empty `results[]`.

## Common headers

```http
Content-Type: application/json; charset=utf-8
X-Request-ID: <task_id>
Cache-Control: no-store
```

## Common endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/healthz` | GET | Liveness probe |
| `/ready` | GET | Readiness probe |
| `/metrics` | GET | Prometheus metrics (Bearer as above) |
| `/openapi.json` | GET | OpenAPI document (served from the embedded copy) |

Unknown paths, including the removed `/welcome` and `/hello_world` HTML
probes, answer `404` with process-level `UnifiedResponse`.

## Model inference request

```json
{
  "req_id": "optional",
  "images": ["base64 encoded image"],
  "params": {},
  "options": {}
}
```

`images` is required and always an array (≥1). `params` / `options` are optional
objects. Model-specific knobs (thresholds, DDPM `timesteps`, …) go in `params`, not
at the root. Generative models still require `images[]` (≥1); the pixels are
ignored, so a dummy base64 string is enough.

### Removed field: `img_data` (HTTP 422)

The following body is **not** a success request. It is rejected even if
`images` is also present:

```json
{
  "req_id": "legacy",
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
decode error) returns its own error status while its batch mates keep their
results; only session-level failures (engine errors) fail every participating
item.

## Async jobs

Long-running inference can be submitted asynchronously when the server enables
`async_enabled`. On the **model port** the paths are unchanged (`POST /jobs`,
`GET /jobs/{id}`, …). Through the **gateway** the same handlers are reached
with a catalog-id prefix; the gateway is stateless and rewrites `Location` /
`poll_url` / `result_url` onto that prefix:

| Gateway | Method | Upstream on the model port |
|---|---|---|
| `/v1/models/{id}/infer` | POST | `{server_uri}` |
| `/v1/models/{id}/jobs` | POST | `/jobs` |
| `/v1/models/{id}/jobs/{job}` | GET | `/jobs/{job}` |
| `/v1/models/{id}/jobs/{job}/wait` | GET | `/jobs/{job}/wait` + query |
| `/v1/models/{id}/jobs/{job}/result` | GET | `/jobs/{job}/result` |
| `{server_uri}` | POST | `{server_uri}` (legacy) |

`GET /v1/models/{id}/infer` and `GET {server_uri}` return `405`. Unknown `{id}`
is the same 404 envelope as an unknown `server_uri`. If the model has
`async_enabled` off, upstream `404` is passed through.

| Endpoint (model port) | Success | Errors |
|---|---|---|
| `POST /jobs` | `202` at **admission** with `job_id`, `state: pending`, `poll_url`, `result_url` | `429` when the admission queue is full |
| `GET /jobs/{id}` | `200` with `state` (`pending`/`running`/`done`/`failed`/`timeout`) | `404` unknown id |
| `GET /jobs/{id}/wait?timeout=N` | `200` when the job is **terminal**, or when the wait budget expires (`timeout` is milliseconds; default 30000, cap 300000). State may still be `pending`/`running` on expiry | `404` unknown id |
| `GET /jobs/{id}/result` | `200` standard envelope (repeatable) | `404` unknown id, `409` not finished (including `pending`/`running`/`failed`/`timeout`) |

`202` means the server accepted the job, not that inference has finished. A
correct client never treats `POST /jobs` like a blocking `/infer`. Submit the
same JSON envelope as `/infer` (`images[]`). Step-by-step customer verification
is in [async-jobs-customer-test.md](async-jobs-customer-test.md).

The job ledger is in-memory (lost on restart). The component design, concurrency
contract and verification gates are documented in
[async-job-table.md](async-job-table.md).
