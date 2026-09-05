# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). 每个版本的条目保留英文原文，中文说明
以引用块附于同版本之下。

## [Unreleased]

### Fixed
- GPU catalog ports: diffusion servers collided with scene segmentation
  (9070–9072). DDPM/DDIM/CLS_COND_DDIM/LDM now listen on 9081–9084 so
  `mortred-supervisor` can init the full GPU catalog (pack autostart still
  loads every profile-matching `conf/server` file).

### Added
- Machine-local autostart pack (`conf/packs/demo.toml`, `MORTRED_PACK`): listed
  catalog ids boot; `MORTRED_AUTOSTART=true` no longer starts the whole zoo.
  Pack `worker_nums` / `model_config` override the child via env without
  rewriting `conf/server` (still `worker_nums=1`).
- Stdlib HTTP inference RPS client (`scripts/server/http_infer_rps.py`): keep-alive
  workers, pre-encoded envelope, serving RPS + latency percentiles, optional `--qps`,
  JSON report. `test_server.py --mode load` wraps catalog/gateway URLs. No locust
  or requests.
- `mortredctl doctor --strict`: fail the doctor when security warnings fire
  (non-loopback plaintext HTTP, short tokens, identical tokens). Default
  `doctor` still warns only.
- Hosted `cpu-profile` fail-closes a multi-family MNN CPU golden set from
  `conf/ci_hosted_golden.json` (classification, NanoDet, DBNet, SuperPoint,
  BiSeNetV2): sha256-locked HF fetch, `MORTRED_CI_REQUIRE_WEIGHTS`, XML
  `skipped=0`. GPU smoke and TensorRT are still maintainer-only. Nightly
  remaining goldens write a skip-inventory artifact. HTTP catalog ids must
  declare a CI tier (`hosted` / `gpu-smoke` / `nightly`).
- Caddy reverse-proxy example (`deploy/caddy/Caddyfile`) as the supported TLS
  front for loopback gateway/supervisor. `mortredctl doctor` prints warnings
  (never fails unless `--strict`) for a non-loopback listen, a token shorter
  than 32 characters, or identical tokens.
- Optional gateway scrape token `MORTRED_METRICS_TOKEN`. Unset keeps
  `GET /metrics` public **on loopback**. A non-loopback gateway refuses to
  start without a distinct scrape Bearer. Managed model `/metrics` requires
  the supervisor internal token when `MORTRED_AUTH_TOKEN` is set.
- Gateway routes `POST /v1/models/{id}/infer` and `/v1/models/{id}/jobs*` to
  the model's loopback port. Job `Location` / `poll_url` / `result_url` are
  rewritten onto that prefix. The legacy `{server_uri}` POST path still works.

### Removed
- Locust demo worker `scripts/server/locust_performance.py` (breaking for anyone
  who invoked `--mode locust`). Use `--mode load` / `http_infer_rps.py`.
- Supervisor `/api/v1/infer`, `/api/v1/jobs*`, and `/api/v1/pipelines*`
  (breaking). Inference and async jobs go through the gateway; there is no
  server-side pipeline on the supervisor. Those paths now return the
  management `{ok, error}` 404. Graceful restart still drains by reading the
  model's `mortred_async_queue_depth` gauge.
- Unused helpers left by catalog and envelope migrations: diffusion
  `create_*_sampler` factories, dead `CvUtils` overlay/base64/tensor-copy
  helpers, unused `std_clip_*` / `std_sam_prompt_input` aliases,
  `build_unified_response_body`, `handle_custom_endpoint`,
  `FilePathUtil::is_dir_exist`, `Timestamp::to_str` / `invalid`,
  `detection_params_parse` (inlined into `DetectionParams::parse`),
  `TypeErasedFactory::register_type` and the `ModelFactory` alias.
- Dead `json_request_parser.h` (`parse_json_request` had no callers; it still
  accepted `img_data` and ignored unknown keys).
- `http_response.h` (`{req_id, code, msg, data}` shim). Process-level JSON
  now uses `UnifiedResponse` from `response_envelope.h`.
- Unused family `create_server` wrappers (every caller already used
  `cv_catalog::create_server`).
- Detector rename shims (`DetectionGeometryScale`,
  `make_detection_geometry_scale`, `scale_detection_bbox` /
  `scale_detection_point`, `validated_f32_output`).
- `task_request` / `go_result` aliases for `InferenceTask` /
  `InferenceResult`.
- Legacy HTML probes `/welcome` and `/hello_world` (breaking). Unknown paths
  now answer `404` with process-level `UnifiedResponse`.

### Changed
- GPU golden smoke is a maintainer gate (same-repo PR and push to main,
  skipped=0). It runs only when repository variable `MORTRED_HAS_GPU_RUNNER=true`;
  otherwise the job is skipped so CI does not wait for a missing runner. Smoke
  engine refresh is `convert_trt_engines.sh --only yolov8` only. Fork PRs never
  run on the self-hosted GPU runner. Require the `inference paths` check, not
  `gpu golden smoke` by name. See `docs/ci-golden-regression.md`.
- Example Docker Compose publishes gateway `:8080` and supervisor `:8787` on
  `127.0.0.1` only. The monitoring stack binds Grafana/Prometheus to loopback,
  requires `GRAFANA_ADMIN_PASSWORD`, and no longer enables Prometheus
  `--web.enable-lifecycle`. Default Prometheus scrape is gateway `/metrics`
  only; supervisor (Bearer) and model (loopback) jobs are commented with the
  real auth and topology constraints.
- Supervisor Web UI and `mortredctl infer` POST the data-plane envelope to
  the gateway (`/v1/models/{id}/infer` on `:8080`) instead of `/api/v1/infer`.
  The gateway accepts `MORTRED_API_TOKEN` and API keys with `admin` (or `all` /
  `inference`) scope, and reflects CORS for the supervisor UI origin.
  `common/response_envelope.h` (encode/decode + field names). Data-plane
  binding is `server/parsed_request.h`; in-process execution types moved from
  `async_job_table.h` to `server/inference_task.h`. Supervisor/CLI go through
  the codec instead of hand-rolled JSON.
- **Process-level JSON is now UnifiedResponse (breaking)**: `/healthz`,
  `/ready`, and 401/404/405/413/415/429 exits emit `{status, status_str,
  task_id, results:[]}` instead of `{req_id, code, msg, data}`. HTTP status
  codes and StatusCode wire integers are unchanged.
- **Gateway / supervisor proxy local failures use UnifiedResponse (breaking)**:
  gateway 401/404/405/502/503 and supervisor `/api/v1/infer` `/jobs*`
  `/pipelines*` failures before upstream now emit `{status, status_str,
  results:[], errors[]}`. HTTP status codes are unchanged. Management APIs
  (`/servers*`, start/stop, logs, supervisor 401/405) still use `{ok, error}`.
- **Unified request/response contract (breaking)**: model endpoints now speak
  the single envelope `{"req_id", "images": [<base64>...], "params", "options"}`
  and answer with `{status, status_str, task_id, model, results[], server_time_ms,
  partial}`. `results[]` is index-aligned with `images[]` and every item carries
  its own status (per-item failure isolation, deadline partials). The legacy
  `img_data` field was removed and answers `422` with a JSON-pointer migration
  hint; unknown fields/params are rejected strictly (`422` + `errors[]`).
- Request-level parameters for the detection family: `score_threshold`,
  `nms_threshold`, `top_k` (validated per model, TOML config stays the default).
- Backpressure, `Retry-After` and queue depth accounting are now per image
  item (`max_request_items`, default 16); one deadline spans queue wait,
  worker wait and inference.
- Gateway forwards the client `Content-Type`/`Accept` verbatim (binary body
  encoding groundwork).
- HTTP and benchmark ELFs no longer share `custom_drivers.cpp`; bench-only
  product rows compile only into the benchmark target (`MORTRED_WITH_CUSTOM_DRIVERS`).
- Gateway and supervisor share `control/http_reply.h` for JSON replies
  (error JSON shapes are unchanged).
- Factory `create_*` / `make_server_worker` / catalog `make_model` /
  `CvWorkerFactory` drop the unused name argument.
- `common` no longer links OpenCV (`cv_utils` is header-only).

### Fixed
- Supervisor Web UI, `/api/v1/infer`, `/api/v1/jobs` and pipelines now speak
  the unified `images[]` / `results[].data` envelope. They previously still
  sent the removed `img_data` field and read the legacy `{data: ...}`
  response, so the built-in test proxy and pipelines could not succeed
  against current model servers.

### Added
- Contract generation chain: `contract_dump` (C++ catalogs as the single
  source) -> `docs/contract_dump.json` -> `gen_openapi.py` -> OpenAPI +
  embedded `/openapi.json`; `scripts/check_contract_sync.py` gates the chain
  in CI (any spec change without regeneration fails the build).

## [0.1.0] - 2026-08-23

### Added
- Deployment profile system (`cpu` | `gpu`): one switch drives the build
  (`MORTRED_BUILD_PROFILE`), the dependency set (`install_deps.sh --cpu`),
  the model catalog (per-server `profile` field + `MORTRED_PROFILE`) and the
  weight subset (`fetch_weights.py --profile`). The cpu profile compiles
  TensorRT out entirely and ships a curated model set (mobilenetv2, resnet50,
  yolov8, hrnet).
- Dual-track distribution: `mortred-cpu` Docker target + `docker compose
  --profile cpu|gpu`, and versioned binary tarballs
  (`make_release_tarball.sh` + in-tarball `install.sh` with systemd wiring).
- Three first-hour entries sharing one core: `curl | bash` bootstrap, docker
  compose, and `mortredctl init / doctor / upgrade`.
- `MORTRED_AUTO_BUILD_ENGINES=true` opt-in engine conversion at container
  start (gpu profile).
- Project version (`--version`), this changelog, and a tag-driven release
  pipeline building both images and both tarballs.

### Fixed
- `mortred-gateway` link failure against vendored OpenSSL after the P0-2
  rework (no CI path compiled the gateway; the new cpu-profile job now does).

> 中文摘要：新增部署 Profile 体系（cpu/gpu 单一事实源贯穿构建、依赖、目录、
> 权重四层）、双轨分发（Docker 双 target + compose profiles；版本化 tarball
> + systemd 安装器）、三入口共享 mortredctl 内核（bootstrap / compose /
> init-doctor-upgrade）、可选首启 engine 转换、项目版本化与发布流水线；
> 修复 gateway 对 vendored OpenSSL 的链接缺口。
