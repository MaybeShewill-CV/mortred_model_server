# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). 每个版本的条目保留英文原文，中文说明
以引用块附于同版本之下。

## [Unreleased]

### Changed
- Release tarball layout is documented as **flat**: `install.sh`, `opt/mortred/`,
  and `deploy/` sit at the archive root (`make_release_tarball.sh` and the gpu
  release job already packed that way). README / deployment / installer comments
  unpack into an empty directory instead of `cd` into a wrapper the packer does
  not create.

> 发布 tarball 按**平铺**写进文档：包根就是 `install.sh`、`opt/mortred/`、
> `deploy/`（打包脚本和 gpu release job 本来就是这样打的）。README / 部署 /
> 安装脚本注释改为解到空目录，不再 `cd` 一个打包器不会生成的 wrapper 目录。

- Entry 2/3 and bootstrap now export all three tokens (`mortredctl init-trust`
  / `supervisor.env`). Compose already required `MORTRED_METRICS_TOKEN`; the
  copied quick-start commands now match. The optional monitoring compose
  bind-mounts a scrape credentials file (not minted by the app compose or
  the release tarball) and provisions the Grafana Prometheus datasource.

> 入口二/三与 bootstrap 改为三个 token（`init-trust` / `supervisor.env`），
> 与 compose 已有的 `:?` scrape 必填对齐。可选监控 compose 挂载 scrape
> credentials 文件（不由应用 compose 或发布 tarball 发明 token），并
> provision Grafana 的 Prometheus 数据源。

- Live documentation now matches the unified HTTP envelope (`images[]` /
  `{status, status_str, task_id, results[]}`). `api-contract`, `api-keys`, and
  the task tutorials no longer present `img_data` or `{code, msg, data}` as
  success examples. Gateway `/metrics` docs require `MORTRED_METRICS_TOKEN`
  on loopback in both languages. README Model Zoo splits HTTP catalog vs
  bench-only.

> 人读文档与统一信封对齐：请求 `images[]`，响应
> `{status, status_str, task_id, results[]}`。`api-contract`、`api-keys` 与任务教程
> 不再把 `img_data` 或 `{code, msg, data}` 写成成功示例。中英监控/部署均写明
> 网关 `/metrics` 含环回也要 `MORTRED_METRICS_TOKEN`。README Model Zoo 区分
> HTTP catalog 与 bench-only。

### Fixed
- `ci_container_boot.sh` asserts the unified infer envelope (`status` /
  `results[]`, including `results[0].status`) instead of the removed `code`
  field. `payload.get("code", 0)` always passed, so a HTTP 200 with
  `status != 0` would still print success.

> `ci_container_boot.sh` 按统一信封断言推理结果（`status` / `results[]`，含
> `results[0].status`），不再读已删除的 `code`。`payload.get("code", 0)` 会永远
> 通过，HTTP 200 且 `status != 0` 仍会打印成功。

- HTTP diffusion workers now seed the sampler input template from the model
  TOML at `init` (`sample_size` / default steps / channels). A missing or
  empty `sample_size` fails init instead of serving `MODEL_EMPTY_INPUT_IMAGE`
  on every request. The async smoke script and DDPM examples use
  `params.timesteps` (not a root `timestep`) and the unified result envelope.
  Few-step CPU generate proof: `model_golden.ddpm_celeba_hq_fewstep` with
  `conf/ci/ddpm_onnx_fewstep.toml` (not in hosted; ONNX is ~143MiB). The case
  checks a 128x128 PNG from the HTTP adapter; it does not pin pixels
  (`random_device` still seeds each run). The async smoke script sets
  `MORTRED_AUTH_TOKEN` and sends `Authorization: Bearer` on `/jobs`
  (empty token is 401; `/healthz` stays public).

> HTTP 扩散工人在 `init` 时从模型 TOML 种入采样模板（`sample_size` / 默认步数 /
> channels）。缺或空 `sample_size` 会 init 失败，而不再每请求
> `MODEL_EMPTY_INPUT_IMAGE`。异步冒烟与 DDPM 示例改用 `params.timesteps`
> （不是根上 `timestep`）和统一结果信封。少步 CPU 出图证明：
> `model_golden.ddpm_celeba_hq_fewstep` + `conf/ci/ddpm_onnx_fewstep.toml`
> （不进 hosted；ONNX 约 143MiB）。该用例断言 HTTP adapter 走出 128x128 PNG，
> 不钉像素（每次 `random_device` 仍重新播种）。异步冒烟会设 `MORTRED_AUTH_TOKEN`
> 并在 `/jobs` 带 `Authorization: Bearer`（空 token 是 401；`/healthz` 仍公开）。

- Supervisor now refuses to listen (when `mortred-gateway.out` is on disk) and
  refuses to spawn `__gateway` unless `MORTRED_METRICS_TOKEN` is set and
  distinct from the inference and management tokens — the same rule the
  gateway already enforced. Missing or colliding scrape secrets are a
  permanent failure, not a restart backoff, so a healthy `:8787` banner can
  no longer hide a gateway crash loop. `/api/v1/keys` remains 404.

> Supervisor 在磁盘上已有 `mortred-gateway.out` 时，若 scrape token 未设置或与
> 推理/管理 token 相同则拒绝 listen；spawn `__gateway` 使用同一谓词，失败记为
> permanent 而非 backoff，避免「健康横幅 + 网关崩溃循环」。`/api/v1/keys` 仍是 404。

- `BaseAiModel::run` (and packed `BackendCvModel::run_batch`) catch throws from
  `run_impl` / OpenCV and return `MODEL_RUN_SESSION_FAILED`, so a workflow go
  thread does not `std::terminate` the process. Packed `run_batch` also
  broadcasts that code onto every `item_status` after a throw (`run_image_batch`
  pre-assigns OK; HTTP would otherwise report success). Diffusion DDPM/DDIM/cls-cond
  postprocess no longer `convertTo(CV_8UC3)` + `COLOR_RGB2BGR` on 1/4-channel
  tensors (LDM latent `channels=4` with `save_raw_output`); display conversion
  is channel-correct and skipped when only raw latents are needed.

> `BaseAiModel::run`（以及 packed `BackendCvModel::run_batch`）接住 `run_impl` /
> OpenCV 的抛出并返回 `MODEL_RUN_SESSION_FAILED`，避免 workflow go 线程
> `std::terminate` 整进程。packed `run_batch` 在 catch 后还会把该错误码盖到
> 每一个 `item_status`（`run_image_batch` 会先全部标 OK；不盖的话 HTTP 会谎报成功）。
> 扩散 DDPM/DDIM/cls-cond 后处理不再对 1/4 通道
> `convertTo(CV_8UC3)` + `COLOR_RGB2BGR`（LDM latent `channels=4` 且
> `save_raw_output`）；出图按通道转换，只要 raw 时跳过。

- `do_work` writes inference metrics and the run-time EWMA before returning
  the worker to the queue (same order as `process_batch`). The destructor
  drain treats an enqueued worker as permission to destroy those members;
  writing them after enqueue raced a timed-out request's destructor.

> `do_work` 在把 worker 还回队列之前写入推理 metrics 和运行时间 EWMA
> （与 `process_batch` 相同）。析构 drain 把「worker 已入队」当作可以销毁
> 这些成员；enqueue 后再写会与超时请求的析构竞态。

- YOLO v5/v6/v7/v8 preprocess now uses Ultralytics-style center letterbox
  (keep-ratio, pad 114) instead of independent-axis stretch, and unmaps
  boxes with the matching pad/scale. Stretch `GeometryScale` remains for
  NanoDet / CenterFace / LibFace. Fork CI proves YOLOv8 decode+NMS+unmap
  via a CI-only ONNX overlay (`conf/ci/yolov8_onnx_hosted.toml`,
  `yolov8s.onnx`); product `yolov8_config.toml` is still TensorRT.

> YOLO v5/v6/v7/v8 预处理改为 Ultralytics 导出同款中心 letterbox（等比、
> 填充 114），框反变换用同一套 pad/scale。NanoDet / CenterFace / LibFace
> 仍走拉伸 `GeometryScale`。Fork CI 用仅 CI 的 ONNX overlay
> （`conf/ci/yolov8_onnx_hosted.toml`，`yolov8s.onnx`）证明 YOLOv8
> decode+NMS+unmap；出厂 `yolov8_config.toml` 仍是 TensorRT。

- Supervisor graceful shutdown no longer hangs. `WFServerBase::stop()` is
  already `shutdown() + wait_finish()` (blocking); the old teardown called
  `wait_finish()` a second time and blocked forever. In production this was
  masked by systemd's `TimeoutStopSec` SIGKILL - `mortred-supervisor` never
  actually exited gracefully. Found by the in-process SupervisorApp teardown
  of this refactor.

> Supervisor 优雅关停不再挂死。`WFServerBase::stop()` 本身就是阻塞的
> `shutdown() + wait_finish()`；旧代码又补了一次 `wait_finish()`，第二次
> 永久阻塞。生产上一直被 systemd `TimeoutStopSec` 的 SIGKILL 掩盖——
> `mortred-supervisor` 此前从未真正优雅退出过。由本次重构的进程内
> SupervisorApp 关停路径暴露。

### Changed
- Control plane de-globalized (internal refactor, no behavior change):
  the gateway/supervisor file-scope globals are gone. `GatewayApp` /
  `SupervisorApp` own their state (catalog, config, tokens, api keys,
  metrics, supervisor); `run()` maps the process environment onto an
  explicit `*InitOptions`, and the app objects live in a new
  workflow-bound `control_workflow` library so tests link them in-process.
  Acceptance: two gateway and two supervisor instances with distinct
  roots/tokens/catalogs serve side by side in one process
  (`gateway_multiinstance_test`, `supervisor_multiinstance_test`).

> 控制面去全局化（内部重构，无行为变化）：网关/supervisor 的文件级全局
> 状态移除，`GatewayApp` / `SupervisorApp` 持有各自状态；`run()` 将进程
> 环境映射为显式的 `*InitOptions`，app 对象移入新的依赖 workflow 的
> `control_workflow` 库以便测试进程内链接。验收：两个网关与两个
> supervisor 实例（不同 root/token/catalog）可同进程并存
> （`gateway_multiinstance_test`、`supervisor_multiinstance_test`）。

### Fixed
- Gateway forwards the raw-body control headers `X-Mortred-Params`,
  `X-Mortred-Options` and `X-Request-ID` to the model server, and echoes
  `X-Request-ID` on its own error replies. Raw-body requests previously lost
  their params/options and client correlation when proxied through the
  gateway (JSON-envelope requests were unaffected). All other client headers
  stay dropped - the default-deny forward list is the header-injection guard.

> 网关现在向模型服务器转发 raw-body 控制头 `X-Mortred-Params` /
> `X-Mortred-Options` / `X-Request-ID`，并在自身错误回复中回显
> `X-Request-ID`。此前 raw-body 请求经网关代理后参数与关联 id 被静默丢弃
> （JSON envelope 路径不受影响）。其余客户端头仍然丢弃——默认拒绝的转发
> 白名单就是防头注入的屏障。

- Supervisor `process()` folds a null request method to `""` like the
  gateway/model-server guards. A malformed request line (workflow leaves
  `get_method()` null) used to construct `std::string` from nullptr - UB,
  typically a crash of the management plane.

> Supervisor 的 `process()` 现在与网关/模型服务器一样把空 method 折叠为
> `""`。此前畸形请求行（workflow 返回 null method）会从 nullptr 构造
> `std::string`——未定义行为，通常表现为管理面崩溃。

- Async job replies are counted in `mortred_http_requests_total`: the `202`
  submit reply, `200` status/wait/result replies and the async error codes
  (404/405/409/429) were previously invisible on the model-server dashboards.
  Async replies deliberately carry no `mortred_http_request_duration_ms`
  sample - that histogram observes inference time and the async HTTP path
  has none.

> 异步任务回复现在计入 `mortred_http_requests_total`：此前 202 提交回复、
> 200 状态/等待/结果回复以及异步错误码（404/405/409/429）在模型服务器监控
> 上不可见。异步回复刻意不产生 `mortred_http_request_duration_ms` 样本——该
> 直方图观测的是推理耗时，异步 HTTP 路径上没有这一耗时。

- Model and gateway mains arm a `ProcessStop` latch (block SIGINT/SIGTERM,
  sigwait thread, idempotent `WaitGroup::done()`) so those signals reach
  `server->stop()` and the impl destructor drain instead of default-killing
  the process. Supervisor already had this path; `kStopGraceMs` SIGKILL
  remains the hung-model backstop.

> 模型进程与网关的 `main` 在 SIGINT/SIGTERM 时会 `done()` WaitGroup，从而走到
> `stop()` 和析构 drain；不再被默认信号直接杀死。Supervisor 本身原本就是这样。

### Changed
- `POST /jobs` flushes HTTP 202 at admission. The runner is a detached
  Workflow go task (`go->start()`), so submit no longer waits for
  `run_items`. `GET /jobs/{id}/wait` hangs the HTTP series on a named
  counter and wakes on a terminal job or the wait budget (milliseconds),
  not on `pending`→`running`. Serial submits can now observe `429` while a
  job is still running. Customer steps:
  [async-jobs-customer-test.md](docs/async-jobs-customer-test.md).

> `POST /jobs` 在准入时立即返回 202，不再等推理结束。`GET …/wait` 在终态或
> wait 预算耗尽时返回（单位毫秒）。客户逐步验收见
> [async-jobs-customer-test.zh-cn.md](docs/async-jobs-customer-test.zh-cn.md)。

### Removed
- Supervisor `GET /api/v1/keys` and `POST /api/v1/keys/reload`. Those routes
  mutated a copy of `ApiKeyManager` that never authenticated inference
  traffic. Edit `conf/api_keys.toml` then restart the gateway child
  (`POST /api/v1/servers/__gateway/restart`). `scope=admin` still does not
  unlock `:8787`; management stays `MORTRED_API_TOKEN` only.

### Changed
- `ApiKeyManager::load` treats a readable empty or comment-only key file as
  success with `key_count()==0` (not a parse failure). The gateway still
  refuses to start with no static token and no keys; with a static token it
  logs a warning and uses that token only.
- Spawn failures (fork/exec, missing exe, monitor respawn) now apply
  `RestartEngine::Decision` the same way as child exits, so backoff is
  scheduled instead of leaving `wanted` true with a dead pid.
- Deleted unused `kAutostartReadyTimeoutMs` (never referenced; 10-minute
  kill is still out of scope).

### Changed
- One model toml per model: removed `*_cpu_config.toml` duplicates. Git
  defaults `backend.device` to `gpu` (including omitted keys). The value is
  `cpu` or `gpu` (`cuda` is rejected). `type=tensorrt` with `device=cpu`
  with `device=cpu` fails at parse, session create, and supervisor spawn.
  CPU catalog entries point at the same files; operators who need CPU inference
  set `device = "cpu"` themselves. YOLOv8 and HRNet are not in the cpu catalog
  (TensorRT).

### Fixed
- Container runtime: `docker_entrypoint.sh` now has a `bash` shebang so
  exec-form `ENTRYPOINT` can start; compose `--profile gpu` builds
  `target: mortred-gpu` (the Dockerfile last stage is that alias, so
  `docker build .` is the GPU runtime again, not `mortred-cpu`).
- TensorRT spawn gate no longer fails when the model toml path is a dummy or
  missing (`SupervisorTest.spawn_passes_model_flag_for_unified_exe`). The gate
  only refuse-spawns after it can read `type=tensorrt` and the engine file is
  missing or empty.
- Calibrate / model-server HTTP start: YOLOV8 TRT can init then fail
  `Cannot start server` when `:9056` is already bound (supervisor still
  serving the pack). Probe now binds `127.0.0.1` like the supervisor, refuses
  a busy port before spawn, and logs host:port/errno on listen failure.
  Bind failure is `start_failed`, not OOM.
- Calibrate no longer treats whole-card `memory.used` as the model's footprint.
  Per-model numbers come from NVML compute-apps (pid, else unique
  `mortred-model-server` name). If WSL has no process row, `gpu_mem_mib_*` is
  the **delta** vs device used sampled before that spawn (`gpu_mem_source=
  device_delta`), not the card total. Joint residency does not sum deltas.
- Calibrate probed `127.0.0.1:0` on Python 3.10: `repo_toml` ignored unquoted
  `port=9002`. Fallback parser now reads integers; calibrate takes port/uri
  from the same `conf/server` file used to spawn.
- `convert_trt_engines.sh` retries without min/opt/maxShapes when TensorRT
  reports a static ONNX (`Static model does not take explicit shapes`). The
  yolov8 profile is for dynamic batch; some weight drops are fixed 1x3x640x640.
- `prepare_pack.sh` runs the `/ready` probe with cwd = `_bin`/`bin`, matching
  supervisor spawn, so `model_config_file_path = "../conf/..."` resolves when
  the script is invoked from the repo root.
- `prepare_pack.sh` stops the `/ready` probe with SIGINT (same as the
  supervisor) instead of SIGTERM, so glog does not dump a failure stack.
  Probe logs go to `logs/prepare-<id>.log`; a failed ready prints that file.
- GPU catalog ports: diffusion servers collided with scene segmentation
  (9070–9072). DDPM/DDIM/CLS_COND_DDIM/LDM now listen on 9081–9084 so
  `mortred-supervisor` can init the full GPU catalog (pack autostart still
  loads every profile-matching `conf/server` file).

### Added
- Release workflow pushes `ghcr.io/...:vX.Y.Z-gpu` and `:latest-gpu` from the
  existing gpu-tarball Docker compile (cpu tags stay in the images job).
- Hosted CI job `container boot (cpu compose)`: compose up → supervisor
  health → gateway `/healthz` → one MOBILENETV2 infer → supervisor/gateway
  process liveness. Infer uses a CI-only `device=cpu` pack overlay (git
  model tomls stay `device=gpu`). Expensive rebuild is path-filtered; GPU
  compose is not claimed on GitHub-hosted runners.
- Trust boundary for P0-4: `mortredctl init-trust` writes gitignored
  `conf/local/trust.env` (inference / management / scrape / internal). Gateway
  and supervisor refuse to start without their tokens, including on loopback.
  `GET /metrics` is never public. Wildcard bind requires `MORTRED_EXPOSE=docker`
  (containers) or `unsafe`. Nginx is the supported TLS edge
  (`mortredctl init-edge --mode lan|acme|files`, `deploy/nginx`, compose
  profile `edge` on Linux host network). Caddy is removed.
- ONNX Runtime CUDA `gpu_mem_limit` defaults to 2048 MiB per session
  (`gpu_mem_limit_mb` / `MORTRED_ORT_GPU_MEM_LIMIT_MB`; `0` = unlimited).
  The previous `gpu_mem_limit = 0` let the CUDA EP arena grow without bound.
- Machine pack ops in [docs/deployment.md](docs/deployment.md) §10
  (and [中文](docs/deployment.zh-cn.md)): autostart listed ids only, `mortredctl
  prepare` for pack TensorRT engines, `mortredctl calibrate` / `--write-pack`
  for `worker_nums` on the pack file (`conf/server` stays `1`).
- Pack worker_nums calibration report (`scripts/calibrate_pack.py`,
  `mortredctl calibrate`): sweep w, HTTP RPS via `http_infer_rps.py`, per-process
  GPU occupancy (NVML pid/name, else pre-spawn device delta), suggested w*,
  optional joint residency. `--write-pack` updates `[pack.<ID>] worker_nums`
  in that pack file only. Does not write `conf/server` (git copies stay
  `worker_nums=1`).
- Pack-scoped TensorRT prepare (`scripts/prepare_pack.sh`, `mortredctl prepare`):
  convert only engines used by `MORTRED_PACK`, refuse spawn if a file is missing
  or empty (no crash-loop), optional `/ready` at `worker_nums=1`.
  `MORTRED_AUTO_BUILD_ENGINES` still converts the whole zoo and stays opt-in.
  `doctor --strict` fails when pack TRT files are missing.
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
