# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). 每个版本的条目保留英文原文，中文说明
以引用块附于同版本之下。

## [Unreleased]

### Changed
- Envelope codec lives in `common/request_envelope.h` and
  `common/response_envelope.h` (encode/decode + field names). Data-plane
  binding is `server/parsed_request.h`; in-process execution types moved from
  `async_job_table.h` to `server/inference_task.h`. Supervisor/CLI go through
  the codec instead of hand-rolled JSON.
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
