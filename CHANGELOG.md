# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/). 每个版本的条目保留英文原文，中文说明
以引用块附于同版本之下。

## [Unreleased]

### Added
- **Per-model autostart budget:** `[servers.<id>] start_timeout_ms` sets that model's readiness budget. It is measured from the model's own launch, so a slow TensorRT engine no longer gets marked failed at a shared 60 s wave deadline while it is still loading, and it no longer shrinks the budget of the models queued behind it. Models without the key keep the `autostart_all` default (60 s).

### Fixed
- **Shutdown drain use-after-free:** `WorkerPool::drain` and `AsyncJobEndpoints::drain` now report whether the deadline cut the wait short, and `~BaseAiServerImpl` aborts on a short drain instead of destroying `_m_metrics` / `_m_workers` that a late go callback still writes. Same fail-closed policy as `StuckWorkerAction::EXIT`, so the supervisor restarts the process. Covered by a death test on an initialized server whose worker never comes back.
- **Batch stop can no longer hang shutdown:** `BatchCollector::stop` takes the same `kDrainTimeoutMs` deadline as the other drains and reports a batch that is still in flight, instead of waiting on it forever; `~BaseAiServerImpl` fails closed on that report too. A throw from a custom `run_batch` is now fenced at the collector, so the worker is checked back in and every item fails with `MODEL_RUN_SESSION_FAILED` (previously the throw escaped the go task and the worker never returned).
- **Readiness probe could wedge the monitor thread:** a whole-second probe budget was written only into `tv_usec` (`1000ms -> 1'000'000`, outside the kernel's accepted range) and the `setsockopt` failure was ignored, so `recv` ran without a timeout against a silent peer. The budget is now split into seconds plus the sub-second remainder, and a rejected `SO_RCVTIMEO` fails the probe closed. The monitor thread that runs probes also owns ordered shutdown, so this could stall stop as well as the probe.
- **DDPM could read past its step schedule:** request `timesteps` (the ParamSpec allows 1..1000) doubles as the p_sample iteration count and the highest `_m_betas` / `_m_alpha_cumprod` index, but those vectors are sized by `[DDPM_SAMPLER] timesteps` at init. An above-schedule request now fails with `INVALID_REQUEST_PARAMETER` (422) instead of indexing out of range, and `timesteps <= 0` fails init instead of building an empty schedule.
- **Admission cap could be overshot:** the queue gate compared `_m_waiting_jobs.load()` against `max_queue_depth` and added the item count later, so concurrent handler threads could all pass the check. The cap was advisory, `set_queue_depth` could publish values above the configured maximum, and `Retry-After` was computed from that inflated depth. `try_reserve_queue` claims the slots in one CAS, so rejection is fail-closed and the published depth matches accepted work. Batch intake needs no separate bound: a queued batch entry is still counted until its request replies.
- **TensorRT ran on the caller's CUDA device:** `cudaSetDevice` was called only in `init`, but CUDA's current device is a per-thread property and `run()` executes on whichever go/worker thread picked the session. With `device_id != 0`, every `cudaMalloc`, async copy and stream sync in that call was issued against the caller's device (default 0), and the destructor freed device memory from a context that did not own it. The session now stores the configured device and re-selects it at `run()` entry and before release.

### Changed
- **The supervisor daemon keeps its own log:** `SupervisorApp::run()` never initialised glog, so the supervisor's own `LOG(...)` lines only reached stderr — the journal under systemd, nothing at all when a script redirected stderr. It now points glog at the same `logs/` directory the children use (still mirrored to stderr) and installs glog's failure handler, so a crash leaves a stack trace next to those files. Initialisation happens before the config is read, so a refusal to start (missing `MORTRED_API_TOKEN`, bad config) is recorded too — the token refusal was the one that first proved the gap. `run()` is the daemon-only path, so the in-process test harness is unaffected.
- **`ImagePipeline` stopped cloning the source image:** measured against the gate recorded in `scripts/microbench/clone_bench.cc`, the constructor clone was 13.4% (640x640), 21.1% (1080p) and 30.2% (4K) of the preprocess chain — over the 10% gate, so it is gone. The constructor now shares the refcounted `cv::Mat`; `to_float()` is the step that detaches the buffer, so the in-place `scale/subtract/divide` reject a chain that reaches them first instead of writing through to the caller's image. Two tests pin the untouched source and the rejection.
- **The status table stopped naming one model's input size:** `MODEL_INPUT_SIZE_REJECTED` carried `"input size must be 256x256"`, wrong for every model that is not Real-ESRGAN; the shared message now says what happened and Real-ESRGAN reports its own requirement plus the size it got. `status` and `status_str` were paired by hand at four reply sites (`base_server_impl`, `item_exec`, `http_wire`, `control/http_reply`); `common::set_status()` now sets both, with a unit test pinning the pair and asserting the shared table does not hardcode a size again.
- **Entry points fail with distinguishable exit codes and validate the port:** `model_server_main` / `model_benchmark_main` used `return -1` for every failure, so a wrapper could not tell bad argv from bad config from a backend that failed to init. They now return sysexits 64 (usage), 78 (config) and 70 (backend/listen), and the listen port is validated to 1..65535 before the `unsigned short` narrowing — a `port=70000` or a malformed `MORTRED_LISTEN_PORT` used to silently bind a different port. A catalog id collision is now a startup failure (78) that names the ids instead of a `--list` row disappearing with a stderr warning.
- **Config typos are locatable:** `mini_toml` now reports dropped lines with their line number, and `ControlConfig::load` warns `path:line: unknown config section/key … ignored` for anything outside the schema instead of dropping it in silence. Validation errors now name the config file too. Unknown keys still do not fail the load — stage 1 diagnoses, it does not change startup semantics.
- **Test isolation and hang detection:** the gateway/supervisor e2e and multi-instance suites derive their temp root from the process id instead of a fixed `/tmp` name, so two concurrent runs (or `ctest -j`) no longer delete each other's fixtures. Every ctest entry gains a `TIMEOUT` (900 s, heavy-golden 1800 s) so a hung binary fails instead of riding ctest's default.
- **Model `/metrics` only accepts the scrape token:** the inference bearer used to authorize `/metrics` on model processes, so an inference credential could read the scrape surface — while the gateway already refused a metrics token equal to the inference token (`trust_tokens.h`). Only `MORTRED_METRICS_TOKEN` authorizes `/metrics` now, and unset means closed rather than a fallback to inference. The supervisor already injects the scrape token into model children; `scripts/calibrate_pack.py` now derives and injects its own.
- **Model families stop duplicating their pre/post chains:** three classification postprocess loops and five segmentation/matting preprocess chains were byte-identical copies. They now call `apply_top_k` (`models/classification/top_k.h`, unit-tested) and `dense_norm_pipeline` (`models/backend/model_runtime.h`). The normalisation constants live in one `k_norm_signed_unit`, shared by the CPU chain and the GPU descriptor registration, so the two paths cannot drift. Net −44 lines; no numerical change.
- **Duplicate output lookup removed, and the clone now has a number:** five model classes (`openai_clip`, `lightglue`, `superpoint`, `fast_sam_segmentor`, `sam_prompt_decoder`) carried their own `find_output` loop next to the shared one in `tensor_contract.h`; they call the shared helper now. A new `scripts/microbench/clone_bench.cc` measures `ImagePipeline`'s constructor clone against the resize/convert chain it precedes, with the accept/reject gate recorded in its header — the measurement that decides whether dropping the clone is worth it.
- **Default `check` is unit + contract only:** the request-path e2e suites (`gateway_e2e_test`, `gateway_multiinstance_test`, `server_e2e_contract_test`, `supervisor_malformed_e2e_test`, `supervisor_multiinstance_test`) and `async_job_stress_test` now carry `e2e`/`stress` labels and are excluded from `--target check` (`-LE valgrind|heavy-golden|e2e|stress`). They start real gateways, supervisors and listeners, which made the default gate slow and fragile — one aborting supervisor multi-instance suite ended a full run at 59/96. CI runs them in a dedicated step, `ctest -L e2e` / `-L stress` locally, and `MORTRED_CHECK_HEAVY=1` still runs everything except valgrind. `test/CHECK_SSOT.md`, its checker and that checker's self-test track the new lanes.

### Removed
- **Dead server-factory registry:** `TypeErasedFactory` / `ServerFactory` had one caller that registered a creator into a process-wide map under a mutex and immediately called `create()` — `std::make_unique` plus global state, and it contradicted the neighbouring lint's own rationale. `cv_catalog::create_server` constructs `CvModelServer` directly now; the header, its 13-test unit suite and the `register_type` lint (the symbol no longer exists, so a reintroduction is a compile error) are gone.

- **Supervisor teardown could abort:** a `Child` owns two log-reader `std::thread`s whose only join site is `handle_exit`. Any other destruction path (catalog reset, or teardown racing the reaper) destroyed a still-joinable thread and `std::terminate`d the process. `~Child` now detaches them: the reader closure owns its pipe fd and a `shared_ptr<LogBuffer>` copy, and joining on the reaper is deliberately avoided so a stalled filesystem cannot stop reaping.

## [1.1.0] - 2026-10-22

### Fixed
- **Graceful draining (P0-5):** model servers and the gateway gain `begin_drain()`; `/ready` flips to 503 and new admissions are rejected before `stop()` while in-flight work finishes in the existing drain. The gateway also gains a `/ready` endpoint (the LB pull-out point); compose `stop_grace_period` now covers drain + stop grace. e2e: 20-iteration ready-503/new-503/in-flight-200 contract on both sides.
- **Supervisor revive race (P0-3b):** the monitor re-checks `wanted`/`stopping` under the spawn lock before respawning, so a REST stop landing between candidate collection and spawn can no longer revive a stopped process.
- **Respawn no longer destroys log evidence (P0-3a):** `LogBuffer::reset()` (truncate in place) is replaced by `rollover()` — the ended run's file is renamed to `<id>.log.run<unix_ms>` (newest 3 kept; size-rotation `.1` untouched) and a fresh log starts. Crash evidence survives restarts.
- **Spawn never waits on log readers (P0-3c):** reader threads hold the ended run's `shared_ptr<LogBuffer>`; spawn swaps in a new buffer over the archived file. The 10 s global-mutex stall in `spawn_locked` (which blocked status/logs/stop) is gone, and `logs()` returns a `shared_ptr` (raw-pointer use-after-free window closed).
- **Prompt stop no longer depends on pipe EOF (latent):** `handle_exit` never notified the condition variable, so stop latency silently relied on the log-reader joiner's notify; a stalled reader would have burned the whole stop grace. `handle_exit` notifies directly (supervisor suite wall time 180 s → 59 s).
- **Diffusion sampling status is thread-safe by construction (P0-4):** `_m_sample_status` moved from instance members into run-local out-params in ddim/ddpm (`p_sample`/`p_sample_once`); no shared mutable run state remains. Server-side `indicators` progress bars (instance members mutated during inference) are deleted. `conf/ci/ddpm_onnx_fewstep.toml` pointed at a removed weight filename, so `model_golden.ddpm_celeba_hq_fewstep` had been an unreachable skip — repaired; the case now passes end-to-end (`NUMERICAL_PASSED`).
- **`-Werror` gate green again in `src/control`:** dead `drop_env_entry` deleted, `spawn_locked`'s now-unused `lock` parameter removed, vestigial `name_len` in `gpu_sampler.h::parse_sample` silenced with the reason stated.
- **API keys support a per-file `[kdf]` salt (P1-1):** `hash = sha256(salt + key)` when `[kdf] salt` is declared; legacy unsalted entries keep working (salted tried first). Keeps O(1) lookup and hardens a stolen config against offline brute force. `mint_api_key.sh` picks the salt up automatically.
- **enlightengan contract tests crashed on a default-constructed model:** the preprocess dtype tail dereferenced the (null) session before init; guarded with `has_session()` (0/20 -> 20/20 passes).
- **mini_toml silently dropped malformed config lines** (key-value outside any section, or missing `=`): load() now reports each dropped line with the file path on stderr (P1-6).
- **Supervisor could revive a process the operator had just stopped** when a restart backoff expired between candidate collection and spawn; fixed with a re-check under the spawn lock (P0-3b).
- **Detector fallback thresholds single-sourced** in `models/object_detection/detection_defaults.h` (were copy-pasted across seven detectors) and the status-8 wire message is now English: "input size must be 256x256" (P1-8, P1-9).

> 模型服务器与网关新增 `begin_drain()`：停止前 `/ready` 翻 503、新请求被拒，在途请求照常跑完；网关新增 `/ready` 端点（LB 摘流点）。supervisor 修复 stop 后进程被复活的竞态。respawn 不再原地截断子进程日志（改为归档保留最近 3 份），spawn 也不再持全局锁等日志 reader（10 秒卡顿消除），`logs()` 改返回 `shared_ptr` 关闭换表 UAF 窗口。顺带修复潜伏 bug：`handle_exit` 从不 notify，停止时效一直依赖 reader EOF 时序。diffusion 采样状态改为局部传递（并发隐患清除）、删除服务端进度条、修复 ddpm golden 用例的权重路径漂移（现为 `NUMERICAL_PASSED`）。`src/control` 的 `-Werror` 门禁恢复绿色。

### Changed
- **Namespace `jinq` renamed to `mortred` across the tree (P1-5):** 294 files. Breaking for external includes (none exist; application project). Ledger snippets under cowork/ keep the original spelling as history.
- **Gateway retries the upstream once on transport errors (P1-7):** `create_http_task` retry_max 0->1 covers the transient window when a model child exits under restart. No circuit breaker by design (loopback upstreams fail connect in microseconds) and the 180s upstream timeouts stay (diffusion sampling legitimately runs minutes; configurable in mortred.toml).
- **`max_connections` default 200 -> 1024 (P1-4):** admission control protects the model, not the socket cap.
- **Non-batch request item serialization (P1-3) closed as by-design:** one worker = one exclusive model session; items cannot interleave. Levers are `worker_nums`, `max_request_items`, and batch co-scheduling.
- **Dead code removed (P1-2):** `server/rate_limiter.h` (+ its unwired test), `ApiKeyManager::reload()` (no production caller; "NO hot reload" was already the documented contract), and the duplicated nvidia-smi probe (~130 lines) merged into one `run_nvidia_smi_csv()` shared by the occupancy snapshot and HUD sampler.
- **Raw-body path single copy- **Raw-body path single copy (P0-2b, measured):** `parse_raw_request` takes the body by value and moves it into the item; the transport copy is the only full-body copy on the raw path. The larger in-situ JSON surgery was measured (-13% envelope stage on the 1.6 MB baseline, below the ≥30% gate) and intentionally dropped; numbers recorded in `cowork/progress-card.md`.

> raw 路径 `parse_raw_request` 改为按值拿 body 并 move 进 item：传输层拷贝成为该路径唯一一次整包拷贝。更大的 in-situ JSON 改造实测仅 -13%（低于 ≥30% 验收门），有意放弃，数字记录在 `cowork/progress-card.md`。

### Added
- **SME-14:** `scripts/check_consistency.py` (and CI) runs `golden_drift_check.py --check` fail-closed against `test/golden_baseline.json`; refresh docs require `--record` in the same PR. Baseline reset to current 30 cases / 26 golden files.
- **SME-12:** Document unsupported boundaries in `docs/unsupported-boundaries.md` (+ zh summary): Linux x64 only; RTDETR scaffold not in catalog; CUDA 12.x + TensorRT **10.x** only (8/9 out); engines must be rebuilt on this GPU/TRT; cpu profile has no TensorRT. Linked from README, oob-main-path, deployment, and rtdetr model doc.
- **SME-11:** Single out-of-box main path: `docs/oob-main-path.md` plus `mortredctl next` (`scripts/mortredctl_next.sh`) prints one next command (three tokens → loopback listen tip → start supervisor → pack prepare/calibrate → `doctor --strict`). `init` / `init-trust` / doctor WARN·FAIL / prepare / calibrate emit `next:` lines. README Quick Start points at the main path; other entries are variants.
- Pack occupancy is fail-closed at supervisor spawn and `mortredctl doctor --strict`.
  TensorRT ids in an active pack need `gpu_mem_mib` from
  `mortredctl calibrate --write-pack` (plus `gpu_mem_at_workers` and a GPU
  fingerprint on `[pack]`). Missing stamps, stale `worker_nums`, and joint-budget
  overflow refuse spawn with the next command in the error (no crash-loop).
  `occupancy_policy=off` / `MORTRED_OCCUPANCY_ENFORCE=0` skip the spawn gate
  (unsafe) and fail `--strict`. `gpu_mem_limit_mb` stays ORT CUDA EP only.

> pack 占用在 supervisor spawn 和 `mortredctl doctor --strict` 上 fail-closed。
> 已启用 pack 里的 TensorRT id 必须有 `calibrate --write-pack` 写入的
> `gpu_mem_mib`（以及 `gpu_mem_at_workers` 与 `[pack]` GPU 指纹）。缺 stamp、
> `worker_nums` 过期、联合预算超限都会拒绝 spawn，错误里是下一条命令，不进
> crash-loop。`occupancy_policy=off` / `MORTRED_OCCUPANCY_ENFORCE=0` 跳过 spawn
> 闸（unsafe），`--strict` 仍失败。`gpu_mem_limit_mb` 仍然只约束 ORT CUDA EP。

### Added
- Cpu source-tree shortest success path: `docs/shortest-path-cpu.md` (linked from
  README Quick Start) and SME-02 evidence under `docs/evidence/`.

> 增加 cpu 源码树最短成功路径文档 `docs/shortest-path-cpu.md`（README Quick Start
> 已链入），以及 SME-02 evidence（`docs/evidence/`）。

### Changed
- **Docs:** drop `oob-main-path.md` (first-hour lane lives in `deployment.md`); remove `sme-oob-checklist.md`, `async-job-table` (+zh), historical `model_inference_benchmark` (+zh), and `onnx-export-guide.md`.
- **Docs:** remove `model-contract-governance.md` (rules live in `model-developer-guide`; image-size caps noted there).
- **Docs:** fold `shortest-path-cpu.md` into `oob-main-path.md` (single out-of-box lane); remove `docs/bench/`, `docs/evidence/`, and `docs/models/` (RTDETR note stays in unsupported-boundaries).
- **Docs:** merge five per-task model-server tutorials into `docs/tutorials_of_model_servers.md` (+ zh-cn); README links updated.
- **Docs prune:** drop completed historical plans (base_server split, unified-contract refactor log, P4 DX plan/todolist/metrics) and large generated architecture HTML/diagrams; README / how-to / developer-guide links retargeted to living guides.

### Fixed
- **P1-9 / A3:** CHANGELOG no longer claims batch uses deleted `submit_and_wait`; describe shared request-level timer + `BatchCollector::submit` snapshot semantics.
- **P1-8 / A2:** label `batch_collector_unittest` as `sanitizer` and build it (with `worker_pool_unittest`) in the TSAN CI gate / tests-only-tsan preset.
- **P1-3 / A1:** `tests-only*` build presets default to target `check` (so EXCLUDE_FROM_ALL unit tests are built); docs drop bare `ctest` after preset build. TSAN preset builds sanitizer-labeled binaries.
- **Round-3 P0:** `check_consistency` convert/CI contract aligned with SME-16 (require unconditional `--skipInference`, refuse `-ge 9` / `--buildOnly`, require `-lt 10` gate and CI `TRT_VERSION_MAJOR=8` failure assert).
- **SME-16:** `convert_trt_engines.sh` refuses TensorRT major &lt; 10 and drops the TRT 8 `--workspace`/`--buildOnly` path; `TRT_VERSION_MAJOR` only overrides failed banner probes (still fail-closed if &lt; 10). Dry-run no longer silently assumes 10 when version cannot be detected.
- **SME-13:** Drop zombie `web_console` / `ServerManager` claims from `CMakeLists.txt`, `test/ready_probe_unittest.cc`, and `scripts/clean_artifacts.sh` (those apps are gone; `_bin`/`_lib` and `ready_probe` belong to supervisor/control).
- **SME-10:** CMake configure fail-closed for full builds: require vendored workflow; assert ONNX Runtime headers `ORT_API_VERSION==29` and refuse leftover `libonnxruntime.so.1.18*`; FATAL fix lines are profile-aware (`--cpu --all` / `--all`) and match `install_deps --check`. README / shortest-path-cpu / deployment §7 document `--check` before cmake.
- **SME-09:** Batch collector honors per-request `deadline` (same clock as the HTTP timer): expired entries TIMEOUT without checkout; checkout wait is `min(config, min remaining deadline)`; re-check after pool wait. In-flight `run_batch` is not cancelled mid-run.
- **SME-08:** `write_slot` uses lock-free claim→write→publish (no dual writers on `outputs`); `submit`/`stop` use `_accepting`+epoch recheck with TIMEOUT fixup (no new mutex). Concurrent unit tests cover same-index races and submit/stop TOCTOU.
- **SME-07:** Model `serve_process` authenticates before per-IP `rate_limit_qps`; `/healthz` `/ready` `/openapi.json` are exempt. Unauthenticated callers get **401** (not **429**) and do not burn the IP budget.
- **SME-06:** `POST /api/v1/servers/{id}/start|stop|restart` returns HTTP **500** (with `ok:false`) on failure instead of always 200; aligns with `graceful_restart`. UI treats non-2xx as failure.
- **SME-05:** Supervisor applies `[supervisor].pack_file` from `mortred.toml` via `ControlConfig::apply_pack` when `MORTRED_PACK` is unset (`resolve_pack_path`: env/CLI wins). Docs/comments matched behavior.
- **SME-04:** `scripts/release_dry_run.sh` locally verifies GHCR IMAGE lowercase, tarball+basename `.sha256`, and bootstrap mismatch refuse; `check_consistency` gates `release.yml` lowercase IMAGE construction.
- **SME-03:** `scripts/check_consistency.py` gates the `ci.yml` mock-trtexec dry-run contract against `convert_trt_engines.sh` (require `--skipInference`, reject `--buildOnly`, TensorRT 10.x mock banner).
- **SME-03:** `pack_trt.py --self-test` isolates `MORTRED_PROFILE` (and uses `profile="any"`) so a live CPU shell cannot fail hermetic consistency.
- `install_deps.sh` toml++ one-shot install: timed curl into a stage file, reject
  truncated headers (<17000 lines), always ensure `TOML_EXCEPTIONS=0`, and do not
  stamp a partial download. SME-01 archives Linux tests-only 53/53 at `d51b0287`
  under `docs/evidence/` plus the living `docs/sme-oob-checklist.md`.

> `install_deps.sh` 的 toml++ 改为限时下载到暂存文件、拒绝残片（&lt;17000 行）、
> 始终保证 `TOML_EXCEPTIONS=0`，不对半截下载盖 stamp。SME-01 在 `docs/evidence/`
> 归档了 Linux tests-only 53/53（`d51b0287`），并加入活清单 `docs/sme-oob-checklist.md`。

- Release workflow lowercases `github.repository` for GHCR image paths so
  `ghcr.io/maybeshewill-cv/mortred_model_server` matches docs (uppercase owner
  refs are illegal). Bootstrap refuses install when a published `.sha256` does
  not match; missing checksum still WARN-and-continue.

> release 工作流把 `github.repository` 转成小写再推 GHCR，与文档中的
> `ghcr.io/maybeshewill-cv/mortred_model_server` 一致（大写 owner 非法）。
> bootstrap 在已发布的 `.sha256` 与包内容不匹配时拒绝安装；缺少校验文件仍
> WARN 后继续。

- `install_deps.sh --workflow` also installs vendored `libssl`/`libcrypto` (same
  path as `--all`). CMake requires `vendored::crypto` whenever workflow is
  present; sanitizer CI only ran `--workflow` and failed at configure.
  `--check` now asserts `libcrypto.so.3` or `libcrypto.so.1.1`.

> `install_deps.sh --workflow` 也会安装 vendored 的 `libssl`/`libcrypto`（与
> `--all` 同路径）。有 workflow 时 CMake 硬要 `vendored::crypto`；sanitizer CI
> 只跑 `--workflow` 会在 configure 阶段挂。`--check` 现在断言 `libcrypto.so.3`
> 或 `libcrypto.so.1.1`。

- `install_deps.sh` binds ORT headers to the pinned library version: wipe/reinstall
  replaces `3rd_party/include/onnxruntime` atomically, `stamp_fresh` invalidates on
  `ORT_API_VERSION` mismatch (and missing `struct CUDAProviderOptions` on gpu), and
  `--check` fail-closes on the same gates. Stops local full-GPU builds from compiling
  `ort_session.cpp` CUDA EP against leftover ORT 1.18 headers while linking 1.29.

> `install_deps.sh` 把 ORT 头与 pin 的库版本强绑定：wipe/重装会整目录替换
> `3rd_party/include/onnxruntime`；`stamp_fresh` 在 `ORT_API_VERSION` 不匹配（gpu
> 下缺 `struct CUDAProviderOptions`）时失效；`--check` 用同一套门闩 fail-closed。
> 避免本地 full-GPU 用残留的 ORT 1.18 头编译 `ort_session.cpp` CUDA EP、却链接 1.29。

- Sync inference replies at `model_run_timeout` without dropping completed
  items. The HTTP series holds a unique-reply counter; per-item compute is a
  detached go chain plus one request-level timer. `results[]` is always length
  N: 504 / status 4 when nothing published; 200 / status 68 / `partial=true`
  when at least one item finished. Batch uses async `BatchCollector::submit`
  with the **same** request-level timer race as the non-batch path: reply
  takes an acquire snapshot of published slots (unpublished → TIMEOUT);
  late `write_slot` completions are dropped after the unique reply.

> 同步推理在 `model_run_timeout` 到期时回包，不再把已完成项丢掉。HTTP series
> 上是唯一回复的 counter；逐项计算是脱离 series 的 go 链，外加一个请求级
> timer。`results[]` 长度恒为 N：没有任何项发布时 504 / status 4；至少一项
> 完成时 200 / status 68 / `partial=true`。凑批走异步 `BatchCollector::submit`，
> 与非凑批共用请求级 timer：回包对已发布 slot 做 acquire 快照（未发布 → TIMEOUT）；
> 唯一回复之后迟到的 `write_slot` 会被丢弃。

- Release checksums use the tarball **basename** so `sha256sum -c` after
  `curl -fLO` prints `OK` (deployment §6.1). `sha256sum /abs/path` wrote a
  runner path that does not exist on the operator machine. `bootstrap.sh`
  downloads `mortred_model_server-<tag>-<profile>-linux-x64.tar.gz` after
  resolving the latest GitHub release tag (there is no `...-latest-...`
  asset). GPU release staging copies into `opt/mortred/` like the cpu packer
  (`cp -a src/. dest/`, not `cp -a src dest/opt/`).

> 发布 `.sha256` 只写 tarball **文件名**，`curl -fLO` 后 `sha256sum -c` 才能
> 打出 `OK`（deployment §6.1）。原先 `sha256sum /绝对路径` 会把 runner 路径写进
> 校验文件。`bootstrap.sh` 先解析 latest tag，再下
> `mortred_model_server-<tag>-<profile>-linux-x64.tar.gz`（没有 `...-latest-...`
> 资产）。gpu staging 与 cpu 一样落到 `opt/mortred/`（`cp -a src/. dest/`，
> 不是 `cp -a src dest/opt/`）。

- `write_prometheus_credentials.sh` `chown`s the scrape secret to the process
  that reads it: uid **65534** (compose `prom/prometheus` nobody) or user
  **prometheus** (apt unit, `/etc/prometheus/*`). Mode stays 600. Compose
  monitoring scrapes `host.docker.internal:8080` via `prometheus.compose.yml`
  (`localhost` inside that container is Prometheus). `deployment` documents
  both the ownership step and the Docker DNS name.

> `write_prometheus_credentials.sh` 把 scrape 密钥 `chown` 给真正读它的进程：
> compose 里 `prom/prometheus` 的 uid **65534**，或 apt 单元的 **prometheus**
> 用户（`/etc/prometheus/*`）。权限仍是 600。compose 监控通过
> `prometheus.compose.yml` 刮 `host.docker.internal:8080`（容器内 `localhost`
> 是 Prometheus 自己）。`deployment` 补了属主步骤和该 Docker 主机名。

### Changed
- GPU line is CUDA 12 + TensorRT 10.3.0.26 + cuDNN 9.10.2 + MNN 3.6.1 +
  ORT 1.29 cuda12. `install_deps` wipe/reinstall removes CUDA 11 / TensorRT 8 /
  ORT 1.18 **libs and mismatched ORT headers**; `--check` requires
  `ORT_API_VERSION` to match the pin. `--cuda-version 11` fails. Engines must
  be rebuilt with matching trtexec. Full GPU CMake refuses AddressSanitizer
  (ASan + TRT/cudart mixed ELF).

> GPU 线改为 CUDA 12 + TensorRT 10.3 + cuDNN 9 + MNN 3.6.1 + ORT 1.29 cuda12。
> `install_deps` 的 wipe/重装会清掉 CUDA 11 / TensorRT 8 / ORT 1.18 **库以及
> 与 pin 不一致的 ORT 头**；`--check` 校验 `ORT_API_VERSION`。`--cuda-version 11`
> 失败。引擎必须用匹配的 trtexec 重建。full GPU CMake 拒绝 AddressSanitizer。

- Entry 1 without Docker resolves the GitHub latest **tag** and downloads
  `mortred_model_server-<version>-<profile>-linux-x64.tar.gz`. Release does not
  publish a `...-latest-...` tarball name (GHCR `:latest-cpu` / `:latest-gpu`
  stay image tags). README / deployment / `bootstrap.sh` no longer claim that
  path always installs: no Release yet prints WARN and the source-build track.
  `verify_deployment.sh --basic` rejects packing or fetching
  `mortred_model_server-latest-`.

> 入口一无 Docker 按 GitHub latest **tag** 下
> `mortred_model_server-<version>-<profile>-linux-x64.tar.gz`。Release 不挂
> `...-latest-...` tarball 文件名（GHCR `:latest-cpu` / `:latest-gpu` 仍是镜像
> tag）。README / deployment / `bootstrap.sh` 不再写成「无 Docker 一定能装上」：
> 还没有 Release 时 WARN 并落到源码构建。`verify_deployment.sh --basic` 禁止
> 打包或下载 `mortred_model_server-latest-`。

- YOLOv5 / v6 / v7 / v8 apply `min_box_area_px` in decoder **network
  (letterboxed) pixels** before letterbox unmap. v5/v6 previously filtered
  source pixels after unmap; v8 ignored the key. Synthetic decode tests
  cover the shared threshold.

> YOLOv5 / v6 / v7 / v8 的 `min_box_area_px` 统一在解码坐标系（letterbox
> **网络像素**）里、unmap 之前过滤。原先 v5/v6 在 unmap 后的源图像像素上比
> 面积，v8 完全不读该键。合成解码单测覆盖这条共用阈值。

- `check_trt_engine_manifest` walks HTTP catalog `[*.backend] type="tensorrt"`
  paths (not the removed `*_TRT` tables) against `conf/trt_engines.json`.
  Scaffold configs (`TODO(new_model)` / not in the HTTP catalog) are skipped.
  `fetch_weights --profile cpu` no longer includes the off-HF hrnet ONNX that
  the cpu catalog does not serve.

> checker 按 HTTP catalog 的 `[*.backend] type="tensorrt"` 对照
> `trt_engines.json`（不再找已删除的 `*_TRT`）。脚手架（`TODO(new_model)` /
> 不在 HTTP catalog）跳过。cpu 权重集合不再含 catalog 不服务、且不在 HF 上的
> hrnet ONNX。

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
