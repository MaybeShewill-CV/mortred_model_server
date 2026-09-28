# 模块评分

- id: src-server-apps
- 权重: 15
- 分数: 7.7
- 读过的路径: `src/apps/model_server_main.cpp`, `src/server/abstract_server.h`, `src/server/base_server_impl.h`, `src/server/async_endpoints.h`

## 优势
- 单进程按 catalog id 启动：`--list` / `--model` / TOML 段与 `--model` 一致性校验齐全，未知模型、bench-only、缺段、model 字段不匹配都会在启动前 stderr 退出，小团队能按错误文案改配置（`model_server_main.cpp`）。
- HTTP 推理面已覆盖交付常用能力：Bearer 鉴权、IP QPS、`/healthz` `/ready` `/metrics` `/openapi.json`、同步 POST、413/415/429（含 Retry-After）、统一 JSON 拒绝与 Prometheus，出错时有状态码与指标可查（`base_server_impl.h` `serve_process`）。
- 异步任务有完整生命周期：`POST /jobs` → 202 + Location、status / wait / result，队列满 429、未完成 409，并与 worker checkout、metrics 挂钩，长耗时 CV 任务可交而不堵同步口（`async_endpoints.h`；`base_server_impl.h` `async_run_job` / `wire_async_endpoints`）。
- 进程生命周期有明确约定：`BaseAiServer` 封装 start/stop/wait_finish；析构 drain worker/async，并写明依赖 supervisor SIGINT→SIGKILL，和「一 catalog 一进程」运维模型对齐（`abstract_server.h`；`base_server_impl.h` 析构注释）。

## 劣势
- `WORKFLOW_library_init` 用 `std::call_once`，同进程多 server（注释写明 tests/gateway）时只认第一次的 compute/handler threads，后起模型无法按自身配置改全局线程池，排障时易误判为「配置没生效」（`abstract_server.h` `init_http_server`）。
- 成功 init 后析构对 worker/async drain **无界等待**；模型挂死时进程停不下来，必须靠外层 supervisor 超时 SIGKILL，小团队若 grace/监控配错会表现为「停服卡住、日志无终态」（`base_server_impl.h` 析构；`async_endpoints.h` `drain` 忙等）。
- 同步路径含 batch、per-item go、timer 与 counter 竞态回包，可定位手段依赖 metrics / 可选 stage_timing；未开 trace 时客户侧超时、部分成功（`DEADLINE_EXCEEDED_PARTIAL`）仍难在现场快速拆到「排队 / 推理 / 回包」哪一段（`base_server_impl.h` `schedule_sync_request` / `reply_sync_request`）。

## 依据
- 入口与配置门禁：`model_server_main.cpp`（约 L33–94）。
- HTTP 与运行时：`base_server_impl.h`（`serve_process`、runtime config、sync/async、metrics）。
- 异步 HTTP 适配：`async_endpoints.h`（`/jobs` 四接口、`drain`）。
- 进程与 Workflow 组装：`abstract_server.h`（`start`/`stop`/`init_http_server` once）。
- 分数 7.7：能支撑小团队经 gateway 交付检测/分割等 HTTP 推理，且有健康检查与指标；未到 9.0，因挂死停服与同进程 Workflow 首启锁定仍会卡住独立排障。

## 未读
无
