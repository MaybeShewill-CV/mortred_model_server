# 模块评分

- id: src-server-apps
- 权重: 15
- 分数: 9.3
- 读过的路径: `src/apps/model_server_main.cpp`, `src/server/abstract_server.h`, `src/server/base_server_impl.h`, `src/server/async_endpoints.h`

## 优势
- 单进程按 catalog id 启动：`--list` / `--model`、未知模型、bench-only、TOML 缺段、`model` 字段与 `--model` 不一致均在启动前 stderr 退出，小团队能按文案改配置（`model_server_main.cpp`）。
- HTTP 推理面覆盖交付常用能力：Bearer 先于 IP QPS、`/healthz` `/ready` `/metrics` `/openapi.json`、同步 POST（JSON/raw）、413/415/429（含 Retry-After）、统一拒绝与 Prometheus，可按状态码与指标定位（`base_server_impl.h` `serve_process`）。
- 异步任务生命周期完整：`POST /jobs` → 202+Location、status / wait / result，队列满 429、未完成 409，与 worker checkout、metrics 挂钩（`async_endpoints.h`；`base_server_impl.h` `async_run_job` / `wire_async_endpoints`）。
- 停服可观测：析构先 `_m_batch.stop()`，再在 `kDrainTimeoutMs` 共享 deadline 上 drain async/worker，超时各路径打 `drain timeout`，随后 `server members destroying after drain`，与外层 SIGKILL 约定写在注释里（`base_server_impl.h` 析构；`async_endpoints.h` `drain`）。

## 劣势
无

## 不计毛病
- 一个 catalog id 一个 OS 进程；超时 drain 之后析构成员，再由外层 SIGKILL（`base_server_impl.h` 析构注释与 `kDrainTimeoutMs` 预算；`async_endpoints.h` `drain` 超时返回）。

## 依据
- 入口与配置门禁：`model_server_main.cpp`（CLI 解析、ProductIndex、TOML 段校验）。
- HTTP 与运行时：`base_server_impl.h`（`serve_process`、sync/async/batch、metrics、有界 drain + destroying 日志）。
- 异步 HTTP 适配：`async_endpoints.h`（`/jobs` 四接口、有界 `drain`）。
- 进程与 Workflow 组装：`abstract_server.h`（`start`/`stop`/`wait_finish`、`init_http_server` 的 `call_once`）。
- 分数 9.3：小团队可按 catalog 独立起进程交付检测/分割等 HTTP 推理，并靠健康检查、指标与可选 stage 埋点定位；同进程多 server 的 Workflow 首启锁定在一进程一模型交付模型下不触发，不计入劣势。旧「无界 drain / 缺 destroying 日志」已不成立。

## 未读
无
