# 模块评分

- id: src-server-apps
- 权重: 15
- 分数: 8.1
- 读过的路径:
  - `src/apps/model_server_main.cpp`
  - `src/server/abstract_server.h`
  - `src/server/base_server_impl.h`
  - `src/server/async_endpoints.h`
  - `src/server/worker_pool.h`
  - `test/worker_pool_unittest.cc`（`drain_timeout_returns_with_remaining_log`）
  - `test/async_job_unittest.cc`（`drain_timeout_returns_with_inflight_log`、`shared_deadline_budget_under_350ms`、`failed_init_destructor_skips_drain`）

## 优势
- 入口与契约清晰：`model_server_main.cpp` 校验 `--model`、catalog、server section；`BaseAiServer` 统一 HTTP 装配；`serve_process` 覆盖 health/ready/metrics/openapi、鉴权与限流、同步推理与 `/jobs` 异步路径。
- 停服 drain 可交付：析构先停 batch，再对 async + worker 共享 `kDrainTimeoutMs=3000` 截止；超时打 `drain timeout` 与剩余数量，失败 init 跳过 drain，避免假死。
- 行为有单测锚点：`worker_pool` / `async_endpoints` 验证超时返回、日志字段与共享 deadline；现场说明 WSL 已通过。

## 劣势
- 超时后析构仍继续：租约未回时成员可先销毁，挂死模型仍依赖外层 `kStopGraceMs` SIGKILL；头文件仍承认 go 抢占微窗口对队列不可见。
- `WorkerPool::drain` 用 `size_approx()` 判齐，高并发下可能误判或晚判。
- `BaseAiServerImpl` 同步 go/timer、batch、async 交织，小团队排障成本仍高。

## 依据
- `worker_pool.h`：`kDrainTimeoutMs`、`drain` 的 `remaining=` 日志。
- `async_endpoints.h`：`drain` 的 `inflight` / `wait_inflight` 超时日志。
- `base_server_impl.h`：析构共享 deadline、失败 init 早退、残余竞态注释。
- `test/worker_pool_unittest.cc`、`test/async_job_unittest.cc`：上述行为与共享预算用例。
- `abstract_server.h`、`model_server_main.cpp`：服务骨架与进程入口。

## 未读
无
