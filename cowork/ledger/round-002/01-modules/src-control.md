# 模块评分

- id: src-control
- 权重: 15
- 分数: 9.0
- 读过的路径: `cowork/ledger/rubric.md`；`src/control/supervisor.h`；`src/control/supervisor.cpp`；`src/control/catalog.h`；`src/control/trust_tokens.h`；`src/control/gpu_sampler.h`；`src/control/occupancy_gate.h`；`src/control/restart_policy.h`；`src/control/gateway/gateway_app.h`；`src/control/gateway/gateway_app.cpp`；`src/control/rate_limit/gcra.h`；`src/control/supervisor/supervisor_app.cpp`；`src/control/cli/cli_app.cpp`；`src/control/supervisor/ui/app.js`；`deploy/mortred-supervisor.service`；`test/gateway_e2e_test.cc`；`test/gateway_multiinstance_test.cc`；`test/trust_tokens_unittest.cc`；`test/supervisor_unittest.cc`

## 已具备

- 驻留集按 catalog id 各一个子进程。`stop_server` / `restart_server` 只向该 pid 发信号。`restart_server` 先停再拉起同一个 id。`test/supervisor_unittest.cc` 的 `manual_stop_is_expected_and_not_restarted`、`restart_action_recovers_running_state` 覆盖单 id。gateway 对 `ECONNREFUSED` 回 503，`test/gateway_e2e_test.cc` 的 `dead_upstream_maps_to_503` 锁定这一映射。停一个 id 不会改其他 child 的 `wanted`。
- `GET /api/v1/status` 对每个 child 给出 `state`、`pid`、`ready`、`restart_count`，有退出记录时给出 `last_exit_status`。`mortredctl status` 把 catalog 与这份状态按 id 合成一张表，非 running 时 info 列写出 `last_exit=`。
- 收到 SIGINT/SIGTERM 后，`perform_ordered_shutdown` 先向全部模型发 SIGINT，宽限后 SIGKILL，再停 `__gateway`。`deploy/mortred-supervisor.service` 为 `KillMode=process`、`KillSignal=SIGTERM`、`TimeoutStopSec=120`。模型子进程在 `execve` 前用 `syscall(SYS_prctl, PR_SET_PDEATHSIG, SIGINT)` 绑父进程死亡。
- gateway 监听前走 `tokens_usable_for_listen`：metrics、推理、管理三个 token 都必须存在，且两两不同。不通过则 `GatewayApp::init` 返回 false，进程以非 0 退出。supervisor 拉起 `__gateway` 前做同一检查，失败则永久失败并跳过模型波。`test/trust_tokens_unittest.cc` 覆盖缺推理 token、缺管理 token、二者相同。
- 打开来源限流后，401 在转发前调用 `release`，把这一次准入退回。`test/gateway_multiinstance_test.cc` 的 `wrong_token_preserves_full_burst` 先用错误 token 打 3 次，随后正确 token 仍拿到完整 burst。未鉴权的推理路由得到 401。`/healthz` 在限流之前返回。
- 模型子进程的环境块在 `fork` 之前写入，`execve` 带这份 `envp`。子进程路径是 `dup2`、`close`、`syscall`、`getppid`、`sigaction`、`sigprocmask`、`chdir`、`execve`。`occupancy_gate.h` 与 `gpu_sampler.h` 的 `nvidia-smi` 子进程使用绝对路径 `execve`。
- `nvidia-smi` 无输出时 `available=false`，首次日志为 `gpu sampler offline (no nvidia-smi / no GPU)`。利用率列为 `N/A` 或 `[N/A]` 时利用率为 -1，`available=false`，首次日志为 `gpu sampler offline (utilization N/A)`。`/api/v1/gpu` 带 `available`。HUD 在 `available` 为假时显示 `GPU offline`。
- 不创建 `/tmp/mortred_stage_timing` 时，带顶层 `server_time_ms` 的成功推理仍进入 gateway `/metrics` 的 `gw_in`、`route`、`auth`、`fwd`、`upstream` 直方图，响应体仍带 `server_time_ms`。样本数不超过 4096 且五段 p50 之和大于 `server_time_ms` p50 的一半时，同一份 `/metrics` 写出 `mortred_gateway_stage_dominant`。`StageMetricsVisibleWithoutFlag` 与 `StageMetricsDominantNamesTheLargestStage` 锁定这两端。

## 问题

P2。同一次成功推理超过 4096 次之后，gateway `/metrics` 不再写明时间主要花在哪一段（PF4 的 gateway 阶段）。

- 出处：`src/control/gateway/gateway_app.h` 的 `k_stage_sample_cap = 4096`；`src/control/gateway/gateway_app.cpp` 的 `record_stage_sample`、`render_stage_metrics`。
- 现在的行为：每次成功推理都累加五段直方图，并把样本留在最多 4096 条的 deque 里。`mortred_gateway_stage_dominant` 只在 `stage_observed_ <= 4096` 且五段 p50 之和大于 `server_time_ms` p50 的一半时写出。第 4097 次成功之后，直方图计数继续增加，这行不再出现。
- 一次外部验收即可确认的变化：用会在响应里写很小 `server_time_ms` 的上游，连续 4097 次成功推理，再带 metrics token 读一次 gateway `/metrics`。正文里仍有五段 `mortred_gateway_stage_duration_ms`，并且有 `mortred_gateway_stage_dominant{model="gateway",stage="..."} 1`，`stage` 为这五段里 p50 最大的名字。

## 范围外

- 重复描述，P1-1：不创建 `/tmp/mortred_stage_timing` 时，gateway `/metrics` 读不到阶段耗时（PF4）。
- 重复描述，P2-2：打开限流后，鉴权失败仍消耗来源 IP 的配额（ST2）。
- 重复描述，P2-3：`fork` 之后、`exec` 之前的库函数跟踪里出现 `prctl`、`execvp` 或 `setenv`（ST5）。
- 重复描述，P2-5：推理 token 或管理 token 缺失，或二者相同，gateway 仍会启动（OP4）。
- 重复描述，P2-6：停止信号没有收成只发给 supervisor 主进程（OP5）。
- P0-1、P1-2、P2-1、P2-4、P2-7、P2-8、P2-9、P2-10、P2-11 不落在本模块。PF1、PF5、TP1 的压测报告清单在已关闭的 P2-11，并发数取已部署 `worker_nums` 的两倍不在本模块另开问题。
- 多机调度、Docker 镜像用户、TensorRT 动态 batch、多个 worker 共享一个 execution context，按 rubric 范围外。

## 依据

- 完成标准里落在本模块的 ST2、ST4、ST5、ST6、OP2、OP3、OP4、OP5，以及 PF4 里「不依赖 `/tmp/mortred_stage_timing` 就能读到五段耗时和响应里的 `server_time_ms`」这一段，对得上当前源码和上列测试。
- PF4 的后半句仍有一条未关闭缺口：样本计数超过 4096 后，同一份 `/metrics` 不再点名主要阶段。直方图还在，所以这台机器的入口、单模型故障和进程恢复不受影响，模块可以放上这一台 Linux GPU 服务器。分数停在 9.0。

## 未覆盖

无
