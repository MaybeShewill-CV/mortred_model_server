# 模块评分

- id: src-control
- 权重: 15
- 分数: 7.6
- 读过的路径: `src/control/supervisor.h`、`src/control/supervisor.cpp`、`src/control/catalog.h`、`src/control/trust_tokens.h`、`src/control/restart_policy.h`、`src/control/gpu_sampler.h`、`src/control/occupancy_gate.h`、`src/control/rate_limit/gcra.h`、`src/control/supervisor/supervisor_app.cpp`、`src/control/gateway/gateway_app.cpp`、`src/control/cli/cli_app.cpp`、`src/common/stage_timing.h`、`conf/mortred.toml`、`test/trust_tokens_unittest.cc`、`test/supervisor_unittest.cc`、`test/gateway_e2e_test.cc`

## 已具备

- 子进程环境在 `fork` 之前写进 `execve` 的环境块。`spawn_locked` 注入 `MORTRED_WORKER_NUMS`、`MORTRED_DECODE_AUTO`、内部 token 和 loopback 监听。出处：`src/control/supervisor.cpp`。
- 每个 id 的状态是 `stopped` / `starting` / `running` / `backoff` / `failed`。退出后 `last_exit_status` 进入状态对象；`GET` 状态接口按 id 返回。`restart_server` 只停、再起这一个 id。出处：`src/control/supervisor.cpp`、`src/control/supervisor/supervisor_app.cpp`、`src/control/cli/cli_app.cpp`。
- 收到 `SIGINT` / `SIGTERM` 后，先向模型发 `SIGINT`，宽限后 `SIGKILL`，再停 gateway。崩溃按 `on-failure` / `always` 退避；60 秒内超过 5 次重启进入 `failed` 并不再自动拉起。出处：`src/control/supervisor.cpp`、`src/control/restart_policy.h`。
- metrics token 为空，或与推理 token、管理 token 相同，gateway 启动失败，supervisor 把 `__gateway` 记成永久失败。出处：`src/control/trust_tokens.h`、`src/control/gateway/gateway_app.cpp`、`test/trust_tokens_unittest.cc`、`test/supervisor_unittest.cc`。
- 未鉴权的推理路由返回 401，不转发。上游 `ECONNREFUSED` 返回 503。`/healthz` 在鉴权之前返回。出处：`src/control/gateway/gateway_app.cpp`、`test/gateway_e2e_test.cc`。
- gateway 按 catalog 的 `server_uri` 转发，不按模型写死路由。出处：`src/control/catalog.h`、`src/control/gateway/gateway_app.cpp`。
- `nvidia-smi` 失败或 utilization 为 `N/A` 时，`available` 为 false，利用率记为 -1，首次日志写 `offline`。`/api/v1/gpu` 带 `available`。出处：`src/control/gpu_sampler.h`、`src/control/supervisor/supervisor_app.cpp`。

## 问题

1. 完成标准 OP4。出处：`src/control/trust_tokens.h`、`src/control/gateway/gateway_app.cpp`、`test/trust_tokens_unittest.cc`。现在的行为：`scrape_token_usable` 只拒绝空的 metrics token，以及 metrics token 与推理 token 或管理 token 相同。推理 token 为空且管理 token 与 metrics token 不同时检查通过；管理 token 为空且推理 token 与 metrics token 不同时同样通过。推理 token 与管理 token 相同、metrics token 为第三个值时也通过。gateway 在「推理 token 或管理 token 至少有一个」或已加载 API key 文件时继续监听。用 `MORTRED_METRICS_TOKEN` 与 `MORTRED_API_TOKEN` 两个不同值、不设置 `MORTRED_GATEWAY_AUTH_TOKEN`、且没有 API key 文件启动 gateway，进程会监听。改完后：上述启动退出非 0，不监听；推理 token 与管理 token 设成同一个值、metrics token 为另一个值时同样拒绝启动。

2. 完成标准 ST2。出处：`src/control/gateway/gateway_app.cpp`、`src/control/rate_limit/gcra.h`、`conf/mortred.toml`。现在的行为：`[gateway.rate_limit] enabled = false` 时不计量。`MORTRED_RATE_LIMIT=enforce` 或把该开关打开后，`check_ip_rate_limit` 在路由和鉴权之前调用 `gcra_admit`，允许的请求会推进 TAT。随后鉴权失败仍返回 401，但这部分请求已经消耗来源 IP 的配额。同一 IP 用错误 token 打满 burst 后，带正确 token 的请求收到 429 和 `Retry-After`。按 key 的配额只在 key 已允许推理之后扣减。改完后：打开同一限流开关，用错误 token 打满原先的 burst 次数，同一 IP 紧接着的合法推理仍然成功，401 不改变该来源的剩余配额。

3. 完成标准 ST5。出处：`src/control/supervisor.cpp`、`src/control/occupancy_gate.h`、`src/control/gpu_sampler.h`。现在的行为：拉起子进程时，`fork` 之后、`execve` 之前调用 `prctl(PR_SET_PDEATHSIG)`。`prctl` 不在 POSIX async-signal-safe 列表中。supervisor 进程里探测 `nvidia-smi` 的子进程在 `fork` 之后调用 `execvp`；`execvp` 同样不在该列表中，可能分配内存。这段路径没有调用 `setenv`。改完后：上述三处子进程在 `exec` 之前只保留 async-signal-safe 调用；`nvidia-smi` 改为带绝对路径的 `execve`。对照这三处子进程代码即可确认 `prctl` 和 `execvp` 已不在 `fork` 与 `exec` 之间。

4. 完成标准 PF4 中 gateway 阶段耗时这一半。出处：`src/control/gateway/gateway_app.cpp`、`src/common/stage_timing.h`。现在的行为：gateway `/metrics` 只有整段 HTTP 耗时。`gw_in`、`route`、`auth`、`fwd`、`upstream` 只在存在 `/tmp/mortred_stage_timing` 时写入 stderr。按本次性能约束做的 `http_infer_rps.py --raw` 读 `/metrics` 时看不到这些阶段。模型响应里的 `server_time_ms` 不在本模块。改完后：不创建该 flag 文件，对一次推理抓取 gateway `/metrics`，能读到上述阶段的耗时；同一次压测可以直接引用这些数。

## 范围外

- 一个 catalog id 一个 OS 进程；停止时宽限后 `SIGKILL`。
- 多机调度、Docker 镜像用户、TensorRT 动态 batch、多个 worker 共享一个 execution context。
- 覆盖集以外的 catalog id。
- MNN 与 ONNX Runtime 的吞吐和延迟。
- `mortredctl doctor --strict` 的退出码在 `scripts` 侧，本模块只在拉起时调用占用门闩。
- 缺少按「本次性能约束」完成的 HTTP RPS 报告本身，不记在本模块。

## 依据

- 分数对照的是这台机器上的监督面：单 id 状态、单独重启、有序停止、坏 token 不进推理、死上游 503、GPU 采样标明 offline，这些在上列文件和测试里对得上。
- OP4、ST2、ST5、PF4 四条在本模块的代码里未满足，所以不到 8.0。监督主路径已经能单独拉起和停掉模型，所以高于 7.0。
- 本次没有跑压测，也没有把缺 GPU 运行写成阻塞。PF4 的未满足来自 gateway 指标面，不来自本次是否执行 GPU 命令。

## 未覆盖

无
